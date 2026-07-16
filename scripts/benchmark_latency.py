"""Benchmark runtime latency for deepfake inference and optional detection pipeline.

Usage examples:
    python scripts/benchmark_latency.py --mode light --frames 120
    python scripts/benchmark_latency.py --mode heavy --video dataset/test/real/sample.mp4 --frames 150 --include-face
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from core.blink_detector import BlinkDetector
from core.face_detector import FaceDetector
from core.fusion_engine import FusionEngine
from core.model_inference import ModelInferenceEngine


def _safe_landmark(landmarks_norm, idx: int):
    if landmarks_norm is None or idx >= len(landmarks_norm):
        return None
    return landmarks_norm[idx]


def _estimate_head_turn_signal(landmarks_norm):
    left_eye = _safe_landmark(landmarks_norm, 33)
    right_eye = _safe_landmark(landmarks_norm, 263)
    nose = _safe_landmark(landmarks_norm, 1)
    if left_eye is None or right_eye is None or nose is None:
        return None
    den = float(right_eye[0]) - float(left_eye[0])
    if abs(den) < 1e-6:
        return None
    mid = (float(left_eye[0]) + float(right_eye[0])) * 0.5
    return (float(nose[0]) - mid) / den


def _estimate_mouth_open_signal(landmarks_norm):
    upper_lip = _safe_landmark(landmarks_norm, 13)
    lower_lip = _safe_landmark(landmarks_norm, 14)
    left_mouth = _safe_landmark(landmarks_norm, 78)
    right_mouth = _safe_landmark(landmarks_norm, 308)
    if upper_lip is None or lower_lip is None or left_mouth is None or right_mouth is None:
        return None
    width = abs(float(right_mouth[0]) - float(left_mouth[0]))
    if width < 1e-6:
        return None
    height = abs(float(lower_lip[1]) - float(upper_lip[1]))
    return height / width


def _iter_frames(video_path: Path | None, max_frames: int):
    if video_path is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(max_frames):
            yield frame
        return

    cap = cv2.VideoCapture(str(video_path))
    try:
        count = 0
        while count < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield frame
            count += 1
    finally:
        cap.release()


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((len(s) - 1) * q))
    return s[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark latency for runtime pipeline")
    parser.add_argument("--mode", choices=["light", "heavy"], default="light")
    parser.add_argument("--video", type=str, default=None, help="Optional input video path")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--include-face", action="store_true", help="Include face detection/blink/fusion in timing")
    args = parser.parse_args()

    cfg = load_config()
    cfg.mode = args.mode

    model_engine = ModelInferenceEngine(
        mode=cfg.mode,
        frame_buffer_size=cfg.frame_buffer_size,
        light_checkpoint_path=cfg.light_checkpoint_path,
        heavy_checkpoint_path=cfg.heavy_checkpoint_path,
        real_class_index=cfg.real_class_index,
        fake_class_index=cfg.fake_class_index,
    )

    face_detector = None
    blink_detector = None
    fusion_engine = None
    if args.include_face:
        face_detector = FaceDetector(
            max_num_faces=1,
            min_detection_confidence=cfg.mp_min_detection_confidence,
            min_tracking_confidence=cfg.mp_min_tracking_confidence,
        )
        blink_detector = BlinkDetector(ear_threshold=cfg.ear_threshold)
        fusion_engine = FusionEngine(
            analysis_duration=cfg.analysis_duration,
            face_hold_duration=cfg.face_hold_duration,
            waiting_face_gap_tolerance=cfg.waiting_face_gap_tolerance,
            analyzing_face_absence_tolerance=cfg.analyzing_face_absence_tolerance,
            verified_threshold=cfg.verified_threshold,
            rejected_threshold=cfg.rejected_threshold,
            challenge_enabled=cfg.liveness_challenge_enabled,
            challenge_timeout=cfg.challenge_timeout,
            challenge_required_successes=cfg.challenge_required_successes,
            challenge_weight=cfg.challenge_weight,
            deepfake_weight=cfg.deepfake_weight,
            blink_weight=cfg.blink_weight,
            risk_smoothing_alpha=cfg.risk_smoothing_alpha,
        )

    video_path = Path(args.video) if args.video else None
    if video_path is not None and not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    samples_ms: list[float] = []
    total_frames = 0

    try:
        for frame in _iter_frames(video_path, max_frames=max(1, args.frames)):
            total_frames += 1
            t0 = time.perf_counter()

            deepfake_score = model_engine.infer_frame(frame)

            if args.include_face and face_detector is not None and fusion_engine is not None and blink_detector is not None:
                detection = face_detector.detect(frame)
                face_present = detection is not None
                blink_score = None
                blink_count = None
                head_turn_signal = None
                mouth_open_signal = None
                if detection is not None and len(detection.landmarks_norm) >= 388:
                    metrics = blink_detector.process(detection.landmarks_norm)
                    blink_score = metrics.blink_pattern_score
                    blink_count = metrics.blink_count
                    head_turn_signal = _estimate_head_turn_signal(detection.landmarks_norm)
                    mouth_open_signal = _estimate_mouth_open_signal(detection.landmarks_norm)

                fusion_engine.update(
                    now_ts=time.time(),
                    face_present=face_present,
                    deepfake_score=deepfake_score,
                    blink_pattern_score=blink_score,
                    blink_count=blink_count,
                    head_turn_signal=head_turn_signal,
                    mouth_open_signal=mouth_open_signal,
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if total_frames > args.warmup:
                samples_ms.append(elapsed_ms)
    finally:
        if face_detector is not None:
            face_detector.close()

    if not samples_ms:
        raise RuntimeError("No benchmark samples collected. Increase --frames or lower --warmup.")

    avg = statistics.mean(samples_ms)
    med = statistics.median(samples_ms)
    p95 = _pct(samples_ms, 0.95)
    p99 = _pct(samples_ms, 0.99)
    fps = 1000.0 / avg if avg > 0 else 0.0

    print("Latency Benchmark")
    print(f"- mode: {args.mode}")
    print(f"- include_face: {args.include_face}")
    print(f"- samples: {len(samples_ms)}")
    print(f"- avg_ms: {avg:.2f}")
    print(f"- median_ms: {med:.2f}")
    print(f"- p95_ms: {p95:.2f}")
    print(f"- p99_ms: {p99:.2f}")
    print(f"- est_fps: {fps:.2f}")


if __name__ == "__main__":
    main()
