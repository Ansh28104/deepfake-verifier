"""Replay-attack harness for repeatable manual verification.

Manifest JSON example:
{
  "scenarios": [
    {"name": "real_baseline", "video": "dataset/test/real/sample.mp4", "expected": "VERIFIED"},
    {"name": "replay_attack", "video": "dataset/test/fake/replay.mp4", "expected": "REJECTED"}
  ]
}

Usage:
    python scripts/replay_attack_harness.py --manifest docs/replay_manifest.sample.json --mode light
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from core.blink_detector import BlinkDetector
from core.face_detector import FaceDetector
from core.fusion_engine import FusionEngine
from core.model_inference import ModelInferenceEngine


@dataclass
class ScenarioResult:
    name: str
    video: str
    expected: str
    actual: str
    risk_score: float
    frames: int
    status: str
    reason: str = ""


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


def _load_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", [])
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("manifest must contain non-empty 'scenarios' list")
    return scenarios


def _run_scenario(
    cfg,
    model_engine: ModelInferenceEngine,
    face_detector: FaceDetector,
    scenario: dict,
    max_frames: int,
) -> ScenarioResult:
    name = str(scenario.get("name", "unnamed"))
    video = str(scenario.get("video", ""))
    expected = str(scenario.get("expected", "SUSPICIOUS")).upper()
    video_path = Path(video)

    if not video_path.exists():
        return ScenarioResult(
            name=name,
            video=video,
            expected=expected,
            actual="SKIPPED",
            risk_score=50.0,
            frames=0,
            status="SKIPPED",
            reason="video file missing",
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

    cap = cv2.VideoCapture(str(video_path))
    frames = 0
    verdict = None
    risk_score = 50.0
    try:
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            detection = face_detector.detect(frame)
            face_present = detection is not None

            deepfake_score = None
            blink_score = None
            blink_count = None
            head_turn_signal = None
            mouth_open_signal = None

            if face_present:
                deepfake_score = model_engine.infer_frame(frame)
                if len(detection.landmarks_norm) >= 388:
                    metrics = blink_detector.process(detection.landmarks_norm)
                    blink_score = metrics.blink_pattern_score
                    blink_count = metrics.blink_count
                    head_turn_signal = _estimate_head_turn_signal(detection.landmarks_norm)
                    mouth_open_signal = _estimate_mouth_open_signal(detection.landmarks_norm)

            pkg = fusion_engine.update(
                now_ts=time.time(),
                face_present=face_present,
                deepfake_score=deepfake_score,
                blink_pattern_score=blink_score,
                blink_count=blink_count,
                head_turn_signal=head_turn_signal,
                mouth_open_signal=mouth_open_signal,
            )
            frames += 1
            risk_score = float(pkg.risk_score)
            if pkg.verdict is not None:
                verdict = str(pkg.verdict)
                break
    finally:
        cap.release()

    actual = verdict or "NO_VERDICT"
    status = "PASS" if actual == expected else "FAIL"
    reason = ""
    if actual == "NO_VERDICT":
        reason = "analysis did not reach verdict in allotted frames"

    return ScenarioResult(
        name=name,
        video=video,
        expected=expected,
        actual=actual,
        risk_score=risk_score,
        frames=frames,
        status=status,
        reason=reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay-attack harness")
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--mode", choices=["light", "heavy"], default="light")
    parser.add_argument("--max-frames", type=int, default=450)
    parser.add_argument("--report-json", type=str, default="docs/replay_harness_report.json")
    parser.add_argument("--fail-on-mismatch", action="store_true")
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
    face_detector = FaceDetector(
        max_num_faces=1,
        min_detection_confidence=cfg.mp_min_detection_confidence,
        min_tracking_confidence=cfg.mp_min_tracking_confidence,
    )

    manifest_path = Path(args.manifest)
    scenarios = _load_manifest(manifest_path)

    results: list[ScenarioResult] = []
    try:
        for scenario in scenarios:
            result = _run_scenario(
                cfg=cfg,
                model_engine=model_engine,
                face_detector=face_detector,
                scenario=scenario,
                max_frames=max(1, args.max_frames),
            )
            results.append(result)
            print(
                f"[{result.status}] {result.name}: expected={result.expected} actual={result.actual} "
                f"risk={result.risk_score:.1f} frames={result.frames}"
            )
            if result.reason:
                print(f"  reason={result.reason}")
    finally:
        face_detector.close()

    pass_count = sum(1 for r in results if r.status == "PASS")
    fail_count = sum(1 for r in results if r.status == "FAIL")
    skipped_count = sum(1 for r in results if r.status == "SKIPPED")

    summary = {
        "mode": args.mode,
        "manifest": str(manifest_path),
        "total": len(results),
        "pass": pass_count,
        "fail": fail_count,
        "skipped": skipped_count,
        "results": [r.__dict__ for r in results],
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Replay Harness Summary")
    print(f"- total: {summary['total']}")
    print(f"- pass: {summary['pass']}")
    print(f"- fail: {summary['fail']}")
    print(f"- skipped: {summary['skipped']}")
    print(f"- report: {report_path}")

    if args.fail_on_mismatch and fail_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
