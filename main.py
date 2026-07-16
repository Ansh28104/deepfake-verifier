"""Entry point for the Deepfake Verifier system."""

from __future__ import annotations

from pathlib import Path
import time

import cv2

from config import load_config
from core.blink_detector import BlinkDetector
from core.camera import CameraStream
from core.face_detector import FaceDetector
from core.fusion_engine import ANALYZING, RESET, VERDICT, FusionEngine
from core.model_inference import ModelInferenceEngine
from display.renderer import Renderer


def _safe_landmark(landmarks_norm, idx: int):
	if landmarks_norm is None or idx >= len(landmarks_norm):
		return None
	return landmarks_norm[idx]


def _estimate_head_turn_signal(landmarks_norm) -> float | None:
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


def _estimate_mouth_open_signal(landmarks_norm) -> float | None:
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


def _estimate_nose_motion_signal(landmarks_norm, prev_nose_xy: tuple[float, float] | None):
	nose = _safe_landmark(landmarks_norm, 1)
	if nose is None:
		return 0.0, prev_nose_xy
	current = (float(nose[0]), float(nose[1]))
	if prev_nose_xy is None:
		return 0.0, current
	dx = current[0] - prev_nose_xy[0]
	dy = current[1] - prev_nose_xy[1]
	# Normalized movement in landmark space.
	motion = min(1.0, ((dx * dx + dy * dy) ** 0.5) * 25.0)
	return motion, current


def run() -> None:
	print("[Startup] Loading configuration...")
	cfg = load_config()
	switch_popup_text: str | None = None
	switch_popup_until: float = 0.0
	debug_overlay: bool = False
	prev_nose_xy: tuple[float, float] | None = None

	screenshots_dir = Path("assets") / "screenshots"
	screenshots_dir.mkdir(parents=True, exist_ok=True)

	camera = None
	face_detector = None
	model_engine = None

	print("[Startup] Initializing model inference engine...")
	model_engine = ModelInferenceEngine(
		mode=cfg.mode,
		frame_buffer_size=cfg.frame_buffer_size,
		light_checkpoint_path=cfg.light_checkpoint_path,
		heavy_checkpoint_path=cfg.heavy_checkpoint_path,
		real_class_index=cfg.real_class_index,
		fake_class_index=cfg.fake_class_index,
	)
	print(f"[Startup] {model_engine.model_status()}")
	if cfg.require_finetuned_weights and not model_engine.using_finetuned:
		print("[Startup] require_finetuned_weights=True but no fine-tuned checkpoint was loaded.")
		print("[Startup] Add checkpoints in models/ and restart.")
		return

	print("[Startup] Initializing face detector...")
	face_detector = FaceDetector(
		max_num_faces=1,
		min_detection_confidence=cfg.mp_min_detection_confidence,
		min_tracking_confidence=cfg.mp_min_tracking_confidence,
	)
	print("[Startup] Initializing blink detector...")
	blink_detector = BlinkDetector(ear_threshold=cfg.ear_threshold)
	print("[Startup] Initializing fusion engine...")
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
	print("[Startup] Initializing renderer...")
	renderer = Renderer(mode=cfg.mode)
	print("[Startup] Opening camera stream...")
	camera = CameraStream(camera_index=cfg.camera_index, unmirror=cfg.camera_unmirror)
	if not getattr(camera, "is_opened", lambda: True)():
		print(
			f"[Startup] Camera could not be opened at index {cfg.camera_index}. "
			"Check the device index, permissions, or another app using the webcam."
		)
		if camera is not None:
			camera.release()
		if face_detector is not None:
			face_detector.close()
		cv2.destroyAllWindows()
		return
	print("[Startup] System ready.")

	try:
		while True:
			ok, frame = camera.read()
			if not ok or frame is None:
				if camera.too_many_failures():
					print("Camera error: too many capture failures.")
					break
				continue

			landmarks = face_detector.detect(frame)
			face_present = landmarks is not None

			deepfake_score = None
			blink_score = None
			blink_count = None
			head_turn_signal = None
			mouth_open_signal = None
			blink_metrics = None
			can_run_expensive = face_present
			if can_run_expensive:
				if len(landmarks.landmarks_norm) >= 388:
					blink_metrics = blink_detector.process(landmarks.landmarks_norm)
					blink_count = blink_metrics.blink_count
					head_turn_signal = _estimate_head_turn_signal(landmarks.landmarks_norm)
					mouth_open_signal = _estimate_mouth_open_signal(landmarks.landmarks_norm)
					nose_motion_signal, prev_nose_xy = _estimate_nose_motion_signal(
						landmarks.landmarks_norm,
						prev_nose_xy,
					)

					head_motion = min(1.0, abs(head_turn_signal) * 4.0) if head_turn_signal is not None else 0.0
					mouth_motion = min(1.0, (mouth_open_signal / 0.14)) if mouth_open_signal is not None else 0.0
					motion_activity = (head_motion + mouth_motion + nose_motion_signal) / 3.0
					motion_risk = 1.0 - motion_activity

					blink_score = (
						0.45 * blink_metrics.blink_pattern_score
						+ 0.30 * blink_metrics.passive_motion_score
						+ 0.25 * motion_risk
					)
				deepfake_score = model_engine.infer_frame(frame)
			else:
				prev_nose_xy = None

			verdict_pkg = fusion_engine.update(
				now_ts=time.time(),
				face_present=face_present,
				deepfake_score=deepfake_score,
				blink_pattern_score=blink_score,
				blink_count=blink_count,
				head_turn_signal=head_turn_signal,
				mouth_open_signal=mouth_open_signal,
			)
			now_ts = time.time()
			active_popup = switch_popup_text if now_ts < switch_popup_until else None
			debug_lines = None
			if debug_overlay:
				buffer_count = len(model_engine.frame_buffer)
				buffer_max = model_engine.frame_buffer.maxlen
				debug_lines = [
					f"Face present: {face_present}",
					f"Deepfake score: {deepfake_score if deepfake_score is not None else 'n/a'}",
					f"Blink score: {blink_score if blink_score is not None else 'n/a'}",
					f"Challenge: {getattr(verdict_pkg, 'challenge_progress', 'n/a')}",
					f"Frame buffer: {buffer_count}/{buffer_max}",
				]
				if blink_metrics is not None:
					debug_lines.append(
						f"EAR: {blink_metrics.ear:0.3f}, blinks: {blink_metrics.blink_count}, "
						f"head: {head_turn_signal if head_turn_signal is not None else 'n/a'}"
					)

			annotated = renderer.render(
				frame,
				verdict_pkg,
				detection=landmarks,
				warning=model_engine.last_warning,
				strict_mode=cfg.require_finetuned_weights,
				popup_message=active_popup,
				debug_overlay=debug_overlay,
				debug_lines=debug_lines,
			)
			cv2.imshow("Deepfake Verifier", annotated)

			key = cv2.waitKey(1) & 0xFF
			if key in (ord("q"), ord("Q")):
				break
			if key in (ord("m"), ord("M")):
				target_mode = "heavy" if cfg.mode == "light" else "light"
				previous_mode = cfg.mode
				model_engine.switch_mode(target_mode)

				if cfg.require_finetuned_weights and not model_engine.using_finetuned:
					print(
						"[ModeSwitch] Blocked: require_finetuned_weights=True and "
						f"target mode '{target_mode}' has no valid fine-tuned checkpoint."
					)
					switch_popup_text = "SWITCH BLOCKED: strict mode requires valid fine-tuned checkpoint"
					switch_popup_until = time.time() + 2.0
					model_engine.switch_mode(previous_mode)
					cfg.mode = previous_mode
				else:
					cfg.mode = target_mode
					print(f"[ModeSwitch] {model_engine.model_status()}")
					renderer.set_mode(cfg.mode)
			if key in (ord("r"), ord("R")):
				fusion_engine.reset()
			if key in (ord("s"), ord("S")):
				timestamp = time.strftime("%Y%m%d-%H%M%S")
				screenshot_path = screenshots_dir / f"deepfake-verifier-{timestamp}.png"
				if cv2.imwrite(str(screenshot_path), annotated):
					print(f"[Capture] Saved screenshot: {screenshot_path}")
					switch_popup_text = f"SCREENSHOT SAVED: {screenshot_path.name}"
					switch_popup_until = time.time() + 2.0
				else:
					print("[Capture] Failed to save screenshot")
			if key in (ord("d"), ord("D")):
				debug_overlay = not debug_overlay
				state_text = "ON" if debug_overlay else "OFF"
				print(f"[Debug] Overlay {state_text}")
				switch_popup_text = f"DEBUG OVERLAY {state_text}"
				switch_popup_until = time.time() + 2.0
	finally:
		if camera is not None:
			camera.release()
		if face_detector is not None:
			face_detector.close()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	run()
