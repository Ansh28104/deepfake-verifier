from __future__ import annotations

from dataclasses import dataclass

import main


@dataclass
class _Cfg:
    mode: str = "light"
    analysis_duration: float = 1.0
    face_hold_duration: float = 0.0
    waiting_face_gap_tolerance: float = 0.1
    analyzing_face_absence_tolerance: float = 0.1
    verified_threshold: float = 35.0
    rejected_threshold: float = 66.0
    camera_index: int = 0
    camera_unmirror: bool = True
    frame_buffer_size: int = 5
    ear_threshold: float = 0.2
    log_level: str = "debug"
    light_checkpoint_path: str = "models/efficientnet_b4_finetuned.pth"
    heavy_checkpoint_path: str = "models/vit_b16_finetuned.pth"
    require_finetuned_weights: bool = False
    mp_min_detection_confidence: float = 0.35
    mp_min_tracking_confidence: float = 0.35
    real_class_index: int = 0
    fake_class_index: int = 1
    liveness_challenge_enabled: bool = True
    challenge_timeout: float = 4.0
    challenge_required_successes: int = 2
    challenge_weight: float = 0.35
    deepfake_weight: float = 0.55
    blink_weight: float = 0.45
    risk_smoothing_alpha: float = 0.35


@dataclass
class _VerdictPackage:
    state: str = "WAITING"
    risk_score: float = 50.0
    time_remaining: float = 0.0
    verdict: str | None = None
    deepfake_score: float = 0.5
    blink_score: float = 0.5


class _FakeCamera:
    def __init__(self, camera_index: int = 0, unmirror: bool = False) -> None:
        self.camera_index = camera_index
        self.unmirror = unmirror
        self.read_calls = 0
        self._failures = 0
        self.released = False
        self.opened = True

    def is_opened(self) -> bool:
        return self.opened

    def read(self):
        self.read_calls += 1
        self._failures += 1
        return False, None

    def too_many_failures(self, max_failures: int = 10) -> bool:
        return self._failures >= max_failures

    def release(self) -> None:
        self.released = True


class _FakeFaceDetector:
    def __init__(self, *args, **kwargs) -> None:
        self.closed = False

    def detect(self, frame):
        return None

    def close(self) -> None:
        self.closed = True


class _FakeModelEngine:
    def __init__(self, *args, **kwargs) -> None:
        self.using_finetuned = False
        self.last_warning = None

    def model_status(self) -> str:
        return "mode=light, source=base pretrained"

    def infer_frame(self, frame):
        return 0.5

    def switch_mode(self, mode: str) -> None:
        return None


class _FakeFusionEngine:
    def __init__(self, *args, **kwargs) -> None:
        self.reset_calls = 0

    def update(self, **kwargs):
        return _VerdictPackage()

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeRenderer:
    def __init__(self, mode: str = "light") -> None:
        self.mode = mode

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def render(self, frame, *args, **kwargs):
        return frame


def test_main_run_exits_and_cleans_up_on_camera_drop(monkeypatch) -> None:
    cfg = _Cfg()
    camera_holder = {}
    face_detector_holder = {}

    def _camera_factory(*args, **kwargs):
        cam = _FakeCamera(*args, **kwargs)
        camera_holder["cam"] = cam
        return cam

    def _face_detector_factory(*args, **kwargs):
        detector = _FakeFaceDetector(*args, **kwargs)
        face_detector_holder["detector"] = detector
        return detector

    monkeypatch.setattr(main, "load_config", lambda: cfg)
    monkeypatch.setattr(main, "CameraStream", _camera_factory)
    monkeypatch.setattr(main, "FaceDetector", _face_detector_factory)
    monkeypatch.setattr(main, "BlinkDetector", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "ModelInferenceEngine", _FakeModelEngine)
    monkeypatch.setattr(main, "FusionEngine", _FakeFusionEngine)
    monkeypatch.setattr(main, "Renderer", _FakeRenderer)

    monkeypatch.setattr(main.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.cv2, "waitKey", lambda *_args, **_kwargs: -1)
    monkeypatch.setattr(main.cv2, "destroyAllWindows", lambda: None)

    main.run()

    assert "cam" in camera_holder
    assert camera_holder["cam"].read_calls >= 10
    assert camera_holder["cam"].released is True

    assert "detector" in face_detector_holder
    assert face_detector_holder["detector"].closed is True


def test_main_run_exits_cleanly_when_camera_never_opens(monkeypatch) -> None:
    cfg = _Cfg()
    camera_holder = {}
    face_detector_holder = {}

    class _ClosedCamera(_FakeCamera):
        def __init__(self, camera_index: int = 0, unmirror: bool = False) -> None:
            super().__init__(camera_index=camera_index, unmirror=unmirror)
            self.opened = False

    def _camera_factory(*args, **kwargs):
        cam = _ClosedCamera(*args, **kwargs)
        camera_holder["cam"] = cam
        return cam

    def _face_detector_factory(*args, **kwargs):
        detector = _FakeFaceDetector(*args, **kwargs)
        face_detector_holder["detector"] = detector
        return detector

    monkeypatch.setattr(main, "load_config", lambda: cfg)
    monkeypatch.setattr(main, "CameraStream", _camera_factory)
    monkeypatch.setattr(main, "FaceDetector", _face_detector_factory)
    monkeypatch.setattr(main, "BlinkDetector", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "ModelInferenceEngine", _FakeModelEngine)
    monkeypatch.setattr(main, "FusionEngine", _FakeFusionEngine)
    monkeypatch.setattr(main, "Renderer", _FakeRenderer)

    monkeypatch.setattr(main.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.cv2, "waitKey", lambda *_args, **_kwargs: -1)
    monkeypatch.setattr(main.cv2, "destroyAllWindows", lambda: None)

    main.run()

    assert "cam" in camera_holder
    assert camera_holder["cam"].read_calls == 0
    assert camera_holder["cam"].released is True

    assert "detector" in face_detector_holder
    assert face_detector_holder["detector"].closed is True
