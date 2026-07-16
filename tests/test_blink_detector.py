from __future__ import annotations

from core.blink_detector import BlinkDetector


def _eye_points_for_ear(ear: float):
    return [
        [0.0, 0.0],
        [0.0, ear / 2.0],
        [0.5, ear / 2.0],
        [1.0, 0.0],
        [0.5, -ear / 2.0],
        [0.0, -ear / 2.0],
    ]


def _dummy_landmarks() -> list[list[float]]:
    return [[0.0, 0.0, 0.0] for _ in range(400)]


def test_missing_landmarks_returns_neutral() -> None:
    detector = BlinkDetector(ear_threshold=0.2)
    m = detector.process(None)
    assert m.ear == 0.0
    assert m.blink_pattern_score == 0.5
    assert m.passive_motion_score == 0.5


def test_blink_pattern_threshold_bands(monkeypatch) -> None:
    detector = BlinkDetector(ear_threshold=0.2)

    def _extract_eye(_landmarks, _indices):
        return _eye_points_for_ear(0.14)

    monkeypatch.setattr(detector, "_extract_eye", _extract_eye)
    m = detector.process(_dummy_landmarks())
    assert m.blink_pattern_score == 0.8

    def _extract_eye_mid(_landmarks, _indices):
        return _eye_points_for_ear(0.18)

    monkeypatch.setattr(detector, "_extract_eye", _extract_eye_mid)
    m = detector.process(_dummy_landmarks())
    assert m.blink_pattern_score == 0.6

    def _extract_eye_high(_landmarks, _indices):
        return _eye_points_for_ear(0.24)

    monkeypatch.setattr(detector, "_extract_eye", _extract_eye_high)
    m = detector.process(_dummy_landmarks())
    assert m.blink_pattern_score == 0.2


def test_blink_count_increments_after_consecutive_closed_frames(monkeypatch) -> None:
    detector = BlinkDetector(ear_threshold=0.2, consecutive_frames=2)

    def _extract_closed(_landmarks, _indices):
        return _eye_points_for_ear(0.12)

    def _extract_open(_landmarks, _indices):
        return _eye_points_for_ear(0.24)

    monkeypatch.setattr(detector, "_extract_eye", _extract_closed)
    detector.process(_dummy_landmarks())
    detector.process(_dummy_landmarks())

    monkeypatch.setattr(detector, "_extract_eye", _extract_open)
    m = detector.process(_dummy_landmarks())

    assert m.blink_count == 1


def test_passive_motion_score_penalizes_static_ear(monkeypatch) -> None:
    detector = BlinkDetector(ear_threshold=0.2)

    def _extract_static(_landmarks, _indices):
        return _eye_points_for_ear(0.24)

    monkeypatch.setattr(detector, "_extract_eye", _extract_static)
    for _ in range(4):
        m = detector.process(_dummy_landmarks())

    assert m.passive_motion_score == 0.75

    ears = iter([0.12, 0.12, 0.26, 0.26, 0.18, 0.18, 0.28, 0.28])

    def _extract_dynamic(_landmarks, _indices):
        return _eye_points_for_ear(next(ears))

    monkeypatch.setattr(detector, "_extract_eye", _extract_dynamic)
    for _ in range(4):
        m = detector.process(_dummy_landmarks())

    assert m.passive_motion_score == 0.25
