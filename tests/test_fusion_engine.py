"""Basic behavior tests for fusion engine state transitions."""

from __future__ import annotations

from core.fusion_engine import FusionEngine


def test_verified_path() -> None:
    engine = FusionEngine(analysis_duration=1.0, face_hold_duration=0.0)
    t0 = 0.0

    # WAITING -> FACE_DETECTED
    engine.update(now_ts=t0, face_present=True)
    # FACE_DETECTED -> ANALYZING
    engine.update(now_ts=t0 + 0.01, face_present=True)

    pkg = None
    for i in range(5):
        pkg = engine.update(
            now_ts=t0 + 0.2 + i * 0.2,
            face_present=True,
            deepfake_score=0.05,
            blink_pattern_score=0.2,
        )

    assert pkg is not None
    assert pkg.risk_score < 35.0


def test_rejected_path() -> None:
    engine = FusionEngine(analysis_duration=1.0, face_hold_duration=0.0)
    t0 = 0.0
    engine.update(now_ts=t0, face_present=True)
    engine.update(now_ts=t0 + 0.01, face_present=True)

    pkg = None
    for i in range(5):
        pkg = engine.update(
            now_ts=t0 + 0.2 + i * 0.2,
            face_present=True,
            deepfake_score=0.95,
            blink_pattern_score=0.9,
        )

    assert pkg is not None
    assert pkg.risk_score >= 66.0


def test_weighted_average_prefers_recent_values() -> None:
    engine = FusionEngine()
    avg = engine._weighted_average([0.1, 0.5, 0.9])
    assert abs(avg - ((0.1 * 1 + 0.5 * 2 + 0.9 * 3) / 6.0)) < 1e-9


def test_challenge_timeout_increases_risk() -> None:
    engine = FusionEngine(
        analysis_duration=10.0,
        face_hold_duration=0.0,
        challenge_enabled=True,
        challenge_timeout=1.0,
        risk_smoothing_alpha=1.0,
    )
    t0 = 0.0

    engine.update(now_ts=t0, face_present=True)
    engine.update(now_ts=t0 + 0.01, face_present=True)

    pkg = engine.update(
        now_ts=t0 + 0.1,
        face_present=True,
        deepfake_score=0.5,
        blink_pattern_score=0.5,
        blink_count=0,
    )
    assert pkg.challenge_prompt is not None

    pkg = engine.update(
        now_ts=t0 + 0.45,
        face_present=True,
        deepfake_score=0.5,
        blink_pattern_score=0.5,
        blink_count=0,
    )
    pkg = engine.update(
        now_ts=t0 + 1.7,
        face_present=True,
        deepfake_score=0.5,
        blink_pattern_score=0.5,
        blink_count=0,
    )
    assert pkg.risk_score > 60.0


def test_threshold_boundaries() -> None:
    engine = FusionEngine(verified_threshold=35.0, rejected_threshold=66.0)
    assert engine._to_verdict(35.0) == "VERIFIED"
    assert engine._to_verdict(35.01) == "SUSPICIOUS"
    assert engine._to_verdict(65.99) == "SUSPICIOUS"
    assert engine._to_verdict(66.0) == "REJECTED"


def test_live_verdict_requires_min_samples() -> None:
    engine = FusionEngine(analysis_duration=10.0, face_hold_duration=0.0, min_live_samples=3)
    t0 = 0.0
    engine.update(now_ts=t0, face_present=True)
    engine.update(now_ts=t0 + 0.01, face_present=True)

    pkg = engine.update(
        now_ts=t0 + 0.2,
        face_present=True,
        deepfake_score=0.05,
        blink_pattern_score=0.2,
    )
    assert pkg.verdict is None

    pkg = engine.update(
        now_ts=t0 + 0.4,
        face_present=True,
        deepfake_score=0.05,
        blink_pattern_score=0.2,
    )
    assert pkg.verdict is None

    pkg = engine.update(
        now_ts=t0 + 0.6,
        face_present=True,
        deepfake_score=0.05,
        blink_pattern_score=0.2,
    )
    assert pkg.verdict == "VERIFIED"
