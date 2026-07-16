from __future__ import annotations

import numpy as np

from core.fusion_engine import FusionEngine
from core.model_inference import ModelInferenceEngine


def test_infer_frame_updates_before_buffer_full() -> None:
    engine = ModelInferenceEngine.__new__(ModelInferenceEngine)
    engine.frame_buffer = __import__("collections").deque(maxlen=4)

    class _Temporal:
        def update(self, value: float) -> float:
            return float(value)

    engine.temporal = _Temporal()
    engine._raw_inference = lambda _frame: 0.73

    score = ModelInferenceEngine.infer_frame(engine, np.zeros((2, 2, 3), dtype=np.uint8))

    assert 0.72 < score < 0.74


def test_fusion_uses_neutral_when_modalities_missing() -> None:
    engine = FusionEngine(analysis_duration=10.0, face_hold_duration=0.0)
    t0 = 0.0

    # Enter ANALYZING quickly.
    engine.update(now_ts=t0, face_present=True)
    engine.update(now_ts=t0 + 0.01, face_present=True)

    # Missing signals should append neutral defaults and keep streams moving.
    pkg = engine.update(
        now_ts=t0 + 0.5,
        face_present=True,
        deepfake_score=None,
        blink_pattern_score=None,
    )

    assert len(engine.deepfake_scores) == 1
    assert len(engine.blink_scores) == 1
    assert abs(engine.deepfake_scores[0] - 0.5) < 1e-9
    assert abs(engine.blink_scores[0] - 0.5) < 1e-9
    assert 49.9 <= pkg.risk_score <= 50.1
