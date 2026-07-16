from __future__ import annotations

import numpy as np
import torch

from core.model_inference import ModelInferenceEngine


class _DummyModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits

    def __call__(self, _x: torch.Tensor) -> torch.Tensor:
        return self._logits


def _make_engine(logits: torch.Tensor, real_idx: int, fake_idx: int):
    engine = ModelInferenceEngine.__new__(ModelInferenceEngine)
    engine.model = _DummyModel(logits)
    engine.real_class_index = real_idx
    engine.fake_class_index = fake_idx
    engine._preprocess = lambda _frame: torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    return engine


def test_binary_inference_uses_configured_indices() -> None:
    # Softmax([1, 3]) ~= [0.1192, 0.8808], so fake=class0 should be low.
    logits = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    engine = _make_engine(logits, real_idx=1, fake_idx=0)

    score = ModelInferenceEngine._raw_inference(engine, np.zeros((2, 2, 3), dtype=np.uint8))

    assert 0.10 < score < 0.14


def test_binary_inference_falls_back_when_indices_invalid() -> None:
    # With invalid indices, fallback is class 1 probability.
    logits = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    engine = _make_engine(logits, real_idx=2, fake_idx=0)

    score = ModelInferenceEngine._raw_inference(engine, np.zeros((2, 2, 3), dtype=np.uint8))

    assert 0.86 < score < 0.90


class _DummyCheckpointModel:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict, strict=False):
        self.loaded = state_dict
        return [], []


def test_load_finetuned_weights_rejects_weak_metadata(monkeypatch) -> None:
    engine = ModelInferenceEngine.__new__(ModelInferenceEngine)
    engine.device = torch.device("cpu")
    engine.light_checkpoint_path = "models/efficientnet_b4_finetuned.pth"
    engine.heavy_checkpoint_path = None
    engine.last_warning = None
    engine.checkpoint_metadata = None
    engine.frame_buffer = __import__("collections").deque(maxlen=8)

    checkpoint_obj = {
        "state_dict": {},
        "metadata": {
            "mode": "light",
            "best_val_acc": 0.0,
            "best_macro_f1": 0.0,
            "class_counts": {
                "train_real": 1,
                "train_fake": 1,
                "val_real": 1,
                "val_fake": 0,
            },
        },
    }

    monkeypatch.setattr("core.model_inference.Path.exists", lambda self: True)
    monkeypatch.setattr("core.model_inference.torch.load", lambda *args, **kwargs: checkpoint_obj)
    monkeypatch.setattr(ModelInferenceEngine, "_is_checkpoint_compatible", lambda *args, **kwargs: (True, None))
    monkeypatch.setattr(ModelInferenceEngine, "_convert_legacy_efficientnet_state_dict", lambda *args, **kwargs: ({}, None))
    monkeypatch.setattr(ModelInferenceEngine, "_adapt_head_for_checkpoint", lambda *args, **kwargs: None)

    ok = ModelInferenceEngine._load_finetuned_weights(engine, _DummyCheckpointModel(), "light")

    assert ok is False
    assert "checkpoint metrics below threshold" in str(engine.last_warning)
    assert engine.checkpoint_metadata is None


def test_temporal_alpha_calibrates_from_checkpoint_metadata() -> None:
    engine = ModelInferenceEngine.__new__(ModelInferenceEngine)

    assert ModelInferenceEngine._temporal_alpha_from_metadata(engine, None) == 0.65
    assert ModelInferenceEngine._temporal_alpha_from_metadata(
        engine,
        {"best_val_acc": 0.97, "best_macro_f1": 0.96},
    ) == 0.55
    assert ModelInferenceEngine._temporal_alpha_from_metadata(
        engine,
        {"best_val_acc": 0.94, "best_macro_f1": 0.92},
    ) == 0.60
