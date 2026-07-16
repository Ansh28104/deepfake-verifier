from __future__ import annotations

from pathlib import Path

import torch

from scripts.verify_checkpoints import _check_file


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def test_verify_detects_vit_imagenet_head_as_not_task_ready(tmp_path) -> None:
    ckpt = tmp_path / "vit_b16_finetuned.pth"
    _save_state(
        ckpt,
        {
            "encoder.layers.encoder_layer_0.ln_1.weight": torch.zeros(768),
            "heads.head.weight": torch.zeros((1000, 768)),
            "heads.head.bias": torch.zeros(1000),
        },
    )

    report = _check_file(ckpt)

    assert report["looks_like"] == "vit"
    assert report["task_ready"] is False
    assert report["out_features"] == 1000
    assert "expected 1 or 2" in str(report["reason"])


def test_verify_accepts_binary_vit_head(tmp_path) -> None:
    ckpt = tmp_path / "vit_b16_finetuned.pth"
    _save_state(
        ckpt,
        {
            "encoder.layers.encoder_layer_0.ln_1.weight": torch.zeros(768),
            "heads.head.weight": torch.zeros((2, 768)),
            "heads.head.bias": torch.zeros(2),
        },
    )

    report = _check_file(ckpt)

    assert report["looks_like"] == "vit"
    assert report["task_ready"] is True
    assert report["out_features"] == 2


def test_verify_accepts_legacy_efficientnet_binary_head(tmp_path) -> None:
    ckpt = tmp_path / "efficientnet_b4_finetuned.pth"
    _save_state(
        ckpt,
        {
            "_conv_stem.weight": torch.zeros((48, 3, 3, 3)),
            "_fc.weight": torch.zeros((1, 1792)),
            "_fc.bias": torch.zeros(1),
        },
    )

    report = _check_file(ckpt)

    assert report["looks_like"] == "efficientnet"
    assert report["task_ready"] is True
    assert report["head_key"] == "_fc.weight"
    assert report["out_features"] == 1


def test_verify_reads_metadata_from_wrapped_checkpoint(tmp_path) -> None:
    ckpt = tmp_path / "wrapped_light.pth"
    _save_state(
        ckpt,
        {
            "state_dict": {
                "classifier.1.weight": torch.zeros((2, 1280)),
                "classifier.1.bias": torch.zeros(2),
            },
            "metadata": {
                "mode": "light",
                "epoch": 3,
                "best_val_acc": 0.93,
                "best_macro_f1": 0.92,
            },
        },
    )

    report = _check_file(ckpt)

    assert report["task_ready"] is True
    assert report["metadata"] is not None
    assert report["metadata"]["mode"] == "light"
    assert report["metadata"]["epoch"] == 3
