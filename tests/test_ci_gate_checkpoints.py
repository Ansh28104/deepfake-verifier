from __future__ import annotations

import json
from pathlib import Path

from scripts.ci_gate_checkpoints import _check_meta


def _write_meta(path: Path, *, val_acc: float, macro_f1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "light",
        "epoch": 3,
        "best_val_acc": val_acc,
        "best_macro_f1": macro_f1,
        "class_counts": {
            "train_real": 12,
            "train_fake": 12,
            "val_real": 4,
            "val_fake": 4,
        },
        "data_roots": ["dataset/ci_smoke"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_check_meta_passes_thresholds(tmp_path) -> None:
    meta = tmp_path / "light.meta.json"
    _write_meta(meta, val_acc=0.93, macro_f1=0.91)

    ok, _msg = _check_meta(meta, min_val_acc=0.90, min_macro_f1=0.88)

    assert ok is True


def test_check_meta_fails_thresholds(tmp_path) -> None:
    meta = tmp_path / "heavy.meta.json"
    _write_meta(meta, val_acc=0.87, macro_f1=0.86)

    ok, _msg = _check_meta(meta, min_val_acc=0.90, min_macro_f1=0.88)

    assert ok is False


def test_check_meta_rejects_missing_class_balance(tmp_path) -> None:
    meta = tmp_path / "bad.meta.json"
    meta.write_text(
        json.dumps(
            {
                "mode": "light",
                "epoch": 1,
                "best_val_acc": 0.93,
                "best_macro_f1": 0.92,
                "class_counts": {
                    "train_real": 10,
                    "train_fake": 0,
                    "val_real": 4,
                    "val_fake": 4,
                },
            }
        ),
        encoding="utf-8",
    )

    ok, _msg = _check_meta(meta, min_val_acc=0.90, min_macro_f1=0.88)

    assert ok is False
