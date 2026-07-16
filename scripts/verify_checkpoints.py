"""Verify deepfake model checkpoints in models/.

Usage:
    python scripts/verify_checkpoints.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj:
            return obj["state_dict"]
        if "model_state_dict" in obj:
            return obj["model_state_dict"]
    return obj


def _check_file(path: Path) -> dict:
    report = {
        "file": str(path),
        "exists": path.exists(),
        "keys": 0,
        "looks_like": "unknown",
        "head_key": None,
        "out_features": None,
        "task_ready": False,
        "metadata": None,
        "reason": None,
        "error": None,
    }

    if not path.exists():
        report["error"] = "file not found"
        return report

    try:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and isinstance(obj.get("metadata"), dict):
            meta = obj["metadata"]
            report["metadata"] = {
                "mode": meta.get("mode"),
                "epoch": meta.get("epoch"),
                "best_val_acc": meta.get("best_val_acc"),
                "best_macro_f1": meta.get("best_macro_f1"),
                "class_counts": meta.get("class_counts"),
                "data_roots": meta.get("data_roots"),
            }
        state = _extract_state_dict(obj)
        if not isinstance(state, dict):
            report["error"] = "checkpoint is not a state_dict-like object"
            return report

        report["keys"] = len(state)
        keys = set(state.keys())

        if any(k.startswith("classifier.") for k in keys):
            report["looks_like"] = "efficientnet"
            if "classifier.1.weight" in state:
                report["head_key"] = "classifier.1.weight"
                report["out_features"] = int(state["classifier.1.weight"].shape[0])
            elif "classifier.weight" in state:
                report["head_key"] = "classifier.weight"
                report["out_features"] = int(state["classifier.weight"].shape[0])
        elif "_fc.weight" in state:
            report["looks_like"] = "efficientnet"
            report["head_key"] = "_fc.weight"
            report["out_features"] = int(state["_fc.weight"].shape[0])

        if any(k.startswith("heads.head") for k in keys):
            report["looks_like"] = "vit"
            if "heads.head.weight" in state:
                report["head_key"] = "heads.head.weight"
                report["out_features"] = int(state["heads.head.weight"].shape[0])

        if report["looks_like"] == "unknown":
            report["reason"] = "architecture markers not recognized"
        elif report["out_features"] is None:
            report["reason"] = "classifier head not found"
        elif report["out_features"] in (1, 2):
            report["task_ready"] = True
        else:
            report["reason"] = (
                f"classifier out_features={report['out_features']} (expected 1 or 2 for deepfake task)"
            )
    except Exception as exc:
        report["error"] = str(exc)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify deepfake model checkpoints")
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Do not return a non-zero exit code when expected checkpoints are not task-ready",
    )
    args = parser.parse_args()

    models_dir = Path("models")
    default_light = models_dir / "efficientnet_b4_finetuned.pth"
    default_heavy = models_dir / "vit_b16_finetuned.pth"
    has_expected_errors = False

    print("[Verify] Checking expected checkpoint files...")
    for p in (default_light, default_heavy):
        r = _check_file(p)
        if r["error"]:
            has_expected_errors = True
            print(f"- {r['file']}: ERROR: {r['error']}")
        elif not r["task_ready"]:
            has_expected_errors = True
            print(
                f"- {r['file']}: NOT TASK-READY, keys={r['keys']}, type={r['looks_like']}, "
                f"head={r['head_key']}, reason={r['reason']}"
            )
        else:
            print(
                f"- {r['file']}: ok, keys={r['keys']}, type={r['looks_like']}, "
                f"head={r['head_key']}, out_features={r['out_features']}"
            )
            if r["metadata"]:
                print(f"  metadata={r['metadata']}")

    print("\n[Verify] Listing all .pth files in models/ ...")
    if not models_dir.exists():
        print("- models/ directory does not exist")
        return

    pths = sorted(models_dir.glob("*.pth"))
    if not pths:
        print("- no .pth files found")
        return

    for path in pths:
        r = _check_file(path)
        if r["error"]:
            print(f"- {path.name}: ERROR: {r['error']}")
        elif not r["task_ready"]:
            print(
                f"- {path.name}: NOT TASK-READY, keys={r['keys']}, type={r['looks_like']}, "
                f"head={r['head_key']}, reason={r['reason']}"
            )
        else:
            print(
                f"- {path.name}: ok, keys={r['keys']}, type={r['looks_like']}, "
                f"head={r['head_key']}, out_features={r['out_features']}"
            )
            if r["metadata"]:
                print(f"  metadata={r['metadata']}")

    if has_expected_errors and not args.no_fail:
        print("\n[Verify] FAILED: Expected checkpoint files are missing or not task-ready deepfake heads.")
        sys.exit(1)

    if has_expected_errors:
        print("\n[Verify] WARNING: Expected checkpoint files are not task-ready, but --no-fail was set.")
    else:
        print("\n[Verify] PASSED: Expected checkpoint files are task-ready.")


if __name__ == "__main__":
    main()
