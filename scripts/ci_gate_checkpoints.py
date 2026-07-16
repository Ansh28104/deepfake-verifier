"""CI gate for checkpoint acceptance based on metadata quality thresholds.

Usage:
    python scripts/ci_gate_checkpoints.py --min-val-acc 0.90 --min-macro-f1 0.88
    python scripts/ci_gate_checkpoints.py --meta-file models/efficientnet_b4_finetuned.meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MIN_VAL_ACC = 0.90
MIN_MACRO_F1 = 0.88


def _read_meta(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"metadata file is not a JSON object: {path}")
    return payload


def _check_meta(path: Path, min_val_acc: float, min_macro_f1: float) -> tuple[bool, str]:
    data = _read_meta(path)

    if data.get("mode") not in ("light", "heavy"):
        return False, f"{path.name}: invalid or missing mode"

    try:
        val_acc = float(data.get("best_val_acc", data.get("val_acc", -1.0)))
        macro_f1 = float(data.get("best_macro_f1", data.get("macro_f1", -1.0)))
    except (TypeError, ValueError) as exc:
        return False, f"{path.name}: invalid numeric metrics: {exc}"

    class_counts = data.get("class_counts")
    if not isinstance(class_counts, dict):
        return False, f"{path.name}: missing class_counts"

    required_counts = ("train_real", "train_fake", "val_real", "val_fake")
    missing_counts = [name for name in required_counts if name not in class_counts]
    if missing_counts:
        return False, f"{path.name}: missing class count fields: {', '.join(missing_counts)}"

    try:
        count_values = {name: int(class_counts[name]) for name in required_counts}
    except (TypeError, ValueError) as exc:
        return False, f"{path.name}: invalid class counts: {exc}"

    if any(value <= 0 for value in count_values.values()):
        return False, f"{path.name}: class counts must be positive (got {count_values})"

    ok_acc = val_acc >= min_val_acc
    ok_f1 = macro_f1 >= min_macro_f1
    ok = ok_acc and ok_f1

    msg = (
        f"{path.name}: val_acc={val_acc:.4f} (min={min_val_acc:.4f}) "
        f"macro_f1={macro_f1:.4f} (min={min_macro_f1:.4f})"
    )
    return ok, msg


def main() -> None:
    parser = argparse.ArgumentParser(description="CI gate for checkpoint metadata")
    parser.add_argument("--min-val-acc", type=float, default=MIN_VAL_ACC)
    parser.add_argument("--min-macro-f1", type=float, default=MIN_MACRO_F1)
    parser.add_argument("--meta-file", action="append", default=None, help="Specific metadata file path (repeatable)")
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail when metadata files are missing")
    args = parser.parse_args()

    if args.meta_file:
        meta_files = [Path(p) for p in args.meta_file]
    else:
        meta_files = sorted(Path("models").glob("*.meta.json"))

    if not meta_files:
        message = "No metadata files found (models/*.meta.json)."
        print(f"[gate] {message}")
        if args.allow_missing:
            print("[gate] PASS (allow-missing enabled)")
            return
        raise SystemExit(1)

    any_fail = False
    for path in meta_files:
        if not path.exists():
            any_fail = True
            print(f"[gate] FAIL: file missing: {path}")
            continue

        try:
            ok, msg = _check_meta(path, args.min_val_acc, args.min_macro_f1)
        except Exception as exc:
            any_fail = True
            print(f"[gate] FAIL: {path.name}: {exc}")
            continue

        if ok:
            print(f"[gate] PASS: {msg}")
        else:
            any_fail = True
            print(f"[gate] FAIL: {msg}")

    if any_fail:
        raise SystemExit(1)

    print("[gate] PASS: all metadata files satisfy acceptance thresholds")


if __name__ == "__main__":
    main()
