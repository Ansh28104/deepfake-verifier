"""Fine-tune Light/Heavy classifiers on deepfake video folders.

Supported dataset layouts:
1) Generic real/fake folders (recursive):
    <root>/real/**/*.mp4
    <root>/fake/**/*.mp4

2) Celeb-DF v2 style:
    <root>/Celeb-real/**/*.mp4
    <root>/YouTube-real/**/*.mp4
    <root>/Celeb-synthesis/**/*.mp4

Examples:
    python scripts/train_finetune.py --mode light --epochs 5 --batch-size 8
    python scripts/train_finetune.py --mode heavy --epochs 5 --batch-size 4
    python scripts/train_finetune.py --mode light --data-root dataset/faceforensics --data-root D:/datasets/Celeb-DF-v2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.efficientnet_light import load_light_model
from core.vit_heavy import load_heavy_model


@dataclass
class VideoItem:
    path: Path
    label: int


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def _collect_from_named_dirs(data_root: Path, names: tuple[str, ...], label: int) -> list[VideoItem]:
    items: list[VideoItem] = []
    for name in names:
        d = data_root / name
        if not d.exists() or not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if _is_video(p):
                items.append(VideoItem(path=p, label=label))
    return items


class VideoFrameDataset(Dataset):
    def __init__(self, items: list[VideoItem], image_size: int, train: bool) -> None:
        self.items = items
        self.train = train
        self.tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5 if train else 0.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def _read_frame(self, path: Path):
        cap = cv2.VideoCapture(str(path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            ok, frame = cap.read()
            cap.release()
            return frame if ok else None

        if self.train:
            target_idx = random.randint(0, max(0, frame_count - 1))
        else:
            target_idx = frame_count // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        frame = self._read_frame(item.path)
        if frame is None:
            frame = (torch.zeros(3, 224, 224).permute(1, 2, 0).numpy() * 255).astype("uint8")
        x = self.tf(frame)
        y = torch.tensor(item.label, dtype=torch.long)
        return x, y


def collect_items(data_root: Path) -> list[VideoItem]:
    # 1) Generic real/fake convention (recursive).
    items = []
    items.extend(_collect_from_named_dirs(data_root, ("real",), label=0))
    items.extend(_collect_from_named_dirs(data_root, ("fake",), label=1))

    # 2) Celeb-DF v2 convention.
    # Real: Celeb-real + YouTube-real
    # Fake: Celeb-synthesis
    items.extend(_collect_from_named_dirs(data_root, ("Celeb-real", "YouTube-real"), label=0))
    items.extend(_collect_from_named_dirs(data_root, ("Celeb-synthesis",), label=1))

    # Deduplicate paths in case multiple conventions overlap under one root.
    dedup: dict[tuple[str, int], VideoItem] = {}
    for item in items:
        key = (str(item.path.resolve()), item.label)
        dedup[key] = item

    return list(dedup.values())


def collect_items_from_roots(data_roots: list[Path]) -> list[VideoItem]:
    combined: list[VideoItem] = []
    for root in data_roots:
        root_items = collect_items(root)
        if root_items:
            print(f"[data] {root}: loaded {len(root_items)} videos")
        else:
            print(f"[data] {root}: no matching videos found")
        combined.extend(root_items)

    dedup: dict[tuple[str, int], VideoItem] = {}
    for item in combined:
        key = (str(item.path.resolve()), item.label)
        dedup[key] = item

    return list(dedup.values())


def split_items(items: list[VideoItem], train_ratio: float, seed: int) -> tuple[list[VideoItem], list[VideoItem]]:
    rng = random.Random(seed)
    items_copy = items[:]
    rng.shuffle(items_copy)
    cut = int(len(items_copy) * train_ratio)
    return items_copy[:cut], items_copy[cut:]


def build_model(mode: str):
    if mode == "heavy":
        model = load_heavy_model()
        if model is None:
            raise RuntimeError("Unable to load heavy model")
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, 2)
        image_size = 224
    else:
        model = load_light_model()
        if model is None:
            raise RuntimeError("Unable to load light model")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        image_size = 380
    return model, image_size


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            total += y.numel()
            correct += (pred == y).sum().item()

            pred_pos = pred == 1
            true_pos = y == 1
            tp += int((pred_pos & true_pos).sum().item())
            fp += int((pred_pos & ~true_pos).sum().item())
            tn += int((~pred_pos & ~true_pos).sum().item())
            fn += int((~pred_pos & true_pos).sum().item())

    acc = (correct / total) if total else 0.0
    precision_fake = tp / max(1, tp + fp)
    recall_fake = tp / max(1, tp + fn)
    f1_fake = 2.0 * precision_fake * recall_fake / max(1e-9, precision_fake + recall_fake)

    precision_real = tn / max(1, tn + fn)
    recall_real = tn / max(1, tn + fp)
    f1_real = 2.0 * precision_real * recall_real / max(1e-9, precision_real + recall_real)

    return {
        "acc": acc,
        "precision_real": precision_real,
        "recall_real": recall_real,
        "f1_real": f1_real,
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "f1_fake": f1_fake,
        "macro_f1": 0.5 * (f1_real + f1_fake),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune deepfake classifier")
    parser.add_argument("--mode", choices=["light", "heavy"], default="light")
    parser.add_argument(
        "--data-root",
        action="append",
        default=None,
        help=(
            "Dataset root path. Repeat this flag to combine datasets. "
            "Supports real/fake layout and Celeb-DF v2 layout. "
            "Default: dataset/faceforensics"
        ),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-min-val-acc", type=float, default=0.0)
    parser.add_argument("--checkpoint-min-macro-f1", type=float, default=0.0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_roots = [Path(p) for p in (args.data_root or ["dataset/faceforensics"])]
    items = collect_items_from_roots(data_roots)
    if len(items) < 2:
        roots_str = ", ".join(str(p) for p in data_roots)
        raise RuntimeError(
            "Not enough videos found. Supported layouts: "
            "<root>/real + <root>/fake, or Celeb-DF v2 folders under each root. "
            f"Checked roots: {roots_str}"
        )

    train_items, val_items = split_items(items, args.train_ratio, args.seed)
    if not val_items:
        raise RuntimeError("Validation split is empty; lower train-ratio or add more data")

    model, image_size = build_model(args.mode)

    train_ds = VideoFrameDataset(train_items, image_size=image_size, train=True)
    val_ds = VideoFrameDataset(val_items, image_size=image_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / ("efficientnet_b4_finetuned.pth" if args.mode == "light" else "vit_b16_finetuned.pth")

    best_val_acc = -1.0
    best_macro_f1 = -1.0
    train_real = sum(1 for item in train_items if item.label == 0)
    train_fake = sum(1 for item in train_items if item.label == 1)
    val_real = sum(1 for item in val_items if item.label == 0)
    val_fake = sum(1 for item in val_items if item.label == 1)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        metrics = evaluate(model, val_loader, device)
        val_acc = float(metrics["acc"])
        macro_f1 = float(metrics["macro_f1"])
        avg_loss = running_loss / max(1, len(train_loader))
        print(
            "epoch="
            f"{epoch}/{args.epochs} "
            f"loss={avg_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"macro_f1={macro_f1:.4f} "
            f"real(f1={metrics['f1_real']:.4f},recall={metrics['recall_real']:.4f}) "
            f"fake(f1={metrics['f1_fake']:.4f},recall={metrics['recall_fake']:.4f})"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1

        should_save = (
            val_acc >= args.checkpoint_min_val_acc
            and macro_f1 >= args.checkpoint_min_macro_f1
            and (val_acc >= best_val_acc or macro_f1 >= best_macro_f1)
        )

        if should_save:
            payload = {
                "state_dict": model.state_dict(),
                "metadata": {
                    "mode": args.mode,
                    "epoch": epoch,
                    "image_size": image_size,
                    "best_val_acc": best_val_acc,
                    "best_macro_f1": best_macro_f1,
                    "val_acc": val_acc,
                    "macro_f1": macro_f1,
                    "class_counts": {
                        "train_real": train_real,
                        "train_fake": train_fake,
                        "val_real": val_real,
                        "val_fake": val_fake,
                    },
                    "data_roots": [str(p) for p in data_roots],
                },
            }
            torch.save(payload, out_path)

            meta_path = out_path.with_suffix(".meta.json")
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(payload["metadata"], f, indent=2)

            print(
                f"[checkpoint] saved -> {out_path} "
                f"(val_acc={val_acc:.4f}, macro_f1={macro_f1:.4f})"
            )

    print(f"done. best_val_acc={best_val_acc:.4f} best_macro_f1={best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
