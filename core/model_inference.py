"""Model inference with dual-mode selection and temporal smoothing."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque

import cv2
import torch
import torch.nn as nn

from core.efficientnet_light import load_light_model
from core.temporal_layer import TemporalLayer
from core.vit_heavy import load_heavy_model


class ModelInferenceEngine:
    """Loads either light/heavy model and emits deepfake confidence in [0, 1]."""

    def __init__(
        self,
        mode: str = "light",
        frame_buffer_size: int = 30,
        light_checkpoint_path: str | None = None,
        heavy_checkpoint_path: str | None = None,
        real_class_index: int = 0,
        fake_class_index: int = 1,
    ) -> None:
        self.mode = mode
        self.frame_buffer: Deque = deque(maxlen=frame_buffer_size)
        self.temporal = TemporalLayer(window_size=frame_buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.light_checkpoint_path = light_checkpoint_path
        self.heavy_checkpoint_path = heavy_checkpoint_path
        self.using_finetuned = False
        self.last_warning: str | None = None
        self.checkpoint_metadata: dict | None = None
        self.real_class_index = real_class_index
        self.fake_class_index = fake_class_index
        self._norm_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.model = self._load_model(mode)

    def _extract_state_dict(self, checkpoint_obj):
        if isinstance(checkpoint_obj, dict):
            if "state_dict" in checkpoint_obj:
                return checkpoint_obj["state_dict"]
            if "model_state_dict" in checkpoint_obj:
                return checkpoint_obj["model_state_dict"]
        return checkpoint_obj

    def _normalize_state_dict_keys(self, state_dict: dict) -> dict:
        normalized = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith("module.") else key
            normalized[new_key] = value
        return normalized

    def _is_checkpoint_compatible(self, mode: str, state_dict: dict) -> tuple[bool, str | None]:
        if mode == "light":
            has_light_markers = any(k.startswith("features.") for k in state_dict)
            has_legacy_light_markers = any(k.startswith("_conv_stem") or k.startswith("_blocks.") for k in state_dict)
            if has_legacy_light_markers:
                return True, None
            if not has_light_markers:
                return False, "checkpoint keys do not match EfficientNet structure"
        else:
            has_vit_markers = any(k.startswith("encoder.") for k in state_dict)
            if not has_vit_markers:
                return False, "checkpoint keys do not match ViT structure"
        return True, None

    def _convert_legacy_efficientnet_state_dict(self, model, state_dict: dict) -> tuple[dict, str | None]:
        """Convert legacy EfficientNet key format to torchvision format.

        Uses strict shape-verified insertion-order mapping. Conversion is accepted
        only if every key tensor shape matches exactly.
        """
        has_legacy = any(k.startswith("_conv_stem") or k.startswith("_blocks.") for k in state_dict)
        has_torchvision = any(k.startswith("features.") for k in state_dict)
        if not has_legacy or has_torchvision:
            return state_dict, None

        model_sd = model.state_dict()
        ck_keys = list(state_dict.keys())
        tv_keys = list(model_sd.keys())

        if len(ck_keys) != len(tv_keys):
            return state_dict, "legacy EfficientNet conversion failed: key count mismatch"

        for idx, ck in enumerate(ck_keys):
            tv = tv_keys[idx]
            if tuple(state_dict[ck].shape) != tuple(model_sd[tv].shape):
                return state_dict, "legacy EfficientNet conversion failed: tensor shape mismatch"

        mapped = {tv_keys[idx]: state_dict[ck_keys[idx]] for idx in range(len(ck_keys))}
        return mapped, None

    def _adapt_head_for_checkpoint(self, model, mode: str, state_dict: dict) -> None:
        if mode == "light" and "classifier.1.weight" in state_dict:
            out_features = int(state_dict["classifier.1.weight"].shape[0])
            in_features = int(model.classifier[1].in_features)
            model.classifier[1] = nn.Linear(in_features, out_features)
        elif mode == "heavy" and "heads.head.weight" in state_dict:
            out_features = int(state_dict["heads.head.weight"].shape[0])
            in_features = int(model.heads.head.in_features)
            model.heads.head = nn.Linear(in_features, out_features)

    def _metadata_quality_ok(self, mode: str, metadata: dict | None) -> tuple[bool, str | None]:
        if not isinstance(metadata, dict):
            return False, "checkpoint metadata missing"

        meta_mode = metadata.get("mode")
        if meta_mode is not None and meta_mode != mode:
            return False, f"checkpoint mode '{meta_mode}' does not match requested mode '{mode}'"

        try:
            val_acc = float(metadata.get("best_val_acc", metadata.get("val_acc", -1.0)))
            macro_f1 = float(metadata.get("best_macro_f1", metadata.get("macro_f1", -1.0)))
        except (TypeError, ValueError) as exc:
            return False, f"invalid checkpoint metrics: {exc}"

        if val_acc < 0.90 or macro_f1 < 0.88:
            return False, f"checkpoint metrics below threshold (val_acc={val_acc:.4f}, macro_f1={macro_f1:.4f})"

        class_counts = metadata.get("class_counts")
        if not isinstance(class_counts, dict):
            return False, "checkpoint metadata missing class_counts"

        required = ("train_real", "train_fake", "val_real", "val_fake")
        missing = [name for name in required if name not in class_counts]
        if missing:
            return False, f"checkpoint metadata missing class count fields: {', '.join(missing)}"

        try:
            counts = {name: int(class_counts[name]) for name in required}
        except (TypeError, ValueError) as exc:
            return False, f"invalid class count values: {exc}"

        if any(value <= 0 for value in counts.values()):
            return False, f"checkpoint class counts must all be positive (got {counts})"

        return True, None

    def _temporal_alpha_from_metadata(self, metadata: dict | None) -> float:
        if not isinstance(metadata, dict):
            return 0.65

        try:
            val_acc = float(metadata.get("best_val_acc", metadata.get("val_acc", 0.0)))
            macro_f1 = float(metadata.get("best_macro_f1", metadata.get("macro_f1", 0.0)))
        except (TypeError, ValueError):
            return 0.65

        if val_acc >= 0.96 and macro_f1 >= 0.95:
            return 0.55
        if val_acc >= 0.93 and macro_f1 >= 0.92:
            return 0.60
        return 0.65

    def _load_finetuned_weights(self, model, mode: str) -> bool:
        self.last_warning = None
        self.checkpoint_metadata = None
        checkpoint_path = self.heavy_checkpoint_path if mode == "heavy" else self.light_checkpoint_path
        if not checkpoint_path:
            return False

        path = Path(checkpoint_path)
        if not path.exists():
            self.last_warning = f"Checkpoint not found: {checkpoint_path}. Using base pretrained weights."
            print(f"[ModelInference] {self.last_warning}")
            return False

        try:
            checkpoint_obj = torch.load(path, map_location=self.device)
            state_dict = self._extract_state_dict(checkpoint_obj)
            state_dict = self._normalize_state_dict_keys(state_dict)
            metadata = checkpoint_obj.get("metadata") if isinstance(checkpoint_obj, dict) else None
        except Exception as exc:
            self.last_warning = f"Failed to read checkpoint {path.name}: {exc}. Using base pretrained weights."
            print(f"[ModelInference] {self.last_warning}")
            return False

        metadata_ok, metadata_reason = self._metadata_quality_ok(mode, metadata)
        if metadata is not None and not metadata_ok:
            self.last_warning = (
                f"Rejected checkpoint {path.name}: {metadata_reason}. "
                "Falling back to base pretrained weights."
            )
            print(f"[ModelInference] {self.last_warning}")
            return False

        self.checkpoint_metadata = metadata if metadata_ok else None

        compatible, reason = self._is_checkpoint_compatible(mode, state_dict)
        if not compatible:
            self.last_warning = (
                f"Rejected checkpoint {path.name}: {reason}. Falling back to base pretrained weights."
            )
            print(f"[ModelInference] {self.last_warning}")
            return False

        if mode == "light":
            state_dict, conversion_error = self._convert_legacy_efficientnet_state_dict(model, state_dict)
            if conversion_error is not None:
                self.last_warning = (
                    f"Rejected checkpoint {path.name}: {conversion_error}. "
                    "Falling back to base pretrained weights."
                )
                print(f"[ModelInference] {self.last_warning}")
                return False

        self._adapt_head_for_checkpoint(model, mode, state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # Strict compatibility gate: reject partial loads.
        if missing or unexpected:
            self.last_warning = (
                f"Rejected checkpoint {path.name}: partial state_dict match "
                f"(missing={len(missing)}, unexpected={len(unexpected)}). "
                "Falling back to base pretrained weights."
            )
            print(f"[ModelInference] {self.last_warning}")
            return False

        return True

    def _load_model(self, mode: str):
        if mode == "heavy":
            model = load_heavy_model()
        else:
            model = load_light_model()

        if model is None:
            return None

        self.using_finetuned = self._load_finetuned_weights(model, mode)
        temporal_alpha = self._temporal_alpha_from_metadata(self.checkpoint_metadata)
        self.temporal = TemporalLayer(window_size=self.frame_buffer.maxlen or len(self.frame_buffer) or 1, ema_alpha=temporal_alpha)
        model = model.to(self.device)
        model.eval()
        return model

    def _input_size(self) -> int:
        return 224 if self.mode == "heavy" else 380

    def _preprocess(self, frame):
        size = self._input_size()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        tensor = (tensor - self._norm_mean) / self._norm_std
        return tensor.to(self.device)

    def switch_mode(self, mode: str) -> None:
        self.mode = mode
        self.model = self._load_model(mode)
        self.frame_buffer.clear()

    def model_status(self) -> str:
        source = "fine-tuned checkpoint" if self.using_finetuned else "base pretrained"
        warning = f", warning={self.last_warning}" if self.last_warning else ""
        return f"mode={self.mode}, source={source}, device={self.device}{warning}"

    def _raw_inference(self, frame) -> float:
        if self.model is None:
            return 0.5

        x = self._preprocess(frame)
        with torch.no_grad():
            out = self.model(x)

        if out.ndim == 1:
            out = out.unsqueeze(0)

        classes = out.shape[-1]
        if classes == 1:
            return float(torch.sigmoid(out[0, 0]).item())
        if classes == 2:
            probs = torch.softmax(out[0], dim=0)
            ri = self.real_class_index
            fi = self.fake_class_index
            if 0 <= ri < classes and 0 <= fi < classes and ri != fi:
                numerator = float(probs[fi].item())
                denominator = float((probs[fi] + probs[ri]).item())
                if denominator > 0:
                    return numerator / denominator
            return float(probs[1].item())

        # Multiclass proxy: derive fake-vs-real score from configured indices.
        probs = torch.softmax(out[0], dim=0)
        ri = self.real_class_index
        fi = self.fake_class_index
        if 0 <= ri < classes and 0 <= fi < classes and ri != fi:
            numerator = float(probs[fi].item())
            denominator = float((probs[fi] + probs[ri]).item())
            if denominator > 0:
                return numerator / denominator

        # Safe fallback for unknown class mapping.
        return 0.5

    def infer_frame(self, frame) -> float:
        self.frame_buffer.append(frame)
        raw_score = self._raw_inference(frame)
        return self.temporal.update(raw_score)