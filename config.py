"""Project-wide runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    mode: str = "light"
    analysis_duration: float = 0.0
    face_hold_duration: float = 0.0
    waiting_face_gap_tolerance: float = 0.4
    analyzing_face_absence_tolerance: float = 2.0
    verified_threshold: float = 35.0
    rejected_threshold: float = 66.0
    camera_index: int = 0
    camera_unmirror: bool = True
    frame_buffer_size: int = 8
    ear_threshold: float = 0.20
    log_level: str = "debug"
    light_checkpoint_path: str = "models/efficientnet_b4_finetuned.pth"
    heavy_checkpoint_path: str = "models/vit_b16_finetuned.pth"
    require_finetuned_weights: bool = False
    mp_min_detection_confidence: float = 0.35
    mp_min_tracking_confidence: float = 0.35
    real_class_index: int = 0
    fake_class_index: int = 1
    liveness_challenge_enabled: bool = False
    challenge_timeout: float = 4.0
    challenge_required_successes: int = 2
    challenge_weight: float = 0.0
    deepfake_weight: float = 0.55
    blink_weight: float = 0.45
    risk_smoothing_alpha: float = 0.8


def _auto_discover_checkpoint(models_dir: Path, keywords: tuple[str, ...]) -> str | None:
    if not models_dir.exists():
        return None

    candidates = []
    for path in models_dir.glob("*.pth"):
        name = path.name.lower()
        if any(k in name for k in keywords):
            candidates.append(path)

    if not candidates:
        return None
    return str(sorted(candidates)[0])


def load_config() -> AppConfig:
    cfg = AppConfig()
    cfg.light_checkpoint_path = str(Path(cfg.light_checkpoint_path))
    cfg.heavy_checkpoint_path = str(Path(cfg.heavy_checkpoint_path))

    models_dir = Path("models")
    light_path = Path(cfg.light_checkpoint_path)
    heavy_path = Path(cfg.heavy_checkpoint_path)

    if not light_path.exists():
        discovered = _auto_discover_checkpoint(models_dir, ("efficientnet", "b4", "light"))
        if discovered:
            cfg.light_checkpoint_path = discovered

    if not heavy_path.exists():
        discovered = _auto_discover_checkpoint(models_dir, ("vit", "b16", "heavy", "transformer"))
        if discovered:
            cfg.heavy_checkpoint_path = discovered

    return cfg
