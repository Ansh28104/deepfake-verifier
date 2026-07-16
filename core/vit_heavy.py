"""Heavy-mode model loader (ViT-B/16)."""

from __future__ import annotations

import torch.nn as nn


def load_heavy_model():
    """Load a ViT-B/16 backbone when torchvision is available.

    Returns None when dependencies are unavailable so development can
    continue with neutral inference outputs.
    """
    try:
        import torchvision.models as models
    except Exception:
        return None

    model = models.vit_b_16(weights="IMAGENET1K_V1")
    in_features = int(model.heads.head.in_features)
    model.heads.head = nn.Linear(in_features, 2)
    return model
