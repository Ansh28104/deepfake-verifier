"""Light-mode model loader (EfficientNet-B4)."""

from __future__ import annotations

import torch.nn as nn


def load_light_model():
    """Load an EfficientNet-B4 backbone when torchvision is available.

    Returns None when dependencies are unavailable so the pipeline can
    continue in scaffold mode.
    """
    try:
        import torchvision.models as models
    except Exception:
        return None

    model = models.efficientnet_b4(weights="IMAGENET1K_V1")
    in_features = int(model.classifier[1].in_features)
    model.classifier[1] = nn.Linear(in_features, 2)
    return model