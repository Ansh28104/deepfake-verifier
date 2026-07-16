from __future__ import annotations

import sys
import types

from core.efficientnet_light import load_light_model
from core.vit_heavy import load_heavy_model


class _FakeEfficientNet:
    def __init__(self) -> None:
        self.classifier = [None, types.SimpleNamespace(in_features=1280)]


class _FakeViT:
    def __init__(self) -> None:
        self.heads = types.SimpleNamespace(head=types.SimpleNamespace(in_features=768))


def test_light_loader_replaces_image_net_head(monkeypatch) -> None:
    models_module = types.ModuleType("torchvision.models")
    models_module.efficientnet_b4 = lambda weights=None: _FakeEfficientNet()

    torchvision_module = types.ModuleType("torchvision")
    torchvision_module.models = models_module

    monkeypatch.setitem(sys.modules, "torchvision", torchvision_module)
    monkeypatch.setitem(sys.modules, "torchvision.models", models_module)

    model = load_light_model()

    assert model is not None
    assert model.classifier[1].out_features == 2


def test_heavy_loader_replaces_image_net_head(monkeypatch) -> None:
    models_module = types.ModuleType("torchvision.models")
    models_module.vit_b_16 = lambda weights=None: _FakeViT()

    torchvision_module = types.ModuleType("torchvision")
    torchvision_module.models = models_module

    monkeypatch.setitem(sys.modules, "torchvision", torchvision_module)
    monkeypatch.setitem(sys.modules, "torchvision.models", models_module)

    model = load_heavy_model()

    assert model is not None
    assert model.heads.head.out_features == 2