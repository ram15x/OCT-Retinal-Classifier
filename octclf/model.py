# builds the backbones, all start from imagenet weights

from __future__ import annotations

from torch import nn
from torchvision import models

from .config import CLASSES

BACKBONES = ("convnext_tiny", "maxvit_t", "efficientnet_v2_s", "resnet50", "resnet18")


def build_model(name: str = "convnext_tiny", pretrained: bool = True, num_classes: int = len(CLASSES)):
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}; choose from {BACKBONES}")
    if name == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = models.convnext_tiny(weights=w)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "maxvit_t":
        w = models.MaxVit_T_Weights.DEFAULT if pretrained else None
        m = models.maxvit_t(weights=w)
        m.classifier[5] = nn.Linear(m.classifier[5].in_features, num_classes, bias=False)
    elif name == "efficientnet_v2_s":
        w = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_v2_s(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    # last conv stage of each backbone, that is where the heatmap looks best
    if name == "convnext_tiny":
        return model.features[-1]
    if name == "efficientnet_v2_s":
        return model.features[-1]
    if name == "maxvit_t":
        return model.blocks[-1]
    return model.layer4[-1]
