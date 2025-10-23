"""Model creation helpers for QC classifier."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


def create_model(num_classes: int = 1, pretrained: bool = False) -> nn.Module:
    """Return a MobileNetV3-Small model with a binary classification head."""
    model = mobilenet_v3_small(weights=None if not pretrained else None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False
    return model


def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)
