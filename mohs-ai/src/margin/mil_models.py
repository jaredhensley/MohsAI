"""Placeholder for future multiple-instance learning models."""
from __future__ import annotations

import torch
from torch import nn


class DummyMILModel(nn.Module):
    """Minimal placeholder network returning zeros."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - stub
        batch = x.shape[0]
        return torch.zeros(batch, 1, device=x.device)


__all__ = ["DummyMILModel"]
