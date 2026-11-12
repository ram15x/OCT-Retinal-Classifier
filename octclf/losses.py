# focal loss, with optional class weights and label smoothing

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets, weight=self.weight, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1.0 - pt) ** self.gamma * ce).mean()
