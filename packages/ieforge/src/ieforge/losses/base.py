from abc import abstractmethod
from typing import Any

import torch
from torch import nn


class IELoss(nn.Module):
    """Abstract base class for information extraction loss functions."""

    @abstractmethod
    def forward(self, outputs: Any, batch: dict[str, Any]) -> torch.Tensor:
        """Compute loss from model outputs and the input batch."""
        ...
