from abc import ABC, abstractmethod
from typing import Any

from torch import nn


class IEModel(nn.Module, ABC):
    """Abstract base class for information extraction models."""

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> Any:
        """Forward pass only. No loss computation, no decoding."""
