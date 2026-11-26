from typing import Any

import torch
from torch.nn import functional

from ieforge.losses.base import IELoss
from ieforge.registry import register


@register("loss", name="mse")
class MSELoss(IELoss):
    """Mean squared error loss for regression outputs."""

    def __init__(self, name: str = "mse") -> None:
        """Initialize the loss."""
        super().__init__()
        self.name = name

    def forward(self, outputs: Any, batch: dict[str, Any]) -> torch.Tensor:
        """Return the model loss or compute MSE from logits and labels."""
        if outputs.loss is not None:
            return outputs.loss  # type: ignore[no-any-return]
        logits = outputs.logits
        labels = batch["labels"].to(logits.dtype)
        if logits.ndim == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        return functional.mse_loss(logits, labels)
