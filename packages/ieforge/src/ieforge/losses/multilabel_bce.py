from typing import Any

import torch
from torch.nn import functional

from ieforge.losses.base import IELoss
from ieforge.registry import register


@register("loss", name="multilabel_bce")
class MultiLabelBCELoss(IELoss):
    """Binary cross-entropy over sequence-classification logits."""

    def __init__(self, name: str = "multilabel_bce") -> None:
        """Initialize the loss."""
        super().__init__()
        self.name = name

    def forward(self, outputs: Any, batch: dict[str, Any]) -> torch.Tensor:
        """Compute BCE-with-logits loss from model outputs and float labels."""
        return functional.binary_cross_entropy_with_logits(
            outputs.logits,
            batch["labels"].float(),
        )
