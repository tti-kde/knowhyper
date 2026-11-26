from typing import Any

import torch

from ieforge.losses.base import IELoss
from ieforge.registry import register


@register("loss", name="token_cross_entropy")
class TokenCrossEntropyLoss(IELoss):
    """Cross-entropy loss over token-level logits, ignoring padding positions."""

    def __init__(
        self,
        name: str = "token_cross_entropy",
        class_weight: list[float] | None = None,
    ) -> None:
        """Initialize the loss with optional per-class weights."""
        super().__init__()
        self.name = name
        self._class_weight = class_weight

    def forward(self, outputs: Any, batch: dict[str, Any]) -> torch.Tensor:
        """Return the model's built-in loss, or compute it from logits and labels."""
        if outputs.loss is not None and self._class_weight is None:
            return outputs.loss  # type: ignore[no-any-return]
        from torch.nn import functional

        logits = outputs.logits
        labels = batch["labels"]
        weight: torch.Tensor | None = None
        if self._class_weight is not None:
            weight = torch.tensor(
                self._class_weight, dtype=logits.dtype, device=logits.device
            )
        return functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            weight=weight,
            ignore_index=-100,
        )
