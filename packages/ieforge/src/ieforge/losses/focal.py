from typing import Any

import torch
from torch import nn

from ieforge.losses.base import IELoss
from ieforge.losses.span_bce import _flatten_span_targets
from ieforge.registry import register


@register("loss", name="focal")
class SpanFocalLoss(IELoss):
    """Focal loss for span classification with extreme class imbalance."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        name: str = "focal",
    ) -> None:
        """Initialize the loss with focusing parameter gamma and balancing alpha."""
        super().__init__()
        self.name = name
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs: Any, batch: dict[str, Any]) -> torch.Tensor:
        """Compute masked focal loss from span logits and one-hot span labels."""
        logits, labels, span_mask = _flatten_span_targets(outputs, batch)

        p = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )

        if self.gamma > 0:
            p_t = p * labels + (1 - p) * (1 - labels)
            focal_weight = (1 - p_t) ** self.gamma
            loss = focal_weight * bce
        else:
            loss = bce

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            loss = alpha_t * loss

        mask = span_mask.unsqueeze(-1).expand_as(loss)
        loss = loss * mask
        num_valid = mask.sum().clamp(min=1)
        return loss.sum() / num_valid
