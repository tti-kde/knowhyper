from typing import Any

import torch
from torch import nn

from ieforge.losses.base import IELoss
from ieforge.registry import register


def _flatten_span_targets(
    outputs: Any, batch: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (logits, labels, span_mask) flattened to [B, W*K, C] / [B, W*K].

    The model emits span logits as `[B, W, K, C]` while the collator stores
    labels as `[B, W*K, C]`; reshaping the former makes the two line up.
    """
    logits = outputs.logits
    labels = batch["span_labels"]
    span_mask = batch["span_mask"]

    if logits.dim() == 4:
        b, w, k, c = logits.shape
        logits = logits.reshape(b, w * k, c)
        labels = labels.reshape(b, w * k, c)
        span_mask = span_mask.reshape(b, w * k)

    return logits, labels, span_mask


@register("loss", name="span_bce")
class SpanBCELoss(IELoss):
    """Masked binary cross-entropy over span candidates.

    Control condition for the focal loss: identical masking and normalisation,
    no focal reweighting. `pos_weight` scales the positive term, which matters
    because span enumeration is extremely imbalanced -- roughly one positive per
    200 candidates on BC5CDR-Chem.
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        name: str = "span_bce",
    ) -> None:
        """Store the positive-class weight."""
        super().__init__()
        self.name = name
        if pos_weight <= 0:
            msg = f"pos_weight must be positive, got {pos_weight}"
            raise ValueError(msg)
        self.pos_weight = pos_weight

    def forward(self, outputs: Any, batch: dict[str, Any]) -> torch.Tensor:
        """Compute mean masked BCE over valid span candidates."""
        logits, labels, span_mask = _flatten_span_targets(outputs, batch)

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
            pos_weight=torch.as_tensor(
                self.pos_weight, device=logits.device, dtype=logits.dtype
            ),
        )

        mask = span_mask.unsqueeze(-1).expand_as(loss)
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)
