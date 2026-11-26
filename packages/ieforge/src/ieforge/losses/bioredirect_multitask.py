"""BioREDirect relation-gated multi-label loss."""

from typing import Any

import torch
from torch import nn

from ieforge.losses.base import IELoss
from ieforge.registry import register


@register("loss", name="bioredirect_multitask")
class BioREDirectMultiTaskLoss(IELoss):
    """Multi-label BCE over 3 heads; novelty/direction only for positive relations.

    Mirrors the official BioREDirect training loop (run_exp.py:181-200): the
    relation head is always supervised; the novelty and direction heads are
    supervised only for instances whose gold relation is positive (i.e. not the
    negative class).
    """

    def __init__(
        self, negative_index: int = 0, name: str = "bioredirect_multitask"
    ) -> None:
        """Initialize with the class index that represents the negative relation."""
        super().__init__()
        self.name = name
        self.negative_index = negative_index
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, outputs: Any, batch: dict[str, Any]) -> torch.Tensor:
        """Compute relation-gated multi-task BCE loss.

        Faithful port of run_exp.py:181-206. Per instance the relation head is
        always summed over classes. For relation-positive instances with
        ``rel.sum() > 0`` the novelty and direction heads are added with the
        official self-weighting factor ``head.sum() / (rel + nov + dir).sum()``.
        The summed batch loss is normalized by the batch size.
        """
        rel_labels = batch["relation_labels"]
        nov_labels = batch["novelty_labels"]
        dir_labels = batch["direction_labels"]
        rel = self._bce(outputs.relation_logits, rel_labels).sum(dim=-1)  # (B,)
        nov = self._bce(outputs.novelty_logits, nov_labels).sum(dim=-1)  # (B,)
        direction = self._bce(outputs.direction_logits, dir_labels).sum(dim=-1)  # (B,)
        positive = rel_labels[:, self.negative_index] < 0.5  # (B,) relation not None
        denom = rel + nov + direction
        # avoid div-by-zero; where denom == 0 the weighted terms are 0 anyway
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        nov_weighted = (nov / safe_denom) * nov
        dir_weighted = (direction / safe_denom) * direction
        # official gates the aux terms on relation-positive AND rel.sum() > 0
        gate = positive & (rel > 0)
        total = rel + (nov_weighted + dir_weighted) * gate
        return total.sum() / rel.shape[0]
