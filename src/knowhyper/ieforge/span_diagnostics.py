"""Accumulate the #218 Phase 2 span diagnostics into a per-cell summary.

`SpanCrossAttentionKnowledgeModel.forward` attaches a per-batch record to its output
(`knowledge_diagnostics`). This module reduces a stream of those records to the numbers
Phase 2 asks each cell to save, so the harness writes one small JSON instead of raw
per-candidate tensors.

Why summarise rather than dump: the record is per CANDIDATE, and a 12,574-instance
train split at max_span_width 12 is tens of millions of candidates. The Phase 2 stop
conditions are all statements about distributions -- "the knowledge branch gradient is
persistently zero", "the gate collapsed", "the residual never moves" -- so means,
extrema and counts answer them while a dump would not fit beside the checkpoint.

Only masked candidates count. Padding candidates have a zero residual by construction,
and averaging over them drags every mean toward zero -- which looks exactly like the
dead branch this record exists to detect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class _Welford:
    """Streaming mean/extrema over an unknown number of batches."""

    count: int = 0
    total: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        flat = values.detach().reshape(-1).float()
        self.count += int(flat.numel())
        self.total += float(flat.sum())
        self.minimum = min(self.minimum, float(flat.min()))
        self.maximum = max(self.maximum, float(flat.max()))

    def as_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"n": 0, "mean": None, "min": None, "max": None}
        return {
            "n": self.count,
            "mean": self.total / self.count,
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass
class SpanDiagnosticsAccumulator:
    """Reduce per-batch diagnostics to the Phase 2 per-cell record."""

    control: str | None = None
    batches: int = 0
    candidates_total: int = 0
    candidates_masked: int = 0
    with_evidence: int = 0
    linked: int = 0
    attn_norm: _Welford = field(default_factory=_Welford)
    residual_norm: _Welford = field(default_factory=_Welford)
    logit_delta_abs: _Welford = field(default_factory=_Welford)
    gate_relevance: _Welford = field(default_factory=_Welford)
    gate_present: bool = False
    # Counted separately from the mean: a residual that is zero on candidates that DO
    # own evidence is the dead-branch signature, and a mean cannot show it.
    zero_residual_with_evidence: int = 0

    def update(
        self, diagnostics: dict[str, Any], *, span_mask: torch.Tensor | None = None
    ) -> None:
        """Fold one batch's record in.

        `span_mask` is the public [B, W, K] candidate mask; when given, only masked
        candidates contribute. Padding candidates are structurally zero, so including
        them biases every mean toward the dead-branch reading.
        """
        control = diagnostics.get("control")
        if self.control is None:
            self.control = control
        elif control != self.control:
            msg = (
                f"diagnostics mix controls {self.control!r} and {control!r}; one "
                "accumulator covers one arm"
            )
            raise ValueError(msg)

        has_evidence = diagnostics["has_evidence"]
        link_mask = diagnostics["link_mask"]
        residual = diagnostics["residual_norm"]
        attn = diagnostics["attn_norm"]
        delta = diagnostics["logit_delta"]

        self.batches += 1
        self.candidates_total += int(has_evidence.numel())

        keep = torch.ones_like(has_evidence, dtype=torch.bool)
        if span_mask is not None:
            keep = span_mask.reshape(has_evidence.shape).bool()
        self.candidates_masked += int(keep.sum())
        self.with_evidence += int((has_evidence.bool() & keep).sum())
        self.linked += int((link_mask.bool() & keep).sum())

        self.attn_norm.update(attn[keep])
        self.residual_norm.update(residual[keep])
        # |logit delta| per candidate: collapse the type axis, which is 1 for every
        # dataset in this campaign but must not be assumed away silently.
        self.logit_delta_abs.update(
            delta.abs().amax(dim=-1).reshape(-1)[keep.reshape(-1)]
        )

        evidenced = has_evidence.bool() & keep
        if bool(evidenced.any()):
            self.zero_residual_with_evidence += int((residual[evidenced] == 0).sum())

        relevance = diagnostics.get("gate_relevance")
        if relevance is not None:
            self.gate_present = True
            self.gate_relevance.update(relevance[keep])

    def as_dict(self) -> dict[str, Any]:
        """Return the record Phase 2 saves per cell."""
        return {
            "control": self.control,
            "batches": self.batches,
            "candidates": {
                "total": self.candidates_total,
                "masked": self.candidates_masked,
                "with_evidence": self.with_evidence,
                "linked": self.linked,
                # knowledge-empty = a masked candidate that received no keys. The
                # stratum Phase 2 wants counted, not inferred from a difference.
                "knowledge_empty": self.candidates_masked - self.with_evidence,
            },
            "attn_norm": self.attn_norm.as_dict(),
            "residual_norm": self.residual_norm.as_dict(),
            "logit_delta_abs": self.logit_delta_abs.as_dict(),
            "gate": {
                "present": self.gate_present,
                **(
                    {"relevance": self.gate_relevance.as_dict()}
                    if self.gate_present
                    else {}
                ),
            },
            "zero_residual_with_evidence": self.zero_residual_with_evidence,
            # The two readings Phase 2 stops on. Reported as explicit booleans rather
            # than left for a reader to derive from the means, because "the branch is
            # dead" and "the effect is null" have the same aggregate F1.
            "verdicts": {
                "residual_always_zero": self.residual_norm.count > 0
                and self.residual_norm.maximum == 0.0,
                "attn_always_zero": self.attn_norm.count > 0
                and self.attn_norm.maximum == 0.0,
                "gate_collapsed": self.gate_present
                and self.gate_relevance.count > 0
                and self.gate_relevance.minimum == self.gate_relevance.maximum,
                "no_candidate_had_evidence": self.candidates_masked > 0
                and self.with_evidence == 0,
            },
        }
