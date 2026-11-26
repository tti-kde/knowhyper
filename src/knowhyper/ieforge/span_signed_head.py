"""The signed, bounded residual head for #233's candidate-owned branch.

Implements the pre-registered form

    z_final(s) = z_anchor(s) + valid(s) * eligible(s) * r(s),    r(s) = b * tanh(raw(s))

`text` and `lexical` ONLY. `graph` keeps `RescueOnlyLateHead`: its 400 recorded #231 cells ran
under a head that is non-negative by construction, so giving them the ability to suppress would
re-interpret completed results.

How this differs from `RescueOnlyLateHead`, and why each difference matters:

* **signed.** `softplus(...)` can only rescue. The registered residual is signed, so a
  candidate whose evidence argues against it can be suppressed.
* **bounded by a FIXED `b`.** `RescueOnlyLateHead` multiplies by a learnable `scale`, so its
  magnitude is bounded only at initialisation. Here `b` is a buffer, not a parameter: `tanh`
  gives `|r| <= b` for every possible raw score, permanently.
* **`valid` is a VALIDITY MASK, not a confidence.** For the text branch,
  `valid = candidate_evidence_mask`. The graph `link_score` / `link_mask` are NOT multiplied in
  again: a candidate that owns text has already been linked upstream, so re-gating on graph
  linkability would be double-conditioning -- and Phase 0 measured `link_score` at AUC
  0.58-0.61 with `link_mask` open ~100% of the time, i.e. no signal added and the treatment
  suppressed. They are recorded as DIAGNOSTICS instead.
* **`eligible` is a separate band**, decided train-only and committed before validation is read.

INITIALISATION. A fully zero final layer would satisfy "initial delta under tolerance" while
producing NO gradient to the evidence encoder -- the arm would report its initialisation and
look like a null. So the final layer gets a small non-zero init and `b` carries the smallness.
`test_signed_bounded_head.py` asserts both properties together.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from knowhyper.ieforge.span_late_head import RescueOnlyLateHead

#: Keeps the initial perturbation under the same tolerance the #231 head is held to, so an
#: initial shift can never be mistaken for an effect.
_INITIAL_TOLERANCE = RescueOnlyLateHead.INITIAL_DELTA_TOLERANCE


class SignedHeadError(ValueError):
    """The signed head cannot be built or used consistently."""


class SignedBoundedLateHead(nn.Module):
    """`delta = valid * eligible * b * tanh(raw(span, evidence))`.

    Shapes: `span_rep` and `evidence` are `[B, S, D]`; `valid` and `eligible` are `[B, S]`
    (bool or float); the returned delta is `[B, S, num_labels]`.
    """

    def __init__(
        self,
        *,
        span_dim: int,
        evidence_dim: int,
        num_labels: int,
        bound: float,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        if bound <= 0:
            msg = (
                f"bound must be positive, got {bound}. It is the hard cap on |delta|; a "
                "non-positive value would make the head a no-op that still trains."
            )
            raise SignedHeadError(msg)
        self.span_dim = int(span_dim)
        self.evidence_dim = int(evidence_dim)
        self.num_labels = int(num_labels)
        # A BUFFER, not a Parameter: a learnable bound would make "bounded" a claim about
        # initialisation rather than about the model.
        self.register_buffer("bound", torch.tensor(float(bound)))

        width = int(hidden) if hidden else self.span_dim
        # Same feature construction as the rescue-only head (span, evidence, their product)
        # so the two differ in the RESIDUAL FORM alone, not in what they can see. `valid` is
        # deliberately NOT a feature: it is a mask applied to the output, so it cannot leak
        # in as a magnitude.
        self.project = nn.Linear(3 * self.span_dim, width)
        self.score = nn.Linear(width, self.num_labels)
        # Small NON-zero init. Zeros would give `tanh(0)=0` with a zero weight, so the
        # gradient to the evidence encoder would be zero at step 0 and the encoder could
        # stall while every "initial delta is tiny" test passed.
        nn.init.normal_(self.score.weight, std=0.02)
        nn.init.zeros_(self.score.bias)
        # The smallness of the initial delta comes from this scale on the pre-tanh score,
        # NOT from a dead final layer: the gradient stays O(1) through tanh at 0.
        self._init_scale = float(_INITIAL_TOLERANCE) / float(bound)

    def forward(
        self,
        *,
        span_rep: torch.Tensor,
        evidence: torch.Tensor,
        valid: torch.Tensor,
        eligible: torch.Tensor,
    ) -> torch.Tensor:
        if evidence.shape[-1] != self.evidence_dim:
            msg = f"evidence width {evidence.shape[-1]} != evidence_dim {self.evidence_dim}"
            raise SignedHeadError(msg)
        if span_rep.shape[:2] != evidence.shape[:2]:
            msg = (
                f"span_rep {tuple(span_rep.shape[:2])} and evidence "
                f"{tuple(evidence.shape[:2])} must agree on [B, S]"
            )
            raise SignedHeadError(msg)
        for name, gate in (("valid", valid), ("eligible", eligible)):
            if gate.shape != span_rep.shape[:2]:
                msg = (
                    f"{name} must be [B, S] = {tuple(span_rep.shape[:2])}, got "
                    f"{tuple(gate.shape)}"
                )
                raise SignedHeadError(msg)

        out_dtype = span_rep.dtype
        span32 = span_rep.float()
        evidence32 = evidence.float()
        features = torch.cat([span32, evidence32, span32 * evidence32], dim=-1)
        raw = self.score(torch.tanh(self.project(features))) * self._init_scale
        # `b * tanh(raw)` -- signed, and |r| <= b for EVERY raw score, permanently.
        bound = cast("torch.Tensor", self.bound)
        residual = bound * torch.tanh(raw)

        # Masks, not magnitudes: booleans and 1.0-floats must give identical results, so a
        # confidence value cannot be smuggled in through `valid`.
        gate = (valid != 0).float() * (eligible != 0).float()
        return (residual * gate.unsqueeze(-1)).to(out_dtype)
