"""Rescue-only late logit head for #231 Variant 1.

The head produces a per-candidate, per-label logit DELTA that is added to the frozen
baseline's logits. #231 requires three properties simultaneously, and the obvious
implementations satisfy only two:

* **non-negative** -- Chem's measured failure under cross-attention is recall loss
  (#218: mid-confidence recall down, precision unharmed), so the first variant must not be
  able to push an existence logit DOWN.
* **near-zero at init** -- the frozen baseline has to be reproduced on the first forward,
  or the contrast measures a perturbation rather than knowledge.
* **live gradient** -- `ReLU(raw)` with `raw` starting at 0 returns exactly 0 with a DEAD
  gradient (PyTorch defines ReLU'(0) = 0), so the head can never leave the origin. That
  implementation passes a non-negativity check and an initial-magnitude check and is
  still useless, which is why `test_the_gradient_is_alive_at_initialisation` exists.

`softplus` gives strict positivity with a finite gradient everywhere, but softplus ALONE
cannot deliver the other two properties together. Measured: in the tail its value and its
derivative are equal, so pushing the initial delta under the 1e-4 tolerance necessarily
pushes the gradient under it too.

| bias | softplus(bias) | d/dx | inside tolerance? | trainable? |
|---|---|---|---|---|
| -12 | 6.1e-06 | 6.1e-06 | yes | **no** |
| -10 | 4.5e-05 | 4.5e-05 | yes | **no** |
| -9 | 1.2e-04 | 1.2e-04 | **no** | yes |

A bias of -12 was the first attempt and it fails in practice, not merely in theory: 50 SGD
steps at lr=1.0 moved the bias by 0.001 and left the delta at 6.1e-06. The liveness test
passed (the gradient is non-zero) while the head was untrainable -- the `ReLU(0)` trap in a
subtler form.

So magnitude and gradient are DECOUPLED: a `softplus` term that starts near 1 (ordinary
gradient) is multiplied by a learnable `scale` that starts at the tolerance. The scale's
gradient is the softplus value (~1), so it trains immediately; the product is small because
the scale is small, not because the gradient is.

Reliability multiplies the delta by the candidate's first-pass link confidence, so a
weakly linked candidate cannot receive a larger intervention than a confidently linked
one. Everything is computed per candidate; no candidate sees another's evidence.
"""

from __future__ import annotations

import torch
from torch import nn


class RescueOnlyLateHead(nn.Module):
    """Per-candidate non-negative logit delta, near-zero at initialisation.

    Shapes: `span_rep` and `evidence` are `[B, S, D]`, `link_score` is `[B, S]`, and the
    returned delta is `[B, S, num_labels]`.
    """

    # The initial delta must be below this. Chosen against the smallest contrast #218 ever
    # resolved at n=10 (0.0040 F1): a starting perturbation two orders of magnitude under
    # that cannot be mistaken for an effect, while leaving softplus a live gradient.
    INITIAL_DELTA_TOLERANCE = 1e-4

    # The raw score starts near 0, so softplus(raw) starts near log(2) ~ 0.69 with an
    # ordinary O(1) gradient. Smallness comes from `scale` below, not from the activation.
    _INITIAL_SCALE = INITIAL_DELTA_TOLERANCE / 10.0

    def __init__(
        self,
        *,
        span_dim: int,
        evidence_dim: int,
        num_labels: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.span_dim = span_dim
        self.evidence_dim = evidence_dim
        self.num_labels = num_labels
        hidden = hidden_dim if hidden_dim is not None else span_dim

        # Interaction term: the elementwise product lets the head condition on whether the
        # evidence AGREES with the span, not merely on the evidence alone -- a head fed
        # only `evidence` could learn "any evidence raises the logit", which is precisely
        # the content-insensitivity #218 measured (real/shuffled gate ratio 0.95-1.01).
        self.project = nn.Linear(span_dim + evidence_dim + span_dim + 1, hidden)
        self.score = nn.Linear(hidden, num_labels)
        self.activation = nn.GELU()

        # Start the raw score near 0 so softplus sits near log(2) with an O(1) gradient.
        # Only the BIAS is zeroed: zeroing `score.weight` too would make `raw` constant,
        # which kills the gradient into `project` and makes the head candidate-BLIND --
        # it then ignores its evidence entirely, exactly the content-insensitivity this
        # variant exists to fix. Caught by the candidate-locality test.
        nn.init.zeros_(self.score.bias)
        # A learnable per-label scale carries the smallness. Its gradient is the softplus
        # value (~0.69), so it moves on the first step -- unlike a deep negative bias,
        # whose gradient is as small as the value it produces.
        self.scale = nn.Parameter(torch.full((num_labels,), self._INITIAL_SCALE))

    def forward(
        self,
        *,
        span_rep: torch.Tensor,
        evidence: torch.Tensor,
        link_score: torch.Tensor,
    ) -> torch.Tensor:
        if evidence.shape[-1] != self.evidence_dim:
            msg = (
                "evidence dimension does not match the head: "
                f"got {evidence.shape[-1]}, expected {self.evidence_dim}"
            )
            raise ValueError(msg)
        if span_rep.shape[-1] != self.span_dim:
            msg = (
                "span dimension does not match the head: "
                f"got {span_rep.shape[-1]}, expected {self.span_dim}"
            )
            raise ValueError(msg)
        if span_rep.shape[:-1] != evidence.shape[:-1]:
            msg = (
                "span and evidence must align on batch/candidate axes: "
                f"{tuple(span_rep.shape)} vs {tuple(evidence.shape)}"
            )
            raise ValueError(msg)

        # The head's parameters are float32 while the campaign trains under bf16-mixed, so
        # compute in float32 and cast back. Running the Linear in bf16 against float32
        # weights raises rather than autocasting when called outside an autocast region,
        # which is how the unit tests exercise it.
        out_dtype = span_rep.dtype
        span32 = span_rep.float()
        evidence32 = evidence.float()
        reliability = link_score.float().unsqueeze(-1).clamp(min=0.0, max=1.0)
        features = torch.cat(
            [span32, evidence32, span32 * evidence32, reliability], dim=-1
        )
        raw = self.score(self.activation(self.project(features)))
        # softplus, not ReLU: strictly positive with a finite gradient at every point.
        magnitude = nn.functional.softplus(raw)
        # `scale.abs()` keeps the product non-negative however the scale is updated; a bare
        # `scale` would let a gradient step take it negative and silently turn the
        # rescue-only head into one that can suppress logits.
        delta = magnitude * self.scale.abs()
        # Reliability scales the intervention; it never changes its sign.
        return (delta * reliability).to(out_dtype)
