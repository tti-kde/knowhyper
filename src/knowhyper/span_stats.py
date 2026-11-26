"""Paired contrasts and Holm correction for the #218 span campaign.

Issue #218 Phase 3 fixes three comparison families and requires Holm correction
WITHIN each family:

1. ``real - shuffled`` per condition: 18 contrasts (2 backbones x 3 datasets x 3
   conditions) -- does knowledge CONTENT matter.
2. ``real - zero`` per condition: 18 contrasts -- does the knowledge PATH matter.
3. ``+attributes - Structure only`` and ``+pretraining - +attributes``: 12
   contrasts -- do the increments matter.

Pooling all 48 into one family would change which contrasts survive, so the split
is part of the pre-registered protocol rather than a caller's choice.

Headline claims use test; validation selects the checkpoint. A non-significant
contrast is NOT a negative result -- record delta, sd, t, p and wins either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from knowhyper.span_campaign import BACKBONES, CONDITIONS, DATASETS

_BACKBONE_NAMES = tuple(BACKBONES)


@dataclass(frozen=True)
class PairedResult:
    """A paired per-seed contrast."""

    delta: float
    sd: float
    t: float
    p: float
    n: int
    wins: int


@dataclass(frozen=True)
class Contrast:
    """One pre-registered comparison."""

    kind: str
    backbone: str
    dataset: str
    condition: str

    @property
    def label(self) -> str:
        return f"{self.kind}/{self.backbone}/{self.dataset}/{self.condition}"


def paired_delta(treated: list[float], control: list[float]) -> PairedResult:
    """Paired difference with a two-sided t test on the per-seed differences."""
    if len(treated) != len(control):
        msg = (
            "paired contrast needs one treated value per control value, got "
            f"{len(treated)} vs {len(control)}"
        )
        raise ValueError(msg)
    n = len(treated)
    if n < 2:
        msg = f"a paired contrast needs at least 2 seeds, got {n}"
        raise ValueError(msg)

    diffs = [a - b for a, b in zip(treated, control, strict=True)]
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(variance)
    wins = sum(1 for d in diffs if d > 0)

    if sd == 0.0:
        # Every seed moved by the same amount. A zero-variance non-zero shift is
        # infinitely significant; a zero shift is not a difference at all.
        t = 0.0 if mean == 0.0 else math.inf * (1.0 if mean > 0 else -1.0)
        p = 1.0 if mean == 0.0 else 0.0
        return PairedResult(delta=mean, sd=0.0, t=t, p=p, n=n, wins=wins)

    t = mean / (sd / math.sqrt(n))
    p = _two_sided_t_p(abs(t), n - 1)
    return PairedResult(delta=mean, sd=sd, t=t, p=p, n=n, wins=wins)


def _two_sided_t_p(t: float, df: int) -> float:
    """Two-sided p for Student's t via the regularised incomplete beta function."""
    if df <= 0:
        msg = f"t test needs df >= 1, got {df}"
        raise ValueError(msg)
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) via a continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x) / b


def _beta_cf(a: float, b: float, x: float, *, iterations: int = 200) -> float:
    """Lentz's algorithm for the beta continued fraction."""
    tiny = 1e-30
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for m in range(1, iterations + 1):
        numerator = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        numerator = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return result


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p values, returned in the INPUT order.

    Step-down: the i-th smallest p is multiplied by (m - i), then made monotone
    non-decreasing and capped at 1.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        scaled = p_values[index] * (m - rank)
        running = max(running, scaled)
        adjusted[index] = min(1.0, running)
    return adjusted


def contrast_families(
    *,
    backbones: tuple[str, ...] = _BACKBONE_NAMES,
    datasets: tuple[str, ...] = DATASETS,
    conditions: tuple[str, ...] = CONDITIONS,
) -> dict[str, list[Contrast]]:
    """Build the three pre-registered families for an explicit matrix scope."""
    real_minus_shuffled = [
        Contrast(
            kind="real_minus_shuffled",
            backbone=backbone,
            dataset=dataset,
            condition=condition,
        )
        for backbone in backbones
        for dataset in datasets
        for condition in conditions
    ]
    real_minus_zero = [
        Contrast(
            kind="real_minus_zero",
            backbone=backbone,
            dataset=dataset,
            condition=condition,
        )
        for backbone in backbones
        for dataset in datasets
        for condition in conditions
    ]
    increments = [
        Contrast(kind=kind, backbone=backbone, dataset=dataset, condition=condition)
        for backbone in backbones
        for dataset in datasets
        for kind, condition in (
            ("attributes_minus_structure", "ev"),
            ("pretraining_minus_attributes", "ev_warm"),
        )
    ]
    return {
        "real_minus_shuffled": real_minus_shuffled,
        "real_minus_zero": real_minus_zero,
        "increments": increments,
    }
