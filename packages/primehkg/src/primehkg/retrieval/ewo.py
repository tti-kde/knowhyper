from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class EwoBreakdown(TypedDict):
    """Typed breakdown for EWO scoring results."""

    ewo: float
    overlap: list[str]
    overlap_count: int
    per_entity_weight: dict[str, float]
    hub_penalty: float


def _hub_weight(degree: int, *, hub_penalty: float) -> float:
    """Downweight very frequent nodes (hub penalty).

    Deterministic, monotonic, bounded in (0, 1].
    """
    d = max(0, int(degree))
    hp = float(hub_penalty)
    if hp <= 0:
        return 1.0
    return 1.0 / ((1.0 + math.log1p(d)) ** hp)


def ewo_score(
    *,
    seed_node_ids: Iterable[str],
    hyperedge_node_ids: Iterable[str],
    node_degrees: Mapping[str, int] | None = None,
    hub_penalty: float = 1.0,
) -> EwoBreakdown:
    """Entity-Weighted Overlap (EWO) with hub downweighting.

    Returns a small explainable dict so callers can surface score breakdowns.
    """
    seed_set = {s for s in seed_node_ids if isinstance(s, str) and s}
    he_set = {n for n in hyperedge_node_ids if isinstance(n, str) and n}
    overlap = sorted(seed_set.intersection(he_set))

    degrees = node_degrees or {}
    per_entity_weight: dict[str, float] = {}
    ewo = 0.0
    for node_id in overlap:
        w = _hub_weight(int(degrees.get(node_id, 0)), hub_penalty=hub_penalty)
        per_entity_weight[node_id] = w
        ewo += w

    return {
        "ewo": ewo,
        "overlap": overlap,
        "overlap_count": len(overlap),
        "per_entity_weight": per_entity_weight,
        "hub_penalty": float(hub_penalty),
    }
