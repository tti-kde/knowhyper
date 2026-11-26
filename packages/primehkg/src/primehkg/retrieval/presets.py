"""Named parameter bundles for ``retrieve_hyperedges`` / batch retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from primehkg.retrieval.batch import QuerySpec


class RetrievalScoringPreset(TypedDict, total=False):
    """Scoring kwargs shared by ``retrieve_hyperedges`` and ``retrieve_hyperedges_batch``."""

    hub_penalty: float
    hub_degree_flag_threshold: int
    semantics_weight: dict[str, float]
    max_hyperedge_size: int | None


class RetrievalPreset(RetrievalScoringPreset, total=False):
    """``retrieve_hyperedges`` / ``QuerySpec`` preset (optional per-query ``top_k``)."""

    top_k: int


# Down-rank hub-dominated phenotype neighborhoods on full PrimeKG coverage builds.
# See docs/superpowers/specs/2026-05-20-phenotype-hub-mitigation-design.md
PHENOTYPE_HUB_RETRIEVAL: Final[RetrievalScoringPreset] = {
    "hub_penalty": 2.0,
    "hub_degree_flag_threshold": 10_000,
    "semantics_weight": {"drug_disease_phenotype": 0.85},
    "max_hyperedge_size": 8,
}

DEFAULT_RETRIEVAL: Final[RetrievalPreset] = {
    "hub_penalty": 1.0,
    "hub_degree_flag_threshold": 100,
}

# Suggested batch defaults when many queries use phenotype / effect/phenotype seeds.
PHENOTYPE_HUB_BATCH_DEFAULTS: Final[RetrievalPreset] = {
    **PHENOTYPE_HUB_RETRIEVAL,
    "top_k": 50,
}


def make_query_spec(
    query_id: str,
    seed_node_ids: Sequence[str],
    *,
    preset: RetrievalPreset | RetrievalScoringPreset | None = None,
    top_k: int | None = None,
    hub_penalty: float | None = None,
    hub_degree_flag_threshold: int | None = None,
    max_hyperedge_size: int | None = None,
    semantics_weight: dict[str, float] | None = None,
) -> QuerySpec:
    """Build a ``QuerySpec`` with optional preset fields and per-query overrides."""
    spec: QuerySpec = {
        "query_id": query_id,
        "seed_node_ids": list(seed_node_ids),
    }
    if preset is not None:
        preset_top_k = preset.get("top_k")
        if preset_top_k is not None:
            spec["top_k"] = int(preset_top_k)
        if "hub_penalty" in preset:
            spec["hub_penalty"] = float(preset["hub_penalty"])
        if "hub_degree_flag_threshold" in preset:
            spec["hub_degree_flag_threshold"] = int(preset["hub_degree_flag_threshold"])
        if "max_hyperedge_size" in preset and preset["max_hyperedge_size"] is not None:
            spec["max_hyperedge_size"] = int(preset["max_hyperedge_size"])
        if "semantics_weight" in preset:
            spec["semantics_weight"] = dict(preset["semantics_weight"])
    if top_k is not None:
        spec["top_k"] = top_k
    if hub_penalty is not None:
        spec["hub_penalty"] = hub_penalty
    if hub_degree_flag_threshold is not None:
        spec["hub_degree_flag_threshold"] = hub_degree_flag_threshold
    if max_hyperedge_size is not None:
        spec["max_hyperedge_size"] = max_hyperedge_size
    if semantics_weight is not None:
        spec["semantics_weight"] = semantics_weight
    return spec
