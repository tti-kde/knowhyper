"""Presets for budgeted subhypergraph sampling (orthogonal to EWO scoring)."""

from __future__ import annotations

from typing import TypedDict

from primehkg.retrieval.batch import DEFAULT_KHOP_BUDGETS


class SubhypergraphPreset(TypedDict, total=False):
    """Budget knobs for k-hop subhypergraph sampling."""

    max_he_level1: int
    max_nodes_level2: int
    max_he_level3: int


DEFAULT_SUBHYPERGRAPH: SubhypergraphPreset = {
    "max_he_level1": DEFAULT_KHOP_BUDGETS.max_he_level1,
    "max_nodes_level2": DEFAULT_KHOP_BUDGETS.max_nodes_level2,
    "max_he_level3": DEFAULT_KHOP_BUDGETS.max_he_level3,
}

PHENOTYPE_HUB_SUBHYPERGRAPH: SubhypergraphPreset = {
    "max_he_level1": 200,
    "max_nodes_level2": 200,
    "max_he_level3": 200,
}
