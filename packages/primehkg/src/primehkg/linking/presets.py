"""Linking preset configurations for DefaultLinkPipeline."""

from __future__ import annotations

from typing import TypedDict


class LinkingPreset(TypedDict, total=False):
    enable_fuzzy: bool
    dense_retrieve_k: int
    rerank_weights: dict[str, float]
    min_score: float
    fuzzy_weights: tuple[float, float]


DEFAULT_LINKING: LinkingPreset = {
    "enable_fuzzy": True,
    "fuzzy_weights": (0.3, 0.7),
    "min_score": 0.0,
}

SAPBERT_LINKING: LinkingPreset = {
    "enable_fuzzy": False,
    "dense_retrieve_k": 50,
    "rerank_weights": {"cosine": 0.7, "jaccard": 0.1, "levenshtein": 0.2},
    "min_score": 0.6,
}
