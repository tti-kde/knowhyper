"""Hybrid dense + lexical reranking for SapBERT linking."""

from __future__ import annotations

from primehkg.linking.fuzzy import normalized_levenshtein, token_jaccard
from primehkg.linking.xref_index import normalize_label


def hybrid_rerank_score(
    query: str,
    surface: str,
    *,
    cosine: float,
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    q = normalize_label(query)
    s = normalize_label(surface)
    j = token_jaccard(q, s)
    lev = normalized_levenshtein(q, s)
    wc = float(weights.get("cosine", 0.7))
    wj = float(weights.get("jaccard", 0.1))
    wl = float(weights.get("levenshtein", 0.2))
    combined = wc * cosine + wj * j + wl * lev
    return combined, {
        "cosine": cosine,
        "jaccard": j,
        "levenshtein": lev,
        "combined": combined,
    }
