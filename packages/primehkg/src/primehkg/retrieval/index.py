from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from primehkg.ids import normalize_node_id

if TYPE_CHECKING:
    from datasets import Dataset

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def build_inverted_index(
    hyperedges: Iterable[Mapping[str, object]],
) -> dict[str, list[str]]:
    """Build a minimal inverted index: node_id -> sorted list of he_id.

    This is intentionally simple and deterministic for v0.
    """
    idx: dict[str, set[str]] = defaultdict(set)
    for he in hyperedges:
        he_id_obj = he.get("he_id")
        node_ids_obj = he.get("node_ids")
        if not isinstance(he_id_obj, str):
            continue
        if not isinstance(node_ids_obj, list):
            continue
        for n in node_ids_obj:
            if not isinstance(n, str):
                continue
            nn = normalize_node_id(n)
            if not nn:
                continue
            idx[nn].add(he_id_obj)

    return {node_id: sorted(he_ids) for node_id, he_ids in sorted(idx.items())}


def build_inverted_index_from_dataset(hyperedges: Dataset) -> dict[str, list[str]]:
    """Build inverted index by streaming a ``Dataset`` without list materialization."""
    return build_inverted_index(hyperedges)


def node_degrees(inverted_index: Mapping[str, list[str]]) -> dict[str, int]:
    """Compute node degrees from an inverted index."""
    return {node_id: len(he_ids) for node_id, he_ids in inverted_index.items()}
