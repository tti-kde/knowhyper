"""Seed-type-filtered vector-local candidate generator.

Adapts primehkg's ``VectorLocalCandidate`` so a modality index is queried ONLY by
seeds of the matching node type (protein index ← protein seeds; drug index ← drug
seeds), avoiding a nonsensical mean over mixed-type seed embeddings. Records each
emitted hyperedge's max contributing-neighbor similarity on ``self.last_similarity``
so a downstream ranker can boost these (near-zero-EWO) hyperedges past the top_k cut.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from primehkg.retrieval.protocols import IndexProvider
    from primehkg.retrieval.vector_index import VectorIndex


class TypedVectorLocalCandidate:
    """Seed-type-filtered vector-local candidate generator.

    Filters seeds by prefix, queries a modality embedding index for similar neighbor
    entities, and returns those neighbors' incident hyperedge ids as extra retrieval
    candidates. Records each emitted hyperedge's max neighbor similarity for use by
    a downstream ranker.
    """

    def __init__(
        self,
        *,
        node_index: VectorIndex,
        seed_prefix: str,
        top_k_nodes: int = 50,
    ) -> None:
        self._node_index = node_index
        self._seed_prefix = seed_prefix
        self._top_k_nodes = top_k_nodes
        self.last_similarity: dict[str, float] = {}

    def generate(
        self,
        *,
        seed_node_ids: Sequence[str],
        query_id: str,
        index: IndexProvider,
        hyperedges: Sequence[Mapping[str, object]],
        query_text: str | None = None,
    ) -> list[str]:
        del query_id, hyperedges, query_text
        self.last_similarity = {}

        typed_seeds = [s for s in seed_node_ids if str(s).startswith(self._seed_prefix)]
        seed_vecs: list[np.ndarray] = []
        for nid in typed_seeds:
            vec = self._node_index.lookup(str(nid))
            if vec is not None:
                seed_vecs.append(vec)
        if not seed_vecs:
            return []

        mean_vec = np.mean(seed_vecs, axis=0).astype(np.float32)
        similar_nodes = self._node_index.topk(mean_vec, self._top_k_nodes)

        seed_set = frozenset(str(s) for s in seed_node_ids)
        inv = index.node_to_hyperedge_ids()
        seed_he_ids: set[str] = set()
        for nid in seed_node_ids:
            for hid in inv.get(str(nid), []):
                seed_he_ids.add(str(hid))

        sim_by_he: dict[str, float] = {}
        for nid, score in similar_nodes:
            if nid in seed_set:
                continue
            for hid in inv.get(nid, []):
                hid_s = str(hid)
                if hid_s in seed_he_ids:
                    continue
                prev = sim_by_he.get(hid_s)
                if prev is None or score > prev:
                    sim_by_he[hid_s] = float(score)

        self.last_similarity = sim_by_he
        return sorted(sim_by_he.keys())
