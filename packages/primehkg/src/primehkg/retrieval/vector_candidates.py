from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from primehkg.retrieval.protocols import IndexProvider
    from primehkg.retrieval.vector_index import VectorIndex


def vector_determinism_version(
    *,
    model_id: str,
    ann_backend: str = "brute_force",
    top_k_nodes: int,
    top_k_he: int,
) -> str:
    return f"vector:{model_id}:{ann_backend}:{top_k_nodes}:{top_k_he}"


class VectorLocalCandidate:
    """Node embedding ANN → similar nodes → their incident HEs as candidates."""

    def __init__(
        self,
        *,
        node_index: VectorIndex,
        top_k_nodes: int = 50,
    ) -> None:
        self._node_index = node_index
        self._top_k_nodes = top_k_nodes

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

        seed_vecs: list[np.ndarray] = []
        for nid in seed_node_ids:
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

        cands: set[str] = set()
        for nid, _score in similar_nodes:
            if nid in seed_set:
                continue
            for hid in inv.get(nid, []):
                hid_s = str(hid)
                if hid_s not in seed_he_ids:
                    cands.add(hid_s)

        return sorted(cands)


class VectorGlobalCandidate:
    """HE embedding ANN → directly matched HEs as candidates."""

    def __init__(
        self,
        *,
        he_index: VectorIndex,
        node_index: VectorIndex,
        top_k_he: int = 100,
        encode_fn: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self._he_index = he_index
        self._node_index = node_index
        self._top_k_he = top_k_he
        self._encode_fn = encode_fn

    def generate(
        self,
        *,
        seed_node_ids: Sequence[str],
        query_id: str,
        index: IndexProvider,
        hyperedges: Sequence[Mapping[str, object]],
        query_text: str | None = None,
    ) -> list[str]:
        del query_id, index, hyperedges

        query_vec: np.ndarray | None = None

        if query_text is not None and self._encode_fn is not None:
            query_vec = np.asarray(self._encode_fn(query_text), dtype=np.float32)
        else:
            seed_vecs: list[np.ndarray] = []
            for nid in seed_node_ids:
                vec = self._node_index.lookup(str(nid))
                if vec is not None:
                    seed_vecs.append(vec)
            if seed_vecs:
                query_vec = np.mean(seed_vecs, axis=0).astype(np.float32)

        if query_vec is None:
            return []

        results = self._he_index.topk(query_vec, self._top_k_he)
        return sorted(he_id for he_id, _score in results)
