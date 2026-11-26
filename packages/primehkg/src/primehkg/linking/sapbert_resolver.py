"""SapBERT dense label linking resolver."""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

from primehkg.linking.embedding_index import LabelEmbeddingIndex
from primehkg.linking.hybrid_rerank import hybrid_rerank_score
from primehkg.linking.presets import SAPBERT_LINKING, LinkingPreset
from primehkg.linking.protocols import LinkCandidate, LinkMatchKind
from primehkg.linking.resolvers import _candidate
from primehkg.linking.sapbert import encode_texts_sapbert


class SapBERTLinkResolver:
    def __init__(
        self,
        index: LabelEmbeddingIndex,
        *,
        preset: LinkingPreset | None = None,
        encode_query: Callable[[str], Any] | None = None,
    ) -> None:
        self._index = index
        self._preset = preset or SAPBERT_LINKING
        self._encode_query = encode_query or self._default_encode_query

    @staticmethod
    def _default_encode_query(query: str) -> Any:
        import numpy as np

        vecs = encode_texts_sapbert([query])
        if vecs.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        return vecs[0]

    def resolve(
        self,
        query: str,
        *,
        namespace: str | None = None,
        node_type_hint: str | Collection[str] | None = None,
        allowed_node_types: Collection[str] | None = None,
        top_k: int = 5,
    ) -> list[LinkCandidate]:
        del node_type_hint, allowed_node_types
        if namespace is not None:
            return []
        q = (query or "").strip()
        if not q:
            return []

        dense_k = int(self._preset.get("dense_retrieve_k", 50))
        weights = dict(
            self._preset.get("rerank_weights") or SAPBERT_LINKING["rerank_weights"]
        )

        query_vec = self._encode_query(q)
        hits = self._index.cosine_topk(query_vec, k=dense_k)

        best_by_node: dict[str, LinkCandidate] = {}
        for row_idx, cosine in hits:
            surface_row = self._index.surfaces[row_idx]
            node_id = surface_row["node_id"]
            surface = surface_row["surface"]
            field = surface_row["field"]
            combined, breakdown = hybrid_rerank_score(
                q,
                surface,
                cosine=cosine,
                weights=weights,
            )
            cand = _candidate(
                node_id=node_id,
                score=combined,
                match_kind=LinkMatchKind.DENSE_LABEL,
                matched_surface=surface,
                matched_field=field,
                score_breakdown=breakdown,
            )
            prev = best_by_node.get(node_id)
            if prev is None or float(cand["score"]) > float(prev["score"]):
                best_by_node[node_id] = cand

        scored = sorted(
            best_by_node.values(),
            key=lambda c: (-float(c["score"]), str(c["node_id"])),
        )
        return scored[: max(0, int(top_k))]
