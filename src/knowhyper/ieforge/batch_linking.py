"""Batched SapBERT entity linking for fast precompute.

The standard ``PrimeHKGLinker.link()`` encodes ONE mention at a time through
SapBERT and does a single-vector cosine search per mention. For N mentions
across a dataset this means N transformer forward passes and N matrix-vector
multiplies — the dominant precompute cost (~118 ms/instance on CPU).

This module provides a batch-oriented alternative:

1. Collect ALL mention texts across the dataset.
2. GPU-batch encode them in one ``cached_encode_texts_sapbert`` call (batch_size
   chunks, but the model loads once and stays on GPU).
3. Batch cosine top-k via a single matrix-matrix multiply (``embeddings @ queries.T``)
   followed by per-query ``argpartition``.
4. Run the cheap exact resolvers + hybrid re-rank per mention (these are fast).

The result is identical to sequential linking — same model, same scoring,
same candidate selection — but ~10-20x faster because the GPU processes all
queries in a few large batches instead of hundreds of single-vector passes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from primehkg.linking.embedding_index import LabelEmbeddingIndex

    from knowhyper.pipeline.types import LinkedEntity, Mention

logger = logging.getLogger(__name__)


def batch_cosine_topk(
    index: LabelEmbeddingIndex,
    query_vecs: np.ndarray,
    k: int,
) -> list[list[tuple[int, float]]]:
    """Batched cosine top-k: one matrix multiply for all queries at once.

    ``query_vecs`` is ``(Q, D)`` L2-normalized. Returns a list of Q result
    lists, each containing up to ``k`` ``(row_index, cosine_score)`` tuples
    sorted by descending score — identical to calling ``index.cosine_topk``
    Q times but ~10-50x faster for Q > 50.
    """
    if k <= 0 or query_vecs.shape[0] == 0:
        return [[] for _ in range(query_vecs.shape[0])]

    scores = index._embeddings @ query_vecs.T  # noqa: SLF001  # (N_labels, Q)
    n_labels = scores.shape[0]
    kk = min(k, n_labels)
    if kk == 0:
        return [[] for _ in range(query_vecs.shape[0])]

    results: list[list[tuple[int, float]]] = []
    for qi in range(query_vecs.shape[0]):
        col = scores[:, qi]
        if kk == n_labels:
            idx = np.argsort(-col)
        else:
            part = np.argpartition(-col, kk - 1)[:kk]
            idx = part[np.argsort(-col[part])]
        results.append([(int(i), float(col[i])) for i in idx])
    return results


def batch_encode_mentions(
    texts: list[str],
    *,
    device: str = "auto",
    batch_size: int = 256,
) -> np.ndarray:
    """Encode all mention texts through SapBERT in one batched pass.

    Returns an ``(N, D)`` L2-normalized float32 array. Uses the cached encoder
    (``sapbert_cache``) so the model loads once and stays on the chosen device.
    """
    from knowhyper.ieforge.sapbert_cache import cached_encode_texts_sapbert

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    return cached_encode_texts_sapbert(texts, batch_size=batch_size, device=device)


def _resolve_exact(
    pipeline: Any,
    mention_text: str,
    *,
    namespace: str | None,
    node_type_hint: str | None,
    top_k: int,
) -> list[Any]:
    """Run ONLY the exact resolvers (no dense/fuzzy) for one mention."""
    from primehkg.linking.build import is_lexical_resolver

    q = (mention_text or "").strip()
    if not q:
        return []
    return [
        cand
        for resolver in pipeline._resolvers  # noqa: SLF001
        if not is_lexical_resolver(resolver)
        for cand in resolver.resolve(
            q, namespace=namespace, node_type_hint=node_type_hint, top_k=top_k
        )
    ]


def _dense_hits_to_candidates(
    index: LabelEmbeddingIndex,
    hits: list[tuple[int, float]],
    query_text: str,
    preset: dict[str, Any],
) -> list[Any]:
    """Convert dense cosine hits to LinkCandidate dicts with hybrid re-rank."""
    from primehkg.linking.hybrid_rerank import hybrid_rerank_score
    from primehkg.linking.protocols import LinkMatchKind
    from primehkg.linking.resolvers import _candidate

    weights = dict(preset.get("rerank_weights") or {})
    best_by_node: dict[str, Any] = {}
    for row_idx, cosine in hits:
        surface_row = index.surfaces[row_idx]
        node_id = surface_row["node_id"]
        surface = surface_row["surface"]
        field = surface_row["field"]
        combined, breakdown = hybrid_rerank_score(
            query_text, surface, cosine=cosine, weights=weights
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
    return sorted(
        best_by_node.values(),
        key=lambda c: (-float(c["score"]), str(c["node_id"])),
    )


def batch_link(
    linker: Any,
    mentions_per_instance: list[list[Mention]],
    *,
    device: str = "auto",
    batch_size: int = 256,
) -> list[list[LinkedEntity]]:
    """Batch-link all mentions across all instances in one GPU pass.

    Returns a list of linked-entity lists, one per instance, in the same order
    as ``mentions_per_instance``. The results are identical to calling
    ``linker.link(mentions)`` per instance, but much faster because all SapBERT
    encoding happens in a single batched GPU call.

    Falls back to sequential linking when the linker has no SapBERT pipeline
    (e.g. exact-only mode without a label_embedding_index).
    """
    from primehkg.linking.build import is_lexical_resolver
    from primehkg.linking.pipeline import (
        _apply_node_type_hint,
        _link_candidate_rank,
        _prefer_link_candidate,
    )

    from knowhyper.ieforge.adapters import link_result_to_linked_entities

    pipeline = linker._ensure_pipeline()  # noqa: SLF001

    sapbert_resolver = None
    for r in pipeline._resolvers:  # noqa: SLF001
        if is_lexical_resolver(r) and hasattr(r, "_index") and hasattr(r, "_preset"):
            sapbert_resolver = r
            break

    if sapbert_resolver is None:
        logger.info("batch_link: no SapBERT resolver, falling back to sequential")
        return [linker.link(mentions) for mentions in mentions_per_instance]

    label_index: LabelEmbeddingIndex = sapbert_resolver._index  # noqa: SLF001
    preset: dict[str, Any] = dict(sapbert_resolver._preset)  # noqa: SLF001
    dense_k = int(preset.get("dense_retrieve_k", 50))
    min_score = float(preset.get("min_score", 0.0))

    all_texts: list[str] = []
    text_map: list[tuple[int, int, Mention]] = []
    for inst_idx, mentions in enumerate(mentions_per_instance):
        for m_idx, mention in enumerate(mentions):
            text = (mention.text or "").strip()
            if text and mention.namespace is None:
                all_texts.append(text)
                text_map.append((inst_idx, m_idx, mention))

    logger.info("batch_link: encoding %d mentions on %s", len(all_texts), device)
    if all_texts:
        all_vecs = batch_encode_mentions(
            all_texts, device=device, batch_size=batch_size
        )
        all_dense_hits = batch_cosine_topk(label_index, all_vecs, dense_k)
    else:
        all_dense_hits = []

    dense_candidates_by_pos: dict[tuple[int, int], list[Any]] = {}
    for idx, (inst_idx, m_idx, mention) in enumerate(text_map):
        hits = all_dense_hits[idx]
        dense_candidates_by_pos[(inst_idx, m_idx)] = _dense_hits_to_candidates(
            label_index, hits, mention.text, preset
        )

    results: list[list[LinkedEntity]] = [[] for _ in mentions_per_instance]
    for inst_idx, mentions in enumerate(mentions_per_instance):
        for m_idx, mention in enumerate(mentions):
            exact = _resolve_exact(
                pipeline,
                mention.text,
                namespace=mention.namespace,
                node_type_hint=mention.entity_type,
                top_k=linker._top_k,  # noqa: SLF001
            )
            dense = dense_candidates_by_pos.get((inst_idx, m_idx), [])

            merged: dict[str, Any] = {}
            for cand in exact:
                merged[cand["node_id"]] = cand
            for cand in dense:
                nid = cand["node_id"]
                prev = merged.get(nid)
                if prev is None or _prefer_link_candidate(cand, prev):
                    merged[nid] = cand

            scored = sorted(
                merged.values(),
                key=_link_candidate_rank,
                reverse=True,
            )
            if mention.entity_type:
                scored = _apply_node_type_hint(scored, mention.entity_type)

            if min_score > 0:
                scored = [c for c in scored if float(c["score"]) >= min_score]

            top = scored[: linker._top_k]  # noqa: SLF001
            link_result = {"query": mention.text, "candidates": top}

            if linker._strict_type_match and mention.entity_type:  # noqa: SLF001
                linked = linker._pick_typed_top(mention, link_result)  # noqa: SLF001
            else:
                linked = link_result_to_linked_entities(mention, link_result)

            if linked is None and linker._should_recall(mention):  # noqa: SLF001
                linked = linker._recall_link(mention)  # noqa: SLF001
            if linked is None and linker._should_expand(mention):  # noqa: SLF001
                linked = linker._expansion_link(mention)  # noqa: SLF001

            if linked is not None:
                results[inst_idx].append(linked)

    return results
