"""Ranker wrapper that adds a similarity bonus to vector-derived hyperedges.

primehkg's EWO score is ``|seed ∩ he_nodes|``-driven; a hyperedge pulled in by a
sequence/SMILES-similar NEIGHBOR contains the neighbor, not the seed, so its EWO is
~0 and it is dropped at the top_k cut. This wrapper scores all candidates, ADDS
``boost * similarity`` (a multiplier cannot rescue a zero base), re-sorts, and
truncates — so similarity-justified neighbor hyperedges can survive ranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Literal

    from primehkg.retrieval.defaults import EwoRanker
    from primehkg.retrieval.retrieve import RetrievalResult

    from knowhyper.ieforge.typed_vector_candidate import TypedVectorLocalCandidate


class VectorBoostRanker:
    """Wrapper around EwoRanker that adds similarity boost to vector-derived hyperedges."""

    def __init__(
        self,
        inner: EwoRanker,
        *,
        candidates: Sequence[TypedVectorLocalCandidate],
        boost: float = 1.0,
    ) -> None:
        self._inner = inner
        self._candidates = list(candidates)
        self._boost = boost

    def _similarity_for(self, he_id: str) -> float:
        best = 0.0
        for cand in self._candidates:
            sim = cand.last_similarity.get(he_id)
            if sim is not None and sim > best:
                best = sim
        return best

    def rank(
        self,
        *,
        seed_node_ids: Sequence[str],
        candidate_hyperedge_ids: Sequence[str],
        hyperedges: Sequence[Mapping[str, object]],
        inverted_index: Mapping[str, list[str]],
        top_k: int,
        reproducibility: Literal["strict", "fast"],
        hub_penalty: float,
        hub_degree_flag_threshold: int,
        semantics_weight: Mapping[str, float] | None,
        evidence_weight: Mapping[str, float] | None,
        max_hyperedge_size: int | None,
        hyperedge_lookup: Callable[[str], Mapping[str, object] | None] | None = None,
    ) -> list[RetrievalResult]:
        cand_ids = list(candidate_hyperedge_ids)
        # Score ALL candidates (top_k=len) so the additive boost is applied before
        # truncation — EwoRanker truncates internally otherwise.
        results = self._inner.rank(
            seed_node_ids=seed_node_ids,
            candidate_hyperedge_ids=cand_ids,
            hyperedges=hyperedges,
            inverted_index=inverted_index,
            top_k=len(cand_ids),
            reproducibility=reproducibility,
            hub_penalty=hub_penalty,
            hub_degree_flag_threshold=hub_degree_flag_threshold,
            semantics_weight=semantics_weight,
            evidence_weight=evidence_weight,
            max_hyperedge_size=max_hyperedge_size,
            hyperedge_lookup=hyperedge_lookup,
        )

        if self._boost != 0.0:
            for r in results:
                bonus = self._boost * self._similarity_for(r["he_id"])
                if bonus:
                    new_score = float(r["score"]) + bonus
                    r["score"] = new_score
                    r["score_breakdown"]["final_score"] = new_score

        results.sort(key=lambda r: (-float(r["score"]), r["he_id"]))
        return results[:top_k]
