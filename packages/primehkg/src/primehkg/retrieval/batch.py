"""Batch retrieval over many seed queries (v2.0)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Required, TypedDict

from primehkg.determinism import RETRIEVAL_DETERMINISM_VERSION
from primehkg.retrieval.khop import (
    BudgetConfig,
    khop_candidates_k2,
    khop_determinism_version,
)
from primehkg.retrieval.retrieve import RetrievalResult, retrieve_hyperedges
from primehkg.retrieval.truncate import HashBasedTruncator

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from primehkg.retrieval.graph_csr.types import GraphCsrInMemory
    from primehkg.retrieval.truncate import Truncator

CandidateMode = Literal[
    "seed_incident",
    "khop_k2",
    "vector_local",
    "vector_global",
    "vector_hybrid",
    "khop_k2_plus_vector",
]

DEFAULT_KHOP_BUDGETS = BudgetConfig(
    max_he_level1=500,
    max_nodes_level2=500,
    max_he_level3=500,
)


class QuerySpec(TypedDict, total=False):
    """One query in a batch retrieval run."""

    query_id: Required[str]
    seed_node_ids: Required[Sequence[str]]
    query_text: str | None
    top_k: int
    hub_penalty: float
    semantics_weight: dict[str, float]
    evidence_weight: dict[str, float]
    max_hyperedge_size: int | None
    hub_degree_flag_threshold: int


class BatchQueryResult(TypedDict):
    """Ranked results for one query id."""

    query_id: str
    reproducibility: Literal["strict", "fast"]
    determinism_version: str | None
    candidate_mode: CandidateMode
    candidate_hyperedge_ids: list[str] | None
    results: list[RetrievalResult]


def batch_determinism_version(
    *,
    reproducibility: Literal["strict", "fast"],
    candidate_mode: CandidateMode,
    budgets: BudgetConfig,
    truncator: Truncator,
    inputs_fingerprint: str | None,
) -> str | None:
    """Compose strict-mode determinism version for batch retrieval."""
    if reproducibility != "strict":
        return None
    if candidate_mode == "seed_incident":
        return RETRIEVAL_DETERMINISM_VERSION
    if candidate_mode in ("vector_local", "vector_global", "vector_hybrid"):
        return RETRIEVAL_DETERMINISM_VERSION
    khop_v = khop_determinism_version(
        budgets=budgets,
        truncator=truncator,
        k=2,
        inputs_fingerprint=inputs_fingerprint,
    )
    return f"{RETRIEVAL_DETERMINISM_VERSION}+{khop_v}"


def retrieve_hyperedges_batch(
    queries: Sequence[QuerySpec],
    hyperedges: Sequence[Mapping[str, object]],
    *,
    reproducibility: Literal["strict", "fast"] = "strict",
    candidate_mode: CandidateMode = "seed_incident",
    budgets: BudgetConfig | None = None,
    truncator: Truncator | None = None,
    graph_csr: GraphCsrInMemory | None = None,
    inputs_fingerprint: str | None = None,
    inverted_index: Mapping[str, list[str]] | None = None,
    default_top_k: int = 20,
    hub_penalty: float = 1.0,
    semantics_weight: Mapping[str, float] | None = None,
    evidence_weight: Mapping[str, float] | None = None,
    max_hyperedge_size: int | None = None,
    hub_degree_flag_threshold: int = 100,
) -> list[BatchQueryResult]:
    """Run ``retrieve_hyperedges`` for many queries over one shared hyperedge slice.

    Queries are processed in **sorted ``query_id`` order** so batch output order is
    deterministic for a given input list.

    When ``candidate_mode="khop_k2"``, candidates come from ``khop_candidates_k2``
    (budgeted k=2 expansion) before EWO ranking. ``determinism_version`` in strict
    mode composes the v2 retrieval version with ``khop_determinism_version``.
    """
    khop_budgets = budgets if budgets is not None else DEFAULT_KHOP_BUDGETS
    khop_truncator = truncator if truncator is not None else HashBasedTruncator()
    det_version = batch_determinism_version(
        reproducibility=reproducibility,
        candidate_mode=candidate_mode,
        budgets=khop_budgets,
        truncator=khop_truncator,
        inputs_fingerprint=inputs_fingerprint,
    )
    ordered = sorted(queries, key=lambda q: str(q["query_id"]))
    batch_out: list[BatchQueryResult] = []
    for q in ordered:
        qid = str(q["query_id"])
        seeds = q["seed_node_ids"]
        tk = int(q["top_k"]) if "top_k" in q else default_top_k
        hp = float(q["hub_penalty"]) if "hub_penalty" in q else hub_penalty
        sem_w = q.get("semantics_weight", semantics_weight)
        ev_w = q.get("evidence_weight", evidence_weight)
        max_sz = q.get("max_hyperedge_size", max_hyperedge_size)
        hub_thr = (
            int(q["hub_degree_flag_threshold"])
            if "hub_degree_flag_threshold" in q
            else hub_degree_flag_threshold
        )

        khop_cands: list[str] | None = None
        if candidate_mode == "khop_k2":
            khop_cands = khop_candidates_k2(
                seed_node_ids=list(seeds),
                hyperedges=hyperedges,
                budgets=khop_budgets,
                truncator=khop_truncator,
                query_id=qid,
                graph_csr=graph_csr,
            )

        results = retrieve_hyperedges(
            seed_node_ids=seeds,
            hyperedges=hyperedges,
            inverted_index=inverted_index,
            top_k=tk,
            hub_penalty=hp,
            semantics_weight=sem_w,
            evidence_weight=ev_w,
            max_hyperedge_size=max_sz,
            hub_degree_flag_threshold=hub_thr,
            reproducibility=reproducibility,
            candidate_hyperedge_ids=khop_cands,
        )
        batch_out.append(
            {
                "query_id": qid,
                "reproducibility": reproducibility,
                "determinism_version": det_version,
                "candidate_mode": candidate_mode,
                "candidate_hyperedge_ids": khop_cands,
                "results": results,
            }
        )
    return batch_out
