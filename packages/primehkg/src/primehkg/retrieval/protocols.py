"""Retrieval pipeline protocols for pluggable candidate/rank/post steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from primehkg.core.dataset import CoreDataset
    from primehkg.retrieval.graph_csr.types import GraphCsrInMemory
    from primehkg.retrieval.khop import Subhypergraph
    from primehkg.retrieval.retrieve import RetrievalResult


class RetrievalPipelineOutput(TypedDict):
    seed_node_ids: list[str]
    candidate_hyperedge_ids: list[str]
    results: list[RetrievalResult]
    subhypergraph: Subhypergraph | None
    context: ContextOutput | None
    reproducibility: Literal["strict", "fast"]
    determinism_version: str | None
    candidate_mode: str
    audit: NotRequired[dict[str, object]]


class IndexProvider(Protocol):
    def node_to_hyperedge_ids(self) -> Mapping[str, Sequence[str]]: ...

    def hyperedge_by_id(self) -> Mapping[str, Mapping[str, object]] | None: ...

    def graph_csr(self) -> GraphCsrInMemory | None: ...

    @property
    def inputs_fingerprint(self) -> str | None: ...


class CandidateGenerator(Protocol):
    def generate(
        self,
        *,
        seed_node_ids: Sequence[str],
        query_id: str,
        index: IndexProvider,
        hyperedges: Sequence[Mapping[str, object]],
        query_text: str | None = None,
    ) -> list[str]: ...


class HyperedgeRanker(Protocol):
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
    ) -> list[RetrievalResult]: ...


class PostProcessor(Protocol):
    def process(
        self,
        *,
        results: list[RetrievalResult],
        seed_node_ids: Sequence[str],
        candidate_hyperedge_ids: Sequence[str],
        index: IndexProvider,
        hyperedges: Sequence[Mapping[str, object]],
        reproducibility: Literal["strict", "fast"],
        determinism_version: str | None,
        candidate_mode: str,
    ) -> RetrievalPipelineOutput: ...


class SubhypergraphSampler(Protocol):
    def sample(
        self,
        *,
        seed_node_ids: Sequence[str],
        query_id: str,
        index: IndexProvider,
        dataset: CoreDataset,
    ) -> list[str]: ...


class ContextOutput(TypedDict):
    """Structured context for LLM consumption (Phase 2)."""

    entities: list[dict[str, str]]
    relationships: list[dict[str, str]]
    sources: list[dict[str, str]]


class ContextFormatter(Protocol):
    """Formats retrieval results into structured LLM context (Phase 2)."""

    def format(
        self,
        *,
        results: list[RetrievalResult],
        seed_node_ids: Sequence[str],
        subhypergraph: Subhypergraph | None,
        dataset: CoreDataset,
    ) -> ContextOutput: ...
