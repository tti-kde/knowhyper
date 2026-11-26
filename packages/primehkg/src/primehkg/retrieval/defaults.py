"""Default retrieval pipeline components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from primehkg.retrieval.batch import (
    DEFAULT_KHOP_BUDGETS,
    CandidateMode,
    batch_determinism_version,
)
from primehkg.retrieval.khop import (
    BudgetConfig,
    khop_candidates_k2,
    materialize_subhypergraph,
)
from primehkg.retrieval.protocols import RetrievalPipelineOutput
from primehkg.retrieval.retrieve import RetrievalResult, retrieve_hyperedges
from primehkg.retrieval.truncate import HashBasedTruncator

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from primehkg.core.dataset import CoreDataset
    from primehkg.retrieval.graph_csr.types import GraphCsrInMemory
    from primehkg.retrieval.protocols import (
        CandidateGenerator,
        ContextFormatter,
        IndexProvider,
        PostProcessor,
        SubhypergraphSampler,
    )


class CoreDatasetIndexProvider:
    def __init__(self, dataset: CoreDataset) -> None:
        self._dataset = dataset

    def node_to_hyperedge_ids(self) -> Mapping[str, Sequence[str]]:
        return self._dataset.inverted_index

    def hyperedge_by_id(self) -> Mapping[str, Mapping[str, object]] | None:
        he_list = self._dataset.hyperedges_as_mappings()
        out: dict[str, Mapping[str, object]] = {}
        for he in he_list:
            hid = he.get("he_id")
            if isinstance(hid, str):
                out[hid] = he
        return out

    def graph_csr(self) -> GraphCsrInMemory | None:
        return self._dataset.graph_csr

    @property
    def inputs_fingerprint(self) -> str | None:
        return self._dataset.inputs_fingerprint


class SeedIncidentCandidate:
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
        inv = index.node_to_hyperedge_ids()
        cands: set[str] = set()
        for nid in seed_node_ids:
            for hid in inv.get(str(nid), []):
                cands.add(hid)
        return sorted(cands)


class UnionCandidate:
    """Merge candidate hyperedge ids from multiple generators (sorted union)."""

    def __init__(self, generators: Sequence[CandidateGenerator]) -> None:
        self._generators = list(generators)

    def generate(
        self,
        *,
        seed_node_ids: Sequence[str],
        query_id: str,
        index: IndexProvider,
        hyperedges: Sequence[Mapping[str, object]],
        query_text: str | None = None,
    ) -> list[str]:
        cands: set[str] = set()
        for gen in self._generators:
            cands.update(
                gen.generate(
                    seed_node_ids=seed_node_ids,
                    query_id=query_id,
                    index=index,
                    hyperedges=hyperedges,
                    query_text=query_text,
                )
            )
        return sorted(cands)


class KhopK2Candidate:
    def __init__(
        self,
        *,
        budgets: BudgetConfig | None = None,
        truncator: HashBasedTruncator | None = None,
    ) -> None:
        self._budgets = budgets if budgets is not None else DEFAULT_KHOP_BUDGETS
        self._truncator = truncator if truncator is not None else HashBasedTruncator()

    def generate(
        self,
        *,
        seed_node_ids: Sequence[str],
        query_id: str,
        index: IndexProvider,
        hyperedges: Sequence[Mapping[str, object]],
        query_text: str | None = None,
    ) -> list[str]:
        del query_text
        return khop_candidates_k2(
            seed_node_ids=list(seed_node_ids),
            hyperedges=hyperedges,
            budgets=self._budgets,
            truncator=self._truncator,
            query_id=query_id,
            graph_csr=index.graph_csr(),
        )


class EwoRanker:
    def rank(
        self,
        *,
        seed_node_ids: Sequence[str],
        candidate_hyperedge_ids: Sequence[str],
        hyperedges: Sequence[Mapping[str, object]],
        inverted_index: Mapping[str, list[str]],
        top_k: int,
        reproducibility: Literal["strict", "fast"],
        hub_penalty: float = 1.0,
        hub_degree_flag_threshold: int = 100,
        semantics_weight: Mapping[str, float] | None = None,
        evidence_weight: Mapping[str, float] | None = None,
        max_hyperedge_size: int | None = None,
        hyperedge_lookup: Callable[[str], Mapping[str, object] | None] | None = None,
    ) -> list[RetrievalResult]:
        return retrieve_hyperedges(
            seed_node_ids=seed_node_ids,
            hyperedges=hyperedges,
            inverted_index=inverted_index,
            top_k=top_k,
            hub_penalty=hub_penalty,
            hub_degree_flag_threshold=hub_degree_flag_threshold,
            semantics_weight=semantics_weight,
            evidence_weight=evidence_weight,
            max_hyperedge_size=max_hyperedge_size,
            reproducibility=reproducibility,
            candidate_hyperedge_ids=list(candidate_hyperedge_ids),
            hyperedge_lookup=hyperedge_lookup,
        )


class ChainedPostProcessor:
    """Run post-processors in sequence; each receives the same input kwargs."""

    def __init__(self, processors: Sequence[PostProcessor]) -> None:
        if not processors:
            msg = "ChainedPostProcessor requires at least one processor"
            raise ValueError(msg)
        self._processors = list(processors)

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
    ) -> RetrievalPipelineOutput:
        out: RetrievalPipelineOutput | None = None
        kwargs = {
            "results": results,
            "seed_node_ids": seed_node_ids,
            "candidate_hyperedge_ids": candidate_hyperedge_ids,
            "index": index,
            "hyperedges": hyperedges,
            "reproducibility": reproducibility,
            "determinism_version": determinism_version,
            "candidate_mode": candidate_mode,
        }
        for proc in self._processors:
            out = proc.process(**kwargs)
            kwargs["results"] = out["results"]
            kwargs["candidate_hyperedge_ids"] = out["candidate_hyperedge_ids"]
        assert out is not None
        return out


class IdentityPostProcessor:
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
    ) -> RetrievalPipelineOutput:
        del index, hyperedges
        return {
            "seed_node_ids": list(seed_node_ids),
            "candidate_hyperedge_ids": list(candidate_hyperedge_ids),
            "results": results,
            "subhypergraph": None,
            "context": None,
            "reproducibility": reproducibility,
            "determinism_version": determinism_version,
            "candidate_mode": candidate_mode,
        }


class SubhypergraphPostProcessor:
    def __init__(self, inner: IdentityPostProcessor | None = None) -> None:
        self._inner = inner if inner is not None else IdentityPostProcessor()

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
    ) -> RetrievalPipelineOutput:
        out = self._inner.process(
            results=results,
            seed_node_ids=seed_node_ids,
            candidate_hyperedge_ids=candidate_hyperedge_ids,
            index=index,
            hyperedges=hyperedges,
            reproducibility=reproducibility,
            determinism_version=determinism_version,
            candidate_mode=candidate_mode,
        )
        if candidate_hyperedge_ids:
            out["subhypergraph"] = materialize_subhypergraph(
                candidate_hyperedge_ids=list(candidate_hyperedge_ids),
                hyperedges=hyperedges,
            )
        return out


class SubhypergraphAttachPostProcessor:
    """Attach a budgeted k-hop subhypergraph after base post-processing."""

    def __init__(
        self,
        inner: PostProcessor,
        *,
        dataset: CoreDataset,
        resolve_samplers: Callable[[], Sequence[SubhypergraphSampler]],
        attach: bool = True,
        csr_sidecar_missing: bool = False,
        require_csr: bool = True,
    ) -> None:
        self._inner = inner
        self._dataset = dataset
        self._resolve_samplers = resolve_samplers
        self._attach = attach
        self._csr_sidecar_missing = csr_sidecar_missing
        self._require_csr = require_csr
        self._query_id = "default"

    def set_query_id(self, query_id: str) -> None:
        self._query_id = query_id

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
    ) -> RetrievalPipelineOutput:
        from primehkg.retrieval.subhypergraph_build import union_subhypergraph_samples

        out = self._inner.process(
            results=results,
            seed_node_ids=seed_node_ids,
            candidate_hyperedge_ids=candidate_hyperedge_ids,
            index=index,
            hyperedges=hyperedges,
            reproducibility=reproducibility,
            determinism_version=determinism_version,
            candidate_mode=candidate_mode,
        )
        if not self._attach:
            return out

        audit = dict(out.get("audit") or {})
        if self._require_csr and (
            self._csr_sidecar_missing or index.graph_csr() is None
        ):
            audit["subhypergraph_unavailable"] = "csr_sidecar_missing"
            if audit:
                out["audit"] = audit
            return out

        if not seed_node_ids:
            out["subhypergraph"] = materialize_subhypergraph(
                candidate_hyperedge_ids=[],
                hyperedge_lookup=self._dataset.get_hyperedge,
            )
            return out

        he_ids = union_subhypergraph_samples(
            self._resolve_samplers(),
            seed_node_ids=seed_node_ids,
            query_id=self._query_id,
            index=index,
            dataset=self._dataset,
        )
        out["subhypergraph"] = materialize_subhypergraph(
            candidate_hyperedge_ids=he_ids,
            hyperedge_lookup=self._dataset.get_hyperedge,
        )
        if audit:
            out["audit"] = audit
        return out


class ContextAttachPostProcessor:
    """Attach structured LLM context after base post-processing."""

    def __init__(
        self,
        inner: PostProcessor,
        *,
        dataset: CoreDataset,
        formatter: ContextFormatter | None = None,
        attach: bool = True,
    ) -> None:
        from primehkg.retrieval.context_format import DefaultContextFormatter

        self._inner = inner
        self._dataset = dataset
        self._formatter = (
            formatter if formatter is not None else DefaultContextFormatter()
        )
        self._attach = attach

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
    ) -> RetrievalPipelineOutput:
        out = self._inner.process(
            results=results,
            seed_node_ids=seed_node_ids,
            candidate_hyperedge_ids=candidate_hyperedge_ids,
            index=index,
            hyperedges=hyperedges,
            reproducibility=reproducibility,
            determinism_version=determinism_version,
            candidate_mode=candidate_mode,
        )
        if not self._attach:
            out["context"] = None
            return out
        context = self._formatter.format(
            results=out["results"],
            seed_node_ids=seed_node_ids,
            subhypergraph=out.get("subhypergraph"),
            dataset=self._dataset,
        )
        out["context"] = context
        return out


def batch_determinism_for_mode(
    *,
    reproducibility: Literal["strict", "fast"],
    candidate_mode: CandidateMode,
    inputs_fingerprint: str | None,
    vector_determinism: str | None = None,
) -> str | None:
    if reproducibility != "strict":
        return None
    base = batch_determinism_version(
        reproducibility=reproducibility,
        candidate_mode=candidate_mode,
        budgets=DEFAULT_KHOP_BUDGETS,
        truncator=HashBasedTruncator(),
        inputs_fingerprint=inputs_fingerprint,
    )
    if vector_determinism is None:
        return base
    if base is None:
        return vector_determinism
    return f"{base}+{vector_determinism}"
