"""Orchestrator for pluggable hyperedge retrieval pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from primehkg.retrieval.batch import CandidateMode, QuerySpec
from primehkg.retrieval.build import build_retrieval_components
from primehkg.retrieval.defaults import ChainedPostProcessor, UnionCandidate
from primehkg.retrieval.presets import DEFAULT_RETRIEVAL, RetrievalPreset
from primehkg.retrieval.protocols import RetrievalPipelineOutput
from primehkg.retrieval.subhypergraph_presets import (
    DEFAULT_SUBHYPERGRAPH,
    SubhypergraphPreset,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np

    from primehkg.core.dataset import CoreDataset
    from primehkg.retrieval.khop import BudgetConfig
    from primehkg.retrieval.protocols import (
        CandidateGenerator,
        ContextFormatter,
        HyperedgeRanker,
        IndexProvider,
        PostProcessor,
        SubhypergraphSampler,
    )


class RetrievalPipeline:
    def __init__(
        self,
        *,
        dataset: CoreDataset,
        candidate: CandidateGenerator,
        ranker: HyperedgeRanker,
        index: IndexProvider | None = None,
        post: PostProcessor | None = None,
        candidate_mode: CandidateMode = "seed_incident",
        default_scoring_preset: RetrievalPreset | None = None,
    ) -> None:
        from primehkg.retrieval.defaults import (
            CoreDatasetIndexProvider,
            IdentityPostProcessor,
        )

        self._dataset = dataset
        self._candidate = candidate
        self._ranker = ranker
        self._index = index if index is not None else CoreDatasetIndexProvider(dataset)
        self._post = post if post is not None else IdentityPostProcessor()
        self._candidate_mode = candidate_mode
        self._default_scoring_preset: RetrievalPreset = (
            default_scoring_preset
            if default_scoring_preset is not None
            else DEFAULT_RETRIEVAL
        )
        self._csr_sidecar_missing = False
        self._attach_subhypergraph = False
        self._default_subhypergraph_preset: SubhypergraphPreset = DEFAULT_SUBHYPERGRAPH
        self._custom_subhypergraph_sampler: SubhypergraphSampler | None = None
        self._extra_subhypergraph_samplers: list[SubhypergraphSampler] = []
        self._subhypergraph_csr_missing = False
        self._subhypergraph_preset_override: SubhypergraphPreset | None = None
        self._vector_sidecar_missing = False
        self._vector_he_sidecar_missing = False
        self._vector_determinism: str | None = None
        self._attach_context = False

    @classmethod
    def from_preset(
        cls,
        dataset: CoreDataset,
        preset: RetrievalPreset,
        *,
        candidate_mode: CandidateMode = "seed_incident",
        extra_candidates: Sequence[CandidateGenerator] = (),
        extra_post: Sequence[PostProcessor] = (),
        khop_budgets: BudgetConfig | None = None,
        post: PostProcessor | None = None,
        subhypergraph_preset: SubhypergraphPreset | None = DEFAULT_SUBHYPERGRAPH,
        attach_subhypergraph: bool = True,
        subhypergraph_sampler: SubhypergraphSampler | None = None,
        extra_subhypergraph_samplers: Sequence[SubhypergraphSampler] = (),
        vector_top_k_nodes: int = 50,
        vector_top_k_he: int = 100,
        encode_fn: Callable[[str], np.ndarray] | None = None,
        query_encode_model: str | None = None,
        query_encode_batch_size: int = 32,
        attach_context: bool = False,
        context_formatter: ContextFormatter | None = None,
    ) -> RetrievalPipeline:
        from primehkg.retrieval.defaults import (
            ContextAttachPostProcessor,
            SubhypergraphAttachPostProcessor,
        )
        from primehkg.retrieval.query_encode import resolve_encode_fn
        from primehkg.retrieval.subhypergraph_build import build_subhypergraph_sampler

        resolved_encode_fn = resolve_encode_fn(
            encode_fn=encode_fn,
            query_encode_model=query_encode_model,
            encode_batch_size=query_encode_batch_size,
        )
        parts, info = build_retrieval_components(
            dataset,
            candidate_mode=candidate_mode,
            preset=preset,
            khop_budgets=khop_budgets,
            post=post,
            vector_top_k_nodes=vector_top_k_nodes,
            vector_top_k_he=vector_top_k_he,
            encode_fn=resolved_encode_fn,
        )
        candidate = parts.candidate
        if extra_candidates:
            candidate = UnionCandidate([candidate, *extra_candidates])
        pipeline_post = parts.post
        if extra_post:
            pipeline_post = ChainedPostProcessor([pipeline_post, *extra_post])

        subhypergraph_csr_missing = False
        require_csr_for_subhypergraph = subhypergraph_sampler is None
        if require_csr_for_subhypergraph and attach_subhypergraph:
            _, sub_info = build_subhypergraph_sampler(
                dataset,
                subhypergraph_preset,
            )
            subhypergraph_csr_missing = sub_info.csr_sidecar_missing

        pipe_ref: list[RetrievalPipeline] = []
        if attach_subhypergraph:

            def resolve_samplers() -> list[SubhypergraphSampler]:
                assert pipe_ref
                return pipe_ref[0].resolve_subhypergraph_samplers()

            pipeline_post = SubhypergraphAttachPostProcessor(
                pipeline_post,
                dataset=dataset,
                resolve_samplers=resolve_samplers,
                attach=True,
                csr_sidecar_missing=subhypergraph_csr_missing,
                require_csr=require_csr_for_subhypergraph,
            )

        if attach_context:
            pipeline_post = ContextAttachPostProcessor(
                pipeline_post,
                dataset=dataset,
                formatter=context_formatter,
                attach=True,
            )

        pipe = cls(
            dataset=dataset,
            candidate=candidate,
            ranker=parts.ranker,
            index=parts.index,
            post=pipeline_post,
            candidate_mode=info.candidate_mode,
            default_scoring_preset=info.scoring_preset,
        )
        if attach_subhypergraph:
            pipe_ref.append(pipe)
        pipe._csr_sidecar_missing = info.csr_sidecar_missing
        pipe._vector_sidecar_missing = info.vector_sidecar_missing
        pipe._vector_he_sidecar_missing = info.vector_he_sidecar_missing
        pipe._vector_determinism = info.vector_determinism
        pipe._attach_subhypergraph = attach_subhypergraph
        pipe._default_subhypergraph_preset = (
            subhypergraph_preset
            if subhypergraph_preset is not None
            else DEFAULT_SUBHYPERGRAPH
        )
        pipe._custom_subhypergraph_sampler = subhypergraph_sampler
        pipe._extra_subhypergraph_samplers = list(extra_subhypergraph_samplers)
        pipe._subhypergraph_csr_missing = subhypergraph_csr_missing
        pipe._attach_context = attach_context
        return pipe

    def resolve_subhypergraph_samplers(self) -> list[SubhypergraphSampler]:
        from primehkg.retrieval.subhypergraph_build import build_subhypergraph_sampler

        if self._custom_subhypergraph_sampler is not None:
            return [
                self._custom_subhypergraph_sampler,
                *self._extra_subhypergraph_samplers,
            ]
        preset = (
            self._subhypergraph_preset_override
            if self._subhypergraph_preset_override is not None
            else self._default_subhypergraph_preset
        )
        primary, _ = build_subhypergraph_sampler(self._dataset, preset)
        return [primary, *self._extra_subhypergraph_samplers]

    def _resolve_preset(
        self,
        scoring_preset: RetrievalPreset | None,
    ) -> RetrievalPreset:
        if scoring_preset is not None:
            return scoring_preset
        return self._default_scoring_preset

    def retrieve(
        self,
        seed_node_ids: Sequence[str],
        *,
        query_id: str = "default",
        query_text: str | None = None,
        top_k: int = 20,
        reproducibility: Literal["strict", "fast"] = "strict",
        scoring_preset: RetrievalPreset | None = None,
        hub_penalty: float | None = None,
        hub_degree_flag_threshold: int | None = None,
        semantics_weight: dict[str, float] | None = None,
        evidence_weight: dict[str, float] | None = None,
        max_hyperedge_size: int | None = None,
        subhypergraph_preset: SubhypergraphPreset | None = None,
    ) -> RetrievalPipelineOutput:
        from primehkg.retrieval.defaults import SubhypergraphAttachPostProcessor

        preset = self._resolve_preset(scoring_preset)
        hp = float(
            hub_penalty if hub_penalty is not None else preset.get("hub_penalty", 1.0)
        )
        hub_thr = int(
            hub_degree_flag_threshold
            if hub_degree_flag_threshold is not None
            else preset.get("hub_degree_flag_threshold", 100)
        )
        sem_w = (
            semantics_weight
            if semantics_weight is not None
            else preset.get("semantics_weight")
        )
        ev_w = (
            evidence_weight
            if evidence_weight is not None
            else preset.get("evidence_weight")
        )
        max_sz = (
            max_hyperedge_size
            if max_hyperedge_size is not None
            else preset.get("max_hyperedge_size")
        )

        hyperedges = (
            self._dataset.hyperedges_as_mappings()
            if self._candidate_mode in ("khop_k2", "khop_k2_plus_vector")
            else []
        )
        cands = self._candidate.generate(
            seed_node_ids=seed_node_ids,
            query_id=query_id,
            index=self._index,
            hyperedges=hyperedges,
            query_text=query_text,
        )
        inv_raw = self._index.node_to_hyperedge_ids()
        inv: dict[str, list[str]] = {k: list(v) for k, v in inv_raw.items()}
        results = self._ranker.rank(
            seed_node_ids=seed_node_ids,
            candidate_hyperedge_ids=cands,
            hyperedges=hyperedges,
            inverted_index=inv,
            top_k=top_k,
            reproducibility=reproducibility,
            hub_penalty=hp,
            hub_degree_flag_threshold=hub_thr,
            semantics_weight=sem_w,
            evidence_weight=ev_w,
            max_hyperedge_size=max_sz,
            hyperedge_lookup=self._dataset.get_hyperedge,
        )
        from primehkg.retrieval.defaults import batch_determinism_for_mode

        det = batch_determinism_for_mode(
            reproducibility=reproducibility,
            candidate_mode=self._candidate_mode,
            inputs_fingerprint=self._index.inputs_fingerprint,
            vector_determinism=self._vector_determinism,
        )
        prev_override = self._subhypergraph_preset_override
        self._subhypergraph_preset_override = subhypergraph_preset
        attach_post = (
            self._post
            if isinstance(self._post, SubhypergraphAttachPostProcessor)
            else None
        )
        if attach_post is not None:
            attach_post.set_query_id(query_id)
        try:
            out = self._post.process(
                results=results,
                seed_node_ids=seed_node_ids,
                candidate_hyperedge_ids=cands,
                index=self._index,
                hyperedges=hyperedges,
                reproducibility=reproducibility,
                determinism_version=det,
                candidate_mode=self._candidate_mode,
            )
        finally:
            self._subhypergraph_preset_override = prev_override

        audit = dict(out.get("audit") or {})
        if self._csr_sidecar_missing:
            audit["csr_sidecar_missing"] = True
        if self._vector_sidecar_missing:
            audit["vector_sidecar_missing"] = True
        if self._vector_he_sidecar_missing:
            audit["vector_he_sidecar_missing"] = True
        if audit:
            out["audit"] = audit
        return out

    def retrieve_batch(
        self,
        queries: Sequence[QuerySpec],
        *,
        default_top_k: int = 20,
        reproducibility: Literal["strict", "fast"] = "strict",
        scoring_preset: RetrievalPreset | None = None,
    ) -> list[RetrievalPipelineOutput]:
        ordered = sorted(queries, key=lambda q: str(q["query_id"]))
        out: list[RetrievalPipelineOutput] = []
        for q in ordered:
            qid = str(q["query_id"])
            seeds = q["seed_node_ids"]
            tk = int(q["top_k"]) if "top_k" in q else default_top_k
            out.append(
                self.retrieve(
                    seed_node_ids=seeds,
                    query_id=qid,
                    query_text=q.get("query_text"),
                    top_k=tk,
                    reproducibility=reproducibility,
                    scoring_preset=scoring_preset,
                    hub_penalty=float(q["hub_penalty"]) if "hub_penalty" in q else None,
                    hub_degree_flag_threshold=int(q["hub_degree_flag_threshold"])
                    if "hub_degree_flag_threshold" in q
                    else None,
                    semantics_weight=q.get("semantics_weight"),
                    evidence_weight=q.get("evidence_weight"),
                    max_hyperedge_size=q.get("max_hyperedge_size"),
                )
            )
        return out


def make_retrieval_pipeline(
    dataset: CoreDataset,
    *,
    candidate_mode: CandidateMode = "seed_incident",
    post: PostProcessor | None = None,
    khop_budgets: BudgetConfig | None = None,
    attach_subhypergraph: bool = True,
    subhypergraph_preset: SubhypergraphPreset | None = DEFAULT_SUBHYPERGRAPH,
) -> RetrievalPipeline:
    return RetrievalPipeline.from_preset(
        dataset,
        DEFAULT_RETRIEVAL,
        candidate_mode=candidate_mode,
        post=post,
        khop_budgets=khop_budgets,
        attach_subhypergraph=attach_subhypergraph,
        subhypergraph_preset=subhypergraph_preset,
    )


def make_vector_hybrid_pipeline(
    dataset: CoreDataset,
    *,
    preset: RetrievalPreset | None = None,
    attach_context: bool = False,
    query_encode_model: str | None = None,
    query_encode_batch_size: int = 32,
    encode_fn: Callable[[str], np.ndarray] | None = None,
    vector_top_k_nodes: int = 50,
    vector_top_k_he: int = 100,
    attach_subhypergraph: bool = True,
    subhypergraph_preset: SubhypergraphPreset | None = DEFAULT_SUBHYPERGRAPH,
    post: PostProcessor | None = None,
) -> RetrievalPipeline:
    """Build a ``vector_hybrid`` pipeline (seed_incident union vector local/global).

    Uses ``DEFAULT_RETRIEVAL`` scoring unless ``preset`` is set. Pass
    ``query_encode_model`` (or ``encode_fn``) when calling ``retrieve(...,
    query_text=...)`` so HE-global vector legs use query embeddings.
    """
    return RetrievalPipeline.from_preset(
        dataset,
        preset if preset is not None else DEFAULT_RETRIEVAL,
        candidate_mode="vector_hybrid",
        attach_context=attach_context,
        query_encode_model=query_encode_model,
        query_encode_batch_size=query_encode_batch_size,
        encode_fn=encode_fn,
        vector_top_k_nodes=vector_top_k_nodes,
        vector_top_k_he=vector_top_k_he,
        attach_subhypergraph=attach_subhypergraph,
        subhypergraph_preset=subhypergraph_preset,
        post=post,
    )


def DefaultRetrievalPipeline(
    dataset: CoreDataset,
    *,
    candidate_mode: CandidateMode = "seed_incident",
    post: PostProcessor | None = None,
    khop_budgets: BudgetConfig | None = None,
) -> RetrievalPipeline:
    import warnings

    warnings.warn(
        "DefaultRetrievalPipeline is deprecated; use make_retrieval_pipeline",
        DeprecationWarning,
        stacklevel=2,
    )
    return make_retrieval_pipeline(
        dataset,
        candidate_mode=candidate_mode,
        post=post,
        khop_budgets=khop_budgets,
    )
