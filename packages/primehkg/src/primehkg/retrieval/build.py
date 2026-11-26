"""Build retrieval pipeline components from presets and candidate mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from primehkg.retrieval.defaults import (
    CoreDatasetIndexProvider,
    EwoRanker,
    IdentityPostProcessor,
    KhopK2Candidate,
    SeedIncidentCandidate,
    UnionCandidate,
)
from primehkg.retrieval.presets import DEFAULT_RETRIEVAL, RetrievalPreset
from primehkg.retrieval.vector_candidates import (
    VectorGlobalCandidate,
    VectorLocalCandidate,
    vector_determinism_version,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from primehkg.core.dataset import CoreDataset
    from primehkg.retrieval.batch import CandidateMode
    from primehkg.retrieval.khop import BudgetConfig
    from primehkg.retrieval.protocols import (
        CandidateGenerator,
        HyperedgeRanker,
        IndexProvider,
        PostProcessor,
    )


@dataclass(frozen=True)
class RetrievalPipelineBuildInfo:
    """Metadata from ``build_retrieval_components``."""

    candidate_mode: CandidateMode
    scoring_preset: RetrievalPreset
    csr_sidecar_missing: bool
    vector_sidecar_missing: bool = False
    vector_he_sidecar_missing: bool = False
    vector_determinism: str | None = None


@dataclass(frozen=True)
class RetrievalPipelineParts:
    """Wired components before ``RetrievalPipeline`` wrapper."""

    candidate: CandidateGenerator
    ranker: HyperedgeRanker
    index: IndexProvider
    post: PostProcessor


def _vector_mode_active(candidate_mode: CandidateMode) -> bool:
    return candidate_mode in (
        "vector_local",
        "vector_global",
        "vector_hybrid",
        "khop_k2_plus_vector",
    )


def build_retrieval_components(
    dataset: CoreDataset,
    *,
    candidate_mode: CandidateMode = "seed_incident",
    preset: RetrievalPreset | None = None,
    khop_budgets: BudgetConfig | None = None,
    post: PostProcessor | None = None,
    vector_top_k_nodes: int = 50,
    vector_top_k_he: int = 100,
    encode_fn: Callable[[str], np.ndarray] | None = None,
) -> tuple[RetrievalPipelineParts, RetrievalPipelineBuildInfo]:
    scoring: RetrievalPreset = preset if preset is not None else DEFAULT_RETRIEVAL
    csr_sidecar_missing = False
    vector_sidecar_missing = False
    vector_he_sidecar_missing = False
    vector_determinism: str | None = None

    if candidate_mode == "khop_k2":
        candidate: CandidateGenerator = KhopK2Candidate(budgets=khop_budgets)
        if dataset.graph_csr is None:
            csr_sidecar_missing = True
    elif _vector_mode_active(candidate_mode):
        node_index = dataset.vector_node_index
        he_index = dataset.vector_hyperedge_index
        vector_sidecar_missing = node_index is None
        vector_he_sidecar_missing = he_index is None

        generators: list[CandidateGenerator] = []
        if candidate_mode == "vector_hybrid":
            generators.append(SeedIncidentCandidate())
        if candidate_mode == "khop_k2_plus_vector":
            generators.append(KhopK2Candidate(budgets=khop_budgets))
            if dataset.graph_csr is None:
                csr_sidecar_missing = True

        if node_index is not None and candidate_mode in (
            "vector_local",
            "vector_hybrid",
            "khop_k2_plus_vector",
        ):
            generators.append(
                VectorLocalCandidate(
                    node_index=node_index,
                    top_k_nodes=vector_top_k_nodes,
                )
            )
        if (
            he_index is not None
            and node_index is not None
            and candidate_mode
            in (
                "vector_global",
                "vector_hybrid",
                "khop_k2_plus_vector",
            )
        ):
            generators.append(
                VectorGlobalCandidate(
                    he_index=he_index,
                    node_index=node_index,
                    top_k_he=vector_top_k_he,
                    encode_fn=encode_fn,
                )
            )

        if not generators:
            candidate = SeedIncidentCandidate()
            vector_sidecar_missing = True
        elif len(generators) == 1:
            candidate = generators[0]
        else:
            candidate = UnionCandidate(generators)

        if node_index is not None:
            meta_model = node_index.model_id
            vector_determinism = vector_determinism_version(
                model_id=meta_model,
                top_k_nodes=vector_top_k_nodes,
                top_k_he=vector_top_k_he,
            )
    else:
        candidate = SeedIncidentCandidate()

    return (
        RetrievalPipelineParts(
            candidate=candidate,
            ranker=EwoRanker(),
            index=CoreDatasetIndexProvider(dataset),
            post=post if post is not None else IdentityPostProcessor(),
        ),
        RetrievalPipelineBuildInfo(
            candidate_mode=candidate_mode,
            scoring_preset=scoring,
            csr_sidecar_missing=csr_sidecar_missing,
            vector_sidecar_missing=vector_sidecar_missing,
            vector_he_sidecar_missing=vector_he_sidecar_missing,
            vector_determinism=vector_determinism,
        ),
    )
