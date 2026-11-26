"""Build and compose subhypergraph samplers for consumer retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from primehkg.retrieval.batch import DEFAULT_KHOP_BUDGETS
from primehkg.retrieval.khop import BudgetConfig, khop_candidates_k2
from primehkg.retrieval.subhypergraph_presets import (
    DEFAULT_SUBHYPERGRAPH,
    SubhypergraphPreset,
)
from primehkg.retrieval.truncate import HashBasedTruncator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from primehkg.core.dataset import CoreDataset
    from primehkg.retrieval.protocols import IndexProvider, SubhypergraphSampler
    from primehkg.retrieval.truncate import Truncator


@dataclass(frozen=True)
class SubhypergraphBuildInfo:
    """Metadata from ``build_subhypergraph_sampler``."""

    preset: SubhypergraphPreset
    csr_sidecar_missing: bool
    sampler_kind: Literal["khop_k2", "custom"]


def _preset_to_budgets(preset: SubhypergraphPreset) -> BudgetConfig:
    return BudgetConfig(
        max_he_level1=int(
            preset.get("max_he_level1", DEFAULT_KHOP_BUDGETS.max_he_level1)
        ),
        max_nodes_level2=int(
            preset.get("max_nodes_level2", DEFAULT_KHOP_BUDGETS.max_nodes_level2)
        ),
        max_he_level3=int(
            preset.get("max_he_level3", DEFAULT_KHOP_BUDGETS.max_he_level3)
        ),
    )


class KhopK2SubhypergraphSampler:
    """Budgeted k=2 neighborhood sampler for subhypergraph materialization."""

    def __init__(
        self,
        *,
        budgets: BudgetConfig,
        truncator: Truncator | None = None,
    ) -> None:
        """Configure budgets and deterministic truncation for k-hop sampling."""
        self._budgets = budgets
        self._truncator = truncator if truncator is not None else HashBasedTruncator()

    def sample(
        self,
        *,
        seed_node_ids: Sequence[str],
        query_id: str,
        index: IndexProvider,
        dataset: CoreDataset,
    ) -> list[str]:
        """Return budgeted k=2 hyperedge ids for materialization."""
        del dataset
        return khop_candidates_k2(
            seed_node_ids=list(seed_node_ids),
            hyperedges=[],
            budgets=self._budgets,
            truncator=self._truncator,
            query_id=query_id,
            graph_csr=index.graph_csr(),
        )


def build_subhypergraph_sampler(
    dataset: CoreDataset,
    preset: SubhypergraphPreset | None = None,
    *,
    truncator: Truncator | None = None,
) -> tuple[SubhypergraphSampler, SubhypergraphBuildInfo]:
    resolved: SubhypergraphPreset = (
        preset if preset is not None else DEFAULT_SUBHYPERGRAPH
    )
    csr_missing = dataset.graph_csr is None
    sampler: SubhypergraphSampler = KhopK2SubhypergraphSampler(
        budgets=_preset_to_budgets(resolved),
        truncator=truncator,
    )
    return (
        sampler,
        SubhypergraphBuildInfo(
            preset=resolved,
            csr_sidecar_missing=csr_missing,
            sampler_kind="khop_k2",
        ),
    )


def union_subhypergraph_samples(
    samplers: Sequence[SubhypergraphSampler],
    *,
    seed_node_ids: Sequence[str],
    query_id: str,
    index: IndexProvider,
    dataset: CoreDataset,
) -> list[str]:
    cands: set[str] = set()
    for sampler in samplers:
        cands.update(
            sampler.sample(
                seed_node_ids=seed_node_ids,
                query_id=query_id,
                index=index,
                dataset=dataset,
            )
        )
    return sorted(cands)
