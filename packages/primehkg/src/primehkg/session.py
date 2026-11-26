"""Optional facade for common load → link → retrieve flows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from primehkg.core.dataset import CoreDataset, load_core_dataset
from primehkg.linking.pipeline import LinkPipeline
from primehkg.linking.protocols import LinkResult
from primehkg.retrieval.pipeline import RetrievalPipeline
from primehkg.retrieval.presets import DEFAULT_RETRIEVAL
from primehkg.retrieval.protocols import RetrievalPipelineOutput

if TYPE_CHECKING:
    from pathlib import Path

    from primehkg.linking.pipeline import LinkPipeline as LinkPipelineT


class PrimeHKGSession:
    """Thin facade wiring default link and retrieval pipelines over a ``CoreDataset``."""

    def __init__(
        self,
        dataset: CoreDataset,
        *,
        link_pipeline: LinkPipelineT | None = None,
        retrieval_pipeline: RetrievalPipeline | None = None,
    ) -> None:
        """Bind ``dataset`` and optional custom pipelines."""
        self.dataset = dataset
        self._link = (
            link_pipeline if link_pipeline is not None else LinkPipeline(dataset)
        )
        self._retrieve = (
            retrieval_pipeline
            if retrieval_pipeline is not None
            else RetrievalPipeline.from_preset(dataset, DEFAULT_RETRIEVAL)
        )

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        hub_repo_id: str | None = None,
        revision: str | None = None,
        expected_fingerprint: str | None = None,
        sidecar_dir: Path | None = None,
        index_mode: Literal["lazy", "eager"] = "lazy",
        hub_fallback: Literal["disk_cache", "error"] = "disk_cache",
    ) -> PrimeHKGSession:
        """Load a build and return a session with default pipelines."""
        ds = load_core_dataset(
            path,
            hub_repo_id=hub_repo_id,
            revision=revision,
            expected_fingerprint=expected_fingerprint,
            sidecar_dir=sidecar_dir,
            index_mode=index_mode,
            hub_fallback=hub_fallback,
        )
        return cls(ds)

    def link_and_retrieve(
        self,
        query: str,
        *,
        namespace: str | None = None,
        link_top_k: int = 3,
        retrieve_top_k: int = 20,
    ) -> tuple[LinkResult, RetrievalPipelineOutput]:
        """Link ``query`` to node ids, then retrieve ranked hyperedges for those seeds."""
        link_out = self._link.link(query, namespace=namespace, top_k=link_top_k)
        seeds = [c["node_id"] for c in link_out["candidates"]]
        if not seeds:
            empty = self._retrieve.retrieve(seed_node_ids=[], top_k=retrieve_top_k)
            return link_out, empty
        ret_out = self._retrieve.retrieve(seed_node_ids=seeds, top_k=retrieve_top_k)
        return link_out, ret_out
