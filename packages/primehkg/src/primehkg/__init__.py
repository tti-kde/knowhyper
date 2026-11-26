"""PrimeHKG: PrimeKG → local hypergraph dataset tables and helpers."""

from typing import Never

from primehkg.core.dataset import CoreDataset, load_core_dataset
from primehkg.linking.build import (
    LinkResolverBuildInfo,
    build_link_resolvers,
    insert_extra_resolvers,
)
from primehkg.linking.pipeline import LinkPipeline
from primehkg.linking.presets import DEFAULT_LINKING, SAPBERT_LINKING
from primehkg.linking.protocols import LinkCandidate, LinkResult
from primehkg.retrieval.build import (
    RetrievalPipelineBuildInfo,
    RetrievalPipelineParts,
    build_retrieval_components,
)
from primehkg.retrieval.pipeline import (
    RetrievalPipeline,
    make_retrieval_pipeline,
    make_vector_hybrid_pipeline,
)
from primehkg.retrieval.presets import DEFAULT_RETRIEVAL, PHENOTYPE_HUB_RETRIEVAL
from primehkg.retrieval.protocols import RetrievalPipelineOutput
from primehkg.retrieval.subhypergraph_build import (
    SubhypergraphBuildInfo,
    build_subhypergraph_sampler,
)
from primehkg.retrieval.subhypergraph_presets import (
    DEFAULT_SUBHYPERGRAPH,
    PHENOTYPE_HUB_SUBHYPERGRAPH,
)
from primehkg.session import PrimeHKGSession

__version__ = "0.4.1"

__all__ = [
    "DEFAULT_LINKING",
    "DEFAULT_RETRIEVAL",
    "DEFAULT_SUBHYPERGRAPH",
    "PHENOTYPE_HUB_RETRIEVAL",
    "PHENOTYPE_HUB_SUBHYPERGRAPH",
    "SAPBERT_LINKING",
    "CoreDataset",
    "DefaultLinkPipeline",
    "DefaultRetrievalPipeline",
    "LinkCandidate",
    "LinkPipeline",
    "LinkResolverBuildInfo",
    "LinkResult",
    "PrimeHKGSession",
    "RetrievalPipeline",
    "RetrievalPipelineBuildInfo",
    "RetrievalPipelineOutput",
    "RetrievalPipelineParts",
    "SubhypergraphBuildInfo",
    "__version__",
    "build_core_dataset",
    "build_link_resolvers",
    "build_retrieval_components",
    "build_subhypergraph_sampler",
    "check_cuda",
    "hello",
    "insert_extra_resolvers",
    "load_core_dataset",
    "make_retrieval_pipeline",
    "make_vector_hybrid_pipeline",
]


def __getattr__(name: str) -> object:
    if name == "DefaultLinkPipeline":
        from primehkg.linking.pipeline import DefaultLinkPipeline

        return DefaultLinkPipeline
    if name == "DefaultRetrievalPipeline":
        from primehkg.retrieval.pipeline import DefaultRetrievalPipeline

        return DefaultRetrievalPipeline
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def build_core_dataset() -> Never:
    msg = "Use primehkg-build CLI or load_core_dataset() for consumer loads."
    raise NotImplementedError(msg)


def hello() -> str:
    """Return a greeting message."""
    return "Hello from primehkg!"


def check_cuda() -> dict[str, bool | str | int]:
    """Check CUDA availability and device information."""
    try:
        import torch
        from torch import version as torch_version

        return {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch_version.cuda or "N/A"
            if torch.cuda.is_available()
            else "N/A",
            "device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else "N/A"
            ),
        }
    except ImportError:
        return {
            "cuda_available": False,
            "cuda_version": "PyTorch not installed",
            "device_count": 0,
            "device_name": "N/A",
        }
