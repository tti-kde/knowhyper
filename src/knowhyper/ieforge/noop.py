"""No-op ieforge knowledge components for KnowHyper.

KnowHyper's injector performs linking/retrieval/encoding itself from the Query,
so the KnowledgePipeline's source/builder/retriever only need to exist and yield
an empty RetrievalResult. These keep the required pipeline slots filled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ieforge.knowledge.core.base import (
    IEKnowledgeBuilder,
    IEKnowledgeRetriever,
    IEKnowledgeSource,
)
from ieforge.knowledge.core.schema import (
    BuiltKnowledge,
    KnowledgeArtifact,
    RetrievalResult,
)
from ieforge.registry import register

if TYPE_CHECKING:
    from ieforge.knowledge.core.schema import Query


@register("knowledge_source", name="noop")
class NoOpKnowledgeSource(IEKnowledgeSource):
    """Source that loads an empty artifact."""

    def load(self) -> KnowledgeArtifact:
        """Return an empty knowledge artifact."""
        return KnowledgeArtifact(payload=None)


@register("knowledge_builder", name="noop")
class NoOpKnowledgeBuilder(IEKnowledgeBuilder):
    """Builder that produces empty built knowledge."""

    @property
    def identity(self) -> str:
        """Stable identity for caches."""
        return "knowhyper-noop"

    def build(self, artifact: KnowledgeArtifact) -> BuiltKnowledge:  # noqa: ARG002
        """Return empty built knowledge."""
        return BuiltKnowledge(handles={})


@register("knowledge_retriever", name="noop")
class NoOpRetriever(IEKnowledgeRetriever):
    """Retriever that returns no facts (the KnowHyper injector retrieves itself)."""

    def retrieve(self, query: Query, knowledge: BuiltKnowledge) -> RetrievalResult:  # noqa: ARG002
        """Return an empty retrieval result for the query."""
        return RetrievalResult(query=query, facts=[])
