"""Protocols for native IEForge knowledge components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ieforge.knowledge.core.schema import (
        BuiltKnowledge,
        KnowledgeArtifact,
        KnowledgeContext,
        Query,
        RetrievalResult,
    )


@runtime_checkable
class KnowledgeSource(Protocol):
    """Loads raw knowledge artifacts."""

    def load(self) -> KnowledgeArtifact:
        """Load a knowledge artifact."""
        ...


@runtime_checkable
class KnowledgeBuilder(Protocol):
    """Builds retrieval-ready knowledge handles from artifacts."""

    @property
    def identity(self) -> str:
        """Stable cache identity for the builder configuration."""
        ...

    def build(self, artifact: KnowledgeArtifact) -> BuiltKnowledge:
        """Build retrieval-ready knowledge from an artifact."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Retrieves relevant facts for a query."""

    def retrieve(self, query: Query, knowledge: BuiltKnowledge) -> RetrievalResult:
        """Retrieve facts from built knowledge."""
        ...


@runtime_checkable
class KnowledgeInjector(Protocol):
    """Converts retrieval results into channel-agnostic context."""

    def inject(self, query: Query, retrieval: RetrievalResult) -> KnowledgeContext:
        """Inject retrieved facts into a knowledge context."""
        ...


@runtime_checkable
class QueryBuilder(Protocol):
    """Builds retrieval queries from IEForge examples."""

    def build(self, example: Any) -> Query:
        """Build a query for an example."""
        ...


@runtime_checkable
class KnowledgeChannel(Protocol):
    """Applies knowledge context to an example representation."""

    def apply(self, example: Any, context: KnowledgeContext) -> Any:
        """Apply a knowledge context to an example."""
        ...


@runtime_checkable
class KnowledgeRunner(Protocol):
    """Runs knowledge augmentation for one example."""

    def run(self, example: Any) -> KnowledgeContext:
        """Return a knowledge context for an example."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Encodes text into vector representations."""

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        ...

    @property
    def identity(self) -> str:
        """Stable identity for embedding cache keys."""
        ...

    def encode(self, texts: list[str]) -> Any:
        """Encode texts as a vector matrix."""
        ...
