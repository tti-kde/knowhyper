"""Abstract base classes for IEForge knowledge components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

    from ieforge.knowledge.core.schema import (
        BuiltKnowledge,
        KnowledgeArtifact,
        KnowledgeContext,
        Query,
        RetrievalResult,
    )


class IEKnowledgeSource(ABC):
    """Abstract base class for knowledge artifact sources."""

    @abstractmethod
    def load(self) -> KnowledgeArtifact:
        """Load a raw knowledge artifact."""
        ...


class IEKnowledgeBuilder(ABC):
    """Abstract base class for retrieval-ready knowledge builders."""

    @property
    @abstractmethod
    def identity(self) -> str:
        """Stable identity for builder configuration and caches."""
        ...

    @abstractmethod
    def build(self, artifact: KnowledgeArtifact) -> BuiltKnowledge:
        """Build retrieval-ready knowledge handles from an artifact."""
        ...


class IEKnowledgeRetriever(ABC):
    """Abstract base class for knowledge retrievers."""

    @abstractmethod
    def retrieve(self, query: Query, knowledge: BuiltKnowledge) -> RetrievalResult:
        """Retrieve facts from built knowledge."""
        ...


class IEKnowledgeInjector(ABC):
    """Abstract base class for knowledge context injectors."""

    @abstractmethod
    def inject(self, query: Query, retrieval: RetrievalResult) -> KnowledgeContext:
        """Convert retrieved facts into a knowledge context."""
        ...


class IEKnowledgeQueryBuilder(ABC):
    """Abstract base class for query builders."""

    @abstractmethod
    def build(self, example: Any) -> Query:
        """Build a retrieval query from an example."""
        ...


class IEKnowledgeChannel(ABC):
    """Abstract base class for per-example knowledge channels."""

    @abstractmethod
    def apply(self, example: Any, context: KnowledgeContext) -> Any:
        """Apply a knowledge context to one example representation."""
        ...


class IEKnowledgeFeatureChannel(ABC):
    """Abstract base class for batch-level knowledge feature channels."""

    @abstractmethod
    def collate(self, contexts: Sequence[KnowledgeContext]) -> Mapping[str, Any]:
        """Collate per-example knowledge contexts into batch features."""
        ...


class IEKnowledgeCollator(ABC):
    """Abstract base class for knowledge-aware batch collators."""

    @abstractmethod
    def __call__(self, examples: Sequence[Any]) -> MutableMapping[str, Any]:
        """Collate examples into a mutable batch mapping."""
        ...


class IEKnowledgeRunner(ABC):
    """Abstract base class for objects that produce knowledge contexts."""

    @abstractmethod
    def run(self, example: Any) -> KnowledgeContext:
        """Return a knowledge context for one example."""
        ...


class IEKnowledgeEmbedder(ABC):
    """Abstract base class for text embedders used by knowledge components."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""
        ...

    @property
    @abstractmethod
    def identity(self) -> str:
        """Stable identity for embedding configuration and caches."""
        ...

    @abstractmethod
    def encode(self, texts: list[str]) -> Any:
        """Encode text strings into vector representations."""
        ...
