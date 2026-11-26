"""Source -> builder -> retriever -> injector orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ieforge.knowledge.core.base import IEKnowledgeRunner

if TYPE_CHECKING:
    from ieforge.knowledge.core.protocols import (
        KnowledgeBuilder,
        KnowledgeInjector,
        KnowledgeSource,
        QueryBuilder,
        Retriever,
    )
    from ieforge.knowledge.core.schema import (
        BuiltKnowledge,
        KnowledgeContext,
        Query,
        RetrievalResult,
    )


class KnowledgePipeline(IEKnowledgeRunner):
    """Lazy knowledge retrieval pipeline for example augmentation."""

    def __init__(
        self,
        source: KnowledgeSource,
        builder: KnowledgeBuilder,
        retriever: Retriever,
        injector: KnowledgeInjector,
        query_builder: QueryBuilder,
        *,
        built_knowledge: BuiltKnowledge | None = None,
    ) -> None:
        """Initialize the pipeline components."""
        self._source = source
        self._builder = builder
        self._retriever = retriever
        self._injector = injector
        self._query_builder = query_builder
        self._built = built_knowledge

    @property
    def knowledge(self) -> BuiltKnowledge:
        """Built knowledge, constructed lazily on first access."""
        if self._built is None:
            self._built = self._builder.build(self._source.load())
        return self._built

    def retrieve(self, example: Any) -> tuple[Query, RetrievalResult]:
        """Build a query and retrieve facts for one example."""
        query = self._query_builder.build(example)
        retrieval = self._retriever.retrieve(query, self.knowledge)
        return query, retrieval

    def inject(self, query: Query, retrieval: RetrievalResult) -> KnowledgeContext:
        """Inject retrieved knowledge into a context."""
        return self._injector.inject(query, retrieval)

    def run(self, example: Any) -> KnowledgeContext:
        """Run retrieval and injection for one example."""
        query, retrieval = self.retrieve(example)
        return self.inject(query, retrieval)
