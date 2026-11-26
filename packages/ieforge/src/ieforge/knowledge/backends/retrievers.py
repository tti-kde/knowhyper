"""Retrievers and injectors for corpus-backed knowledge."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ieforge.knowledge.core.base import IEKnowledgeInjector, IEKnowledgeRetriever
from ieforge.knowledge.core.schema import Fact, KnowledgeContext, RetrievalResult
from ieforge.registry import register

if TYPE_CHECKING:
    from ieforge.knowledge.backends.stores import (
        InMemoryDocumentStore,
        NumpyVectorStore,
    )
    from ieforge.knowledge.core.protocols import Embedder
    from ieforge.knowledge.core.schema import BuiltKnowledge, Query

_DEFAULT_TEMPLATE = "[Knowledge]\n{facts}\n[/Knowledge]"


@register("knowledge_retriever", name="dense")
class DenseRetriever(IEKnowledgeRetriever):
    """Dense vector retriever over corpus-backed knowledge stores."""

    def __init__(self, embedder: Embedder, top_k: int = 5) -> None:
        """Create a retriever with an embedder and result limit."""
        if top_k <= 0:
            msg = "top_k must be > 0"
            raise ValueError(msg)
        self._embedder = embedder
        self._top_k = top_k

    @property
    def top_k(self) -> int:
        """Maximum number of facts returned for each query."""
        return self._top_k

    def retrieve(self, query: Query, knowledge: BuiltKnowledge) -> RetrievalResult:
        """Retrieve relevant facts from built knowledge."""
        docs: InMemoryDocumentStore = knowledge.handles["documents"]
        vecs: NumpyVectorStore = knowledge.handles["vectors"]
        if len(vecs) == 0:
            return RetrievalResult(query=query, facts=[])
        hits = vecs.search(self._embedder.encode([query.text])[0], k=self._top_k)
        facts: list[Fact] = []
        scores: dict[str, float] = {}
        for doc_id, score in hits:
            facts.append(
                Fact(
                    id=doc_id,
                    text=docs.get(doc_id),
                    source="documents",
                    metadata=docs.get_metadata(doc_id),
                )
            )
            scores[doc_id] = score
        return RetrievalResult(query=query, facts=facts, scores=scores)


@register("knowledge_injector", name="text_context")
class TextContextInjector(IEKnowledgeInjector):
    """Formats retrieved facts into a text context channel."""

    def __init__(
        self,
        template: str = _DEFAULT_TEMPLATE,
        separator: str = "\n",
        max_chars: int | None = None,
    ) -> None:
        """Create an injector with a facts template and optional truncation."""
        if "{facts}" not in template:
            msg = "template must contain a '{facts}' placeholder"
            raise ValueError(msg)
        self._template = template
        self._separator = separator
        self._max_chars = max_chars

    def inject(self, query: Query, retrieval: RetrievalResult) -> KnowledgeContext:
        """Format retrieval results as knowledge context."""
        if not retrieval.facts:
            return KnowledgeContext(
                query=query,
                facts=[],
                text_context="",
                traces=retrieval.traces,
                metadata={"scores": retrieval.scores, **retrieval.metadata},
            )
        facts_text = self._separator.join(fact.text for fact in retrieval.facts)
        if self._max_chars is not None and len(facts_text) > self._max_chars:
            facts_text = facts_text[: self._max_chars]
        return KnowledgeContext(
            query=query,
            facts=retrieval.facts,
            text_context=self._template.format(facts=facts_text),
            traces=retrieval.traces,
            metadata={"scores": retrieval.scores, **retrieval.metadata},
        )
