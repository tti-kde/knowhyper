"""Deterministic graph-feature injectors for fixtures and tests."""

from __future__ import annotations

import torch

from ieforge.knowledge.core.base import IEKnowledgeInjector
from ieforge.knowledge.core.schema import KnowledgeContext, Query, RetrievalResult
from ieforge.registry import register


@register("knowledge_injector", name="constant_graph_features")
class ConstantGraphFeatureInjector(IEKnowledgeInjector):
    """Creates deterministic graph-feature tensors from retrieval results."""

    def __init__(
        self,
        node_count: int,
        hyperedge_count: int,
        hidden_size: int,
        scale: float = 1.0,
        include_text_context: bool = False,
    ) -> None:
        """Create a graph-feature injector with fixed feature shapes."""
        if node_count <= 0:
            msg = "node_count must be > 0"
            raise ValueError(msg)
        if hyperedge_count <= 0:
            msg = "hyperedge_count must be > 0"
            raise ValueError(msg)
        if hidden_size <= 0:
            msg = "hidden_size must be > 0"
            raise ValueError(msg)
        self._node_count = node_count
        self._hyperedge_count = hyperedge_count
        self._hidden_size = hidden_size
        self._scale = scale
        self._include_text_context = include_text_context

    @property
    def node_count(self) -> int:
        """Number of node feature rows produced by this injector."""
        return self._node_count

    @property
    def hyperedge_count(self) -> int:
        """Number of hyperedge feature rows produced by this injector."""
        return self._hyperedge_count

    @property
    def hidden_size(self) -> int:
        """Embedding dimension produced by this injector."""
        return self._hidden_size

    def inject(self, query: Query, retrieval: RetrievalResult) -> KnowledgeContext:
        """Return retrieval metadata plus fixed graph-feature tensors."""
        node_embeds = (
            torch.arange(
                self._node_count * self._hidden_size,
                dtype=torch.float32,
            ).reshape(self._node_count, self._hidden_size)
            * self._scale
        )
        hyperedge_embeds = (
            torch.arange(
                self._hyperedge_count * self._hidden_size,
                dtype=torch.float32,
            ).reshape(self._hyperedge_count, self._hidden_size)
            * self._scale
        )
        graph_embeds = (
            torch.arange(self._hidden_size, dtype=torch.float32) * self._scale
        )
        text_context = None
        if self._include_text_context:
            text_context = "\n".join(fact.text for fact in retrieval.facts)

        return KnowledgeContext(
            query=query,
            facts=retrieval.facts,
            text_context=text_context,
            graph_features={
                "node_embeds": node_embeds,
                "node_attention_mask": torch.ones(self._node_count, dtype=torch.long),
                "hyperedge_embeds": hyperedge_embeds,
                "hyperedge_attention_mask": torch.ones(
                    self._hyperedge_count,
                    dtype=torch.long,
                ),
                "graph_embeds": graph_embeds,
            },
            traces=retrieval.traces,
            metadata={"scores": retrieval.scores, **retrieval.metadata},
        )
