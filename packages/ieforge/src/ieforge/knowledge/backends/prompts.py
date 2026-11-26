"""Deterministic soft-prompt injectors for fixtures and tests."""

from __future__ import annotations

import torch

from ieforge.knowledge.core.base import IEKnowledgeInjector
from ieforge.knowledge.core.schema import KnowledgeContext, Query, RetrievalResult
from ieforge.registry import register


@register("knowledge_injector", name="constant_prompt")
class ConstantPromptInjector(IEKnowledgeInjector):
    """Creates a deterministic soft-prompt context from retrieval results."""

    def __init__(
        self,
        prompt_length: int,
        hidden_size: int,
        scale: float = 1.0,
        include_text_context: bool = False,
    ) -> None:
        """Create a prompt injector with a fixed prompt shape."""
        if prompt_length <= 0:
            msg = "prompt_length must be > 0"
            raise ValueError(msg)
        if hidden_size <= 0:
            msg = "hidden_size must be > 0"
            raise ValueError(msg)
        self._prompt_length = prompt_length
        self._hidden_size = hidden_size
        self._scale = scale
        self._include_text_context = include_text_context

    @property
    def prompt_length(self) -> int:
        """Number of soft-prompt positions produced by this injector."""
        return self._prompt_length

    @property
    def hidden_size(self) -> int:
        """Embedding dimension produced by this injector."""
        return self._hidden_size

    def inject(self, query: Query, retrieval: RetrievalResult) -> KnowledgeContext:
        """Return retrieval metadata plus fixed prompt tensors."""
        prompt_embeds = (
            torch.arange(
                self._prompt_length * self._hidden_size,
                dtype=torch.float32,
            ).reshape(self._prompt_length, self._hidden_size)
            * self._scale
        )
        prompt_attention_mask = torch.ones(self._prompt_length, dtype=torch.long)
        text_context = None
        if self._include_text_context:
            text_context = "\n".join(fact.text for fact in retrieval.facts)

        return KnowledgeContext(
            query=query,
            facts=retrieval.facts,
            text_context=text_context,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            traces=retrieval.traces,
            metadata={"scores": retrieval.scores, **retrieval.metadata},
        )
