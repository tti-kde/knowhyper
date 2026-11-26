"""Core data structures for IEForge knowledge augmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Query:
    """Retrieval query produced from an IEForge example."""

    text: str
    entities: list[str] = field(default_factory=list)
    task: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Fact:
    """Knowledge fact returned by a retriever."""

    id: str
    text: str
    entities: list[str] = field(default_factory=list)
    relation: str | None = None
    source: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    """Debug trace emitted during retrieval or injection."""

    steps: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Retriever output before channel-specific injection."""

    query: Query
    facts: list[Fact]
    scores: dict[str, float] = field(default_factory=dict)
    traces: list[RetrievalTrace] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeContext:
    """Channel-agnostic knowledge payload attached to an example."""

    query: Query
    facts: list[Fact]
    text_context: str | None = None
    prompt_embeds: Any | None = None
    prompt_attention_mask: Any | None = None
    graph_features: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    traces: list[RetrievalTrace] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeArtifact:
    """Raw knowledge artifact loaded from a source."""

    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuiltKnowledge:
    """Prepared knowledge handles consumed by retrievers."""

    handles: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
