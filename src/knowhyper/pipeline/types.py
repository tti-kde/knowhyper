"""Upstream-independent domain types passed between KnowHyper pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from knowhyper.encoders.allset import Incidence


@dataclass(frozen=True)
class Mention:
    """A surface mention of an entity in source text."""

    text: str
    char_span: tuple[int, int] | None = None
    entity_type: str | None = None
    # Optional in-document full form of an abbreviation mention (e.g. "Ang II" ->
    # "angiotensin II"), used as a fallback linking aid; None when not an abbrev.
    expansion: str | None = None
    # Optional xref namespace (e.g. "mesh"/"omim") for id-based linking; None for
    # a surface-text mention, which the linker resolves without an xref lookup.
    namespace: str | None = None
    # Optional target-pair ownership for pair-aware text serialization. Linkers
    # ignore it; retrieval provenance preserves it across cache precompute.
    pair_role: str | None = None


@dataclass(frozen=True)
class LinkedEntity:
    """A mention linked to a knowledge-graph node with a match score."""

    mention: Mention
    node_id: str
    score: float
    match_kind: str
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class HyperedgeRef:
    """A reference to a hyperedge connecting a set of nodes."""

    he_id: str
    semantics: str
    node_ids: tuple[str, ...]
    roles: Mapping[str, str] = field(default_factory=dict)
    polarity: str = "unknown"
    score: float = 0.0
    score_breakdown: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SubHypergraph:
    """A retrieved sub-hypergraph seeded from linked entities."""

    seed_node_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    hyperedges: tuple[HyperedgeRef, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeRepr:
    """Encoded representation of a sub-hypergraph for integration."""

    node_embeds: Tensor | None = None
    node_mask: Tensor | None = None
    hyperedge_embeds: Tensor | None = None
    hyperedge_mask: Tensor | None = None
    graph_embed: Tensor | None = None
    text_context: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    incidence: Incidence | None = None


@dataclass
class IntegrationPayload:
    """What the Integrator hands to the ieforge adapter."""

    prompt_embeds: Tensor | None = None
    prompt_attention_mask: Tensor | None = None
    graph_features: dict[str, Any] = field(default_factory=dict)
    text_context: str | None = None
