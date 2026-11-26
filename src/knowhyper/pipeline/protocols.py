"""Stage protocols for the KnowHyper knowledge pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ieforge.knowledge.core.schema import Query

    from knowhyper.cache.keys import InstanceKey
    from knowhyper.encoders.allset import Incidence
    from knowhyper.pipeline.types import (
        IntegrationPayload,
        KnowledgeRepr,
        LinkedEntity,
        Mention,
        SubHypergraph,
    )


@runtime_checkable
class EntityLinker(Protocol):
    """Resolve surface mentions to knowledge-graph nodes."""

    def link(self, mentions: Sequence[Mention]) -> list[LinkedEntity]:
        """Link mentions to knowledge-graph nodes."""
        ...


@runtime_checkable
class SubHGRetriever(Protocol):
    """Retrieve a sub-hypergraph seeded from linked entities."""

    def retrieve(
        self,
        linked: Sequence[LinkedEntity],
        *,
        top_k: int,
        instance: InstanceKey | None = None,
    ) -> SubHypergraph:
        """Retrieve a sub-hypergraph for the given linked entities.

        ``instance`` optionally identifies the dataset example so a caching
        retriever can key the result; non-caching retrievers ignore it.
        """
        ...


@runtime_checkable
class SubHGEncoder(Protocol):
    """Encode a sub-hypergraph into a knowledge representation.

    Every encoder implements ``encode``. The optional ``build_incidence`` (for the
    GPU-batched cross-attention path) is a SEPARATE protocol :class:`IncidenceEncoder`
    so encoders that only do pooled/soft-prompt encoding (e.g. StructureOnly) still
    satisfy ``SubHGEncoder``; the injector duck-types ``build_incidence`` only when
    the integrator declares ``needs_incidence``.
    """

    def encode(self, sub: SubHypergraph) -> KnowledgeRepr:
        """Encode the sub-hypergraph into a knowledge representation."""
        ...


@runtime_checkable
class IncidenceEncoder(Protocol):
    """Optional encoder capability: build a leaf incidence for batched GPU encode."""

    def build_incidence(self, sub: SubHypergraph) -> Incidence:
        """Build an incidence structure for batched GPU encoding."""
        ...


@runtime_checkable
class Integrator(Protocol):
    """Turn a knowledge representation into an integration payload."""

    def to_payload(self, repr_: KnowledgeRepr) -> IntegrationPayload:
        """Convert a knowledge representation into an integration payload."""
        ...


@runtime_checkable
class TextSerializer(Protocol):
    """Render a retrieved sub-hypergraph as text for the R2a/R2b text arms.

    Made explicit because the injector used to reach for ``items_key`` with a
    ``getattr`` default: a serializer that forgot to declare one would publish
    under R2a's key, the collator would look under its own and find nothing, and
    the arm would train with **no knowledge at all** while still reporting a valid
    fingerprint. That is the same silent-empty-payload failure the per-arm keys
    exist to prevent, so the requirement belongs in the type.
    """

    #: Metadata key the injector publishes items under. Each arm declares its own
    #: so a mismatched collator yields empty knowledge loudly rather than
    #: consuming the other arm's payload.
    items_key: str

    def serialize(self, query: Query, sub: SubHypergraph) -> Any:
        """Serialize the retrieved sub-hypergraph into text plus items."""
        ...
