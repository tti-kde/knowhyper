"""Link layer protocols and result types."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict

if TYPE_CHECKING:
    from collections.abc import Collection


class LinkMatchKind(StrEnum):
    EXACT_NODE_ID = "exact_node_id"
    EXACT_XREF = "exact_xref"
    EXACT_LABEL = "exact_label"
    FUZZY_LABEL = "fuzzy_label"
    DENSE_LABEL = "dense_label"


class LinkCandidate(TypedDict):
    node_id: str
    score: float
    match_kind: LinkMatchKind
    matched_surface: str
    matched_field: str
    score_breakdown: dict[str, float]
    audit_flags: list[str]


class LinkResult(TypedDict):
    query: str
    namespace_hint: str | None
    candidates: list[LinkCandidate]
    determinism_version: str
    audit: NotRequired[dict[str, object]]


class LinkResolver(Protocol):
    def resolve(
        self,
        query: str,
        *,
        namespace: str | None = None,
        node_type_hint: str | Collection[str] | None = None,
        allowed_node_types: Collection[str] | None = None,
        top_k: int = 5,
    ) -> list[LinkCandidate]: ...


class LinkPipeline(Protocol):
    def link(
        self,
        query: str,
        *,
        namespace: str | None = None,
        node_type_hint: str | Collection[str] | None = None,
        allowed_node_types: Collection[str] | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> LinkResult: ...
