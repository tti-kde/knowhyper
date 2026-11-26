from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Polarity = Literal["positive", "negative", "unknown", "undirected"]


class NodeRow(BaseModel):
    """Row schema for `nodes` table (HF dataset)."""

    node_id: str
    node_type: str
    name: str
    synonyms: list[str] = Field(default_factory=list)
    xrefs: dict[str, Any] = Field(default_factory=dict)
    source_meta: dict[str, Any] = Field(default_factory=dict)
    feature_refs: dict[str, Any] = Field(default_factory=dict)


class HyperedgeRow(BaseModel):
    """Row schema for `hyperedges` table (HF dataset)."""

    he_id: str
    semantics: str
    evidence_grade: str
    polarity: Polarity
    node_ids: list[str]
    roles: dict[str, str]
    support_id: str


class SupportRow(BaseModel):
    """Row schema for `support` table (HF dataset)."""

    support_id: str
    sources: list[dict[str, Any]]
    support_triplets: list[dict[str, Any]]
    rule_trace: dict[str, Any]
    confidence: float | None = None
