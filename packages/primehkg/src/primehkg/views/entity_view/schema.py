from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from primehkg.schema import Polarity


class AttributeNodeRow(BaseModel):
    """One (entity, modality) attribute node in the entity-attribute view."""

    attribute_node_id: str
    entity_node_id: str
    modality: str
    node_type: str = "attribute"
    feature_refs: dict[str, Any] = Field(default_factory=dict)


class AttributeHyperedgeRow(BaseModel):
    """An ``entity_attribute`` hyperedge binding an anchor + its attribute nodes."""

    he_id: str
    semantics: str = "entity_attribute"
    evidence_grade: str = "structural"
    polarity: Polarity = "undirected"
    node_ids: list[str]
    roles: dict[str, str]
    support_id: str
