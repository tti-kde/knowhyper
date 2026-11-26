from __future__ import annotations

from typing import TYPE_CHECKING, Any

from primehkg.ids import he_id
from primehkg.views.entity_view.modalities import extract_entity_modalities

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def _attribute_node_id(entity_node_id: str, modality: str) -> str:
    return f"attr:{entity_node_id}:{modality}"


def _iter_entity_view(
    nodes: Iterable[dict[str, Any]],
    *,
    source_fingerprint: str,
) -> Iterator[tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]]:
    """Yield (attribute_node_rows, he_row, support_row) per qualifying entity.

    Streaming: one entity is fully processed and yielded before the next is read,
    so the full node list is never materialized downstream.
    """
    for node in nodes:
        entity_node_id = node.get("node_id")
        if not isinstance(entity_node_id, str) or not entity_node_id.strip():
            continue
        modalities = extract_entity_modalities(node)
        if not modalities:
            continue

        ordered = sorted(modalities)  # deterministic modality order
        attr_rows: list[dict[str, Any]] = []
        node_ids = [entity_node_id]
        roles: dict[str, str] = {entity_node_id: "anchor"}
        for modality in ordered:
            anode_id = _attribute_node_id(entity_node_id, modality)
            attr_rows.append(
                {
                    "attribute_node_id": anode_id,
                    "entity_node_id": entity_node_id,
                    "modality": modality,
                    "node_type": "attribute",
                    "feature_refs": {modality: modalities[modality]},
                }
            )
            node_ids.append(anode_id)
            roles[anode_id] = f"attribute:{modality}"

        rule_trace = {
            "view": "entity_attribute",
            "source_fingerprint": source_fingerprint,
            "modalities": ordered,
        }
        hid = he_id(
            semantics="entity_attribute",
            node_ids=node_ids,
            roles=roles,
            rule_trace=rule_trace,
        )
        # Content-address the support id from the (already content-addressed) he_id
        # so it stays stable regardless of input node order (matches other builders'
        # `sup:<hid>` convention).
        support_id = f"sup:{hid}"
        he_row = {
            "he_id": hid,
            "semantics": "entity_attribute",
            "evidence_grade": "structural",
            "polarity": "undirected",
            "node_ids": node_ids,
            "roles": roles,
            "support_id": support_id,
        }
        support_row = {
            "support_id": support_id,
            "sources": [],
            "support_triplets": [],
            "rule_trace": rule_trace,
            "confidence": None,
        }
        yield attr_rows, he_row, support_row


def build_entity_view_tables(
    nodes: Iterable[dict[str, Any]],
    *,
    source_fingerprint: str,
) -> dict[str, list[dict[str, Any]]]:
    """Materialize the three entity-view tables from core node rows."""
    attribute_nodes: list[dict[str, Any]] = []
    hyperedges: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    for attr_rows, he_row, support_row in _iter_entity_view(
        nodes, source_fingerprint=source_fingerprint
    ):
        attribute_nodes.extend(attr_rows)
        hyperedges.append(he_row)
        support.append(support_row)
    return {
        "entity_view_nodes": attribute_nodes,
        "entity_view_hyperedges": hyperedges,
        "entity_view_support": support,
    }
