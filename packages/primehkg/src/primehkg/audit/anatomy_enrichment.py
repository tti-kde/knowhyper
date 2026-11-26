from __future__ import annotations

from typing import Any

ANATOMY_NODE_TYPE = "anatomy"


def _node_type(node: dict[str, Any]) -> str:
    raw = node.get("node_type")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    node_id = node.get("node_id")
    if isinstance(node_id, str) and node_id.startswith("primekg:anatomy:"):
        return ANATOMY_NODE_TYPE
    return "unknown"


def _has_xrefs_anatomy(node: dict[str, Any]) -> bool:
    xrefs = node.get("xrefs")
    if not isinstance(xrefs, dict):
        return False
    anatomy_vals = xrefs.get("anatomy")
    return isinstance(anatomy_vals, list) and any(
        isinstance(v, str) and v for v in anatomy_vals
    )


def _feature_ref_modality_present(
    node: dict[str, Any], *, bucket: str, modality: str
) -> bool:
    feature_refs = node.get("feature_refs")
    if not isinstance(feature_refs, dict):
        return False
    items = feature_refs.get(bucket)
    if not isinstance(items, list):
        return False
    for item in items:
        if isinstance(item, dict) and item.get("modality") == modality:
            return True
    return False


def compute_anatomy_enrichment_coverage(
    nodes: list[dict[str, Any]],
    *,
    source: str,
    source_path: str,
) -> dict[str, Any]:
    """Summarize anatomy xref and feature_refs coverage for anatomy core nodes."""
    typed_nodes = [
        node
        for node in nodes
        if isinstance(node, dict) and _node_type(node) == ANATOMY_NODE_TYPE
    ]
    count = len(typed_nodes)
    with_xrefs = sum(1 for n in typed_nodes if _has_xrefs_anatomy(n))
    with_ontology = sum(
        1
        for n in typed_nodes
        if _feature_ref_modality_present(n, bucket="ontology", modality="anatomy_term")
    )
    with_text = sum(
        1
        for n in typed_nodes
        if _feature_ref_modality_present(n, bucket="text", modality="description")
    )
    return {
        "source": source,
        "source_path": source_path,
        "by_node_type": {
            ANATOMY_NODE_TYPE: {
                "nodes": count,
                "with_xrefs_anatomy": with_xrefs,
                "with_ontology_ref": with_ontology,
                "with_text_ref": with_text,
                "xrefs_anatomy_rate": with_xrefs / count if count else 0.0,
                "text_ref_rate": with_text / count if count else 0.0,
            }
        },
    }
