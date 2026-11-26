from __future__ import annotations

from typing import Any

GO_NODE_TYPES = frozenset(
    {"biological_process", "molecular_function", "cellular_component"}
)


def _node_type(node: dict[str, Any]) -> str:
    raw = node.get("node_type")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    node_id = node.get("node_id")
    if isinstance(node_id, str) and node_id.startswith("primekg:"):
        parts = node_id.split(":")
        if len(parts) >= 3:
            return parts[1]
    return "unknown"


def _has_xrefs_go(node: dict[str, Any]) -> bool:
    xrefs = node.get("xrefs")
    if not isinstance(xrefs, dict):
        return False
    go_vals = xrefs.get("go")
    return isinstance(go_vals, list) and any(isinstance(v, str) and v for v in go_vals)


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


def compute_go_enrichment_coverage(
    nodes: list[dict[str, Any]],
    *,
    source: str,
    source_path: str,
) -> dict[str, Any]:
    """Summarize GO xref and feature_refs coverage for GO-family core nodes."""
    by_type: dict[str, list[dict[str, Any]]] = {t: [] for t in GO_NODE_TYPES}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = _node_type(node)
        if node_type in GO_NODE_TYPES:
            by_type[node_type].append(node)

    by_node_type: dict[str, Any] = {}
    for node_type in sorted(GO_NODE_TYPES):
        typed_nodes = by_type[node_type]
        count = len(typed_nodes)
        with_xrefs = sum(1 for n in typed_nodes if _has_xrefs_go(n))
        with_ontology = sum(
            1
            for n in typed_nodes
            if _feature_ref_modality_present(n, bucket="ontology", modality="go_term")
        )
        with_text = sum(
            1
            for n in typed_nodes
            if _feature_ref_modality_present(n, bucket="text", modality="definition")
        )
        by_node_type[node_type] = {
            "nodes": count,
            "with_xrefs_go": with_xrefs,
            "with_ontology_ref": with_ontology,
            "with_text_ref": with_text,
            "xrefs_go_rate": with_xrefs / count if count else 0.0,
            "text_ref_rate": with_text / count if count else 0.0,
        }

    return {
        "source": source,
        "source_path": source_path,
        "by_node_type": by_node_type,
    }
