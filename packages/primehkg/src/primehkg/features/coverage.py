from __future__ import annotations

from collections import defaultdict
from typing import Any

from primehkg.nodes.normalize import infer_node_type_and_name
from primehkg.policy.modality_coverage import (
    load_modality_coverage,
    node_types_for_profile,
)
from primehkg.policy.node_property_registry import (
    NodePropertyRegistry,
    load_node_property_registry,
)


def _resolve_node_type(node: dict[str, Any]) -> str:
    raw = node.get("node_type")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    node_id = node.get("node_id")
    if not isinstance(node_id, str) or not node_id:
        return "unknown"
    return infer_node_type_and_name(node_id)[0]


def _index_text_modalities(
    text_rows: list[dict[str, Any]],
) -> dict[str, set[str]]:
    by_node: dict[str, set[str]] = defaultdict(set)
    for row in text_rows:
        node_id = row.get("node_id")
        modality = row.get("modality")
        if isinstance(node_id, str) and isinstance(modality, str):
            by_node[node_id].add(modality)
    return by_node


def _index_sequence_types(
    sequence_rows: list[dict[str, Any]],
) -> dict[str, set[str]]:
    by_node: dict[str, set[str]] = defaultdict(set)
    for row in sequence_rows:
        node_id = row.get("node_id")
        seq_type = row.get("sequence_type")
        if isinstance(node_id, str) and isinstance(seq_type, str):
            by_node[node_id].add(seq_type)
    return by_node


def _has_modality(
    node_id: str,
    modality: str,
    *,
    text_by_node: dict[str, set[str]],
    seq_by_node: dict[str, set[str]],
) -> bool:
    if modality == "smiles":
        return "smiles" in seq_by_node.get(node_id, set())
    if modality == "protein":
        return "protein" in seq_by_node.get(node_id, set())
    if modality == "nucleotide":
        return "nucleotide" in seq_by_node.get(node_id, set())
    return modality in text_by_node.get(node_id, set())


def _node_covered(
    node_id: str,
    primary_group: tuple[str, ...],
    *,
    text_by_node: dict[str, set[str]],
    seq_by_node: dict[str, set[str]],
) -> bool:
    return any(
        _has_modality(
            node_id,
            mod,
            text_by_node=text_by_node,
            seq_by_node=seq_by_node,
        )
        for mod in primary_group
    )


def compute_modality_coverage(
    *,
    nodes: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
    profile: str = "full-v1-p1",
) -> dict[str, Any]:
    """Compute OR-primary-group modality coverage KPI for profile-scoped node types."""
    policy = load_modality_coverage()
    text_by_node = _index_text_modalities(text_rows)
    seq_by_node = _index_sequence_types(sequence_rows)

    nodes_by_type: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        node_type = _resolve_node_type(node)
        nodes_by_type[node_type].append(node_id)

    by_node_type: dict[str, Any] = {}
    all_pass = True

    for node_type in node_types_for_profile(profile, policy):
        type_policy = policy.by_node_type[node_type]
        node_ids = nodes_by_type.get(node_type, [])
        count = len(node_ids)
        covered = sum(
            1
            for nid in node_ids
            if _node_covered(
                nid,
                type_policy.primary_group,
                text_by_node=text_by_node,
                seq_by_node=seq_by_node,
            )
        )
        rate = covered / count if count else 0.0
        type_pass = count == 0 or rate >= type_policy.pass_threshold
        if not type_pass:
            all_pass = False

        entry: dict[str, Any] = {
            "nodes": count,
            "covered": covered,
            "rate": rate,
            "pass": type_pass,
            "primary_group": list(type_policy.primary_group),
        }

        if "smiles_rate" in type_policy.stretch_metrics:
            smiles_count = sum(
                1 for nid in node_ids if "smiles" in seq_by_node.get(nid, set())
            )
            entry["smiles_rate"] = smiles_count / count if count else 0.0

        if "sequence_rate" in type_policy.stretch_metrics:
            seq_count = sum(
                1 for nid in node_ids if "protein" in seq_by_node.get(nid, set())
            )
            entry["sequence_rate"] = seq_count / count if count else 0.0

        if node_type == "drug":
            text_modality_counts: dict[str, int] = defaultdict(int)
            for nid in node_ids:
                for mod in text_by_node.get(nid, set()):
                    text_modality_counts[mod] += 1
            entry["text_modalities"] = dict(sorted(text_modality_counts.items()))

        by_node_type[node_type] = entry

    return {
        "profile": profile,
        "pass": all_pass,
        "threshold": policy.pass_threshold,
        "by_node_type": by_node_type,
    }


def _index_property_ids(
    property_rows: list[dict[str, Any]],
) -> dict[str, set[str]]:
    by_node: dict[str, set[str]] = defaultdict(set)
    for row in property_rows:
        node_id = row.get("node_id")
        property_id = row.get("property_id")
        if isinstance(node_id, str) and isinstance(property_id, str):
            by_node[node_id].add(property_id)
    return by_node


def compute_property_coverage(
    *,
    nodes: list[dict[str, Any]],
    property_rows: list[dict[str, Any]],
    registry: NodePropertyRegistry | None = None,
) -> dict[str, Any]:
    """Stretch KPI: per-property fill rates (not used for --strict-coverage)."""
    reg = registry if registry is not None else load_node_property_registry()
    props_by_node = _index_property_ids(property_rows)

    nodes_by_type: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        nodes_by_type[_resolve_node_type(node)].append(node_id)

    def _any_property_block(node_type: str, *, stretch_target: float) -> dict[str, Any]:
        node_ids = nodes_by_type.get(node_type, [])
        count = len(node_ids)
        filled = sum(1 for nid in node_ids if props_by_node.get(nid))
        fill_rate = filled / count if count else 0.0
        return {
            "node_type": node_type,
            "count": count,
            "filled": filled,
            "fill_rate": fill_rate,
            "stretch_target": stretch_target,
            "stretch_pass": fill_rate >= stretch_target,
        }

    by_property_id: dict[str, Any] = {}
    for property_id, prop_def in sorted(reg.by_id.items()):
        for node_type in prop_def.node_types:
            node_ids = nodes_by_type.get(node_type, [])
            count = len(node_ids)
            filled = sum(
                1 for nid in node_ids if property_id in props_by_node.get(nid, set())
            )
            fill_rate = filled / count if count else 0.0
            entry = {
                "node_type": node_type,
                "fill_rate": fill_rate,
                "count": count,
                "filled": filled,
            }
            key = (
                property_id
                if len(prop_def.node_types) == 1
                else f"{property_id}:{node_type}"
            )
            by_property_id[key] = entry

    return {
        "pass": None,
        "any_property": {
            "drug": _any_property_block("drug", stretch_target=0.80),
            "disease": _any_property_block("disease", stretch_target=0.50),
            "protein": _any_property_block("protein", stretch_target=0.70),
            "phenotype": _any_property_block("effect/phenotype", stretch_target=0.85),
        },
        "by_property_id": by_property_id,
    }
