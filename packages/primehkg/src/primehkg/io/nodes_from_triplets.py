from __future__ import annotations

from typing import Any

from primehkg.ids import normalize_node_id
from primehkg.nodes.normalize import infer_node_type_and_name
from primehkg.schema import NodeRow


def build_minimal_nodes_from_triplets(
    triplets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive a minimal ``nodes`` table from triplet endpoints (v0).

    - Collects normalized ``s`` and ``t`` values from each triplet (string values only).
    - Deterministic row order: sorted unique ``node_id``.
    - Populates `NodeRow`-compatible dicts with empty ``synonyms`` / ``xrefs`` /
      ``feature_refs`` (enrichment is out of scope for v0).

    This does **not** load entity metadata from PrimeKG node files; it only
    reflects ids seen in the edge list. See ``docs/builders/minimal-nodes-from-triplets.md``.
    """
    ids: set[str] = set()
    for tr in triplets:
        s = tr.get("s")
        t = tr.get("t")
        if isinstance(s, str):
            ns = normalize_node_id(s)
            if ns:
                ids.add(ns)
        if isinstance(t, str):
            nt = normalize_node_id(t)
            if nt:
                ids.add(nt)

    out: list[dict[str, Any]] = []
    for node_id in sorted(ids):
        node_type, name = infer_node_type_and_name(node_id)
        row = NodeRow(
            node_id=node_id,
            node_type=node_type,
            name=name,
            synonyms=[],
            xrefs={},
            source_meta={},
            feature_refs={},
        )
        out.append(row.model_dump(mode="python"))
    return out


def extend_nodes_with_node_ids(
    base_nodes: list[dict[str, Any]], *, node_ids: set[str]
) -> list[dict[str, Any]]:
    """Ensure the `nodes` table includes all ids referenced by hyperedges.

    Build CLI can add curated hyperedges whose endpoints do not appear in PrimeKG triplets.
    We still want those endpoints represented in `nodes` so downstream consumers can join.
    """
    existing = {
        row.get("node_id") for row in base_nodes if isinstance(row.get("node_id"), str)
    }
    missing = sorted(nid for nid in node_ids if nid not in existing)
    if not missing:
        return base_nodes

    out = list(base_nodes)
    for node_id in missing:
        node_type, name = infer_node_type_and_name(node_id)
        row = NodeRow(
            node_id=node_id,
            node_type=node_type,
            name=name,
            synonyms=[],
            xrefs={},
            source_meta={},
            feature_refs={},
        )
        out.append(row.model_dump(mode="python"))

    out.sort(key=lambda r: str(r.get("node_id", "")))
    return out
