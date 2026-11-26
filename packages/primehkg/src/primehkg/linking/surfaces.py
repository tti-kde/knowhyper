"""Collect linkable label surfaces from node rows."""

from __future__ import annotations

from typing import Any

from primehkg.linking.xref_index import normalize_label


def collect_label_surfaces(nodes: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return one row per name/synonym surface, normalized for linking."""
    rows: list[dict[str, str]] = []
    for row in nodes:
        node_id = row.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip():
            continue
        nid = node_id.strip()

        # Purely-numeric surfaces (e.g. a bare ClinVar VariationID like "100" used as a node's
        # name/synonym) are ambiguous across node_types and redundant with exact_node_id
        # resolution — drop them from the label index. See
        # docs/adr/2026-06-13-variant-node-core-scope.md.
        name = row.get("name")
        if isinstance(name, str) and name.strip():
            norm = normalize_label(name)
            if norm and not norm.isdigit():
                rows.append({"node_id": nid, "field": "name", "surface": norm})

        syns = row.get("synonyms")
        if isinstance(syns, list):
            for syn in syns:
                if isinstance(syn, str) and syn.strip():
                    norm = normalize_label(syn)
                    if norm and not norm.isdigit():
                        rows.append(
                            {"node_id": nid, "field": "synonym", "surface": norm}
                        )
    return rows
