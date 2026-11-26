from __future__ import annotations

from collections import defaultdict
from typing import Any

MONDO_AMBIGUITY_HINT_KEY = "usage_scope_override"


def _is_mondo_definition(row: dict[str, Any]) -> bool:
    return (
        str(row.get("source", "")).strip() == "MONDO"
        and str(row.get("source_field", "")).strip() == "definition"
    )


def resolve_mondo_definition_ambiguity(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Exact-hash collapse MONDO defs; hint multi-distinct nodes for demotion."""
    passthrough: list[dict[str, Any]] = []
    by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _is_mondo_definition(row):
            by_node[str(row["node_id"])].append(row)
        else:
            passthrough.append(row)

    rows_in = sum(len(v) for v in by_node.values())
    resolved: list[dict[str, Any]] = []
    nodes_single = 0
    nodes_multi = 0
    rows_demoted = 0
    rows_after = 0

    for node_id in sorted(by_node):
        group = by_node[node_id]
        # stable collapse: sort then keep first per content_hash
        group_sorted = sorted(
            group,
            key=lambda r: (
                str(r.get("content_hash") or ""),
                str(r.get("upstream_record_id") or ""),
                str(r.get("text") or ""),
            ),
        )
        seen: dict[str, dict[str, Any]] = {}
        for row in group_sorted:
            h = str(row.get("content_hash") or "")
            if h not in seen:
                seen[h] = dict(row)

        kept = list(seen.values())
        rows_after += len(kept)
        if len(kept) >= 2:
            nodes_multi += 1
            for row in kept:
                row[MONDO_AMBIGUITY_HINT_KEY] = "review_required"
                rows_demoted += 1
        else:
            nodes_single += 1
        resolved.extend(kept)

    stats = {
        "rows_in": rows_in,
        "rows_after_exact_dedupe": rows_after,
        "nodes_single": nodes_single,
        "nodes_multi_distinct": nodes_multi,
        "rows_demoted": rows_demoted,
    }
    return [*passthrough, *resolved], stats
