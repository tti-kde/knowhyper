from __future__ import annotations

from typing import Any, TypedDict


class HyperedgeDedupStats(TypedDict):
    """Row counts recorded when deduplicating hyperedges by ``he_id``."""

    rows_before: int
    rows_after: int
    duplicate_he_id_skipped: int
    unique_he_id_after: int
    sources_merged: int


def dedupe_hyperedges_by_id(
    hyperedges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], HyperedgeDedupStats, dict[str, list[str]]]:
    """Dedupe hyperedges by ``he_id``, keeping the first occurrence.

    Returns the deduped list, stats, and a mapping ``surviving_support_id -> [dropped support_id,
    ...]`` for every he_id that collapsed >=1 duplicate, so callers can merge support sources.
    """
    seen: dict[str, str] = {}  # he_id -> surviving support_id
    out: list[dict[str, Any]] = []
    skipped = 0
    collapsed: dict[str, list[str]] = {}  # surviving support_id -> dropped support_ids
    for he in hyperedges:
        hid = he.get("he_id")
        if not isinstance(hid, str) or not hid:
            out.append(he)
            continue
        if hid in seen:
            skipped += 1
            dropped_sid = he.get("support_id")
            surviving_sid = seen[hid]
            if isinstance(dropped_sid, str) and dropped_sid:
                collapsed.setdefault(surviving_sid, []).append(dropped_sid)
            continue
        surviving_sid = he.get("support_id")
        seen[hid] = surviving_sid if isinstance(surviving_sid, str) else ""
        out.append(he)
    stats: HyperedgeDedupStats = {
        "rows_before": len(hyperedges),
        "rows_after": len(out),
        "duplicate_he_id_skipped": skipped,
        "unique_he_id_after": len(out),
        "sources_merged": len(collapsed),
    }
    return out, stats, collapsed
