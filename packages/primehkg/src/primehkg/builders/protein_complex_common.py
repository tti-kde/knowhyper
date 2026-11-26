"""Shared constants for the curated ``protein_complex`` family.

CORUM and Complex Portal both contribute ``protein_complex`` hyperedges. To let an identical
GeneID member set from either source collapse to one ``he_id``, both builders hash the SAME
source-independent ``rule_trace``; per-source provenance lives on the support row instead.
"""

from __future__ import annotations

from typing import Any

# The ONLY rule_trace hashed into a protein_complex he_id. Source-independent on purpose.
PROTEIN_COMPLEX_RULE_TRACE: dict[str, Any] = {
    "kind": "curated_db",
    "semantics_builder": "protein_complex",
    "version": 1,
}


def dedupe_protein_complex_segment(
    segment: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Collapse same-``he_id`` protein_complex rows within ONE builder's output.

    Two complexes from the same source with an identical resolved member set hash to the same
    source-independent ``he_id``; emit ONE hyperedge + ONE support row, accumulating both source
    complex_ids in ``support_triplets`` and de-duping ``sources``. Keeps every builder's output
    free of duplicate ``he_id`` / ``support_id`` before ``merge_hypergraph_segments``.
    """
    hyperedges = segment.get("hyperedges", [])
    support = segment.get("support", [])

    # Index support rows by support_id; first wins, later ones get merged in.
    support_by_id: dict[str, dict[str, Any]] = {}
    for srow in support:
        sid = srow.get("support_id")
        if not isinstance(sid, str) or not sid:
            msg = f"protein_complex support row missing support_id: {srow!r}"
            raise ValueError(msg)
        if sid not in support_by_id:
            support_by_id[sid] = srow
        else:
            survivor = support_by_id[sid]
            for src in srow.get("sources", []):
                if src not in survivor.setdefault("sources", []):
                    survivor["sources"].append(src)
            for trip in srow.get("support_triplets", []):
                if trip not in survivor.setdefault("support_triplets", []):
                    survivor["support_triplets"].append(trip)

    # Deduplicate hyperedges by he_id (first wins).
    out_he: list[dict[str, Any]] = []
    seen_he: set[str] = set()
    for he in hyperedges:
        hid = he.get("he_id")
        if not isinstance(hid, str) or not hid:
            out_he.append(he)
            continue
        if hid in seen_he:
            continue
        seen_he.add(hid)
        out_he.append(he)

    out_support = list(support_by_id.values())
    return {"hyperedges": out_he, "support": out_support}
