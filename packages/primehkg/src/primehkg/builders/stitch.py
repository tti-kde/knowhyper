from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from primehkg.ids import he_id
from primehkg.io.curated_node_ids import (
    stitch_chemical_to_node_id,
    stitch_protein_to_node_id,
)
from primehkg.io.curated_sources import load_stitch_rows


def _role_for(node_id: str) -> str:
    if node_id.startswith(("primekg:drug:", "chebi:")):
        return "drug"
    return "protein"


def build_stitch_drug_protein_hyperedges(
    stitch_tsv: str | Path,
    *,
    cid_map: dict[str, dict[str, str | None]],
    ensp_to_geneid: dict[str, str],
    score_threshold: int = 700,
    stitch_version: str,
    existing_node_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Build size-2 undirected ``drug_protein`` hyperedges from resolvable STITCH pairs.

    One hyperedge per resolvable STITCH pair with ``combined_score >= score_threshold``.
    Pairs resolving to the same (chemical-node, protein-node) dedup, max-aggregating
    the score as confidence. Emits a STITCH-specific he_id (distinct from
    registry_atomic's ``drug_protein``).

    When ``existing_node_ids`` is provided (a set of node_ids already present in the
    graph), an edge is emitted only when BOTH resolved endpoints are in that set, so
    the builder never introduces new nodes (#408). When ``existing_node_ids is None``
    (the default), no such filtering is applied and behavior is unchanged.

    Returns ``{"hyperedges", "support", "stats"}`` where ``stats`` is a coverage
    counter dict (``total_rows``, ``rows_passing_threshold``,
    ``chemical_resolved``/``chemical_unresolved``,
    ``protein_resolved``/``protein_unresolved``, ``protein_resolved_rate``,
    ``edges_dropped_nonexisting``, ``edges_emitted``). Resolution counters are tallied
    over threshold-passing rows only (so
    ``chemical_resolved + chemical_unresolved == rows_passing_threshold``);
    ``protein_resolved_rate`` is the fraction of threshold-passing rows whose protein
    endpoint resolved to an existing node (approximates the ENSP->GeneID resolution
    rate; #408). ``edges_dropped_nonexisting`` counts resolved pairs dropped because an
    endpoint was not in ``existing_node_ids`` (0 when the param is None).
    """
    rows = load_stitch_rows(stitch_tsv)

    hyperedges: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    support_by_hid: dict[str, dict[str, Any]] = {}

    total_rows = 0
    rows_passing_threshold = 0
    chemical_resolved = 0
    chemical_unresolved = 0
    protein_resolved = 0
    protein_unresolved = 0
    edges_dropped_nonexisting = 0

    for row in rows:
        total_rows += 1
        # Unlike signor (which admits a scoreless edge with confidence=None), a
        # STITCH row without an integer combined_score cannot be threshold-checked,
        # so it is dropped.
        raw = (row.get("score") or "").strip()
        try:
            score = int(raw)
        except ValueError:
            continue
        if score < score_threshold:
            continue
        rows_passing_threshold += 1
        chem = stitch_chemical_to_node_id(row["chemical"], cid_map)
        prot = stitch_protein_to_node_id(row["protein"], ensp_to_geneid)
        if chem is None:
            chemical_unresolved += 1
        else:
            chemical_resolved += 1
        if prot is None:
            protein_unresolved += 1
        else:
            protein_resolved += 1
        if chem is None or prot is None or chem == prot:
            continue
        # #408: never introduce new nodes — drop pairs whose endpoints are not
        # already present in the graph.
        if existing_node_ids is not None and (
            chem not in existing_node_ids or prot not in existing_node_ids
        ):
            edges_dropped_nonexisting += 1
            continue

        node_ids = sorted([chem, prot])
        roles = {n: _role_for(n) for n in node_ids}
        rule_trace = {
            "builder": "stitch",
            "version": 1,
            "kind": "curated_db",
            "relation": "drug_protein",
        }
        hid = he_id(
            semantics="drug_protein",
            node_ids=node_ids,
            roles=roles,
            rule_trace=rule_trace,
        )
        confidence = score / 1000.0
        if hid in support_by_hid:
            existing = support_by_hid[hid]["confidence"]
            if existing is None or confidence > existing:
                support_by_hid[hid]["confidence"] = confidence
            continue
        support_id = f"sup:{hid}"
        support_row = {
            "support_id": support_id,
            "sources": [{"name": "STITCH", "version": stitch_version}],
            "support_triplets": [{"s": chem, "r": "drug_protein", "t": prot}],
            "rule_trace": rule_trace,
            "confidence": confidence,
        }
        support_by_hid[hid] = support_row
        support.append(support_row)
        hyperedges.append(
            {
                "he_id": hid,
                "semantics": "drug_protein",
                "evidence_grade": "curated_db",
                "polarity": "undirected",
                "node_ids": node_ids,
                "roles": roles,
                "support_id": support_id,
            }
        )

    stats: dict[str, Any] = {
        "total_rows": total_rows,
        "rows_passing_threshold": rows_passing_threshold,
        "chemical_resolved": chemical_resolved,
        "chemical_unresolved": chemical_unresolved,
        "protein_resolved": protein_resolved,
        "protein_unresolved": protein_unresolved,
        "protein_resolved_rate": (
            protein_resolved / rows_passing_threshold if rows_passing_threshold else 0.0
        ),
        "edges_dropped_nonexisting": edges_dropped_nonexisting,
        "edges_emitted": len(hyperedges),
    }

    return {"hyperedges": hyperedges, "support": support, "stats": stats}
