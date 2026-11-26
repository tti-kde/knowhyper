from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from primehkg.ids import he_id
from primehkg.io.curated_node_ids import ctd_chemical_to_node_id
from primehkg.io.curated_sources import load_ctd_rows


def _polarity_from_verbs(verbs: set[str]) -> str:
    has_inc = "increases" in verbs
    has_dec = "decreases" in verbs
    if has_inc and not has_dec:
        return "positive"
    if has_dec and not has_inc:
        return "negative"
    return "unknown"


def build_ctd_regulation_hyperedges(
    ctd_tsv: str | Path,
    *,
    mesh_to_cid: dict[str, str],
    cid_map: dict[str, dict[str, str | None]],
    existing_node_ids: set[str],
    ctd_version: str,
) -> dict[str, Any]:
    """Size-2 ``regulation`` hyperedges from CTD curated chemical->gene interactions.

    Keeps only rows whose actions include ``increases``/``decreases`` (drops
    ``affects``-only). roles regulator(chemical)/regulated(protein); polarity
    positive(increases-only)/negative(decreases-only)/unknown(both verbs in one row,
    or same-pair cross-row conflict). Existing-node-only (no new nodes). CTD-specific
    he_id (``rule_trace.builder == "ctd"``), distinct from SIGNOR's ``regulation``.

    Returns ``{"hyperedges", "support", "stats"}`` where each support row mirrors
    signor's shape (``support_id``, ``sources``, ``support_triplets``, ``rule_trace``,
    ``confidence``; ``confidence`` is always ``None`` — CTD curated rows carry no
    numeric score) and ``stats`` is a coverage counter dict (``total_rows``,
    ``directed_rows``, ``chemical_resolved``/``chemical_unresolved``,
    ``gene_resolved``/``gene_unresolved``, ``edges_emitted``,
    ``edges_dropped_nonexisting``, ``polarity_conflicts``).
    """
    rows = load_ctd_rows(ctd_tsv)

    def _triplet(r: dict[str, Any]) -> dict[str, str]:
        # Mirror signor's pmid-only support-triplet idiom while retaining the
        # genuinely-useful CTD ``action`` provenance. Include ``pmid`` only when
        # non-empty (never emit an empty-string pmid). ``action`` is always present
        # for an emitted edge (the verb filter guarantees it is non-empty).
        if r["action"]:
            t: dict[str, str] = {"action": r["action"]}
            if r["pmid"]:
                t["pmid"] = r["pmid"]
            return t
        return {"pmid": r["pmid"]} if r["pmid"] else {}

    hyperedges: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    support_by_hid: dict[str, dict[str, Any]] = {}
    he_by_hid: dict[str, dict[str, Any]] = {}

    stats: dict[str, Any] = {
        "total_rows": 0,
        "directed_rows": 0,
        "chemical_resolved": 0,
        "chemical_unresolved": 0,
        "gene_resolved": 0,
        "gene_unresolved": 0,
        "edges_emitted": 0,
        "edges_dropped_nonexisting": 0,
        "polarity_conflicts": 0,
    }

    for row in rows:
        stats["total_rows"] += 1
        verbs = {t.split("^")[0] for t in row["action"].split("|")}
        if not (verbs & {"increases", "decreases"}):
            continue
        stats["directed_rows"] += 1

        chem = ctd_chemical_to_node_id(row["chemical_mesh"], mesh_to_cid, cid_map)
        if chem is None:
            stats["chemical_unresolved"] += 1
        else:
            stats["chemical_resolved"] += 1
        # CTD genes are Entrez ids -> primekg:protein:<entrez>; "resolved" means the
        # node already exists in the graph (existing-node-only builder, no new nodes).
        prot = f"primekg:protein:{row['gene_entrez']}"
        prot_ok = prot in existing_node_ids
        if prot_ok:
            stats["gene_resolved"] += 1
        else:
            stats["gene_unresolved"] += 1

        # Self-loop / unresolved-chemical pairs are out of scope entirely.
        if chem is None or chem == prot:
            continue
        # A pair whose chemical resolved but an endpoint is not present in the graph
        # is dropped to avoid introducing new nodes (mirrors stitch #408).
        if chem not in existing_node_ids or not prot_ok:
            stats["edges_dropped_nonexisting"] += 1
            continue

        node_ids = sorted([chem, prot])
        roles = {chem: "regulator", prot: "regulated"}
        polarity = _polarity_from_verbs(verbs)
        rule_trace = {
            "builder": "ctd",
            "version": 1,
            "kind": "curated_db",
            "relation": "regulation",
        }
        hid = he_id(
            semantics="regulation",
            node_ids=node_ids,
            roles=roles,
            rule_trace=rule_trace,
        )
        if hid in he_by_hid:
            existing_he = he_by_hid[hid]
            if (
                existing_he["polarity"] != polarity
                and existing_he["polarity"] != "unknown"
            ):
                existing_he["polarity"] = "unknown"
                stats["polarity_conflicts"] += 1
            triplet = _triplet(row)
            if triplet:
                support_by_hid[hid]["support_triplets"].append(triplet)
            continue

        support_id = f"sup:{hid}"
        triplet = _triplet(row)
        support_row = {
            "support_id": support_id,
            "sources": [{"name": "CTD", "version": ctd_version}],
            "support_triplets": [triplet] if triplet else [],
            "rule_trace": rule_trace,
            "confidence": None,
        }
        support_by_hid[hid] = support_row
        support.append(support_row)
        he = {
            "he_id": hid,
            "semantics": "regulation",
            "evidence_grade": "curated_db",
            "polarity": polarity,
            "node_ids": node_ids,
            "roles": roles,
            "support_id": support_id,
        }
        he_by_hid[hid] = he
        hyperedges.append(he)
        stats["edges_emitted"] += 1

    return {"hyperedges": hyperedges, "support": support, "stats": stats}
