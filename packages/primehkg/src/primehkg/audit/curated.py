from __future__ import annotations

from collections import Counter
from typing import Any

from primehkg.audit.bundle_family import (
    _drug_count_histogram,
    _low_incidence_coverage,
    _roles_map,
)

_PROTEIN_COMPLEX = "protein_complex"


def _hyperedge_node_ids(he: dict[str, Any]) -> list[str]:
    node_ids_obj = he.get("node_ids")
    if not isinstance(node_ids_obj, list):
        return []
    return [n for n in node_ids_obj if isinstance(n, str)]


def node_id_set_from_nodes(nodes: list[dict[str, Any]] | None) -> set[str] | None:
    if nodes is None:
        return None
    out: set[str] = set()
    for row in nodes:
        if not isinstance(row, dict):
            continue
        nid = row.get("node_id")
        if isinstance(nid, str) and nid:
            out.add(nid)
    return out


def corum_protein_complex_metrics(
    hyperedges: list[dict[str, Any]],
    *,
    corum_tsv: str | None = None,
    known_node_ids: set[str] | None = None,
    uniprot_to_geneid: dict[str, str] | None = None,
    long_tail_quantile: float = 0.2,
    top_k: int = 10,
    support: list[dict[str, Any]] | None = None,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Audit metrics for ``protein_complex`` hyperedges, optionally scoped to a source.

    When *source_name* and *support* are both provided, only hyperedges whose
    matching support row (by ``support_id``) lists *source_name* in its ``sources``
    are counted.  When either is ``None`` the old behaviour is preserved (all
    ``protein_complex`` hyperedges are counted).
    """
    # Build support_id -> set(source names) map for efficient look-up.
    source_set_by_support_id: dict[str, set[str]] = {}
    if support is not None and source_name is not None:
        for s in support:
            if not isinstance(s, dict):
                continue
            sid = s.get("support_id")
            if not isinstance(sid, str):
                continue
            names: set[str] = set()
            for src in s.get("sources", []):
                if isinstance(src, dict) and isinstance(src.get("name"), str):
                    names.add(src["name"])
            source_set_by_support_id[sid] = names

    def _he_matches_source(he: dict[str, Any]) -> bool:
        if source_name is None:
            return True
        sid = he.get("support_id")
        if not isinstance(sid, str):
            return False
        return source_name in source_set_by_support_id.get(sid, set())

    hes = [
        he
        for he in hyperedges
        if isinstance(he, dict)
        and he.get("semantics") == _PROTEIN_COMPLEX
        and _he_matches_source(he)
    ]

    ingest: dict[str, Any] = {"corum_tsv": corum_tsv}
    if corum_tsv:
        from primehkg.io.curated_node_ids import primekg_protein_node_id_from_uniprot
        from primehkg.io.curated_sources import load_corum_participants

        complexes = load_corum_participants(corum_tsv)
        participant_rows = sum(len(c["uniprot_ids"]) for c in complexes.values())
        complexes_total = len(complexes)
        complexes_eligible = sum(
            1 for c in complexes.values() if len(c["uniprot_ids"]) >= 2
        )
        ingest.update(
            {
                "complexes_total": complexes_total,
                "complexes_eligible_ge_2_proteins": complexes_eligible,
                "participant_rows": participant_rows,
            }
        )
        if known_node_ids is not None:
            mapped_rows = 0
            unmapped_rows = 0
            for c in complexes.values():
                for uid in c["uniprot_ids"]:
                    nid = primekg_protein_node_id_from_uniprot(
                        uid, uniprot_to_geneid=uniprot_to_geneid
                    )
                    if nid in known_node_ids:
                        mapped_rows += 1
                    else:
                        unmapped_rows += 1
            ingest["participant_mapping"] = {
                "mapped_rows": mapped_rows,
                "unmapped_rows": unmapped_rows,
                "total_rows": mapped_rows + unmapped_rows,
                "fraction_mapped": (
                    float(mapped_rows) / float(mapped_rows + unmapped_rows)
                    if mapped_rows + unmapped_rows
                    else 0.0
                ),
            }

    if not hes:
        return {"hyperedge_count": 0, "ingest": ingest}

    size_counts: Counter[int] = Counter()
    protein_incidence: Counter[str] = Counter()
    bundle_involves_low_protein: Counter[str] = Counter()

    for he in hes:
        node_ids = _hyperedge_node_ids(he)
        size_counts[len(node_ids)] += 1
        protein_incidence.update(node_ids)

    lq = long_tail_quantile if 0.0 < long_tail_quantile < 1.0 else 0.2
    low_proteins = (
        {
            k
            for k, _ in sorted(
                protein_incidence.items(), key=lambda kv: (int(kv[1]), kv[0])
            )[: max(1, int(len(protein_incidence) * lq))]
        }
        if protein_incidence
        else set()
    )

    emitted_participants: set[str] = set()
    for he in hes:
        roles = _roles_map(he)
        members = (
            [nid for nid, role in roles.items() if role == "protein"]
            if roles
            else _hyperedge_node_ids(he)
        )
        emitted_participants.update(members)
        he_id = str(he.get("he_id", ""))
        bundle_involves_low_protein[he_id] = any(p in low_proteins for p in members)

    emitted_mapping: dict[str, Any] | None = None
    if known_node_ids is not None:
        mapped = emitted_participants & known_node_ids
        unmapped = emitted_participants - known_node_ids
        total = len(emitted_participants)
        emitted_mapping = {
            "mapped_participants": len(mapped),
            "unmapped_participants": len(unmapped),
            "total_participants": total,
            "fraction_mapped": float(len(mapped)) / float(total) if total else 0.0,
        }

    top_proteins = sorted(
        protein_incidence.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))
    )[:top_k]
    total = len(hes)

    return {
        "hyperedge_count": total,
        "complex_size": {
            "histogram": _drug_count_histogram(size_counts),
            "min": min(size_counts) if size_counts else None,
            "max": max(size_counts) if size_counts else None,
        },
        "top_proteins_by_complex_participation": {
            "top_k": top_k,
            "top_nodes": [[nid, int(c)] for nid, c in top_proteins],
        },
        "long_tail_protein_coverage": _low_incidence_coverage(
            protein_incidence, bundle_involves_low_protein, quantile=lq
        ),
        "emitted_participant_mapping": emitted_mapping,
        "ingest": ingest,
    }
