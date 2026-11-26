from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from primehkg.ids import he_id
from primehkg.io.curated_node_ids import (
    curated_chebi_node_id,
    primekg_protein_node_id_from_uniprot,
)
from primehkg.io.curated_sources import load_signor_rows


def _parse_score(raw: str) -> float | None:
    s = (raw or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _polarity_from_effect(effect: str) -> str:
    e = effect.strip().lower()
    if e.startswith("up-regulates"):
        return "positive"
    if e.startswith("down-regulates"):
        return "negative"
    return "unknown"


def _resolve_signor_entity(
    typ: str, ident: str, db: str, uniprot_to_geneid: dict[str, str] | None
) -> str | None:
    d = db.strip().lower()
    if d == "uniprot" or typ.strip().lower() == "protein":
        if not ident.strip():
            return None
        return primekg_protein_node_id_from_uniprot(
            ident, uniprot_to_geneid=uniprot_to_geneid
        )
    if "chebi" in d:
        if not ident.strip():
            return None
        return curated_chebi_node_id(ident)
    return None


def build_signor_regulation_hyperedges(
    signor_tsv: str | Path,
    *,
    signor_version: str,
    uniprot_to_geneid: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """One size-2 directional `regulation` hyperedge per resolvable SIGNOR row."""
    rows = load_signor_rows(signor_tsv)

    hyperedges: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    support_by_hid: dict[str, dict[str, Any]] = {}

    for row in rows:
        src = _resolve_signor_entity(
            row["type_a"], row["id_a"], row["db_a"], uniprot_to_geneid
        )
        tgt = _resolve_signor_entity(
            row["type_b"], row["id_b"], row["db_b"], uniprot_to_geneid
        )
        if src is None or tgt is None or src == tgt:
            continue

        polarity = _polarity_from_effect(row["effect"])
        node_ids = sorted([src, tgt])
        roles = {src: "regulator", tgt: "regulated"}
        rule_trace = {
            "builder": "signor",
            "version": 1,
            "kind": "curated_db",
            "effect": row["effect"],
            "mechanism": row["mechanism"],
        }
        hid = he_id(
            semantics="regulation",
            node_ids=node_ids,
            roles=roles,
            rule_trace=rule_trace,
        )
        score = _parse_score(row["score"])
        if hid in support_by_hid:
            existing = support_by_hid[hid]["confidence"]
            if score is not None and (existing is None or score > existing):
                support_by_hid[hid]["confidence"] = score
            continue
        support_id = f"sup:{hid}"
        support_triplets = [{"pmid": row["pmid"]}] if row["pmid"] else []
        support_row = {
            "support_id": support_id,
            "sources": [{"name": "SIGNOR", "version": signor_version}],
            "support_triplets": support_triplets,
            "rule_trace": rule_trace,
            "confidence": score,
        }
        support_by_hid[hid] = support_row
        support.append(support_row)
        hyperedges.append(
            {
                "he_id": hid,
                "semantics": "regulation",
                "evidence_grade": "curated_db",
                "polarity": polarity,
                "node_ids": node_ids,
                "roles": roles,
                "support_id": support_id,
            }
        )

    return {"hyperedges": hyperedges, "support": support}
