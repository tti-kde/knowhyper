from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from primehkg.ids import he_id
from primehkg.io.curated_sources import (
    load_drugcomb_drug_identifiers,
    load_drugcomb_pairs,
)

DRUGCOMB_SYNERGY_TAU = 5.0  # ZIP cutoff for synergy / antagonism polarity


def _polarity_from_median_zip(median_zip: float | None) -> str:
    if median_zip is None:
        return "unknown"
    if median_zip >= DRUGCOMB_SYNERGY_TAU:
        return "positive"
    if median_zip <= -DRUGCOMB_SYNERGY_TAU:
        return "negative"
    return "unknown"


def build_drugcomb_combination_hyperedges(
    summary_csv: str | Path,
    identifiers_tsv: str | Path,
    *,
    drugcomb_version: str,
    known_node_ids: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """One size-2 ``drug_combination`` hyperedge per resolvable DrugComb drug pair.

    Synergy is aggregated across cell lines via the median ``synergy_zip``; polarity is
    positive (synergy) / negative (antagonism) / unknown by the +/-5 ZIP cutoff.
    """
    name_to_db = load_drugcomb_drug_identifiers(identifiers_tsv)
    pairs = load_drugcomb_pairs(
        summary_csv, name_to_drugbank=name_to_db, known_node_ids=known_node_ids
    )

    hyperedges: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    seen: set[str] = set()

    for pair in pairs:
        node_ids = pair["node_ids"]
        if len(node_ids) != 2:
            continue
        roles = dict.fromkeys(node_ids, "combination_member")
        median_zip = pair["median_zip"]
        median_zip_rounded = round(median_zip, 4) if median_zip is not None else None
        polarity = _polarity_from_median_zip(median_zip_rounded)
        rule_trace = {
            "builder": "drugcomb",
            "version": 1,
            "kind": "curated_db",
            "median_zip": median_zip_rounded,
            "n_measurements": pair["n_measurements"],
            "n_cell_lines": pair["n_cell_lines"],
        }
        hid = he_id(
            semantics="drug_combination",
            node_ids=node_ids,
            roles=roles,
            rule_trace=rule_trace,
        )
        if hid in seen:
            continue
        seen.add(hid)
        support_id = f"sup:{hid}"
        support.append(
            {
                "support_id": support_id,
                "sources": [{"name": "DrugComb", "version": drugcomb_version}],
                "support_triplets": [
                    {
                        "source": "DrugComb",
                        "drug_names": pair["drug_names"],
                        "median_zip": median_zip_rounded,
                        "n_measurements": pair["n_measurements"],
                        "n_cell_lines": pair["n_cell_lines"],
                    }
                ],
                "rule_trace": rule_trace,
                "confidence": None,
            }
        )
        hyperedges.append(
            {
                "he_id": hid,
                "semantics": "drug_combination",
                "evidence_grade": "curated_db",
                "polarity": polarity,
                "node_ids": node_ids,
                "roles": roles,
                "support_id": support_id,
            }
        )

    return {"hyperedges": hyperedges, "support": support}
