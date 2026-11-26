from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from primehkg.builders.protein_complex_common import (
    PROTEIN_COMPLEX_RULE_TRACE,
    dedupe_protein_complex_segment,
)
from primehkg.ids import he_id
from primehkg.io.curated_node_ids import primekg_protein_node_id_from_uniprot
from primehkg.io.curated_sources import load_complexportal_participants


def build_complexportal_protein_complex_hyperedges(
    complexportal_tsv: str | Path,
    *,
    complexportal_version: str,
    uniprot_to_geneid: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """One variable-arity ``protein_complex`` hyperedge per Complex Portal complex (size>=2).

    Uses the same source-independent ``PROTEIN_COMPLEX_RULE_TRACE`` as the CORUM builder, so an
    identical GeneID member set from either source collapses to one ``he_id`` (sources merged at
    dedup time). Per-source provenance lives on the support row.
    """
    complexes = load_complexportal_participants(complexportal_tsv)

    hyperedges: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []

    for complex_id in sorted(complexes):
        uniprot_ids = complexes[complex_id]["uniprot_ids"]
        if len(uniprot_ids) < 2:
            continue

        node_ids = sorted(
            {
                primekg_protein_node_id_from_uniprot(
                    uid, uniprot_to_geneid=uniprot_to_geneid
                )
                for uid in uniprot_ids
            }
        )
        if len(node_ids) < 2:
            continue
        roles = dict.fromkeys(node_ids, "protein")

        hid = he_id(
            semantics="protein_complex",
            node_ids=node_ids,
            roles=roles,
            rule_trace=PROTEIN_COMPLEX_RULE_TRACE,
        )
        support_id = f"sup:{hid}"

        support.append(
            {
                "support_id": support_id,
                "sources": [
                    {"name": "Complex Portal", "version": complexportal_version}
                ],
                "support_triplets": [
                    {"source": "Complex Portal", "complex_id": complex_id}
                ],
                "rule_trace": {
                    "builder": "complexportal",
                    "builder_version": 1,
                    "kind": "curated_db",
                    "complex_id": complex_id,
                },
                "confidence": None,
            }
        )
        hyperedges.append(
            {
                "he_id": hid,
                "semantics": "protein_complex",
                "evidence_grade": "curated_db",
                "polarity": "unknown",
                "node_ids": node_ids,
                "roles": roles,
                "support_id": support_id,
            }
        )

    return dedupe_protein_complex_segment(
        {"hyperedges": hyperedges, "support": support}
    )
