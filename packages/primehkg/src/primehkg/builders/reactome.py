from __future__ import annotations

from typing import TYPE_CHECKING, Any

from primehkg.ids import he_id
from primehkg.io.curated_node_ids import (
    curated_chebi_node_id,
    primekg_protein_node_id_from_uniprot,
)
from primehkg.io.curated_sources import load_reactome_reaction_rows

if TYPE_CHECKING:
    from pathlib import Path


_ROLE_VOCAB = {
    "input",
    "output",
    "catalyst",
    "positive_regulator",
    "negative_regulator",
    "cofactor",
    "participant",
}


def _normalize_role(role: str) -> str | None:
    r = role.strip().lower()
    if r in _ROLE_VOCAB:
        return r
    return None


def _normalize_entity_id(
    entity_id: str, uniprot_to_geneid: dict[str, str] | None = None
) -> str:
    raw = entity_id.strip()
    upper = raw.upper()

    if upper.startswith("CHEBI:"):
        return curated_chebi_node_id(raw)

    acc = upper
    for prefix in ("UNIPROTKB:", "UNIPROT:", "UNIPROTID:", "SWISSPROT:"):
        if acc.startswith(prefix):
            acc = acc.removeprefix(prefix)
            break

    return primekg_protein_node_id_from_uniprot(
        acc, uniprot_to_geneid=uniprot_to_geneid
    )


def build_reactome_reaction_event_hyperedges(
    reactome_tsv: str | Path,
    *,
    reactome_version: str,
    uniprot_to_geneid: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    reactions = load_reactome_reaction_rows(reactome_tsv)

    hyperedges: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []

    role_priority = {
        "input": 1,
        "output": 2,
        "catalyst": 3,
        "cofactor": 4,
        "positive_regulator": 5,
        "negative_regulator": 6,
        "participant": 7,
    }

    for reaction_id in sorted(reactions):
        roles_by_node: dict[str, str] = {}
        for row in reactions[reaction_id]:
            role = _normalize_role(row["role"])
            if role is None:
                continue
            node_id = _normalize_entity_id(row["entity_id"], uniprot_to_geneid)
            if node_id in roles_by_node:
                prev = roles_by_node[node_id]
                if prev == role:
                    continue
                if role_priority[role] < role_priority[prev]:
                    roles_by_node[node_id] = role
                continue
            roles_by_node[node_id] = role

        if not roles_by_node:
            continue

        node_ids = sorted(roles_by_node.keys())
        rule_trace = {
            "builder": "reactome",
            "version": 2,
            "kind": "curated_db",
            "reaction_id": reaction_id,
        }

        hid = he_id(
            semantics="reaction_event",
            node_ids=node_ids,
            roles=roles_by_node,
            rule_trace=rule_trace,
        )
        support_id = f"sup:{hid}"

        support.append(
            {
                "support_id": support_id,
                "sources": [{"name": "Reactome", "version": reactome_version}],
                "support_triplets": [],
                "rule_trace": rule_trace,
                "confidence": None,
            }
        )
        hyperedges.append(
            {
                "he_id": hid,
                "semantics": "reaction_event",
                "evidence_grade": "curated_db",
                "polarity": "unknown",
                "node_ids": node_ids,
                "roles": roles_by_node,
                "support_id": support_id,
            }
        )

    return {"hyperedges": hyperedges, "support": support}
