from __future__ import annotations

from collections import defaultdict
from typing import Any

from primehkg.ids import he_id, normalize_node_id


def build_protein_disease_bundle_hyperedges(
    triplets: list[dict[str, Any]],
    *,
    source_version: str = "unknown",
    max_diseases_per_protein: int = 25,
    min_diseases_per_bundle: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    protein_to_diseases: dict[str, set[str]] = defaultdict(set)
    disease_protein_edges: dict[tuple[str, str], set[tuple[str, str, str]]] = (
        defaultdict(set)
    )

    valid: list[tuple[str, str, str]] = []
    for tr in triplets:
        s = tr.get("s")
        r = tr.get("r")
        t = tr.get("t")
        if not (isinstance(s, str) and isinstance(r, str) and isinstance(t, str)):
            continue
        s = normalize_node_id(s)
        r = r.strip()
        t = normalize_node_id(t)
        if not (s and r and t):
            continue
        valid.append((s, r, t))

    for s, r, t in sorted(valid, key=lambda x: (x[1], x[0], x[2])):
        if (
            r == "disease_protein"
            and s.startswith("primekg:disease:")
            and t.startswith("primekg:protein:")
        ):
            protein_to_diseases[t].add(s)
            disease_protein_edges[(s, t)].add((s, r, t))

    hyperedges: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []

    for protein in sorted(protein_to_diseases):
        diseases_sorted = sorted(protein_to_diseases[protein])
        if len(diseases_sorted) < int(min_diseases_per_bundle):
            continue
        keep = diseases_sorted[: max(0, int(max_diseases_per_protein))]
        if len(keep) < int(min_diseases_per_bundle):
            continue

        support_set: set[tuple[str, str, str]] = set()
        for d in keep:
            support_set.update(disease_protein_edges.get((d, protein), set()))

        support_triplets = [
            {"s": s, "r": r, "t": t}
            for s, r, t in sorted(support_set, key=lambda x: (x[1], x[0], x[2]))
        ]

        rule_trace = {
            "builder": "protein_disease_bundle",
            "version": 1,
            "kind": "composed",
            "join_keys": ["disease_protein"],
            "generation_policy": {
                "max_diseases_per_protein": int(max_diseases_per_protein),
                "min_diseases_per_bundle": int(min_diseases_per_bundle),
                "tie_break": "disease_id_lex",
            },
        }

        node_ids = [protein, *keep]
        roles = {protein: "protein", **dict.fromkeys(keep, "disease")}
        hid = he_id(
            semantics="protein_disease_bundle",
            node_ids=node_ids,
            roles=roles,
            rule_trace=rule_trace,
        )
        support_id = f"sup:{hid}"

        supports.append(
            {
                "support_id": support_id,
                "sources": [{"name": "PrimeKG", "version": source_version}],
                "support_triplets": support_triplets,
                "rule_trace": rule_trace,
                "confidence": None,
            }
        )
        hyperedges.append(
            {
                "he_id": hid,
                "semantics": "protein_disease_bundle",
                "evidence_grade": "kg_composed",
                "polarity": "unknown",
                "node_ids": node_ids,
                "roles": roles,
                "support_id": support_id,
            }
        )

    return {"hyperedges": hyperedges, "support": supports}
