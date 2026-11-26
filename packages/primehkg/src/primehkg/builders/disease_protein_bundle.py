from __future__ import annotations

from collections import defaultdict
from typing import Any

from primehkg.ids import he_id, normalize_node_id


def build_disease_protein_bundle_hyperedges(
    triplets: list[dict[str, Any]],
    *,
    source_version: str = "unknown",
    max_proteins_per_disease: int = 25,
    min_proteins_per_bundle: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    disease_to_proteins: dict[str, set[str]] = defaultdict(set)
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

    # Deterministic processing regardless of input ordering.
    for s, r, t in sorted(valid, key=lambda x: (x[1], x[0], x[2])):
        if (
            r == "disease_protein"
            and s.startswith("primekg:disease:")
            and t.startswith("primekg:protein:")
        ):
            disease_to_proteins[s].add(t)
            disease_protein_edges[(s, t)].add((s, r, t))

    hyperedges: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []

    for disease in sorted(disease_to_proteins):
        proteins_sorted = sorted(disease_to_proteins[disease])
        if len(proteins_sorted) < int(min_proteins_per_bundle):
            continue
        keep = proteins_sorted[: max(0, int(max_proteins_per_disease))]
        if len(keep) < int(min_proteins_per_bundle):
            continue

        support_set: set[tuple[str, str, str]] = set()
        for p in keep:
            support_set.update(disease_protein_edges.get((disease, p), set()))

        support_triplets = [
            {"s": s, "r": r, "t": t}
            for s, r, t in sorted(support_set, key=lambda x: (x[1], x[0], x[2]))
        ]

        rule_trace = {
            "builder": "disease_protein_bundle",
            "version": 1,
            "kind": "composed",
            "join_keys": ["disease_protein"],
            "generation_policy": {
                "max_proteins_per_disease": int(max_proteins_per_disease),
                "min_proteins_per_bundle": int(min_proteins_per_bundle),
                "tie_break": "protein_id_lex",
            },
        }

        node_ids = [disease, *keep]
        roles = {disease: "disease", **dict.fromkeys(keep, "protein")}
        hid = he_id(
            semantics="disease_protein_bundle",
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
                "semantics": "disease_protein_bundle",
                "evidence_grade": "kg_composed",
                "polarity": "unknown",
                "node_ids": node_ids,
                "roles": roles,
                "support_id": support_id,
            }
        )

    return {"hyperedges": hyperedges, "support": supports}
