from __future__ import annotations

from collections import defaultdict
from typing import Any

from primehkg.builders.drug_disease_polarity import (
    ingest_drug_disease_clinical_edge,
    moa_triad_rule_trace,
    resolve_drug_disease_polarity,
)
from primehkg.ids import he_id, normalize_node_id
from primehkg.policy.registry import load_relation_registry, normalize_relation


def build_moa_triad_hyperedges(
    triplets: list[dict[str, Any]],
    *,
    source_version: str = "unknown",
) -> dict[str, list[dict[str, Any]]]:
    """Compose MoA triads (drug, protein, disease) from KG binary edges.

    Composition rules (v0):
    - `drug_protein`: drug targets protein (s=drug, t=protein)
    - `disease_protein`: disease associated with protein (s=disease, t=protein)
    - `indication` and/or `contraindication`: drug treats / contraindicated for disease

    Output:
    - hyperedges with `semantics="moa_triad"`, roles {drug, protein, disease}
    - evidence_grade: kg_composed
    - polarity: shared-target aligned (mixed clinical evidence -> unknown)
    - support table entries with `support_triplets` containing contributing edges
    """
    protein_to_drugs: dict[str, set[str]] = defaultdict(set)
    protein_to_diseases: dict[str, set[str]] = defaultdict(set)

    drug_disease_indications: dict[tuple[str, str], set[tuple[str, str, str]]] = (
        defaultdict(set)
    )
    drug_disease_contraindications: dict[tuple[str, str], set[tuple[str, str, str]]] = (
        defaultdict(set)
    )

    drug_protein_edges: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(
        set
    )
    disease_protein_edges: dict[tuple[str, str], set[tuple[str, str, str]]] = (
        defaultdict(set)
    )

    reg = load_relation_registry()

    valid_triplets: list[tuple[str, str, str]] = []
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
        valid_triplets.append((s, r, t))

    # Deterministic processing regardless of input ordering.
    for s, r, t in sorted(valid_triplets, key=lambda x: (x[1], x[0], x[2])):
        rr = normalize_relation(reg, r)
        if rr == "drug_protein":
            protein_to_drugs[t].add(s)
            drug_protein_edges[(s, t)].add((s, r, t))
            continue
        if rr == "disease_protein":
            protein_to_diseases[t].add(s)
            disease_protein_edges[(s, t)].add((s, r, t))
            continue
        if rr in ("indication", "contraindication"):
            ingest_drug_disease_clinical_edge(
                source=s,
                target=t,
                triplet=(s, rr, t),
                indications=drug_disease_indications,
                contraindications=drug_disease_contraindications,
            )
            continue

    hyperedges: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []

    for protein in sorted(set(protein_to_drugs) | set(protein_to_diseases)):
        drugs = sorted(protein_to_drugs.get(protein, set()))
        diseases = sorted(protein_to_diseases.get(protein, set()))
        if not (drugs and diseases):
            continue

        for drug in drugs:
            for disease in diseases:
                # Support attachment: include contributing edges deterministically.
                support_triplets_set: set[tuple[str, str, str]] = set()
                dp = drug_protein_edges.get((drug, protein))
                if not dp:
                    msg = (
                        "internal error: missing drug_protein evidence for composed moa_triad "
                        f"(drug={drug!r}, protein={protein!r})"
                    )
                    raise RuntimeError(msg)
                support_triplets_set.update(dp)

                dpr = disease_protein_edges.get((disease, protein))
                if not dpr:
                    msg = (
                        "internal error: missing disease_protein evidence for composed moa_triad "
                        f"(disease={disease!r}, protein={protein!r})"
                    )
                    raise RuntimeError(msg)
                support_triplets_set.update(dpr)
                support_triplets_set.update(
                    drug_disease_indications.get((drug, disease), set())
                )
                support_triplets_set.update(
                    drug_disease_contraindications.get((drug, disease), set())
                )

                support_triplets = [
                    {"s": s, "r": r, "t": t}
                    for s, r, t in sorted(
                        support_triplets_set, key=lambda x: (x[1], x[0], x[2])
                    )
                ]

                pol, resolution = resolve_drug_disease_polarity(
                    drug=drug,
                    disease=disease,
                    indications=drug_disease_indications,
                    contraindications=drug_disease_contraindications,
                )
                rule_trace = moa_triad_rule_trace(polarity=pol, resolution=resolution)

                roles = {drug: "drug", protein: "protein", disease: "disease"}
                hid = he_id(
                    semantics="moa_triad",
                    node_ids=[drug, protein, disease],
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
                        "semantics": "moa_triad",
                        "evidence_grade": "kg_composed",
                        "polarity": pol,
                        "node_ids": [drug, protein, disease],
                        "roles": roles,
                        "support_id": support_id,
                    }
                )

    return {"hyperedges": hyperedges, "support": supports}
