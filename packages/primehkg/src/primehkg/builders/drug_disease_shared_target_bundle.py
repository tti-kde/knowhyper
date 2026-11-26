from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, Literal

from primehkg.ids import he_id, normalize_node_id
from primehkg.policy.registry import load_relation_registry, normalize_relation

_DrugDiseasePolarity = Literal["positive", "negative", "unknown"]


def _pair_pol(
    *,
    drug: str,
    disease: str,
    indications: dict[tuple[str, str], set[tuple[str, str, str]]],
    contras: dict[tuple[str, str], set[tuple[str, str, str]]],
) -> _DrugDiseasePolarity:
    has_ind = bool(indications.get((drug, disease)))
    has_contra = bool(contras.get((drug, disease)))
    if has_ind and has_contra:
        return "unknown"
    if has_ind:
        return "positive"
    if has_contra:
        return "negative"
    return "unknown"


def build_drug_disease_shared_target_bundle_hyperedges(
    triplets: list[dict[str, Any]],
    *,
    source_version: str = "unknown",
    max_drugs_per_bundle: int = 2,
    max_eligible_drugs: int = 8,
    max_bundles_per_protein_disease: int = 32,
) -> dict[str, list[dict[str, Any]]]:
    """Compose bundles: multiple drugs share a protein target under a shared disease.

    Each emitted hyperedge has:
    - m >= 2 distinct drugs
    - each drug has `drug_protein` to the shared protein
    - each drug has `indication` and/or `contraindication` to the shared disease
    - if any drug has mixed indication+contraindication, that drug's pair polarity is unknown and it is excluded
    - if drugs disagree (some positive, some negative), bundle polarity is `unknown` (mixed evidence)
    """
    reg = load_relation_registry()

    drug_protein_edges: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(
        set
    )
    disease_protein_edges: dict[tuple[str, str], set[tuple[str, str, str]]] = (
        defaultdict(set)
    )
    indications: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
    contras: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)

    protein_to_drugs: dict[str, set[str]] = defaultdict(set)
    protein_to_diseases: dict[str, set[str]] = defaultdict(set)

    valid: list[tuple[str, str, str]] = []
    for tr in triplets:
        s = tr.get("s")
        r = tr.get("r")
        t = tr.get("t")
        if not (isinstance(s, str) and isinstance(r, str) and isinstance(t, str)):
            continue
        s = normalize_node_id(s)
        rr = normalize_relation(reg, r.strip())
        t = normalize_node_id(t)
        if not (s and rr and t):
            continue
        valid.append((s, rr, t))

    for s, r, t in sorted(valid, key=lambda x: (x[1], x[0], x[2])):
        if (
            r == "drug_protein"
            and s.startswith("primekg:drug:")
            and t.startswith("primekg:protein:")
        ):
            drug_protein_edges[(s, t)].add((s, r, t))
            protein_to_drugs[t].add(s)
            continue
        if (
            r == "disease_protein"
            and s.startswith("primekg:disease:")
            and t.startswith("primekg:protein:")
        ):
            disease_protein_edges[(s, t)].add((s, r, t))
            protein_to_diseases[t].add(s)
            continue
        if (
            r == "indication"
            and s.startswith("primekg:drug:")
            and t.startswith("primekg:disease:")
        ):
            indications[(s, t)].add((s, r, t))
            continue
        if (
            r == "contraindication"
            and s.startswith("primekg:drug:")
            and t.startswith("primekg:disease:")
        ):
            contras[(s, t)].add((s, r, t))
            continue

    hyperedges: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []

    for protein in sorted(set(protein_to_drugs) & set(protein_to_diseases)):
        drugs = sorted(protein_to_drugs[protein])
        diseases = sorted(protein_to_diseases[protein])
        if len(drugs) < 2 or not diseases:
            continue

        for disease in diseases:
            eligible = [
                d
                for d in drugs
                if (
                    _pair_pol(
                        drug=d,
                        disease=disease,
                        indications=indications,
                        contras=contras,
                    )
                    != "unknown"
                )
                and (d, protein) in drug_protein_edges
            ]
            eligible = sorted(set(eligible))[: max(0, int(max_eligible_drugs))]
            if len(eligible) < 2:
                continue

            max_r = min(len(eligible), int(max_drugs_per_bundle))
            if max_r < 2:
                continue

            bundles_emitted = 0
            bundle_cap = max(0, int(max_bundles_per_protein_disease))
            for r in range(2, max_r + 1):
                for combo in combinations(eligible, r):
                    if bundle_cap and bundles_emitted >= bundle_cap:
                        break
                    pols = {
                        _pair_pol(
                            drug=d,
                            disease=disease,
                            indications=indications,
                            contras=contras,
                        )
                        for d in combo
                    }
                    if pols == {"positive"}:
                        pol: _DrugDiseasePolarity = "positive"
                        pres = "all_positive"
                    elif pols == {"negative"}:
                        pol = "negative"
                        pres = "all_negative"
                    else:
                        pol = "unknown"
                        pres = "mixed_drug_disease_evidence"

                    support_triplets_set: set[tuple[str, str, str]] = set()
                    for d in combo:
                        support_triplets_set.update(drug_protein_edges[(d, protein)])
                        support_triplets_set.update(
                            indications.get((d, disease), set())
                        )
                        support_triplets_set.update(contras.get((d, disease), set()))
                    support_triplets_set.update(
                        disease_protein_edges[(disease, protein)]
                    )

                    support_triplets = [
                        {"s": s, "r": rr, "t": t}
                        for s, rr, t in sorted(
                            support_triplets_set, key=lambda x: (x[1], x[0], x[2])
                        )
                    ]

                    node_ids = [protein, disease, *sorted(combo)]
                    roles: dict[str, str] = {protein: "protein", disease: "disease"}
                    for d in combo:
                        roles[d] = "drug"

                    rule_trace: dict[str, Any] = {
                        "builder": "drug_disease_shared_target_bundle",
                        "version": 1,
                        "kind": "composed",
                        "join_logic": "shared_protein_across_drugs_with_shared_disease_context",
                        "polarity_resolution": pres,
                        "polarity": pol,
                        "max_drugs_per_bundle": int(max_drugs_per_bundle),
                        "max_eligible_drugs": int(max_eligible_drugs),
                        "max_bundles_per_protein_disease": int(
                            max_bundles_per_protein_disease
                        ),
                        "source_version": str(source_version),
                    }

                    hid = he_id(
                        semantics="drug_disease_shared_target_bundle",
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
                            "semantics": "drug_disease_shared_target_bundle",
                            "evidence_grade": "kg_composed",
                            "polarity": pol,
                            "node_ids": node_ids,
                            "roles": roles,
                            "support_id": support_id,
                        }
                    )
                    bundles_emitted += 1
                if bundle_cap and bundles_emitted >= bundle_cap:
                    break

    return {"hyperedges": hyperedges, "support": supports}
