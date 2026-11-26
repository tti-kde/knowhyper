from __future__ import annotations

from collections import defaultdict
from typing import Any

from primehkg.ids import he_id, normalize_node_id
from primehkg.policy.registry import (
    PHENOTYPE_BUILDER_RELATIONS,
    load_relation_registry,
    normalize_relation,
)
from primehkg.policy.relation_canonicalization import (
    CanonicalRelationFact,
    canonicalize_relation_fact,
)

PHENOTYPE_BUILDER_VERSION = 2


def build_phenotype_hyperedges(
    triplets: list[dict[str, Any]],
    *,
    emit_composed: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    drug_to_pheno: dict[str, set[str]] = defaultdict(set)
    disease_to_pheno: dict[str, set[str]] = defaultdict(set)

    atomic_hyperedges: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []

    atomic_groups: dict[tuple[str, str, str], set[tuple[str, str, str]]] = defaultdict(
        set
    )
    canonical_facts: dict[tuple[str, str, str], CanonicalRelationFact] = {}

    def add_atomic_group(
        semantics: str,
        fact: CanonicalRelationFact,
        roles: dict[str, str],
        support_triplets: list[dict[str, str]],
    ) -> dict[str, Any]:
        support_id = f"sup:{len(supports) + 1}"
        rule_trace = {
            "builder": "phenotype",
            "version": PHENOTYPE_BUILDER_VERSION,
            "kind": "atomic",
            "relation": fact.relation,
            "canonicalization": "phenotype_owned_role_oriented",
        }
        supports.append(
            {
                "support_id": support_id,
                "sources": [{"name": "PrimeKG", "version": "unknown"}],
                "support_triplets": support_triplets,
                "rule_trace": rule_trace,
                "confidence": None,
            }
        )

        node_ids = [fact.source, fact.target]
        hid = he_id(
            semantics=semantics,
            node_ids=node_ids,
            roles=roles,
            rule_trace=rule_trace,
        )
        he = {
            "he_id": hid,
            "semantics": semantics,
            "evidence_grade": "kg_atomic",
            "polarity": "positive",
            "node_ids": node_ids,
            "roles": roles,
            "support_id": support_id,
        }
        atomic_hyperedges.append(he)
        return he

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

    registry = load_relation_registry()
    triplets_sorted = sorted(valid_triplets, key=lambda x: (x[1], x[0], x[2]))
    for s, r, t in triplets_sorted:
        rr = normalize_relation(registry, r)
        if rr not in PHENOTYPE_BUILDER_RELATIONS:
            continue
        ent = registry.get(rr)
        if ent is None or ent.status != "active":
            continue
        fact = canonicalize_relation_fact(rr, s, t)
        if fact is None:
            continue
        source_row = (s, rr, t)
        atomic_groups[fact.key].add(source_row)
        canonical_facts[fact.key] = fact
        if rr == "drug_effect":
            drug_to_pheno[fact.source].add(fact.target)
        elif rr == "disease_phenotype_positive":
            disease_to_pheno[fact.source].add(fact.target)

    for key in sorted(atomic_groups):
        fact = canonical_facts[key]
        source_triplets = [
            {"s": s, "r": r, "t": t}
            for s, r, t in sorted(
                atomic_groups[key], key=lambda row: (row[1], row[0], row[2])
            )
        ]
        if fact.relation == "drug_effect":
            add_atomic_group(
                "ae_drug_phenotype",
                fact,
                {fact.source: "drug", fact.target: "phenotype"},
                source_triplets,
            )
        elif fact.relation == "disease_phenotype_positive":
            add_atomic_group(
                "disease_phenotype_context",
                fact,
                {fact.source: "disease", fact.target: "phenotype"},
                source_triplets,
            )

    composed_hyperedges: list[dict[str, Any]] = []
    if emit_composed:
        for d in sorted(drug_to_pheno):
            phenos = drug_to_pheno[d]
            for s in sorted(disease_to_pheno):
                s_phenos = disease_to_pheno[s]
                for p in sorted(phenos.intersection(s_phenos)):
                    support_id = f"sup:{len(supports) + 1}"
                    rule_trace = {
                        "builder": "phenotype",
                        "version": 1,
                        "kind": "composed",
                        "logic": "shared_phenotype",
                    }
                    supports.append(
                        {
                            "support_id": support_id,
                            "sources": [{"name": "PrimeKG", "version": "unknown"}],
                            "support_triplets": [
                                {"s": d, "r": "drug_effect", "t": p},
                                {"s": s, "r": "disease_phenotype_positive", "t": p},
                            ],
                            "rule_trace": rule_trace,
                            "confidence": None,
                        }
                    )

                    roles = {d: "drug", s: "disease", p: "phenotype"}
                    hid = he_id(
                        semantics="drug_disease_phenotype",
                        node_ids=[d, s, p],
                        roles=roles,
                        rule_trace=rule_trace,
                    )
                    composed_hyperedges.append(
                        {
                            "he_id": hid,
                            "semantics": "drug_disease_phenotype",
                            "evidence_grade": "kg_composed",
                            "polarity": "unknown",
                            "node_ids": [d, s, p],
                            "roles": roles,
                            "support_id": support_id,
                        }
                    )

    return {"hyperedges": atomic_hyperedges + composed_hyperedges, "support": supports}
