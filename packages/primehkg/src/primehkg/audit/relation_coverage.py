from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Literal, TypedDict

from primehkg.ids import normalize_node_id
from primehkg.policy.atomic_admissions import (
    COVERAGE_REGISTRY_ADMISSIONS,
    active_relations_for_admission,
)
from primehkg.policy.registry import (
    PHENOTYPE_BUILDER_RELATIONS,
    RelationEntry,
    load_relation_registry,
    normalize_relation,
)
from primehkg.policy.relation_canonicalization import canonicalize_relation_fact

RelationOwner = Literal[
    "registry_atomic",
    "phenotype",
    "composed_support_only",
    "ignored_with_reason",
]


class RelationCoverageAuditRow(TypedDict):
    """Per-relation canonicalization and emission coverage metrics."""

    input_rows: int
    canonicalizable_rows: int
    canonical_facts: int
    reverse_duplicate_rows: int
    exact_duplicate_rows: int
    invalid_endpoint_rows: int
    unsupported_policy_rows: int
    orientation_policy: str
    owner: RelationOwner
    emitted_atomic_hyperedges: int
    used_by_composed_builders: list[str]
    notes: str


def _active_registry_atomic_relations(
    registry: dict[str, RelationEntry],
) -> frozenset[str]:
    out: set[str] = set()
    for admission in COVERAGE_REGISTRY_ADMISSIONS:
        out.update(active_relations_for_admission(admission, registry))
    return frozenset(out)


COMPOSED_BUILDER_RELATIONS: dict[str, tuple[str, ...]] = {
    "drug_protein": ("moa_triad", "drug_disease_shared_target_bundle"),
    "disease_protein": (
        "moa_triad",
        "drug_disease_shared_target_bundle",
        "disease_protein_bundle",
        "protein_disease_bundle",
    ),
    "indication": ("moa_triad", "drug_disease_shared_target_bundle"),
    "contraindication": ("moa_triad", "drug_disease_shared_target_bundle"),
    "protein_protein": ("protein_protein_bundle",),
    "drug_effect": ("phenotype",),
    "disease_phenotype_positive": ("phenotype",),
}

ATOMIC_SEMANTICS_BY_RELATION: dict[str, str] = {
    "drug_effect": "ae_drug_phenotype",
    "disease_phenotype_positive": "disease_phenotype_context",
    "off-label use": "off_label_use",
}


def _owner_for_relation(
    relation: str,
    registry: dict[str, RelationEntry],
) -> RelationOwner:
    if relation in PHENOTYPE_BUILDER_RELATIONS:
        return "phenotype"
    if relation in _active_registry_atomic_relations(registry):
        return "registry_atomic"
    if relation in COMPOSED_BUILDER_RELATIONS:
        return "composed_support_only"
    return "ignored_with_reason"


def _atomic_semantics_for_relation(relation: str) -> str:
    return ATOMIC_SEMANTICS_BY_RELATION.get(relation, relation)


def _atomic_counts_by_semantics(
    hyperedges: list[dict[str, Any]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for he in hyperedges:
        if not isinstance(he, dict):
            continue
        if he.get("evidence_grade") != "kg_atomic":
            continue
        semantics = he.get("semantics")
        if not isinstance(semantics, str):
            continue
        semantics = semantics.strip()
        if semantics:
            counts[semantics] += 1
    return counts


def _notes_for_relation(
    relation: str,
    registry: dict[str, RelationEntry],
    owner: RelationOwner,
) -> str:
    entry = registry.get(relation)
    if entry is not None:
        if entry.status == "defer" and entry.defer_reason is not None:
            return f"{entry.defer_reason}: {entry.notes}"
        return entry.notes
    if owner == "composed_support_only":
        return (
            "Consumed as support by composed builders; not emitted as registry atomic."
        )
    return "Relation is not covered by the active registry admissions or builders."


def build_relation_coverage_audit(
    triplets: list[dict[str, Any]],
    hyperedges: list[dict[str, Any]],
) -> dict[str, RelationCoverageAuditRow]:
    registry = load_relation_registry()
    atomic_counts = _atomic_counts_by_semantics(hyperedges)

    input_rows_by_relation: Counter[str] = Counter()
    canonicalizable_rows_by_relation: Counter[str] = Counter()
    reverse_duplicate_rows_by_relation: Counter[str] = Counter()
    exact_duplicate_rows_by_relation: Counter[str] = Counter()
    invalid_endpoint_rows_by_relation: Counter[str] = Counter()
    unsupported_policy_rows_by_relation: Counter[str] = Counter()
    canonical_keys_by_relation: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    raw_orientations_by_key: dict[tuple[str, str, str], set[tuple[str, str]]] = (
        defaultdict(set)
    )
    policies_by_relation: dict[str, set[str]] = defaultdict(set)

    for triplet in triplets:
        if not isinstance(triplet, dict):
            continue
        raw_relation = triplet.get("r")
        raw_source = triplet.get("s")
        raw_target = triplet.get("t")
        if not (
            isinstance(raw_relation, str)
            and isinstance(raw_source, str)
            and isinstance(raw_target, str)
        ):
            continue

        relation = normalize_relation(registry, raw_relation)
        source = normalize_node_id(raw_source)
        target = normalize_node_id(raw_target)
        if not (relation and source and target):
            continue

        input_rows_by_relation[relation] += 1
        fact = canonicalize_relation_fact(relation, source, target)
        if fact is None:
            invalid_endpoint_rows_by_relation[relation] += 1
            continue
        canonicalizable_rows_by_relation[relation] += 1
        if fact.orientation_policy == "unsupported":
            unsupported_policy_rows_by_relation[relation] += 1
        raw_orientation = (source, target)
        seen_orientations = raw_orientations_by_key[fact.key]
        if seen_orientations:
            if raw_orientation in seen_orientations:
                exact_duplicate_rows_by_relation[relation] += 1
            else:
                reverse_duplicate_rows_by_relation[relation] += 1
        seen_orientations.add(raw_orientation)
        canonical_keys_by_relation[relation].add(fact.key)
        policies_by_relation[relation].add(fact.orientation_policy)

    audit: dict[str, RelationCoverageAuditRow] = {}
    for relation in sorted(input_rows_by_relation):
        input_rows = int(input_rows_by_relation[relation])
        canonicalizable_rows = int(canonicalizable_rows_by_relation[relation])
        canonical_facts = len(canonical_keys_by_relation[relation])
        reverse_duplicate_rows = int(reverse_duplicate_rows_by_relation[relation])
        exact_duplicate_rows = int(exact_duplicate_rows_by_relation[relation])
        invalid_endpoint_rows = int(invalid_endpoint_rows_by_relation[relation])
        unsupported_policy_rows = int(unsupported_policy_rows_by_relation[relation])
        policies = policies_by_relation[relation]
        if len(policies) == 1:
            orientation_policy = next(iter(policies))
        elif len(policies) > 1:
            orientation_policy = "mixed"
        else:
            orientation_policy = "unsupported"

        owner = _owner_for_relation(relation, registry)
        audit[relation] = {
            "input_rows": input_rows,
            "canonicalizable_rows": canonicalizable_rows,
            "canonical_facts": canonical_facts,
            "reverse_duplicate_rows": reverse_duplicate_rows,
            "exact_duplicate_rows": exact_duplicate_rows,
            "invalid_endpoint_rows": invalid_endpoint_rows,
            "unsupported_policy_rows": unsupported_policy_rows,
            "orientation_policy": orientation_policy,
            "owner": owner,
            "emitted_atomic_hyperedges": int(
                atomic_counts[_atomic_semantics_for_relation(relation)]
            ),
            "used_by_composed_builders": sorted(
                COMPOSED_BUILDER_RELATIONS.get(relation, ())
            ),
            "notes": _notes_for_relation(relation, registry, owner),
        }
    return audit
