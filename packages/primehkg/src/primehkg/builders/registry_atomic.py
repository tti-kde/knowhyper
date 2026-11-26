from __future__ import annotations

from collections import defaultdict
from typing import Any

from primehkg.ids import he_id, normalize_node_id
from primehkg.policy.atomic_admissions import (
    AtomicAdmission,
    active_relations_for_admission,
)
from primehkg.policy.registry import (
    RelationEntry,
    load_relation_registry,
    normalize_relation,
)
from primehkg.policy.relation_canonicalization import (
    SYMMETRIC_RELATIONS,
    canonicalize_relation_fact,
)

# PrimeKG relation strings that are not valid ``he_id`` semantics slugs.
_ATOMIC_SEMANTICS_SLUG: dict[str, str] = {
    "off-label use": "off_label_use",
}


def _kg_atomic_semantics(relation: str) -> str:
    return _ATOMIC_SEMANTICS_SLUG.get(relation, relation)


def _registry_atomic_he_id_rule_trace(
    *, relation: str, admission_id: str, cap: int | None
) -> dict[str, Any]:
    """``he_id`` payload (legacy keys) — unchanged since pre-0.2.0 builder rename."""
    trace: dict[str, Any] = {
        "builder": "registry_atomic_wave",
        "version": 1,
        "kind": "atomic",
        "relation": relation,
        "wave_id": admission_id,
    }
    if cap is not None:
        trace["max_out_edges_per_anchor"] = int(cap)
    return trace


def _role_from_node_id(node_id: str) -> str:
    if not node_id.startswith("primekg:"):
        return "entity"
    body = node_id[len("primekg:") :]
    typ, _, _ = body.partition(":")
    return typ or "entity"


def _polarity_for_relation(relation: str) -> str:
    if relation in {
        "contraindication",
        "disease_phenotype_negative",
        "anatomy_protein_absent",
    }:
        return "negative"
    if relation in SYMMETRIC_RELATIONS:
        return "undirected"
    return "positive"


def _admitted_relations_by_admission(
    admissions: tuple[AtomicAdmission, ...], registry: dict[str, RelationEntry]
) -> dict[str, str]:
    """Map normalized relation string to owning ``admission_id`` (first wins on overlap)."""
    out: dict[str, str] = {}
    for admission in admissions:
        for rel in active_relations_for_admission(admission, registry):
            out.setdefault(rel, admission.admission_id)
    return out


def build_registry_atomic_hyperedges(
    triplets: list[dict[str, Any]],
    *,
    admissions: tuple[AtomicAdmission, ...],
    source_version: str = "unknown",
    waves: tuple[AtomicAdmission, ...] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Emit one ``kg_atomic`` hyperedge per triplet for registry-active relations."""
    if waves is not None:
        admissions = waves
    registry = load_relation_registry()
    admitted = _admitted_relations_by_admission(admissions, registry)
    if not admitted:
        return {"hyperedges": [], "support": []}

    admissions_by_id = {a.admission_id: a for a in admissions}
    anchor_edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    seen_triplets: set[tuple[str, str, str]] = set()

    hyperedges: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []

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
        rr = normalize_relation(registry, r)
        admission_id = admitted.get(rr)
        if admission_id is None:
            continue
        fact = canonicalize_relation_fact(rr, s, t)
        if fact is None:
            continue
        src, tgt = fact.source, fact.target
        triplet_key = fact.key
        if triplet_key in seen_triplets:
            continue
        seen_triplets.add(triplet_key)
        admission = admissions_by_id[admission_id]
        cap = admission.max_out_edges_per_anchor
        if cap is not None:
            key = (rr, src)
            if anchor_edge_counts[key] >= cap:
                continue
            anchor_edge_counts[key] += 1
        roles = {src: _role_from_node_id(src), tgt: _role_from_node_id(tgt)}
        support_id = f"sup:{len(supports) + 1}"
        rule_trace = {
            "builder": "registry_atomic",
            "version": 1,
            "kind": "atomic",
            "relation": rr,
            "admission_id": admission_id,
        }
        if cap is not None:
            rule_trace["max_out_edges_per_anchor"] = int(cap)
        supports.append(
            {
                "support_id": support_id,
                "sources": [{"name": "PrimeKG", "version": source_version}],
                "support_triplets": [{"s": src, "r": rr, "t": tgt}],
                "rule_trace": rule_trace,
                "confidence": None,
            }
        )
        semantics = _kg_atomic_semantics(rr)
        hid = he_id(
            semantics=semantics,
            node_ids=[src, tgt],
            roles=roles,
            rule_trace=_registry_atomic_he_id_rule_trace(
                relation=rr,
                admission_id=admission_id,
                cap=cap,
            ),
        )
        hyperedges.append(
            {
                "he_id": hid,
                "semantics": semantics,
                "evidence_grade": "kg_atomic",
                "polarity": _polarity_for_relation(rr),
                "node_ids": [src, tgt],
                "roles": roles,
                "support_id": support_id,
            }
        )

    return {"hyperedges": hyperedges, "support": supports}
