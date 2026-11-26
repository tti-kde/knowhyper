from __future__ import annotations

from collections import Counter
from typing import Any

_DRUG_DISEASE_SHARED_TARGET = "drug_disease_shared_target_bundle"
_DISEASE_PROTEIN_BUNDLE = "disease_protein_bundle"
_PROTEIN_DISEASE_BUNDLE = "protein_disease_bundle"
_CLINICAL_RELATIONS = frozenset({"indication", "contraindication"})
_DISEASE_PROTEIN_RELATION = "disease_protein"


def _roles_map(he: dict[str, Any]) -> dict[str, str]:
    roles_obj = he.get("roles")
    if not isinstance(roles_obj, dict):
        return {}
    return {str(k): str(v) for k, v in roles_obj.items() if isinstance(k, str) and v}


def _drug_count_histogram(counts: Counter[int]) -> dict[str, int]:
    return {str(m): int(counts[m]) for m in sorted(counts)}


def _low_incidence_coverage(
    incidence: Counter[str],
    involved_flags: Counter[str],
    *,
    quantile: float,
) -> dict[str, Any]:
    if not incidence:
        return {
            "quantile": quantile,
            "low_entity_count": 0,
            "hyperedge_count": 0,
            "low_entity_hyperedge_count": 0,
            "coverage": 0.0,
        }
    items = sorted(incidence.items(), key=lambda kv: (int(kv[1]), str(kv[0])))
    cutoff_idx = max(1, int(len(items) * quantile))
    low_set = {k for k, _ in items[:cutoff_idx]}
    total = sum(involved_flags.values())
    low_involved = sum(1 for flag in involved_flags.values() if flag)
    return {
        "quantile": quantile,
        "low_entity_count": len(low_set),
        "hyperedge_count": total,
        "low_entity_hyperedge_count": low_involved,
        "coverage": float(low_involved) / float(total) if total else 0.0,
    }


def _drug_support_complete(
    *,
    drug: str,
    protein: str,
    disease: str,
    triplets: list[dict[str, Any]],
) -> bool:
    has_target = False
    has_clinical = False
    for tr in triplets:
        if not isinstance(tr, dict):
            continue
        s = tr.get("s")
        r = tr.get("r")
        t = tr.get("t")
        if not (isinstance(s, str) and isinstance(r, str) and isinstance(t, str)):
            continue
        if s == drug and t == protein and r == "drug_protein":
            has_target = True
        if s == drug and t == disease and r in _CLINICAL_RELATIONS:
            has_clinical = True
    return has_target and has_clinical


def _generation_policy_cap(rule_trace: dict[str, Any], cap_key: str) -> int | None:
    policy = rule_trace.get("generation_policy")
    if not isinstance(policy, dict):
        return None
    cap = policy.get(cap_key)
    return int(cap) if isinstance(cap, int) else None


def _member_has_disease_protein_edge(
    *,
    disease: str,
    protein: str,
    triplets: list[dict[str, Any]],
) -> bool:
    for tr in triplets:
        if not isinstance(tr, dict):
            continue
        s = tr.get("s")
        r = tr.get("r")
        t = tr.get("t")
        if (
            isinstance(s, str)
            and isinstance(r, str)
            and isinstance(t, str)
            and s == disease
            and t == protein
            and r == _DISEASE_PROTEIN_RELATION
        ):
            return True
    return False


def _star_bundle_family_metrics(
    hyperedges: list[dict[str, Any]],
    *,
    semantics: str,
    anchor_role: str,
    member_role: str,
    support_triplets_by_id: dict[str, list[dict[str, Any]]],
    rule_trace_by_id: dict[str, dict[str, Any]],
    cap_policy_key: str,
    member_count_field: str,
    hub_nodes_field: str,
    long_tail_member_field: str,
    long_tail_anchor_field: str,
    long_tail_quantile: float = 0.2,
    top_k: int = 10,
) -> dict[str, Any]:
    """Metrics for star-shaped coverage bundles (one anchor + many members)."""
    hes = [
        he
        for he in hyperedges
        if isinstance(he, dict) and he.get("semantics") == semantics
    ]
    if not hes:
        return {"hyperedge_count": 0}

    member_counts: Counter[int] = Counter()
    hub_participation: Counter[str] = Counter()
    anchor_incidence: Counter[str] = Counter()
    member_incidence: Counter[str] = Counter()
    bundles_at_cap = 0
    complete_bundles = 0
    bundle_involves_low_member: Counter[str] = Counter()
    bundle_involves_low_anchor: Counter[str] = Counter()

    for he in hes:
        roles = _roles_map(he)
        anchor = next((nid for nid, role in roles.items() if role == anchor_role), "")
        members = sorted(nid for nid, role in roles.items() if role == member_role)

        member_counts[len(members)] += 1
        member_incidence.update(members)
        if anchor:
            anchor_incidence[anchor] += 1
        hub_participation.update(members)

        sid = he.get("support_id")
        trips: list[dict[str, Any]] = []
        rt: dict[str, Any] = {}
        if isinstance(sid, str) and sid:
            trips = support_triplets_by_id.get(sid, [])
            rt = rule_trace_by_id.get(sid, {})

        cap = _generation_policy_cap(rt, cap_policy_key)
        if cap is not None and len(members) == cap:
            bundles_at_cap += 1

        if anchor_role == "disease" and member_role == "protein" and anchor and members:
            all_complete = all(
                _member_has_disease_protein_edge(
                    disease=anchor, protein=p, triplets=trips
                )
                for p in members
            )
        elif (
            anchor_role == "protein" and member_role == "disease" and anchor and members
        ):
            all_complete = all(
                _member_has_disease_protein_edge(
                    disease=d, protein=anchor, triplets=trips
                )
                for d in members
            )
        else:
            all_complete = False

        if all_complete:
            complete_bundles += 1

    lq = long_tail_quantile if 0.0 < long_tail_quantile < 1.0 else 0.2
    low_members = (
        {
            k
            for k, _ in sorted(
                member_incidence.items(), key=lambda kv: (int(kv[1]), kv[0])
            )[: max(1, int(len(member_incidence) * lq))]
        }
        if member_incidence
        else set()
    )
    low_anchors = (
        {
            k
            for k, _ in sorted(
                anchor_incidence.items(), key=lambda kv: (int(kv[1]), kv[0])
            )[: max(1, int(len(anchor_incidence) * lq))]
        }
        if anchor_incidence
        else set()
    )

    for he in hes:
        roles = _roles_map(he)
        anchor = next((nid for nid, role in roles.items() if role == anchor_role), "")
        members = [nid for nid, role in roles.items() if role == member_role]
        he_id = str(he.get("he_id", ""))
        bundle_involves_low_member[he_id] = any(m in low_members for m in members)
        bundle_involves_low_anchor[he_id] = bool(anchor and anchor in low_anchors)

    total = len(hes)
    top_hub = sorted(
        hub_participation.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))
    )[:top_k]

    return {
        "hyperedge_count": total,
        member_count_field: {
            "histogram": _drug_count_histogram(member_counts),
            "min": min(member_counts) if member_counts else None,
            "max": max(member_counts) if member_counts else None,
        },
        hub_nodes_field: {
            "top_k": top_k,
            "top_nodes": [[nid, int(c)] for nid, c in top_hub],
        },
        "bundles_at_generation_cap": {
            "count": bundles_at_cap,
            "fraction": float(bundles_at_cap) / float(total) if total else 0.0,
            "cap_policy_key": cap_policy_key,
        },
        "support_completeness": {
            "complete": complete_bundles,
            "total": total,
            "fraction": float(complete_bundles) / float(total) if total else 0.0,
        },
        long_tail_member_field: _low_incidence_coverage(
            member_incidence, bundle_involves_low_member, quantile=lq
        ),
        long_tail_anchor_field: _low_incidence_coverage(
            anchor_incidence, bundle_involves_low_anchor, quantile=lq
        ),
    }


def disease_protein_bundle_metrics(
    hyperedges: list[dict[str, Any]],
    *,
    support_triplets_by_id: dict[str, list[dict[str, Any]]],
    rule_trace_by_id: dict[str, dict[str, Any]],
    long_tail_quantile: float = 0.2,
    top_k: int = 10,
) -> dict[str, Any]:
    return _star_bundle_family_metrics(
        hyperedges,
        semantics=_DISEASE_PROTEIN_BUNDLE,
        anchor_role="disease",
        member_role="protein",
        support_triplets_by_id=support_triplets_by_id,
        rule_trace_by_id=rule_trace_by_id,
        cap_policy_key="max_proteins_per_disease",
        member_count_field="protein_count_per_bundle",
        hub_nodes_field="top_proteins_by_bundle_participation",
        long_tail_member_field="long_tail_protein_coverage",
        long_tail_anchor_field="long_tail_disease_coverage",
        long_tail_quantile=long_tail_quantile,
        top_k=top_k,
    )


def protein_disease_bundle_metrics(
    hyperedges: list[dict[str, Any]],
    *,
    support_triplets_by_id: dict[str, list[dict[str, Any]]],
    rule_trace_by_id: dict[str, dict[str, Any]],
    long_tail_quantile: float = 0.2,
    top_k: int = 10,
) -> dict[str, Any]:
    return _star_bundle_family_metrics(
        hyperedges,
        semantics=_PROTEIN_DISEASE_BUNDLE,
        anchor_role="protein",
        member_role="disease",
        support_triplets_by_id=support_triplets_by_id,
        rule_trace_by_id=rule_trace_by_id,
        cap_policy_key="max_diseases_per_protein",
        member_count_field="disease_count_per_bundle",
        hub_nodes_field="top_diseases_by_bundle_participation",
        long_tail_member_field="long_tail_disease_coverage",
        long_tail_anchor_field="long_tail_protein_coverage",
        long_tail_quantile=long_tail_quantile,
        top_k=top_k,
    )


def drug_disease_shared_target_bundle_metrics(
    hyperedges: list[dict[str, Any]],
    *,
    support_triplets_by_id: dict[str, list[dict[str, Any]]],
    rule_trace_by_id: dict[str, dict[str, Any]],
    long_tail_quantile: float = 0.2,
    top_k: int = 10,
) -> dict[str, Any]:
    """Family-specific audit metrics for ``drug_disease_shared_target_bundle`` hyperedges."""
    hes = [
        he
        for he in hyperedges
        if isinstance(he, dict) and he.get("semantics") == _DRUG_DISEASE_SHARED_TARGET
    ]
    if not hes:
        return {"hyperedge_count": 0}

    drug_counts: Counter[int] = Counter()
    protein_counts: Counter[str] = Counter()
    polarity_resolution: Counter[str] = Counter()
    bundles_at_cap = 0
    complete_bundles = 0
    drug_incidence: Counter[str] = Counter()
    disease_incidence: Counter[str] = Counter()
    bundle_involves_low_drug: Counter[str] = Counter()
    bundle_involves_low_disease: Counter[str] = Counter()

    for he in hes:
        roles = _roles_map(he)
        drugs = sorted(nid for nid, role in roles.items() if role == "drug")
        protein = next((nid for nid, role in roles.items() if role == "protein"), "")
        disease = next((nid for nid, role in roles.items() if role == "disease"), "")

        drug_counts[len(drugs)] += 1
        if protein:
            protein_counts[protein] += 1
        if disease:
            disease_incidence[disease] += 1
        drug_incidence.update(drugs)

        sid = he.get("support_id")
        trips: list[dict[str, Any]] = []
        rt: dict[str, Any] = {}
        if isinstance(sid, str) and sid:
            trips = support_triplets_by_id.get(sid, [])
            rt = rule_trace_by_id.get(sid, {})

        pres = rt.get("polarity_resolution")
        if isinstance(pres, str) and pres:
            polarity_resolution[pres] += 1

        max_drugs = rt.get("max_drugs_per_bundle")
        if isinstance(max_drugs, int) and len(drugs) == max_drugs:
            bundles_at_cap += 1

        if protein and disease and drugs:
            all_complete = all(
                _drug_support_complete(
                    drug=d, protein=protein, disease=disease, triplets=trips
                )
                for d in drugs
            )
            if all_complete:
                complete_bundles += 1

    lq = long_tail_quantile if 0.0 < long_tail_quantile < 1.0 else 0.2
    low_drugs = (
        {
            k
            for k, _ in sorted(
                drug_incidence.items(), key=lambda kv: (int(kv[1]), kv[0])
            )[: max(1, int(len(drug_incidence) * lq))]
        }
        if drug_incidence
        else set()
    )
    low_diseases = (
        {
            k
            for k, _ in sorted(
                disease_incidence.items(), key=lambda kv: (int(kv[1]), kv[0])
            )[: max(1, int(len(disease_incidence) * lq))]
        }
        if disease_incidence
        else set()
    )

    for he in hes:
        roles = _roles_map(he)
        drugs = [nid for nid, role in roles.items() if role == "drug"]
        disease = next((nid for nid, role in roles.items() if role == "disease"), "")
        he_id = str(he.get("he_id", ""))
        bundle_involves_low_drug[he_id] = any(d in low_drugs for d in drugs)
        bundle_involves_low_disease[he_id] = bool(disease and disease in low_diseases)

    total = len(hes)
    top_proteins = sorted(
        protein_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))
    )[:top_k]

    return {
        "hyperedge_count": total,
        "drug_count_per_bundle": {
            "histogram": _drug_count_histogram(drug_counts),
            "min": min(drug_counts) if drug_counts else None,
            "max": max(drug_counts) if drug_counts else None,
        },
        "top_proteins_by_bundle_count": {
            "top_k": top_k,
            "top_nodes": [[pid, int(c)] for pid, c in top_proteins],
        },
        "polarity_resolution_counts": {
            k: int(polarity_resolution[k]) for k in sorted(polarity_resolution)
        },
        "bundles_at_max_drug_cap": {
            "count": bundles_at_cap,
            "fraction": float(bundles_at_cap) / float(total) if total else 0.0,
        },
        "support_completeness": {
            "complete": complete_bundles,
            "total": total,
            "fraction": float(complete_bundles) / float(total) if total else 0.0,
        },
        "long_tail_drug_coverage": _low_incidence_coverage(
            drug_incidence, bundle_involves_low_drug, quantile=lq
        ),
        "long_tail_disease_coverage": _low_incidence_coverage(
            disease_incidence, bundle_involves_low_disease, quantile=lq
        ),
    }
