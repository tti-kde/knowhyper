from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from primehkg.audit.bundle_family import (
    disease_protein_bundle_metrics,
    drug_disease_shared_target_bundle_metrics,
    protein_disease_bundle_metrics,
)
from primehkg.audit.hub_risk_thresholds import (
    default_hub_risk_threshold,
    hub_risk_threshold_for_semantics,
    hub_risk_thresholds_by_semantics,
)
from primehkg.nodes.normalize import infer_node_type_and_name
from primehkg.nodes.synonyms import is_genuine_synonym


def _hyperedge_semantics(he: dict[str, Any]) -> str:
    sem = he.get("semantics")
    if not isinstance(sem, str):
        return "unknown"
    sem = sem.strip()
    return sem or "unknown"


def _hyperedge_node_ids(he: dict[str, Any]) -> list[str]:
    node_ids_obj = he.get("node_ids")
    if not isinstance(node_ids_obj, list):
        return []
    return [n for n in node_ids_obj if isinstance(n, str)]


def _unique_node_ids_per_hyperedge(he: dict[str, Any]) -> list[str]:
    node_ids_obj = he.get("node_ids")
    if not isinstance(node_ids_obj, list):
        return []
    return sorted({n for n in node_ids_obj if isinstance(n, str) and n})


def _node_ids_in_hyperedges(hyperedges: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for he in hyperedges:
        if not isinstance(he, dict):
            continue
        node_ids = he.get("node_ids")
        if not isinstance(node_ids, list):
            continue
        for nid in node_ids:
            if isinstance(nid, str) and nid:
                out.add(nid)
    return out


def _compute_node_connectivity(
    nodes: list[dict[str, Any]] | None,
    hyperedges: list[dict[str, Any]],
) -> dict[str, float | int] | None:
    """Compare ``nodes`` table universe to hyperedge participation (C-layer KPI)."""
    from primehkg.audit.curated import node_id_set_from_nodes

    table = node_id_set_from_nodes(nodes)
    if not table:
        return None
    in_he = _node_ids_in_hyperedges(hyperedges)
    participating = table & in_he
    isolated = table - in_he
    n_table = len(table)
    return {
        "nodes_table": n_table,
        "nodes_in_hyperedges": len(in_he),
        "nodes_he_isolated": len(isolated),
        "endpoint_he_coverage": float(len(participating)) / float(n_table)
        if n_table
        else 0.0,
    }


def _mapping_nonempty(value: object) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    for v in value.values():
        if isinstance(v, list):
            if any(isinstance(x, str) and x.strip() for x in v):
                return True
            if any(isinstance(x, dict) and x for x in v):
                return True
            continue
        if isinstance(v, str) and v.strip():
            return True
        if isinstance(v, dict) and v:
            return True
    return False


def _compute_node_field_completeness(
    nodes: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Fraction of ``nodes`` rows with enrichment-populated fields (T1 / #199)."""
    if not nodes:
        return None

    rows = [
        n for n in nodes if isinstance(n, dict) and isinstance(n.get("node_id"), str)
    ]
    n_table = len(rows)
    if n_table == 0:
        return None

    counts = {
        "with_nonempty_xrefs": 0,
        "with_nonempty_synonyms": 0,
        "with_nonempty_feature_refs": 0,
        "with_nonempty_source_meta": 0,
        "with_enriched_display_name": 0,
    }
    for row in rows:
        node_id = str(row["node_id"])
        _, fallback_name = infer_node_type_and_name(node_id)
        name = row.get("name")
        if isinstance(name, str) and name.strip() and name.strip() != fallback_name:
            counts["with_enriched_display_name"] += 1
        syns = row.get("synonyms")
        if isinstance(syns, list) and any(
            is_genuine_synonym(s, name=name, node_id=node_id) for s in syns
        ):
            counts["with_nonempty_synonyms"] += 1
        if _mapping_nonempty(row.get("xrefs")):
            counts["with_nonempty_xrefs"] += 1
        if _mapping_nonempty(row.get("feature_refs")):
            counts["with_nonempty_feature_refs"] += 1
        if _mapping_nonempty(row.get("source_meta")):
            counts["with_nonempty_source_meta"] += 1

    fractions = {
        k: float(v) / float(n_table) if n_table else 0.0 for k, v in counts.items()
    }
    return {
        "nodes_table": n_table,
        "counts": counts,
        "fractions": fractions,
    }


def count_metabolite_unresolved_names(nodes: list[dict[str, Any]]) -> int:
    """Count metabolite nodes whose display name is still the bare ChEBI number.

    A metabolite is "unresolved" when its name equals the ChEBI local id
    (``chebi:<local>`` -> name ``<local>``), i.e. no dictionary name was applied.
    """
    n = 0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("node_type") or "") != "metabolite":
            continue
        node_id = str(node.get("node_id") or "")
        name = str(node.get("name") or "")
        local = node_id.rsplit(":", 1)[-1] if ":" in node_id else node_id
        if name == local:
            n += 1
    return n


def _compute_general_stats_for_hyperedges(
    hyperedges: list[dict[str, Any]],
) -> dict[str, float | int]:
    he_nodes: list[list[str]] = [
        _unique_node_ids_per_hyperedge(he) for he in hyperedges
    ]
    he_nodes = [ns for ns in he_nodes if ns]

    num_hyperedges = len(he_nodes)
    all_nodes: set[str] = set()
    node_degree: Counter[str] = Counter()
    sizes: list[int] = []
    num_connections = 0

    for ns in he_nodes:
        sizes.append(len(ns))
        num_connections += len(ns)
        all_nodes.update(ns)
        node_degree.update(ns)

    num_nodes = len(all_nodes)
    if num_hyperedges > 0:
        avg_hyperedge_size = num_connections / num_hyperedges
        min_hyperedge_size = min(sizes)
        max_hyperedge_size = max(sizes)
    else:
        avg_hyperedge_size = 0.0
        min_hyperedge_size = 0
        max_hyperedge_size = 0

    if num_nodes > 0:
        avg_node_degree = num_connections / num_nodes
        min_node_degree = min(node_degree.values()) if node_degree else 0
        max_node_degree = max(node_degree.values()) if node_degree else 0
    else:
        avg_node_degree = 0.0
        min_node_degree = 0
        max_node_degree = 0

    denom = num_nodes * num_hyperedges
    density = (num_connections / denom) if denom > 0 else 0.0

    return {
        "num_hyperedges": num_hyperedges,
        "num_nodes_in_hyperedges": num_nodes,
        "num_connections": num_connections,
        "avg_hyperedge_size": float(avg_hyperedge_size),
        "min_hyperedge_size": int(min_hyperedge_size),
        "max_hyperedge_size": int(max_hyperedge_size),
        "avg_node_degree": float(avg_node_degree),
        "min_node_degree": int(min_node_degree),
        "max_node_degree": int(max_node_degree),
        "density": float(density),
    }


def _hyperedge_polarity(he: dict[str, Any]) -> str:
    pol = he.get("polarity")
    return pol if isinstance(pol, str) else ""


def _hyperedge_evidence_grade(he: dict[str, Any]) -> str:
    ev = he.get("evidence_grade")
    return ev if isinstance(ev, str) else ""


def _normalize_evidence_grade_label(he: dict[str, Any]) -> str:
    ev = _hyperedge_evidence_grade(he).strip()
    return ev or "unknown"


def _normalize_polarity_label(he: dict[str, Any]) -> str:
    pol = _hyperedge_polarity(he).strip().lower()
    return pol or "unknown"


def _summarize_label_counts(labels: list[str]) -> dict[str, Any]:
    """Global counts and fractions for a categorical hyperedge field."""
    counts: Counter[str] = Counter(labels)
    total = sum(int(v) for v in counts.values())
    keys = sorted(counts)
    return {
        "total": total,
        "counts": {k: int(counts[k]) for k in keys},
        "fractions": {
            k: (float(counts[k]) / float(total) if total else 0.0) for k in keys
        },
    }


def _compute_hyperedge_classification_summary(
    hyperedges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Cross-semantics evidence_grade and polarity KPIs for ``build_report.json``."""
    report_hes = [he for he in hyperedges if isinstance(he, dict)]
    return {
        "evidence_grade": _summarize_label_counts(
            [_normalize_evidence_grade_label(he) for he in report_hes]
        ),
        "polarity": _summarize_label_counts(
            [_normalize_polarity_label(he) for he in report_hes]
        ),
    }


def _size_histogram(sizes: list[int]) -> dict[str, int]:
    bins = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8+": 0}
    for s in sizes:
        if s <= 0:
            continue
        if s >= 8:
            bins["8+"] += 1
        else:
            bins[str(s)] += 1
    return bins


def _empty_size_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "min": None,
        "max": None,
        "histogram": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8+": 0},
    }


def _update_size_stats(stats: dict[str, Any], size: int) -> None:
    if size <= 0:
        return
    stats["count"] = int(stats["count"]) + 1
    stats["min"] = size if stats["min"] is None else min(int(stats["min"]), size)
    stats["max"] = size if stats["max"] is None else max(int(stats["max"]), size)
    hist: dict[str, int] = stats["histogram"]
    if size >= 8:
        hist["8+"] += 1
    else:
        hist[str(size)] += 1


def _summarize_builder_segments(
    segments: list[tuple[str, dict[str, list[dict[str, Any]]]]],
) -> dict[str, Any]:
    """Per merge-segment slug: row counts, semantics histogram, compact general stats."""
    out: dict[str, Any] = {}
    for slug, part in segments:
        hes = [h for h in part.get("hyperedges", []) if isinstance(h, dict)]
        sups = [s for s in part.get("support", []) if isinstance(s, dict)]
        sem_counts: Counter[str] = Counter(_hyperedge_semantics(h) for h in hes)
        out[slug] = {
            "hyperedge_count": len(hes),
            "support_count": len(sups),
            "semantics_counts": {k: int(sem_counts[k]) for k in sorted(sem_counts)},
            "general_stats": _compute_general_stats_for_hyperedges(hes),
        }
    return out


def build_audit_report(
    hyperedges: list[dict[str, Any]],
    *,
    supports: list[dict[str, Any]] | None = None,
    nodes: list[dict[str, Any]] | None = None,
    corum_tsv: str | None = None,
    uniprot_to_geneid: dict[str, str] | None = None,
    long_tail_quantile: float = 0.2,
    builder_segments: list[tuple[str, dict[str, list[dict[str, Any]]]]] | None = None,
    relation_coverage_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic JSON-serializable audit report from hyperedges.

    Notes:
    - `long_tail_quantile` must be in (0, 1); otherwise it falls back to `0.2` (silent clamp).
    - If `supports` contains duplicate `support_id` values, their `support_triplets` lists are merged
      in encounter order.
    """
    # Collect sizes and incidence counts per semantics.
    size_stats_by_sem: dict[str, dict[str, Any]] = defaultdict(_empty_size_stats)
    incidence_by_sem: dict[str, Counter[str]] = defaultdict(Counter)
    he_incidence_by_sem: dict[str, Counter[str]] = defaultdict(Counter)
    he_count_by_sem: Counter[str] = Counter()

    composed_counts: dict[str, Counter[str]] = defaultdict(Counter)
    composed_relation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    support_triplets_by_id: dict[str, list[dict[str, Any]]] = {}
    rule_trace_by_id: dict[str, dict[str, Any]] = {}

    if supports is not None:
        for s in supports:
            if not isinstance(s, dict):
                continue
            sid = s.get("support_id")
            if not isinstance(sid, str) or not sid:
                continue
            trips = s.get("support_triplets")
            if isinstance(trips, list):
                filtered = [t for t in trips if isinstance(t, dict)]
                if sid in support_triplets_by_id:
                    support_triplets_by_id[sid].extend(filtered)
                else:
                    support_triplets_by_id[sid] = list(filtered)
            rt = s.get("rule_trace")
            if isinstance(rt, dict):
                rule_trace_by_id[sid] = rt

    for he in hyperedges:
        if not isinstance(he, dict):
            continue
        sem = _hyperedge_semantics(he)
        node_ids = _hyperedge_node_ids(he)
        size = len(node_ids)

        he_count_by_sem[sem] += 1
        _update_size_stats(size_stats_by_sem[sem], size)

        # Node participation frequency: count unique nodes per hyperedge (duplicate ids in a single
        # hyperedge do not inflate participation).
        incidence_by_sem[sem].update(sorted(set(node_ids)))
        # Hub ratios are defined per-hyperedge: count unique-node participation per hyperedge.
        he_incidence_by_sem[sem].update(sorted(set(node_ids)))

        if _hyperedge_evidence_grade(he) == "kg_composed":
            pol = _hyperedge_polarity(he) or "unknown"
            composed_counts[sem][pol] += 1
            sid = he.get("support_id")
            if isinstance(sid, str) and sid in support_triplets_by_id:
                for tr in support_triplets_by_id[sid]:
                    r = tr.get("r")
                    if isinstance(r, str) and r:
                        composed_relation_counts[sem][r] += 1

    # Hyperedge size distribution per semantics.
    size_dist: dict[str, Any] = {
        sem: size_stats_by_sem[sem] for sem in sorted(size_stats_by_sem)
    }

    # Top-k node participation per semantics (incidence count).
    top_k = 10
    participation: dict[str, Any] = {}
    for sem in sorted(incidence_by_sem):
        cnt = incidence_by_sem[sem]
        top = sorted(cnt.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:top_k]
        participation[sem] = {
            "top_k": top_k,
            "top_nodes": [[k, int(v)] for k, v in top],
        }

    # Hub risk: top-1 node accounts for > per-semantics threshold of hyperedges.
    default_threshold = default_hub_risk_threshold()
    thresholds_by_semantics = hub_risk_thresholds_by_semantics()
    hub_flagged: dict[str, Any] = {}
    for sem in sorted(he_count_by_sem):
        total = int(he_count_by_sem[sem])
        if total <= 0:
            continue
        cnt = he_incidence_by_sem.get(sem, Counter())
        if not cnt:
            continue
        top_node, top_count = min(cnt.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        ratio = float(top_count) / float(total)
        threshold_used = hub_risk_threshold_for_semantics(sem)
        if ratio > threshold_used:
            hub_flagged[sem] = {
                "top_node": top_node,
                "top_node_hyperedge_count": int(top_count),
                "hyperedge_count": total,
                "top1_ratio": ratio,
                "threshold": threshold_used,
            }

    # Composed-fact trace stats: counts per semantics and polarity.
    composed: dict[str, Any] = {}
    for sem in sorted(composed_counts):
        composed[sem] = {
            pol: int(composed_counts[sem][pol]) for pol in sorted(composed_counts[sem])
        }

    # Long-tail coverage: fraction of hyperedges involving low-incidence nodes per semantics.
    # We define low-incidence nodes as the bottom `long_tail_quantile` of nodes by incidence.
    lq = float(long_tail_quantile)
    if not (0.0 < lq < 1.0):
        lq = 0.2
    long_tail: dict[str, Any] = {}
    low_nodes_by_sem: dict[str, set[str]] = {}
    for sem in sorted(incidence_by_sem):
        cnt = incidence_by_sem[sem]
        if not cnt:
            continue
        items = sorted(cnt.items(), key=lambda kv: (int(kv[1]), str(kv[0])))
        cutoff_idx = max(1, int(len(items) * lq))
        low_nodes_by_sem[sem] = {k for k, _ in items[:cutoff_idx]}

    low_involved_by_sem: Counter[str] = Counter()
    for he in hyperedges:
        if not isinstance(he, dict):
            continue
        sem = _hyperedge_semantics(he)
        low_nodes = low_nodes_by_sem.get(sem)
        if not low_nodes:
            continue
        for nid in _hyperedge_node_ids(he):
            if nid in low_nodes:
                low_involved_by_sem[sem] += 1
                break

    for sem in sorted(low_nodes_by_sem):
        total = int(he_count_by_sem[sem])
        if total <= 0:
            continue
        low_nodes = low_nodes_by_sem[sem]
        low_involved = int(low_involved_by_sem.get(sem, 0))
        long_tail[sem] = {
            "quantile": lq,
            "low_node_count": len(low_nodes),
            "hyperedge_count": total,
            "low_node_hyperedge_count": low_involved,
            "coverage": float(low_involved) / float(total),
        }

    # Composed relation contribution stats (requires `supports`).
    composed_relations: dict[str, Any] = {}
    for sem in sorted(composed_relation_counts):
        top = sorted(
            composed_relation_counts[sem].items(),
            key=lambda kv: (-int(kv[1]), str(kv[0])),
        )
        composed_relations[sem] = [[r, int(c)] for r, c in top]

    # === General structure stats (global + per-semantics) ===
    report_hyperedges = [he for he in hyperedges if isinstance(he, dict)]
    general_stats = _compute_general_stats_for_hyperedges(report_hyperedges)

    by_sem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for he in report_hyperedges:
        by_sem[_hyperedge_semantics(he)].append(he)
    general_stats_by_semantics = {
        sem: _compute_general_stats_for_hyperedges(by_sem[sem])
        for sem in sorted(by_sem)
    }

    family_kw = {
        "support_triplets_by_id": support_triplets_by_id,
        "rule_trace_by_id": rule_trace_by_id,
        "long_tail_quantile": lq,
        "top_k": top_k,
    }
    semantics_family_metrics: dict[str, Any] = {
        "drug_disease_shared_target_bundle": drug_disease_shared_target_bundle_metrics(
            report_hyperedges, **family_kw
        ),
        "disease_protein_bundle": disease_protein_bundle_metrics(
            report_hyperedges, **family_kw
        ),
        "protein_disease_bundle": protein_disease_bundle_metrics(
            report_hyperedges, **family_kw
        ),
    }

    curated_source_metrics: dict[str, Any] = {}
    if corum_tsv or any(
        he.get("semantics") == "protein_complex" for he in report_hyperedges
    ):
        from primehkg.audit.curated import (
            corum_protein_complex_metrics,
            node_id_set_from_nodes,
        )

        known = node_id_set_from_nodes(nodes)
        curated_source_metrics["corum"] = corum_protein_complex_metrics(
            report_hyperedges,
            corum_tsv=corum_tsv,
            known_node_ids=known,
            long_tail_quantile=lq,
            top_k=top_k,
            uniprot_to_geneid=uniprot_to_geneid,
            support=supports,
            source_name="CORUM",
        )
        curated_source_metrics["complexportal"] = corum_protein_complex_metrics(
            report_hyperedges,
            corum_tsv=None,
            known_node_ids=known,
            long_tail_quantile=lq,
            top_k=top_k,
            uniprot_to_geneid=uniprot_to_geneid,
            support=supports,
            source_name="Complex Portal",
        )

    node_connectivity = _compute_node_connectivity(nodes, report_hyperedges)
    node_field_completeness = _compute_node_field_completeness(nodes)
    metabolite_unresolved_names = (
        count_metabolite_unresolved_names(nodes) if nodes else None
    )

    report: dict[str, Any] = {
        "hyperedge_classification_summary": _compute_hyperedge_classification_summary(
            report_hyperedges
        ),
        "hyperedge_size": size_dist,
        "node_participation": participation,
        "hub_risk": {
            "default_threshold": default_threshold,
            "threshold": default_threshold,
            "thresholds_by_semantics": thresholds_by_semantics,
            "flagged": hub_flagged,
        },
        "composed_fact_trace": composed,
        "long_tail_coverage": long_tail,
        "composed_relation_contributions": composed_relations,
        "general_stats": general_stats,
        "general_stats_by_semantics": general_stats_by_semantics,
        "semantics_family_metrics": semantics_family_metrics,
        "curated_source_metrics": curated_source_metrics,
    }
    if node_connectivity is not None:
        report["node_connectivity"] = node_connectivity
    if node_field_completeness is not None:
        report["node_field_completeness"] = node_field_completeness
    if metabolite_unresolved_names is not None:
        report["metabolite_unresolved_names"] = metabolite_unresolved_names
    if builder_segments is not None:
        report["builder_segments"] = _summarize_builder_segments(builder_segments)
    if relation_coverage_audit is not None:
        report["relation_coverage_audit"] = relation_coverage_audit
    return report
