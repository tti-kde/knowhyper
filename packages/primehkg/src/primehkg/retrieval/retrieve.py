from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

from primehkg.ids import normalize_node_id
from primehkg.retrieval.ewo import EwoBreakdown, ewo_score
from primehkg.retrieval.index import build_inverted_index, node_degrees

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


class ScoreBreakdown(TypedDict):
    """Explainable score components returned by `retrieve_hyperedges()`."""

    ewo: EwoBreakdown
    size_penalty: float
    semantics_weight: float
    evidence_weight: float
    final_score: float


class RetrievalResult(TypedDict):
    """One ranked retrieval result row returned by `retrieve_hyperedges()`."""

    he_id: str
    hyperedge: dict[str, object]
    score: float
    score_breakdown: ScoreBreakdown
    audit_flags: list[str]


def retrieve_hyperedges(
    *,
    seed_node_ids: Sequence[str],
    hyperedges: Sequence[Mapping[str, object]],
    inverted_index: Mapping[str, list[str]] | None = None,
    top_k: int = 20,
    hub_penalty: float = 1.0,
    semantics_weight: Mapping[str, float] | None = None,
    evidence_weight: Mapping[str, float] | None = None,
    max_hyperedge_size: int | None = None,
    hub_degree_flag_threshold: int = 100,
    reproducibility: Literal["strict", "fast"] = "strict",
    candidate_hyperedge_ids: Sequence[str] | None = None,
    hyperedge_lookup: Callable[[str], Mapping[str, object] | None] | None = None,
) -> list[RetrievalResult]:
    """Retrieve and rank hyperedges using inverted-index + EWO ranking.

    Output is "explainable" (per-hyperedge `score_breakdown` + `audit_flags`).

    ``reproducibility``:
    - ``strict`` (default): full audit flags including ``large_hyperedge`` and
      ``weak_evidence`` when applicable.
    - ``fast``: same scores and ranking as ``strict``; omits non-critical flags
      ``large_hyperedge`` and ``weak_evidence`` for lower overhead (documented).
    """
    he_by_id: dict[str, Mapping[str, object]] = {}
    if hyperedge_lookup is not None and candidate_hyperedge_ids is not None:
        for hid in candidate_hyperedge_ids:
            if not isinstance(hid, str) or hid in he_by_id:
                continue
            he = hyperedge_lookup(hid)
            if he is not None:
                he_by_id[hid] = he
    else:
        for he in hyperedges:
            hid = he.get("he_id")
            if isinstance(hid, str) and hid not in he_by_id:
                he_by_id[hid] = he

    inv = (
        dict(inverted_index)
        if inverted_index is not None
        else build_inverted_index(hyperedges)
    )
    deg = node_degrees(inv)

    # Candidate generation: optional precomputed set (e.g. k-hop), else seed-incident union.
    if candidate_hyperedge_ids is not None:
        cand_ids: set[str] = set()
        for hid in candidate_hyperedge_ids:
            if not isinstance(hid, str):
                continue
            if hid in he_by_id:
                cand_ids.add(hid)
            elif hyperedge_lookup is not None:
                he = hyperedge_lookup(hid)
                if he is not None:
                    he_by_id[hid] = he
                    cand_ids.add(hid)
    else:
        cand_ids = set()
        for s in seed_node_ids:
            if not isinstance(s, str):
                continue
            ns = normalize_node_id(s)
            if not ns:
                continue
            cand_ids.update(inv.get(ns, []))

        # Fallback: if no candidates, score all (still deterministic).
        if not cand_ids:
            cand_ids = set(he_by_id.keys())

    sem_w: dict[str, float] = (
        dict(semantics_weight) if semantics_weight is not None else {}
    )
    ev_w: dict[str, float] = (
        dict(evidence_weight) if evidence_weight is not None else {}
    )

    results: list[RetrievalResult] = []
    for hid in sorted(cand_ids):
        he = he_by_id.get(hid)
        if he is None:
            continue

        node_ids_obj = he.get("node_ids")
        node_ids: list[str] = []
        if isinstance(node_ids_obj, list):
            for n in node_ids_obj:
                if not isinstance(n, str):
                    continue
                nn = normalize_node_id(n)
                if nn:
                    node_ids.append(nn)
        if max_hyperedge_size is not None and len(node_ids) > max_hyperedge_size:
            continue

        he_out: dict[str, object] = dict(he)
        he_out["node_ids"] = node_ids
        roles_obj = he_out.get("roles")
        if isinstance(roles_obj, dict):
            roles_norm: dict[str, str] = {}
            for k, v in roles_obj.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    continue
                nk = normalize_node_id(k)
                nv = v.strip()
                if nk and nv:
                    roles_norm[nk] = nv
            he_out["roles"] = roles_norm

        sem_obj = he.get("semantics")
        sem: str = sem_obj if isinstance(sem_obj, str) else ""
        ev_obj = he.get("evidence_grade")
        ev: str = ev_obj if isinstance(ev_obj, str) else ""

        if sem == "reaction_event":
            roles_obj = he_out.get("roles")
            if isinstance(roles_obj, dict):
                roles = {
                    k: v
                    for k, v in roles_obj.items()
                    if isinstance(k, str) and isinstance(v, str)
                }
                groups: dict[str, list[str]] = {}
                for nid in node_ids:
                    role = roles.get(nid)
                    if role is not None and role.strip():
                        groups.setdefault(role.strip(), []).append(nid)
                he_out["role_groups"] = {k: groups[k] for k in sorted(groups)}

        ewo = ewo_score(
            seed_node_ids=seed_node_ids,
            hyperedge_node_ids=node_ids,
            node_degrees=deg,
            hub_penalty=hub_penalty,
        )

        size_penalty = 1.0 / max(1, len(node_ids))
        semantics_mul = float(sem_w.get(sem, 1.0))
        evidence_mul = float(ev_w.get(ev, 1.0))

        score = float(ewo["ewo"]) * semantics_mul * evidence_mul * size_penalty

        audit_flags: list[str] = []
        if int(ewo["overlap_count"]) == 0:
            audit_flags.append("no_seed_overlap")
        if any(deg.get(n, 0) >= hub_degree_flag_threshold for n in ewo["overlap"]):  # type: ignore[arg-type]
            audit_flags.append("hub_dominated_risk")
        if reproducibility == "strict":
            if len(node_ids) >= 8:
                audit_flags.append("large_hyperedge")
            if ev and ev not in {"kg_atomic", "kg_composed", "curated_db"}:
                audit_flags.append("weak_evidence")

        score_breakdown: ScoreBreakdown = {
            "ewo": ewo,
            "size_penalty": size_penalty,
            "semantics_weight": semantics_mul,
            "evidence_weight": evidence_mul,
            "final_score": score,
        }

        results.append(
            {
                "he_id": hid,
                "hyperedge": he_out,
                "score": score,
                "score_breakdown": score_breakdown,
                "audit_flags": audit_flags,
            }
        )

    # Deterministic sort: score desc, then he_id asc.
    results.sort(key=lambda r: (-float(r["score"]), str(r["he_id"])))
    return results[: max(0, int(top_k))]
