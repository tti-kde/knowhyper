"""Hyperedge signature index for coarse candidate filtering (spike / #133)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from primehkg.ids import normalize_node_id

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class HyperedgeSignature:
    """Coarse hyperedge bucket key (KHG spec recommendation)."""

    semantics: str
    evidence_grade: str
    size: int
    roles_multiset: tuple[str, ...]
    node_types_multiset: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SignatureFilter:
    """Optional allow-lists for coarse filtering before EWO ranking."""

    semantics_allow: frozenset[str] | None = None
    evidence_grade_allow: frozenset[str] | None = None
    min_size: int | None = None
    max_size: int | None = None
    roles_any: frozenset[str] | None = None
    node_types_any: frozenset[str] | None = None


def infer_node_type_from_id(node_id: str) -> str:
    """Infer node type from PrimeKG-style ids (`primekg:<type>:...`)."""
    n = normalize_node_id(node_id)
    if n.startswith("primekg:"):
        parts = n.split(":")
        if len(parts) >= 3 and parts[1]:
            return parts[1]
    return "unknown"


def hyperedge_signature(he: Mapping[str, object]) -> HyperedgeSignature | None:
    """Compute a deterministic signature for one hyperedge row."""
    he_id_obj = he.get("he_id")
    if not isinstance(he_id_obj, str):
        return None

    sem_obj = he.get("semantics")
    ev_obj = he.get("evidence_grade")
    semantics = sem_obj if isinstance(sem_obj, str) else ""
    evidence_grade = ev_obj if isinstance(ev_obj, str) else ""

    node_ids_obj = he.get("node_ids")
    node_ids: list[str] = []
    if isinstance(node_ids_obj, list):
        for n in node_ids_obj:
            if isinstance(n, str):
                nn = normalize_node_id(n)
                if nn:
                    node_ids.append(nn)

    roles_obj = he.get("roles")
    roles: list[str] = []
    if isinstance(roles_obj, dict):
        roles = [
            v.strip() for v in roles_obj.values() if isinstance(v, str) and v.strip()
        ]

    node_types = sorted({infer_node_type_from_id(n) for n in node_ids})
    return HyperedgeSignature(
        semantics=semantics,
        evidence_grade=evidence_grade,
        size=len(node_ids),
        roles_multiset=tuple(sorted(roles)),
        node_types_multiset=tuple(node_types),
    )


def build_signature_index(
    hyperedges: Iterable[Mapping[str, object]],
) -> dict[HyperedgeSignature, list[str]]:
    """Map signature → sorted he_id list (deterministic)."""
    buckets: dict[HyperedgeSignature, set[str]] = defaultdict(set)
    for he in hyperedges:
        sig = hyperedge_signature(he)
        he_id_obj = he.get("he_id")
        if sig is None or not isinstance(he_id_obj, str):
            continue
        buckets[sig].add(he_id_obj)

    def _sig_sort_key(
        sig: HyperedgeSignature,
    ) -> tuple[str, str, int, tuple[str, ...], tuple[str, ...]]:
        return (
            sig.semantics,
            sig.evidence_grade,
            sig.size,
            sig.roles_multiset,
            sig.node_types_multiset,
        )

    return {
        sig: sorted(ids)
        for sig, ids in sorted(buckets.items(), key=lambda item: _sig_sort_key(item[0]))
    }


def build_he_id_to_signature(
    hyperedges: Iterable[Mapping[str, object]],
) -> dict[str, HyperedgeSignature]:
    """Map he_id → signature for fast per-candidate lookup."""
    out: dict[str, HyperedgeSignature] = {}
    for he in hyperedges:
        he_id_obj = he.get("he_id")
        sig = hyperedge_signature(he)
        if isinstance(he_id_obj, str) and sig is not None:
            out[he_id_obj] = sig
    return out


def signature_matches(sig: HyperedgeSignature, filt: SignatureFilter) -> bool:
    """Return True when hyperedge signature passes all set filters."""
    if filt.semantics_allow is not None and sig.semantics not in filt.semantics_allow:
        return False
    if (
        filt.evidence_grade_allow is not None
        and sig.evidence_grade not in filt.evidence_grade_allow
    ):
        return False
    if filt.min_size is not None and sig.size < int(filt.min_size):
        return False
    if filt.max_size is not None and sig.size > int(filt.max_size):
        return False
    if filt.roles_any is not None and not (
        set(sig.roles_multiset) & set(filt.roles_any)
    ):
        return False
    return filt.node_types_any is None or bool(
        set(sig.node_types_multiset) & set(filt.node_types_any)
    )


def filter_he_ids_by_signature(
    candidate_he_ids: Sequence[str],
    he_to_sig: Mapping[str, HyperedgeSignature],
    filt: SignatureFilter,
) -> list[str]:
    """Coarse-filter candidate he_ids; output order follows sorted input ids."""
    kept: list[str] = []
    for hid in sorted({h for h in candidate_he_ids if isinstance(h, str)}):
        sig = he_to_sig.get(hid)
        if sig is None:
            continue
        if signature_matches(sig, filt):
            kept.append(hid)
    return kept


def measure_candidate_reduction(
    *,
    before: Sequence[str],
    after: Sequence[str],
) -> dict[str, int | float]:
    """Summarize how many candidates were removed by coarse filtering."""
    n_before = len(before)
    n_after = len(after)
    removed = n_before - n_after
    ratio = float(removed) / float(n_before) if n_before else 0.0
    return {
        "candidates_before": n_before,
        "candidates_after": n_after,
        "candidates_removed": removed,
        "reduction_ratio": ratio,
    }
