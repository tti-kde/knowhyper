from __future__ import annotations

from typing import TYPE_CHECKING, Any

from primehkg.nodes.normalize import normalize_namespace, normalize_xref_id
from primehkg.nodes.provenance import sort_xref_provenance, xref_prov

if TYPE_CHECKING:
    from collections.abc import Iterable


def _iter_strs(values: object) -> Iterable[str]:
    if isinstance(values, list | tuple | set):
        for v in values:
            if isinstance(v, str):
                yield v


def _dedupe_sorted(values: Iterable[str]) -> list[str]:
    return sorted({v for v in values if isinstance(v, str) and v})


def _dedupe_prov(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[object, ...]] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        key = (
            r.get("namespace"),
            r.get("id"),
            r.get("source"),
            r.get("source_version"),
            r.get("confidence"),
            r.get("evidence"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def merge_xrefs(
    *,
    base: dict[str, list[str]],
    incoming: dict[str, list[str]],
    incoming_source: str,
    incoming_source_version: str = "unknown",
    evidence: str | None = None,
    confidence: float | None = None,
    provenance: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    out: dict[str, list[str]] = {}
    prov: list[dict[str, Any]] = list(provenance or [])

    for raw_ns, raw_ids in (base or {}).items():
        if not isinstance(raw_ns, str):
            continue
        ns = normalize_namespace(raw_ns)
        ids_norm = _dedupe_sorted(
            normalize_xref_id(ns, rid) for rid in _iter_strs(raw_ids)
        )
        if ids_norm:
            out[ns] = ids_norm

    for raw_ns, raw_ids in (incoming or {}).items():
        if not isinstance(raw_ns, str):
            continue
        ns = normalize_namespace(raw_ns)
        ids_norm = _dedupe_sorted(
            normalize_xref_id(ns, rid) for rid in _iter_strs(raw_ids)
        )
        if not ids_norm:
            continue

        cur = out.get(ns, [])
        out[ns] = _dedupe_sorted([*cur, *ids_norm])

        prov.extend(
            [
                xref_prov(
                    namespace=ns,
                    xref_id=xid,
                    source=incoming_source,
                    source_version=incoming_source_version,
                    confidence=confidence,
                    evidence=evidence,
                )
                for xid in ids_norm
            ]
        )

    prov_sorted = sort_xref_provenance(_dedupe_prov(prov))
    return out, prov_sorted
