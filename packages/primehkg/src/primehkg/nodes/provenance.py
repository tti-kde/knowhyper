from __future__ import annotations

from typing import Any

from primehkg.nodes.normalize import normalize_namespace, normalize_xref_id


def _norm_str(value: str | None, *, fallback: str) -> str:
    out = (value or "").strip()
    return out or fallback


def _norm_opt_str(value: str | None) -> str | None:
    out = (value or "").strip()
    return out or None


def xref_prov(
    *,
    namespace: str,
    xref_id: str,
    source: str,
    source_version: str = "unknown",
    confidence: float | None = None,
    evidence: str | None = None,
) -> dict[str, Any]:
    ns = normalize_namespace(namespace)
    xid = normalize_xref_id(ns, xref_id)
    return {
        "namespace": ns,
        "id": xid,
        "source": _norm_str(source, fallback="unknown"),
        "source_version": _norm_str(source_version, fallback="unknown"),
        "confidence": confidence,
        "evidence": _norm_opt_str(evidence),
    }


def sort_xref_provenance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(r: dict[str, Any]) -> tuple[object, ...]:
        conf = r.get("confidence")
        conf_key = (1, 0.0) if conf is None else (0, float(conf))
        return (
            str(r.get("namespace") or ""),
            str(r.get("id") or ""),
            str(r.get("source") or ""),
            str(r.get("source_version") or ""),
            str(r.get("evidence") or ""),
            conf_key,
        )

    return sorted(rows, key=key)
