from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from primehkg.io.primekg import iter_primekg_rows


def collect_primekg_endpoint_name_updates(
    path: str | Path,
    *,
    strict: bool = True,
    source_version: str = "unknown",
) -> dict[str, dict[str, Any]]:
    """Build ``node_updates`` name provenance from optional endpoint labels in ``kg.csv``.

    When the CSV header includes endpoint name columns (see ``iter_primekg_rows``),
    each non-empty label becomes one ``names`` provenance record for that endpoint
    ``node_id`` (the normalized ``s`` / ``t`` value).

    Deterministic: rows are processed in file order; per node, duplicate
    ``(name, source, source_version, evidence)`` tuples are dropped, then sorted.
    """
    ver = (source_version or "unknown").strip() or "unknown"
    by_node: dict[str, set[tuple[str, str, str, str]]] = {}

    for row in iter_primekg_rows(path, strict=strict):
        for node_id, name, row_src, evidence in _row_endpoint_names(row):
            by_node.setdefault(node_id, set()).add(
                (name, row_src or "PrimeKG", ver, evidence)
            )

    out: dict[str, dict[str, Any]] = {}
    for node_id, tupleset in sorted(by_node.items()):
        names = [
            {
                "name": n,
                "source": src,
                "source_version": sv,
                "evidence": ev,
            }
            for n, src, sv, ev in sorted(tupleset)
        ]
        if names:
            out[node_id] = {"names": names}

    return out


def _row_endpoint_names(row: dict[str, str]) -> list[tuple[str, str, str, str]]:
    """Return list of (node_id, name, row_source_or_empty, evidence)."""
    s = row["s"]
    t = row["t"]
    out: list[tuple[str, str, str, str]] = []

    if "name_s" in row:
        ev = row.get("evidence_s") or "kg_csv_endpoint_name"
        out.append((s, row["name_s"], row.get("src_s") or "", ev))
    if "name_t" in row:
        ev = row.get("evidence_t") or "kg_csv_endpoint_name"
        out.append((t, row["name_t"], row.get("src_t") or "", ev))

    return out
