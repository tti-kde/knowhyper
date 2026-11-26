from __future__ import annotations

import json
from pathlib import Path


def read_xrefs_json(path: str | Path) -> dict[str, dict[str, list[str]]]:
    """Read a local JSON mapping ``node_id -> {namespace: [id...]}``.

    This is intended for **local-only** ingestion (fixtures, private resources) and is
    deterministic given file bytes.
    """
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "xrefs json must be an object mapping node_id -> xrefs"
        raise TypeError(msg)

    out: dict[str, dict[str, list[str]]] = {}
    for node_id, xrefs in raw.items():
        if not isinstance(node_id, str) or not node_id.strip():
            continue
        if not isinstance(xrefs, dict):
            continue
        cleaned: dict[str, list[str]] = {}
        for ns, ids in xrefs.items():
            if not isinstance(ns, str) or not isinstance(ids, list):
                continue
            ns2 = ns.strip()
            if not ns2:
                continue
            filtered = [x.strip() for x in ids if isinstance(x, str) and x.strip()]
            if filtered:
                cleaned[ns2] = filtered
        if cleaned:
            out[node_id.strip()] = cleaned
    return out
