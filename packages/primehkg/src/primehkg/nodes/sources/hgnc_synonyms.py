from __future__ import annotations

import json
from pathlib import Path


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, list):
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
    return []


def read_hgnc_synonym_updates(
    path: str | Path, *, source_version: str | None = None
) -> dict[str, dict[str, object]]:
    p = Path(path)
    version = source_version if source_version is not None else p.name
    data = json.loads(p.read_text(encoding="utf-8"))
    docs = data.get("response", {}).get("docs")
    if not isinstance(docs, list):
        msg = "HGNC JSON missing response.docs list"
        raise ValueError(msg)  # noqa: TRY004
    by_entrez: dict[str, set[str]] = {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        entrez = str(doc.get("entrez_id") or "").strip()
        if not entrez:
            continue
        surfaces = _as_str_list(doc.get("alias_symbol")) + _as_str_list(
            doc.get("prev_symbol")
        )
        if not surfaces:
            continue
        by_entrez.setdefault(entrez, set()).update(surfaces)
    out: dict[str, dict[str, object]] = {}
    for entrez in sorted(by_entrez, key=lambda x: (len(x), x)):
        node_id = f"primekg:protein:{entrez}"
        synonyms = [
            {"name": name, "source": "HGNC", "source_version": version}
            for name in sorted(by_entrez[entrez])
        ]
        out[node_id] = {
            "synonyms": synonyms,
            "source": "HGNC",
            "source_version": version,
        }
    return out
