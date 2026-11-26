from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _strip_header_map(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}
    return {h.strip(): h for h in fieldnames if h is not None}


def read_chebi_synonym_updates(
    path: str | Path,
    *,
    source_version: str | None = None,
    delimiter: str = "\t",
) -> dict[str, dict[str, object]]:
    """Group ChEBI synonym TSV rows into node synonym updates.

    Expected columns: ``chebi_node_id``, ``surface`` (``name_type`` optional).
    Keys are ``chebi:*``. Non-``chebi:`` ids and empty surfaces are skipped.
    """
    p = Path(path)
    version = source_version if source_version is not None else p.name
    by_node: dict[str, set[str]] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames
        if fieldnames is None or not any((h or "").strip() for h in fieldnames):
            msg = "ChEBI synonyms TSV must have a header row"
            raise ValueError(msg)
        cols = _strip_header_map(fieldnames)
        required = {"chebi_node_id", "surface"}
        missing = required - set(cols)
        if missing:
            msg = f"ChEBI synonyms TSV missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        for row in reader:
            node_id = (row.get(cols["chebi_node_id"]) or "").strip()
            surface = (row.get(cols["surface"]) or "").strip()
            if not node_id or not node_id.startswith("chebi:") or not surface:
                continue
            by_node.setdefault(node_id, set()).add(surface)

    out: dict[str, dict[str, object]] = {}
    for node_id in sorted(by_node):
        synonyms = [
            {
                "name": name,
                "source": "ChEBI",
                "source_version": version,
            }
            for name in sorted(by_node[node_id])
        ]
        out[node_id] = {
            "synonyms": synonyms,
            "source": "ChEBI",
            "source_version": version,
        }
    return out
