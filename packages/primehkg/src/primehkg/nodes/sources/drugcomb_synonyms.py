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


def read_drugcomb_synonym_updates(
    path: str | Path,
    *,
    source_version: str | None = None,
    delimiter: str = "\t",
) -> dict[str, dict[str, object]]:
    """Group DrugComb identifiers TSV rows into node synonym updates.

    Expected columns: ``name``, ``drugbank_id``. Fetch already expands dname and
    ``;``-split synonyms into one row per surface. Keys are ``primekg:drug:DB*``.
    Empty names, empty ids, and non-``DB*`` ids are skipped.
    """
    p = Path(path)
    version = source_version if source_version is not None else p.name
    by_db: dict[str, set[str]] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames
        if fieldnames is None or not any((h or "").strip() for h in fieldnames):
            msg = "DrugComb identifiers TSV must have a header row"
            raise ValueError(msg)
        cols = _strip_header_map(fieldnames)
        required = {"name", "drugbank_id"}
        missing = required - set(cols)
        if missing:
            msg = (
                f"DrugComb identifiers TSV missing required columns: {sorted(missing)}"
            )
            raise ValueError(msg)
        for row in reader:
            name = (row.get(cols["name"]) or "").strip()
            db = (row.get(cols["drugbank_id"]) or "").strip()
            if not name or not db or not db.startswith("DB"):
                continue
            by_db.setdefault(db, set()).add(name)

    out: dict[str, dict[str, object]] = {}
    for db in sorted(by_db):
        node_id = f"primekg:drug:{db}"
        synonyms = [
            {
                "name": name,
                "source": "DrugComb",
                "source_version": version,
            }
            for name in sorted(by_db[db])
        ]
        out[node_id] = {
            "synonyms": synonyms,
            "source": "DrugComb",
            "source_version": version,
        }
    return out
