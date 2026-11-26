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


def read_chebi_name_updates(
    path: str | Path, *, source_version: str, delimiter: str = "\t"
) -> dict[str, dict[str, object]]:
    """Read a ``chebi_names.tsv`` mapping ChEBI node ids to canonical names.

    Required columns:
    - chebi_node_id: ``chebi:<accession>`` node id
    - name: canonical compound name

    Returns ``node_updates`` keyed by ``chebi_node_id``, each value carrying a
    single display-name candidate consumed by ``enrich_nodes_v1`` /
    ``_choose_display_name``. Rows whose id does not start with ``chebi:`` or
    whose name is empty are skipped. Raises ``ValueError`` on a missing header,
    or when at least one data row was read but none yielded a valid ``chebi:``
    id (a strong "wrong file / transposed columns" signal).
    """
    p = Path(path)
    out: dict[str, dict[str, object]] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames
        if fieldnames is None or not any((h or "").strip() for h in fieldnames):
            msg = "ChEBI names TSV must have a header row"
            raise ValueError(msg)
        cols = _strip_header_map(fieldnames)
        required = {"chebi_node_id", "name"}
        missing = required - set(cols)
        if missing:
            msg = f"ChEBI names TSV header missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        read = kept = 0
        for row in reader:
            read += 1
            node_id = (row.get(cols["chebi_node_id"]) or "").strip()
            name = (row.get(cols["name"]) or "").strip()
            if not node_id.startswith("chebi:") or not name:
                continue
            kept += 1
            out[node_id] = {
                "names": [
                    {"name": name, "source": "chebi", "source_version": source_version}
                ]
            }
        if read and not kept:
            msg = (
                f"ChEBI names TSV had {read} data rows but 0 valid 'chebi:' ids "
                "— wrong columns or transposed file?"
            )
            raise ValueError(msg)
    return out
