from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from primehkg.nodes.feature_refs import text_pointer

if TYPE_CHECKING:
    from collections.abc import Sequence


def _strip_header_map(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}
    return {h.strip(): h for h in fieldnames if h is not None}


def read_chebi_definition_updates(
    path: str | Path, *, source_version: str, delimiter: str = "\t"
) -> dict[str, dict[str, object]]:
    """Read a ``chebi_definitions.tsv`` mapping ChEBI node ids to definitions.

    Required columns:
    - chebi_node_id: ``chebi:<accession>`` node id
    - definition: compound definition text

    Returns ``node_updates`` keyed by ``chebi_node_id``, each value carrying a
    ``feature_refs.text`` indirection pointer consumed by ``enrich_nodes_v1``.
    Rows whose id does not start with ``chebi:`` or whose definition is empty
    are skipped. Raises ``ValueError`` on a missing header, missing required
    columns, or when at least one data row was read but none yielded a valid
    ``chebi:`` id (a strong "wrong file / transposed columns" signal).
    """
    p = Path(path)
    out: dict[str, dict[str, object]] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames
        if fieldnames is None or not any((h or "").strip() for h in fieldnames):
            msg = "ChEBI definitions TSV must have a header row"
            raise ValueError(msg)
        cols = _strip_header_map(fieldnames)
        required = {"chebi_node_id", "definition"}
        missing = required - set(cols)
        if missing:
            msg = f"ChEBI definitions TSV header missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        read = kept = 0
        for row in reader:
            read += 1
            node_id = (row.get(cols["chebi_node_id"]) or "").strip()
            definition = (row.get(cols["definition"]) or "").strip()
            if not node_id.startswith("chebi:") or not definition:
                continue
            kept += 1
            out[node_id] = {
                "feature_refs": {
                    "text": [
                        text_pointer(
                            modality="definition",
                            source="chebi",
                            source_version=source_version,
                        )
                    ]
                }
            }
        if read and not kept:
            msg = (
                f"ChEBI definitions TSV had {read} data rows but 0 valid 'chebi:' "
                "ids — wrong columns or transposed file?"
            )
            raise ValueError(msg)
    return out
