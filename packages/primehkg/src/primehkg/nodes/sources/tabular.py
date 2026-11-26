from __future__ import annotations

import csv
from pathlib import Path


def read_xrefs_tsv(
    path: str | Path,
    *,
    node_id_col: str = "node_id",
    namespace_col: str = "namespace",
    xref_id_col: str = "id",
    delimiter: str = "\t",
) -> dict[str, dict[str, list[str]]]:
    """Read a simple delimited file containing one xref per row.

    Expected columns (defaults):
    - node_id
    - namespace
    - id
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None or not any(
            (h or "").strip() for h in reader.fieldnames
        ):
            msg = "xrefs tsv must have a header row"
            raise ValueError(msg)
        missing = {node_id_col, namespace_col, xref_id_col} - set(reader.fieldnames)
        if missing:
            msg = f"xrefs tsv missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        out: dict[str, dict[str, list[str]]] = {}
        for row in reader:
            node_id = (row.get(node_id_col) or "").strip()
            ns = (row.get(namespace_col) or "").strip()
            xid = (row.get(xref_id_col) or "").strip()
            if not node_id or not ns or not xid:
                continue
            out.setdefault(node_id, {}).setdefault(ns, []).append(xid)
        return out
