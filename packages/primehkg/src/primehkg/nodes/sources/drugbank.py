from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from primehkg.nodes.normalize import normalize_xref_id

if TYPE_CHECKING:
    from collections.abc import Sequence


def _strip_header_map(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}
    return {h.strip(): h for h in fieldnames if h is not None}


def read_drugbank_updates(
    path: str | Path, *, delimiter: str = "\t"
) -> dict[str, dict[str, object]]:
    """Read a local TSV mapping PrimeKG drug nodes to DrugBank xrefs (and optional name).

    Required columns:
    - node_id: PrimeKG node id (e.g. primekg:drug:D1)
    - drugbank_id: DrugBank identifier (e.g. DB00945 or drugbank:DB00945)

    Optional columns:
    - drug_name: display name candidate
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames
        if fieldnames is None or not any((h or "").strip() for h in fieldnames):
            msg = "DrugBank updates must have a header row"
            raise ValueError(msg)

        cols = _strip_header_map(fieldnames)
        required = {"node_id", "drugbank_id"}
        missing = required - set(cols)
        if missing:
            msg = f"DrugBank updates missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        out: dict[str, dict[str, object]] = {}
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            drugbank_id_raw = (row.get(cols["drugbank_id"]) or "").strip()
            drug_name = (row.get(cols.get("drug_name", "")) or "").strip()
            if not node_id or not drugbank_id_raw:
                continue

            drugbank_id = normalize_xref_id("drugbank", drugbank_id_raw)
            update: dict[str, object] | None = out.get(node_id)
            if update is None:
                update = {
                    "xrefs": {},
                    "source": "DrugBank",
                    "source_version": p.name,
                    "evidence": "local_drugbank_mapping",
                }
                out[node_id] = update

            xrefs_val = update.get("xrefs")
            existing_list: list[str] = []
            if isinstance(xrefs_val, dict):
                xrefs_val2 = cast("dict[str, object]", xrefs_val)
                existing = xrefs_val2.get("drugbank")
                if isinstance(existing, list):
                    existing_list = [v for v in existing if isinstance(v, str) and v]
            update["xrefs"] = {"drugbank": sorted({*existing_list, drugbank_id})}

            if drug_name:
                names_val = update.get("names")
                names_list: list[dict[str, Any]] = []
                if isinstance(names_val, list):
                    for n in names_val:
                        if isinstance(n, dict):
                            n2 = cast("dict[str, object]", n)
                            name = n2.get("name")
                            if isinstance(name, str) and name:
                                names_list.append(cast("dict[str, Any]", n))
                names_list.append(
                    {
                        "name": drug_name,
                        "source": "DrugBank",
                        "source_version": p.name,
                        "evidence": "local_drugbank_mapping",
                    }
                )
                dedup: dict[str, dict[str, Any]] = {}
                for n in names_list:
                    name = n.get("name")
                    if isinstance(name, str) and name:
                        dedup[name] = n
                update["names"] = [dedup[k] for k in sorted(dedup)]

        return out
