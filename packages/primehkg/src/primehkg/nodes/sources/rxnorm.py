from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence


def _strip_header_map(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}
    return {h.strip(): h for h in fieldnames if h is not None}


def read_rxnorm_updates(
    path: str | Path, *, delimiter: str = "\t"
) -> dict[str, dict[str, object]]:
    """Read a local TSV mapping PrimeKG drug nodes to RxNorm xrefs (and optional name).

    Required columns:
    - node_id: PrimeKG node id (e.g. primekg:drug:D1)
    - rxnorm_id: RxCUI (typically digits)

    Optional columns:
    - rxnorm_name: display name candidate
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames
        if fieldnames is None or not any((h or "").strip() for h in fieldnames):
            msg = "RxNorm updates must have a header row"
            raise ValueError(msg)

        cols = _strip_header_map(fieldnames)
        required = {"node_id", "rxnorm_id"}
        missing = required - set(cols)
        if missing:
            msg = f"RxNorm updates missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        out: dict[str, dict[str, object]] = {}
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            rxcui = (row.get(cols["rxnorm_id"]) or "").strip()
            rxnorm_name = (row.get(cols.get("rxnorm_name", "")) or "").strip()
            if not node_id or not rxcui:
                continue

            update: dict[str, object] | None = out.get(node_id)
            if update is None:
                update = {
                    "xrefs": {},
                    "source": "RxNorm",
                    "source_version": p.name,
                    "evidence": "local_rxnorm_mapping",
                }
                out[node_id] = update

            xrefs_val = update.get("xrefs")
            existing_list: list[str] = []
            if isinstance(xrefs_val, dict):
                xrefs_val2 = cast("dict[str, object]", xrefs_val)
                existing = xrefs_val2.get("rxnorm")
                if isinstance(existing, list):
                    existing_list = [v for v in existing if isinstance(v, str) and v]
            update["xrefs"] = {"rxnorm": sorted({*existing_list, rxcui})}

            if rxnorm_name:
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
                        "name": rxnorm_name,
                        "source": "RxNorm",
                        "source_version": p.name,
                        "evidence": "local_rxnorm_mapping",
                    }
                )
                dedup: dict[str, dict[str, Any]] = {}
                for n in names_list:
                    name = n.get("name")
                    if isinstance(name, str) and name:
                        dedup[name] = n
                update["names"] = [dedup[k] for k in sorted(dedup)]

        return out
