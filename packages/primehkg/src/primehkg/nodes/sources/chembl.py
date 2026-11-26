from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from primehkg.nodes.feature_refs import (
    merge_feature_refs,
    property_pointer,
    smiles_pointer,
)
from primehkg.policy.node_property_registry import (
    chembl_column_to_property_id,
    load_node_property_registry,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _strip_header_map(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}
    return {h.strip(): h for h in fieldnames if h is not None}


def read_chembl_updates(
    path: str | Path, *, delimiter: str = "\t"
) -> dict[str, dict[str, object]]:
    """Read a local TSV mapping PrimeKG drug nodes to ChEMBL xrefs (and optional name).

    Required columns:
    - node_id: PrimeKG node id (e.g. primekg:drug:D1)
    - chembl_id: ChEMBL compound id (e.g. CHEMBL25)

    Optional columns:
    - chembl_name: display name candidate
    """
    p = Path(path)
    col_to_pid = chembl_column_to_property_id(load_node_property_registry())
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames
        if fieldnames is None or not any((h or "").strip() for h in fieldnames):
            msg = "ChEMBL updates must have a header row"
            raise ValueError(msg)

        cols = _strip_header_map(fieldnames)
        required = {"node_id", "chembl_id"}
        missing = required - set(cols)
        if missing:
            msg = f"ChEMBL updates missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        out: dict[str, dict[str, object]] = {}
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            chembl_id = (row.get(cols["chembl_id"]) or "").strip()
            chembl_name = (row.get(cols.get("chembl_name", "")) or "").strip()
            if not node_id or not chembl_id:
                continue

            update: dict[str, object] | None = out.get(node_id)
            if update is None:
                update = {
                    "xrefs": {},
                    "feature_refs": {},
                    "source": "ChEMBL",
                    "source_version": p.name,
                    "evidence": "local_chembl_mapping",
                }
                out[node_id] = update

            xrefs_val = update.get("xrefs")
            existing_list: list[str] = []
            if isinstance(xrefs_val, dict):
                xrefs_val2 = cast("dict[str, object]", xrefs_val)
                existing = xrefs_val2.get("chembl")
                if isinstance(existing, list):
                    existing_list = [v for v in existing if isinstance(v, str) and v]
            update["xrefs"] = {"chembl": sorted({*existing_list, chembl_id})}

            if "canonical_smiles" in cols:
                smiles = (row.get(cols["canonical_smiles"]) or "").strip()
                if smiles:
                    existing_fr = update.get("feature_refs")
                    base_fr: dict[str, Any] = (
                        cast("dict[str, Any]", existing_fr)
                        if isinstance(existing_fr, dict)
                        else {}
                    )
                    update["feature_refs"] = merge_feature_refs(
                        base_fr,
                        {
                            "smiles": [
                                smiles_pointer(source="ChEMBL", source_version=p.name)
                            ]
                        },
                    )

            property_refs: list[dict[str, str]] = []
            for col_name, property_id in col_to_pid.items():
                col = cols.get(col_name)
                if col is None:
                    continue
                raw = (row.get(col) or "").strip()
                if not raw:
                    continue
                property_refs.append(
                    property_pointer(
                        property_id=property_id,
                        source="ChEMBL",
                        source_version=p.name,
                    )
                )
            if property_refs:
                existing_fr = update.get("feature_refs")
                base_fr = (
                    cast("dict[str, Any]", existing_fr)
                    if isinstance(existing_fr, dict)
                    else {}
                )
                update["feature_refs"] = merge_feature_refs(
                    base_fr,
                    {"property": property_refs},
                )

            if chembl_name:
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
                        "name": chembl_name,
                        "source": "ChEMBL",
                        "source_version": p.name,
                        "evidence": "local_chembl_mapping",
                    }
                )
                dedup: dict[str, dict[str, Any]] = {}
                for n in names_list:
                    name = n.get("name")
                    if isinstance(name, str) and name:
                        dedup[name] = n
                update["names"] = [dedup[k] for k in sorted(dedup)]

        return out
