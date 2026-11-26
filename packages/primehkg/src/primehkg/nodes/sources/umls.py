from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from primehkg.nodes.feature_refs import merge_feature_refs, text_pointer
from primehkg.nodes.normalize import normalize_xref_id

if TYPE_CHECKING:
    from collections.abc import Sequence


def _strip_header_map(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}
    return {h.strip(): h for h in fieldnames if h is not None}


def read_umls_updates(
    path: str | Path, *, delimiter: str = "\t"
) -> dict[str, dict[str, object]]:
    """Read a local TSV mapping PrimeKG nodes to UMLS CUIs (and optional name).

    Required columns:
    - node_id: PrimeKG node id (e.g. primekg:disease:D1, primekg:drug:D1, ...)
    - umls_id: UMLS CUI (e.g. C0000005)

    Optional columns:
    - umls_name: display name candidate
    - definition: optional text payload (indirection pointer only; no inline text in core)
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames
        if fieldnames is None or not any((h or "").strip() for h in fieldnames):
            msg = "UMLS updates must have a header row"
            raise ValueError(msg)

        cols = _strip_header_map(fieldnames)
        required = {"node_id", "umls_id"}
        missing = required - set(cols)
        if missing:
            msg = f"UMLS updates missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        out: dict[str, dict[str, object]] = {}
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            umls_id_raw = (row.get(cols["umls_id"]) or "").strip()
            umls_name = (row.get(cols.get("umls_name", "")) or "").strip()
            definition = (row.get(cols.get("definition", "")) or "").strip()
            if not node_id or not umls_id_raw:
                continue

            umls_id = normalize_xref_id("umls", umls_id_raw)
            update: dict[str, object] | None = out.get(node_id)
            if update is None:
                update = {
                    "xrefs": {},
                    "feature_refs": {},
                    "source": "UMLS",
                    "source_version": p.name,
                    "evidence": "local_umls_mapping",
                }
                out[node_id] = update

            xrefs_val = update.get("xrefs")
            existing_list: list[str] = []
            if isinstance(xrefs_val, dict):
                xrefs_val2 = cast("dict[str, object]", xrefs_val)
                existing = xrefs_val2.get("umls")
                if isinstance(existing, list):
                    existing_list = [v for v in existing if isinstance(v, str) and v]
            update["xrefs"] = {"umls": sorted({*existing_list, umls_id})}

            if umls_name:
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
                        "name": umls_name,
                        "source": "UMLS",
                        "source_version": p.name,
                        "evidence": "local_umls_mapping",
                    }
                )
                dedup: dict[str, dict[str, Any]] = {}
                for n in names_list:
                    name = n.get("name")
                    if isinstance(name, str) and name:
                        dedup[name] = n
                update["names"] = [dedup[k] for k in sorted(dedup)]

            if definition:
                existing_fr = update.get("feature_refs")
                base_fr: dict[str, Any] = (
                    cast("dict[str, Any]", existing_fr)
                    if isinstance(existing_fr, dict)
                    else {}
                )
                update["feature_refs"] = merge_feature_refs(
                    base_fr,
                    {
                        "text": [
                            text_pointer(
                                modality="definition",
                                source="UMLS",
                                source_version=p.name,
                            )
                        ]
                    },
                )
            elif umls_name:
                existing_fr = update.get("feature_refs")
                base_fr = (
                    cast("dict[str, Any]", existing_fr)
                    if isinstance(existing_fr, dict)
                    else {}
                )
                update["feature_refs"] = merge_feature_refs(
                    base_fr,
                    {
                        "text": [
                            text_pointer(
                                modality="description",
                                source="UMLS",
                                source_version=p.name,
                            )
                        ]
                    },
                )

        return out
