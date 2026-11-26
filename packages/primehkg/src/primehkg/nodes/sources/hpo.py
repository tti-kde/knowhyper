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


def read_hpo_updates(
    path: str | Path, *, delimiter: str = "\t"
) -> dict[str, dict[str, object]]:
    """Read a local TSV mapping PrimeKG nodes to HPO term ids (and optional label).

    Required columns:
    - ``node_id``: PrimeKG node id (e.g. ``primekg:phenotype:123``)
    - ``hp_id``: HPO identifier (e.g. ``HP:0001250`` or ``0001250``)

    Optional columns:
    - ``hp_label``: display name candidate (stored as name provenance + text feature ref)
    - ``definition``: optional text payload (preferred over hp_label for features)
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames
        if fieldnames is None or not any((h or "").strip() for h in fieldnames):
            msg = "HPO updates must have a header row"
            raise ValueError(msg)

        cols = _strip_header_map(fieldnames)
        required = {"node_id", "hp_id"}
        missing = required - set(cols)
        if missing:
            msg = f"HPO updates missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        out: dict[str, dict[str, object]] = {}
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            hp_id_raw = (row.get(cols["hp_id"]) or "").strip()
            hp_label = (row.get(cols.get("hp_label", "")) or "").strip()
            definition = (row.get(cols.get("definition", "")) or "").strip()
            if not node_id or not hp_id_raw:
                continue

            hp_id = normalize_xref_id("hpo", hp_id_raw)
            update: dict[str, object] | None = out.get(node_id)
            if update is None:
                update = {
                    "xrefs": {},
                    "feature_refs": {},
                    "source": "HPO",
                    "source_version": p.name,
                    "evidence": "local_hpo_mapping",
                }
                out[node_id] = update

            xrefs_val = update.get("xrefs")
            hpo_existing: list[str] = []
            if isinstance(xrefs_val, dict):
                xrefs_val2 = cast("dict[str, object]", xrefs_val)
                existing = xrefs_val2.get("hpo")
                if isinstance(existing, list):
                    hpo_existing = [v for v in existing if isinstance(v, str) and v]
            xrefs_dict: dict[str, list[str]] = {"hpo": sorted({*hpo_existing, hp_id})}
            update["xrefs"] = xrefs_dict

            if hp_label:
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
                        "name": hp_label,
                        "source": "HPO",
                        "source_version": p.name,
                        "evidence": "local_hpo_mapping",
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
                                source="HPO",
                                source_version=p.name,
                            )
                        ]
                    },
                )
            elif hp_label:
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
                                source="HPO",
                                source_version=p.name,
                            )
                        ]
                    },
                )

        return out
