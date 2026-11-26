from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from primehkg.nodes.feature_refs import (
    merge_feature_refs,
    ontology_pointer,
    text_pointer,
)
from primehkg.nodes.normalize import normalize_xref_id

if TYPE_CHECKING:
    from collections.abc import Sequence

PATHWAY_NODE_PREFIX = "primekg:pathway:"


def _strip_header_map(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}
    return {h.strip(): h for h in fieldnames if h is not None}


def read_pathway_updates(
    path: str | Path, *, delimiter: str = "\t"
) -> dict[str, dict[str, object]]:
    """Read ``pathway_map.tsv`` into node enrichment updates for pathway nodes."""
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames
        if fieldnames is None or not any((h or "").strip() for h in fieldnames):
            msg = "pathway updates must have a header row"
            raise ValueError(msg)

        cols = _strip_header_map(fieldnames)
        required = {"node_id", "reactome_id"}
        missing = required - set(cols)
        if missing:
            msg = f"pathway updates missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        out: dict[str, dict[str, object]] = {}
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            reactome_id_raw = (row.get(cols["reactome_id"]) or "").strip()
            pathway_name = (row.get(cols.get("pathway_name", "")) or "").strip()
            definition = (row.get(cols.get("description", "")) or "").strip()
            if not node_id or not reactome_id_raw:
                continue
            if not node_id.startswith(PATHWAY_NODE_PREFIX):
                continue

            reactome_id = normalize_xref_id("reactome", reactome_id_raw)

            update: dict[str, object] | None = out.get(node_id)
            if update is None:
                update = {
                    "xrefs": {},
                    "feature_refs": {},
                    "source": "PrimeKG",
                    "source_version": p.name,
                    "evidence": "local_pathway_mapping",
                }
                out[node_id] = update

            xrefs_val = update.get("xrefs")
            reactome_existing: list[str] = []
            if isinstance(xrefs_val, dict):
                xrefs_val2 = cast("dict[str, object]", xrefs_val)
                existing = xrefs_val2.get("reactome")
                if isinstance(existing, list):
                    reactome_existing = [
                        v for v in existing if isinstance(v, str) and v
                    ]
            update["xrefs"] = {"reactome": sorted({*reactome_existing, reactome_id})}

            if pathway_name:
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
                        "name": pathway_name,
                        "source": "PrimeKG",
                        "source_version": p.name,
                        "evidence": "local_pathway_mapping",
                    }
                )
                dedup: dict[str, dict[str, Any]] = {}
                for n in names_list:
                    name = n.get("name")
                    if isinstance(name, str) and name:
                        dedup[name] = n
                update["names"] = [dedup[k] for k in sorted(dedup)]

            fr_patch: dict[str, list[dict[str, str]]] = {
                "ontology": [
                    ontology_pointer(
                        modality="pathway_term",
                        source="PrimeKG",
                        source_version=p.name,
                        namespace="pathway",
                    )
                ]
            }
            if definition:
                fr_patch["text"] = [
                    text_pointer(
                        modality="description",
                        source="PrimeKG",
                        source_version=p.name,
                    )
                ]

            existing_fr = update.get("feature_refs")
            base_fr: dict[str, Any] = (
                cast("dict[str, Any]", existing_fr)
                if isinstance(existing_fr, dict)
                else {}
            )
            update["feature_refs"] = merge_feature_refs(base_fr, fr_patch)

        return out
