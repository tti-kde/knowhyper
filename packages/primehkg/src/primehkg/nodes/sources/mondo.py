from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from primehkg.nodes.feature_refs import merge_feature_refs, text_pointer
from primehkg.nodes.normalize import normalize_xref_id

if TYPE_CHECKING:
    from collections.abc import Sequence

_OBO_XREF_NAMESPACES = frozenset({"MESH", "OMIM", "UMLS"})


def parse_mondo_obo_xrefs(
    obo_path: str | Path,
) -> dict[str, dict[str, list[str]]]:
    """Parse a MONDO OBO file and return ``{MONDO:ID: {ns: [xref_ids]}}``."""
    result: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    current: str | None = None
    with Path(obo_path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("["):
                current = None
            elif s.startswith("id: MONDO:"):
                current = s.removeprefix("id: ").strip()
            elif current and s.startswith("xref:"):
                raw = s.removeprefix("xref:").strip().split("{", 1)[0].strip()
                prefix = raw.split(":", 1)[0] if ":" in raw else ""
                if prefix in _OBO_XREF_NAMESPACES:
                    result[current][prefix.lower()].add(raw)
    return {
        mid: {ns: sorted(ids) for ns, ids in nss.items()} for mid, nss in result.items()
    }


def _strip_header_map(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}
    return {h.strip(): h for h in fieldnames if h is not None}


def read_mondo_updates(
    path: str | Path,
    *,
    delimiter: str = "\t",
    mondo_obo: str | Path | None = None,
) -> dict[str, dict[str, object]]:
    """Read a local TSV that maps PrimeKG disease nodes to MONDO xrefs (and optional name).

    Required columns:
    - node_id: PrimeKG node id (e.g. primekg:disease:D0001)
    - mondo_id: MONDO identifier (e.g. MONDO:0005148 or 0005148)

    Optional columns:
    - mondo_name: display name candidate (stored as name provenance + text feature ref)
    - definition: optional text payload (preferred over mondo_name for features)

    When *mondo_obo* is provided, MONDO OBO xrefs (MeSH, OMIM, UMLS) are
    transitively added to each disease node through its MONDO IDs.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames
        if fieldnames is None or not any((h or "").strip() for h in fieldnames):
            msg = "MONDO updates must have a header row"
            raise ValueError(msg)

        cols = _strip_header_map(fieldnames)
        required = {"node_id", "mondo_id"}
        missing = required - set(cols)
        if missing:
            msg = f"MONDO updates missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        out: dict[str, dict[str, object]] = {}
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            mondo_id_raw = (row.get(cols["mondo_id"]) or "").strip()
            mondo_name = (row.get(cols.get("mondo_name", "")) or "").strip()
            definition = (row.get(cols.get("definition", "")) or "").strip()
            if not node_id or not mondo_id_raw:
                continue

            mondo_id = normalize_xref_id("mondo", mondo_id_raw)
            update: dict[str, object] | None = out.get(node_id)
            if update is None:
                update = {
                    "xrefs": {},
                    "feature_refs": {},
                    "source": "MONDO",
                    "source_version": p.name,
                    "evidence": "local_mondo_mapping",
                }
                out[node_id] = update

            xrefs_val = update.get("xrefs")
            mondo_existing: list[str] = []
            if isinstance(xrefs_val, dict):
                xrefs_val2 = cast("dict[str, object]", xrefs_val)
                existing = xrefs_val2.get("mondo")
                if isinstance(existing, list):
                    mondo_existing = [v for v in existing if isinstance(v, str) and v]
            xrefs_dict: dict[str, list[str]] = {
                "mondo": sorted({*mondo_existing, mondo_id})
            }
            update["xrefs"] = xrefs_dict

            if mondo_name:
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
                        "name": mondo_name,
                        "source": "MONDO",
                        "source_version": p.name,
                        "evidence": "local_mondo_mapping",
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
                                source="MONDO",
                                source_version=p.name,
                            )
                        ]
                    },
                )
            elif mondo_name:
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
                                source="MONDO",
                                source_version=p.name,
                            )
                        ]
                    },
                )

        if mondo_obo is not None:
            obo_xrefs = parse_mondo_obo_xrefs(mondo_obo)
            for update in out.values():
                xrefs_raw = update.get("xrefs")
                if not isinstance(xrefs_raw, dict):
                    continue
                xrefs_val = cast("dict[str, object]", xrefs_raw)
                mondo_ids: list[str] = []
                existing_mondo = xrefs_val.get("mondo")
                if isinstance(existing_mondo, list):
                    mondo_ids = [v for v in existing_mondo if isinstance(v, str)]
                merged: dict[str, set[str]] = {}
                for mid in mondo_ids:
                    for ns, ids in obo_xrefs.get(mid, {}).items():
                        merged.setdefault(ns, set()).update(ids)
                if merged:
                    xd = cast("dict[str, list[str]]", xrefs_val)
                    for ns, ids in sorted(merged.items()):
                        prev = set(xd.get(ns, []))
                        xd[ns] = sorted(prev | ids)
                    update["xrefs"] = xd

        return out
