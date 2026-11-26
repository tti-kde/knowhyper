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


def read_hgnc_updates(
    path: str | Path, *, delimiter: str = "\t"
) -> dict[str, dict[str, object]]:
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames
        if fieldnames is None or not any((h or "").strip() for h in fieldnames):
            msg = "HGNC updates must have a header row"
            raise ValueError(msg)

        cols = _strip_header_map(fieldnames)
        required = {"hgnc_id", "symbol"}
        missing = required - set(cols)
        if missing:
            msg = f"HGNC updates missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        out: dict[str, dict[str, object]] = {}
        for row in reader:
            hgnc_id = (row.get(cols["hgnc_id"]) or "").strip()
            symbol = (row.get(cols["symbol"]) or "").strip()
            if not hgnc_id or not symbol:
                continue

            normalized_hgnc_id = _normalize_hgnc_id(hgnc_id)
            node_id = f"primekg:gene:{normalized_hgnc_id}"
            xrefs = _build_xrefs(row, cols, symbol)
            update: dict[str, object] = {
                "names": [
                    {
                        "name": symbol,
                        "source": "HGNC",
                        "source_version": p.name,
                        "evidence": "local_hgnc_export",
                    }
                ],
                "xrefs": xrefs,
                "source": "HGNC",
                "source_version": p.name,
                "evidence": "local_hgnc_export",
            }
            out[node_id] = update
        return out


def _build_xrefs(
    row: dict[str, str | None], cols: dict[str, str], symbol: str
) -> dict[str, list[str]]:
    xrefs: dict[str, list[str]] = {}

    hgnc_id = (row.get(cols["hgnc_id"]) or "").strip()
    values = {
        "hgnc": [_normalize_hgnc_id(hgnc_id)] if hgnc_id else [],
        "symbol": [symbol],
        "uniprot": _split_multi_value(row.get(cols.get("uniprot_ids", ""))),
        "ensembl": _split_multi_value(row.get(cols.get("ensembl_gene_id", ""))),
        "ncbigene": _split_multi_value(row.get(cols.get("entrez_id", ""))),
    }

    for namespace, ids in values.items():
        cleaned = sorted(
            {value.strip() for value in ids if isinstance(value, str) and value.strip()}
        )
        if cleaned:
            xrefs[namespace] = cleaned

    return xrefs


def _normalize_hgnc_id(hgnc_id: str) -> str:
    value = hgnc_id.strip()
    if value.upper().startswith("HGNC:"):
        suffix = value.split(":", 1)[1]
        return f"HGNC:{suffix}"
    return f"HGNC:{value}"


def _split_multi_value(value: str | None) -> list[str]:
    raw = (value or "").strip()
    if not raw:
        return []
    parts: list[str] = []
    for chunk in raw.replace(";", "|").split("|"):
        item = chunk.strip()
        if item:
            parts.append(item)
    return parts
