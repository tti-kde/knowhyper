from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from primehkg.io.curated_node_ids import normalize_uniprot_accession
from primehkg.nodes.feature_refs import merge_feature_refs, sequence_pointer

if TYPE_CHECKING:
    from collections.abc import Sequence

_REQUIRED_COLUMNS = {"Entry"}


def _strip_header_map(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}
    return {h.strip(): h for h in fieldnames if h is not None}


def read_uniprot_updates(
    path: str | Path, *, delimiter: str = "\t"
) -> dict[str, dict[str, object]]:
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames
        if fieldnames is None or not any((h or "").strip() for h in fieldnames):
            msg = "UniProt updates missing required columns: ['Entry']"
            raise ValueError(msg)

        cols = _strip_header_map(fieldnames)
        missing = _REQUIRED_COLUMNS - set(cols)
        if missing:
            msg = f"UniProt updates missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        out: dict[str, dict[str, object]] = {}
        for row in reader:
            entry = (row.get(cols["Entry"]) or "").strip()
            if not entry:
                continue

            node_id = f"primekg:protein:{entry}"
            protein_names = (row.get(cols.get("Protein names", "")) or "").strip()
            gene_names = (row.get(cols.get("Gene Names", "")) or "").strip()
            gene_names_primary = (
                row.get(cols.get("Gene Names (primary)", "")) or ""
            ).strip()
            name = protein_names or entry
            xrefs = _build_xrefs(entry, row.get(cols.get("Cross-reference (HGNC)", "")))
            feature_refs: dict[str, object] = {}
            if "Sequence" in cols:
                sequence = (row.get(cols["Sequence"]) or "").strip()
                if sequence:
                    feature_refs = merge_feature_refs(
                        {},
                        {
                            "sequence": [
                                sequence_pointer(
                                    source="UniProt",
                                    source_version=p.name,
                                    sequence_type="protein",
                                )
                            ]
                        },
                    )
            out[node_id] = {
                "names": [
                    {
                        "name": name,
                        "source": "UniProt",
                        "source_version": p.name,
                        "evidence": "local_uniprot_export",
                    }
                ],
                "xrefs": xrefs,
                "feature_refs": feature_refs,
                "source": "UniProt",
                "source_version": p.name,
                "evidence": "local_uniprot_export",
            }
            _ = (gene_names, gene_names_primary)
        return out


def read_uniprot_geneid_map(
    path: str | Path, *, delimiter: str = "\t"
) -> dict[str, str]:
    """Map UniProt accession -> NCBI GeneID from a UniProt export TSV.

    Reads the `Entry` and `GeneID` columns. Rows with empty Entry or empty GeneID are
    skipped. Multiple GeneIDs (";"-separated) -> the first. Keys are normalized ACs
    (uppercased). Returns {} if the `GeneID` column is absent. Raises ValueError if `Entry`
    is absent.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        cols = _strip_header_map(reader.fieldnames)
        if "Entry" not in cols:
            msg = "UniProt updates missing required columns: ['Entry']"
            raise ValueError(msg)
        if "GeneID" not in cols:
            return {}

        out: dict[str, str] = {}
        for row in reader:
            entry = (row.get(cols["Entry"]) or "").strip()
            gene_id = (row.get(cols["GeneID"]) or "").strip()
            if not entry or not gene_id:
                continue
            first = gene_id.split(";")[0].strip()
            if not first:
                continue
            out[normalize_uniprot_accession(entry)] = first
        return out


def _build_xrefs(entry: str, hgnc_value: str | None) -> dict[str, list[str]]:
    xrefs: dict[str, list[str]] = {"uniprot": [entry]}
    hgnc_ids = _split_multi_value(hgnc_value)
    if hgnc_ids:
        xrefs["hgnc"] = hgnc_ids
    return xrefs


def _split_multi_value(value: str | None) -> list[str]:
    raw = (value or "").strip()
    if not raw:
        return []
    values: list[str] = []
    for chunk in raw.replace(";", "|").split("|"):
        item = chunk.strip()
        if item:
            values.append(item)
    return sorted(dict.fromkeys(values))
