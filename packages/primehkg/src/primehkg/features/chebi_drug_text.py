from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from primehkg.features.extract import _text_row
from primehkg.io.curated_node_ids import curated_chebi_node_id
from primehkg.io.curated_sources import load_unichem_cid_map


@dataclass(frozen=True, slots=True)
class ChebiDrugProjectionStats:
    """Counts from unambiguous UniChem drugbank↔chebi ChEBI definition projection."""

    kept: int
    dropped_ambiguous_drugbank: int
    dropped_ambiguous_chebi: int
    dropped_missing_definition: int
    dropped_missing_endpoint: int


def _load_chebi_definitions(path: Path) -> dict[str, str]:
    definitions: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            return definitions
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        if "chebi_node_id" not in cols or "definition" not in cols:
            return definitions
        for row in reader:
            node_id = (row.get(cols["chebi_node_id"]) or "").strip()
            definition = (row.get(cols["definition"]) or "").strip()
            if node_id and definition:
                definitions[node_id] = definition
    return definitions


def project_chebi_definitions_to_drugs(
    *,
    chebi_definitions_tsv: Path,
    unichem_cid_map_tsv: Path,
    allowed_node_ids: frozenset[str] | None = None,
) -> tuple[list[dict[str, Any]], ChebiDrugProjectionStats]:
    """Project ChEBI definitions onto drug nodes via unambiguous UniChem pairs."""
    cid_map = load_unichem_cid_map(unichem_cid_map_tsv)
    definitions = _load_chebi_definitions(Path(chebi_definitions_tsv))

    drugbank_to_chebi: dict[str, set[str]] = defaultdict(set)
    chebi_to_drugbank: dict[str, set[str]] = defaultdict(set)
    pairs: set[tuple[str, str]] = set()

    for entry in cid_map.values():
        drugbank = entry.get("drugbank")
        chebi = entry.get("chebi")
        if not drugbank or not chebi:
            continue
        drugbank_to_chebi[drugbank].add(chebi)
        chebi_to_drugbank[chebi].add(drugbank)
        pairs.add((drugbank, chebi))

    rows_out: list[dict[str, Any]] = []
    kept = 0
    dropped_ambiguous_drugbank = 0
    dropped_ambiguous_chebi = 0
    dropped_missing_definition = 0
    dropped_missing_endpoint = 0
    source_version = Path(chebi_definitions_tsv).name

    for drugbank, chebi in sorted(pairs):
        if len(drugbank_to_chebi[drugbank]) != 1:
            dropped_ambiguous_drugbank += 1
            continue
        if len(chebi_to_drugbank[chebi]) != 1:
            dropped_ambiguous_chebi += 1
            continue

        chebi_node_id = curated_chebi_node_id(chebi)
        drug_node_id = f"primekg:drug:{drugbank}"
        definition = definitions.get(chebi_node_id)
        if definition is None:
            dropped_missing_definition += 1
            continue
        if allowed_node_ids is not None and drug_node_id not in allowed_node_ids:
            dropped_missing_endpoint += 1
            continue

        rows_out.append(
            _text_row(
                node_id=drug_node_id,
                modality="definition",
                text=definition,
                source="chebi",
                source_version=source_version,
                source_field="definition",
                payload_class="semantic_definition",
            )
        )
        kept += 1

    stats = ChebiDrugProjectionStats(
        kept=kept,
        dropped_ambiguous_drugbank=dropped_ambiguous_drugbank,
        dropped_ambiguous_chebi=dropped_ambiguous_chebi,
        dropped_missing_definition=dropped_missing_definition,
        dropped_missing_endpoint=dropped_missing_endpoint,
    )
    return rows_out, stats
