from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from primehkg.features.extract import _strip_header_map, _text_row
from primehkg.features.property_rows import property_row
from primehkg.features.tab_parse import parse_tab_numeric_cell
from primehkg.policy.node_property_registry import (
    NodePropertyRegistry,
    drug_tab_column_to_property_id,
    load_node_property_registry,
)

MODALITY_COLUMNS = (
    ("description", "description"),
    ("indication", "indication"),
    ("mechanism_of_action", "mechanism_of_action"),
    ("half_life", "half_life"),
    ("protein_binding", "protein_binding"),
    ("pharmacodynamics", "pharmacodynamics"),
    ("state", "state"),
    ("atc_1", "atc_1"),
    ("atc_2", "atc_2"),
    ("atc_3", "atc_3"),
    ("atc_4", "atc_4"),
    ("category", "category"),
    ("group", "group"),
    ("pathway", "pathway"),
)


def read_drug_features_text_rows(
    path: str | Path,
    *,
    index_to_node_id: dict[int, str],
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Emit one text row per non-empty drug tab field (never pick-one)."""
    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "node_index" not in cols:
            return rows_out
        source_version = p.name
        for row in reader:
            idx_raw = (row.get(cols["node_index"]) or "").strip()
            if not idx_raw:
                continue
            try:
                idx = int(idx_raw)
            except ValueError:
                continue
            node_id = index_to_node_id.get(idx)
            if not node_id:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            for col_name, modality in MODALITY_COLUMNS:
                col = cols.get(col_name)
                if col is None:
                    continue
                text = (row.get(col) or "").strip().strip('"')
                if not text:
                    continue
                rows_out.append(
                    _text_row(
                        node_id=node_id,
                        modality=modality,
                        text=text,
                        source="PrimeKG",
                        source_version=source_version,
                        source_field=col_name,
                        payload_class=(
                            "semantic_definition"
                            if col_name == "description"
                            else "short_label"
                        ),
                    )
                )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["modality"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def read_drug_features_property_rows(
    path: str | Path,
    *,
    index_to_node_id: dict[int, str],
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
    registry: NodePropertyRegistry | None = None,
) -> list[dict[str, Any]]:
    """Emit numeric drug tab columns as ``node_features_property`` rows."""
    reg = registry if registry is not None else load_node_property_registry()
    col_to_pid = drug_tab_column_to_property_id(reg)
    if not col_to_pid:
        return []

    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "node_index" not in cols:
            return rows_out
        source_version = p.name
        for row in reader:
            idx_raw = (row.get(cols["node_index"]) or "").strip()
            if not idx_raw:
                continue
            try:
                idx = int(idx_raw)
            except ValueError:
                continue
            node_id = index_to_node_id.get(idx)
            if not node_id:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            for col_name, property_id in col_to_pid.items():
                col = cols.get(col_name)
                if col is None:
                    continue
                raw = (row.get(col) or "").strip()
                value = parse_tab_numeric_cell(raw)
                if value is None:
                    continue
                prop_def = reg.by_id[property_id]
                rows_out.append(
                    property_row(
                        node_id=node_id,
                        property_id=property_id,
                        value_numeric=value,
                        dtype=prop_def.dtype,
                        source="PrimeKG",
                        source_version=source_version,
                        unit=prop_def.unit,
                    )
                )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["property_id"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out
