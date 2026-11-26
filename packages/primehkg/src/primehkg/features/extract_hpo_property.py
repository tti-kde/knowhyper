from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from primehkg.features.extract import _strip_header_map
from primehkg.features.property_rows import property_row
from primehkg.features.tab_parse import parse_tab_numeric_cell
from primehkg.policy.node_property_registry import (
    NodePropertyRegistry,
    hpo_column_to_property_id,
    load_node_property_registry,
)


def read_hpo_property_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
    registry: NodePropertyRegistry | None = None,
) -> list[dict[str, Any]]:
    """Emit HPO map TSV numeric columns as ``node_features_property`` rows."""
    reg = registry if registry is not None else load_node_property_registry()
    col_to_pid = hpo_column_to_property_id(reg)
    if not col_to_pid:
        return []

    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "node_id" not in cols:
            return rows_out
        source_version = p.name
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            if not node_id:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            for col_name, property_id in col_to_pid.items():
                col = cols.get(col_name)
                if col is None:
                    continue
                raw = (row.get(col) or "").strip()
                if not raw:
                    continue
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
                        source="HPO",
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
