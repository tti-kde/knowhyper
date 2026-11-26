from __future__ import annotations

from typing import Any


def property_row(
    *,
    node_id: str,
    property_id: str,
    value_numeric: float | None = None,
    value_text: str | None = None,
    dtype: str,
    source: str,
    source_version: str,
    unit: str | None = None,
    license_tag: str | None = None,
) -> dict[str, Any]:
    """Build one ``node_features_property`` row."""
    has_num = value_numeric is not None
    has_txt = value_text is not None and str(value_text).strip() != ""
    if not has_num and not has_txt:
        msg = "property_row requires value_numeric and/or value_text"
        raise ValueError(msg)
    row: dict[str, Any] = {
        "node_id": node_id,
        "property_id": property_id,
        "dtype": dtype,
        "source": source,
        "source_version": source_version,
    }
    if has_num:
        assert value_numeric is not None
        row["value_numeric"] = float(value_numeric)
    if has_txt:
        row["value_text"] = str(value_text).strip()
    if unit is not None:
        row["unit"] = unit
    if license_tag is not None:
        row["license_tag"] = license_tag
    return row
