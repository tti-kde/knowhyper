from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class PropertyVector:
    """Dense property-vector payload for one node before encoding."""

    node_id: str
    property_ids: tuple[str, ...]
    values: list[float]
    missing_mask: list[float]
    source: str
    source_version: str


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: float) -> str:
    return f"{value:.8g}"


def property_vector_input_text(
    *,
    property_ids: tuple[str, ...],
    values: list[float],
    missing_mask: list[float],
) -> str:
    parts = [
        f"{property_id}={_format_float(value)}"
        for property_id, value in zip(property_ids, values, strict=True)
    ]
    missing = ",".join(_format_float(value) for value in missing_mask)
    return ";".join([*parts, f"missing={missing}"])


def build_property_vectors(
    *,
    nodes: Iterable[dict[str, Any]],
    property_rows: Iterable[dict[str, Any]],
    registry_order: tuple[str, ...],
) -> list[PropertyVector]:
    node_ids = [str(row["node_id"]) for row in nodes if row.get("node_id")]
    values_by_node: dict[str, dict[str, float]] = {node_id: {} for node_id in node_ids}
    source_by_node: dict[str, str] = {}
    version_by_node: dict[str, str] = {}
    allowed_properties = set(registry_order)

    for row in property_rows:
        node_id = str(row.get("node_id", ""))
        property_id = str(row.get("property_id", ""))
        if node_id not in values_by_node or property_id not in allowed_properties:
            continue
        value = _as_float(row.get("value_numeric"))
        if value is None:
            value = _as_float(row.get("value_text"))
        if value is None:
            continue
        values_by_node[node_id][property_id] = value
        source = row.get("source")
        source_version = row.get("source_version")
        if isinstance(source, str) and source:
            source_by_node.setdefault(node_id, source)
        if isinstance(source_version, str) and source_version:
            version_by_node.setdefault(node_id, source_version)

    vectors: list[PropertyVector] = []
    for node_id in node_ids:
        node_values = values_by_node[node_id]
        if not node_values:
            continue
        values: list[float] = []
        missing_mask: list[float] = []
        for property_id in registry_order:
            if property_id in node_values:
                values.append(node_values[property_id])
                missing_mask.append(0.0)
            else:
                values.append(0.0)
                missing_mask.append(1.0)
        vectors.append(
            PropertyVector(
                node_id=node_id,
                property_ids=registry_order,
                values=values,
                missing_mask=missing_mask,
                source=source_by_node.get(node_id, "node_features_property"),
                source_version=version_by_node.get(node_id, "mixed"),
            )
        )
    return vectors
