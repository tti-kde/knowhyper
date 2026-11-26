from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import yaml

if TYPE_CHECKING:
    from pathlib import Path

SourceKind = Literal[
    "primekg_tab",
    "chembl_tsv",
    "drugbank_tsv",
    "uniprot_tsv",
    "hpo_tsv",
    "core_nodes_meta",
]
_VALID_SOURCE_KINDS: frozenset[str] = frozenset(get_args(SourceKind))


@dataclass(frozen=True, slots=True)
class PropertySourceMapping:
    """One source mapping for a property definition."""

    kind: SourceKind
    column: str | None = None
    tab: str | None = None
    meta_key: str | None = None
    field: str | None = None


@dataclass(frozen=True, slots=True)
class PropertyDefinition:
    """Registry entry for a stable ``property_id``."""

    property_id: str
    display_name: str
    dtype: str
    unit: str | None
    node_types: tuple[str, ...]
    sources: tuple[PropertySourceMapping, ...]
    categories: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class NodePropertyRegistry:
    """Loaded ``node_property_registry.yaml``."""

    version: int
    by_id: dict[str, PropertyDefinition]


def load_node_property_registry_bytes() -> bytes:
    return (
        resources.files("primehkg.policy")
        .joinpath("node_property_registry.yaml")
        .read_bytes()
    )


def node_property_registry_fingerprint() -> str:
    return hashlib.sha256(load_node_property_registry_bytes()).hexdigest()


def _parse_sources(raw: Any, *, property_id: str) -> tuple[PropertySourceMapping, ...]:
    if not isinstance(raw, list):
        msg = f"node property registry: {property_id!r} sources must be a list"
        raise TypeError(msg)
    out: list[PropertySourceMapping] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip()
        if kind not in _VALID_SOURCE_KINDS:
            msg = f"node property registry: invalid source kind for {property_id!r}"
            raise ValueError(msg)
        if kind == "core_nodes_meta":
            meta_key = str(item.get("meta_key", "")).strip()
            field = str(item.get("field", "")).strip()
            if not meta_key or not field:
                msg = (
                    f"node property registry: {property_id!r} core_nodes_meta "
                    "requires meta_key and field"
                )
                raise ValueError(msg)
            out.append(
                PropertySourceMapping(
                    kind=cast("SourceKind", kind),
                    meta_key=meta_key,
                    field=field,
                )
            )
            continue
        column = str(item.get("column", "")).strip()
        if not column:
            msg = f"node property registry: invalid source for {property_id!r}"
            raise ValueError(msg)
        tab = item.get("tab")
        tab_s = str(tab).strip() if tab is not None else None
        if kind == "primekg_tab" and not tab_s:
            msg = f"node property registry: {property_id!r} primekg_tab requires tab"
            raise ValueError(msg)
        out.append(
            PropertySourceMapping(
                kind=cast("SourceKind", kind),
                column=column,
                tab=tab_s,
            )
        )
    return tuple(out)


def load_node_property_registry(
    path: Path | None = None,
) -> NodePropertyRegistry:
    raw = load_node_property_registry_bytes() if path is None else path.read_bytes()
    obj: Any = yaml.safe_load(raw)
    if not isinstance(obj, dict):
        msg = "node property registry: top-level must be a mapping"
        raise TypeError(msg)
    version = obj.get("version")
    if not isinstance(version, int) or version < 1:
        msg = "node property registry: version must be an integer >= 1"
        raise ValueError(msg)
    rows = obj.get("properties")
    if not isinstance(rows, list):
        msg = "node property registry: properties must be a list"
        raise TypeError(msg)
    by_id: dict[str, PropertyDefinition] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        property_id = str(row.get("property_id", "")).strip()
        if not property_id:
            msg = "node property registry: property_id must be non-empty"
            raise ValueError(msg)
        if property_id in by_id:
            msg = f"node property registry: duplicate property_id {property_id!r}"
            raise ValueError(msg)
        node_types_raw = row.get("node_types", [])
        if not isinstance(node_types_raw, list) or not node_types_raw:
            msg = f"node property registry: {property_id!r} node_types required"
            raise ValueError(msg)
        node_types = tuple(str(x).strip() for x in node_types_raw if str(x).strip())
        unit_raw = row.get("unit")
        unit = None if unit_raw is None else str(unit_raw)
        categories_raw = row.get("categories")
        categories = (
            tuple(str(c) for c in categories_raw)
            if isinstance(categories_raw, list) and categories_raw
            else None
        )
        by_id[property_id] = PropertyDefinition(
            property_id=property_id,
            display_name=str(row.get("display_name", property_id)),
            dtype=str(row.get("dtype", "float")),
            unit=unit,
            node_types=node_types,
            sources=_parse_sources(row.get("sources"), property_id=property_id),
            categories=categories,
        )
    if not by_id:
        msg = "node property registry: properties must be non-empty"
        raise ValueError(msg)
    return NodePropertyRegistry(version=version, by_id=by_id)


def chembl_column_to_property_id(
    registry: NodePropertyRegistry,
) -> dict[str, str]:
    """Map ChEMBL TSV column name → property_id."""
    out: dict[str, str] = {}
    for prop in registry.by_id.values():
        for src in prop.sources:
            if src.kind == "chembl_tsv" and src.column:
                out[src.column] = prop.property_id
    return out


def drug_tab_column_to_property_id(
    registry: NodePropertyRegistry,
) -> dict[str, str]:
    """Map drug_features.tab column → property_id."""
    out: dict[str, str] = {}
    for prop in registry.by_id.values():
        for src in prop.sources:
            if src.kind == "primekg_tab" and src.tab == "drug_features" and src.column:
                out[src.column] = prop.property_id
    return out


def disease_tab_column_to_property_id(
    registry: NodePropertyRegistry,
) -> dict[str, str]:
    """Map disease_features.tab column → property_id."""
    out: dict[str, str] = {}
    for prop in registry.by_id.values():
        for src in prop.sources:
            if (
                src.kind == "primekg_tab"
                and src.tab == "disease_features"
                and src.column
            ):
                out[src.column] = prop.property_id
    return out


def drugbank_column_to_property_id(
    registry: NodePropertyRegistry,
) -> dict[str, str]:
    """Map drugbank_properties.tsv column name → property_id."""
    out: dict[str, str] = {}
    for prop in registry.by_id.values():
        for src in prop.sources:
            if src.kind == "drugbank_tsv" and src.column:
                out[src.column] = prop.property_id
    return out


def uniprot_column_to_property_id(
    registry: NodePropertyRegistry,
) -> dict[str, str]:
    """Map uniprot.tsv column name → property_id."""
    out: dict[str, str] = {}
    for prop in registry.by_id.values():
        for src in prop.sources:
            if src.kind == "uniprot_tsv" and src.column:
                out[src.column] = prop.property_id
    return out


def hpo_column_to_property_id(
    registry: NodePropertyRegistry,
) -> dict[str, str]:
    """Map hpo_map.tsv column name → property_id."""
    out: dict[str, str] = {}
    for prop in registry.by_id.values():
        for src in prop.sources:
            if src.kind == "hpo_tsv" and src.column:
                out[src.column] = prop.property_id
    return out


def variant_meta_field_to_property_id(
    registry: NodePropertyRegistry,
) -> dict[tuple[str, str], str]:
    """Map (meta_key, field) -> property_id for core_nodes_meta sources."""
    out: dict[tuple[str, str], str] = {}
    for prop in registry.by_id.values():
        for src in prop.sources:
            if src.kind == "core_nodes_meta" and src.meta_key and src.field:
                out[(src.meta_key, src.field)] = prop.property_id
    return out
