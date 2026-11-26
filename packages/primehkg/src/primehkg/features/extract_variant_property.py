from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from primehkg.features.property_rows import property_row
from primehkg.policy.node_property_registry import (
    NodePropertyRegistry,
    load_node_property_registry,
    variant_meta_field_to_property_id,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

# Chromosome ordering is a universal biological constant -> code, not registry.
# ClinVar's `chromosome` field uses bare designators ("17", "X", "MT") with no "chr" prefix.
_CHROM_ORDINAL: dict[str, int] = {str(i): i for i in range(1, 23)}
_CHROM_ORDINAL.update({"X": 23, "Y": 24, "MT": 25})


def _loads_source_meta(raw: Any) -> dict[str, Any]:
    """Parse a node's source_meta (JSON string on disk, dict in memory)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return obj if isinstance(obj, dict) else {}
    return {}


def read_variant_property_rows(
    core_nodes: Iterable[dict[str, Any]],
    *,
    allowed_node_ids: frozenset[str] | None = None,
    registry: NodePropertyRegistry | None = None,
    collect_unknowns: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, set[str]]]:
    """Emit GNN-learnable property rows from variant nodes' source_meta.clinvar_variant.

    Reads core ``nodes`` rows directly (no external file). For each
    ``node_type == "variant"`` node, parses ``source_meta["clinvar_variant"]`` and emits up to
    4 ordinal/int property rows. Unknown variant_type / chromosome values and missing/non-numeric
    coords are skipped (non-fatal); set ``collect_unknowns=True`` to also receive the skipped values.
    """
    reg = registry if registry is not None else load_node_property_registry()
    meta_to_pid = variant_meta_field_to_property_id(reg)
    type_pid = meta_to_pid.get(("clinvar_variant", "type"))
    chrom_pid = meta_to_pid.get(("clinvar_variant", "chromosome"))
    start_pid = meta_to_pid.get(("clinvar_variant", "start"))
    stop_pid = meta_to_pid.get(("clinvar_variant", "stop"))

    type_categories = reg.by_id[type_pid].categories if type_pid else None
    # 1-based ordinal: 0 is reserved as the property-vector fill for a MISSING value
    # (build_property_vectors fills absent properties with 0.0). Starting categories at 1
    # keeps a missing/unknown variant_type (-> 0.0) distinguishable from the first real
    # category, which would otherwise collide with the most common type (SNV).
    type_index = (
        {c: i for i, c in enumerate(type_categories, start=1)}
        if type_categories
        else {}
    )

    rows_out: list[dict[str, Any]] = []
    unknowns: dict[str, set[str]] = {"variant_type": set(), "variant_chromosome": set()}

    def _emit(node_id: str, pid: str | None, value_numeric: float) -> None:
        if pid is None:
            return
        prop = reg.by_id[pid]
        rows_out.append(
            property_row(
                node_id=node_id,
                property_id=pid,
                value_numeric=value_numeric,
                dtype=prop.dtype,
                source="ClinVar",
                source_version="clinvar_variant_v1",
                unit=prop.unit,
            )
        )

    for node in core_nodes:
        if not isinstance(node, dict) or node.get("node_type") != "variant":
            continue
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        if allowed_node_ids is not None and node_id not in allowed_node_ids:
            continue
        cv = _loads_source_meta(node.get("source_meta")).get("clinvar_variant")
        if not isinstance(cv, dict):
            continue

        t = str(cv.get("type", "")).strip()
        if t and type_pid:
            if t in type_index:
                _emit(node_id, type_pid, float(type_index[t]))
            else:
                unknowns["variant_type"].add(t)

        chrom = str(cv.get("chromosome", "")).strip()
        if chrom and chrom_pid:
            if chrom in _CHROM_ORDINAL:
                _emit(node_id, chrom_pid, float(_CHROM_ORDINAL[chrom]))
            else:
                unknowns["variant_chromosome"].add(chrom)

        for raw_key, pid in (("start", start_pid), ("stop", stop_pid)):
            raw = str(cv.get(raw_key, "")).strip()
            if not raw or pid is None:
                continue
            try:
                _emit(node_id, pid, float(int(raw)))
            except (ValueError, TypeError):
                continue

    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["property_id"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    if collect_unknowns:
        return rows_out, unknowns
    return rows_out
