"""Optional materialization of unambiguous MeSH ids into ``nodes.xrefs``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from primehkg.io.hf_dataset import normalize_cell
from primehkg.nodes.merge import merge_xrefs
from primehkg.nodes.normalize import normalize_xref_id

if TYPE_CHECKING:
    from collections.abc import Collection

    from datasets import Dataset

    from primehkg.linking.chemical_xref_index import ChemicalXrefIndex


def _coerce_xrefs(value: Any) -> dict[str, list[str]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
    if not isinstance(value, dict):
        return {}
    out: dict[str, list[str]] = {}
    for ns, ids in value.items():
        if not isinstance(ns, str):
            continue
        if isinstance(ids, list):
            out[ns] = [str(i) for i in ids if isinstance(i, str | int)]
        elif isinstance(ids, str):
            out[ns] = [ids]
    return out


def select_unambiguous_mesh_targets(
    index: ChemicalXrefIndex,
    *,
    allowed_node_types: Collection[str] | None = ("drug", "chebi"),
) -> dict[str, str]:
    """Return ``{mesh_id: node_id}`` for MeSH keys with exactly one live terminal.

    Ambiguous MeSH ids (multiple live drug/chebi candidates) are omitted — those
    stay sidecar-only per the chemical xref ADR.
    """
    mesh_ids = index.iter_external_ids("mesh")
    targets: dict[str, str] = {}
    for mesh in mesh_ids:
        cands = index.lookup(
            mesh,
            namespace="mesh",
            allowed_node_types=allowed_node_types,
            live_only=True,
        )
        if len(cands) != 1:
            continue
        targets[mesh] = cands[0].node_id
    return targets


def materialize_unambiguous_mesh_xrefs(
    nodes: Dataset,
    index: ChemicalXrefIndex,
    *,
    allowed_node_types: Collection[str] | None = ("drug", "chebi"),
    source: str = "chemical_xref_v1",
    source_version: str = "chemical_xref_mappings_v1",
) -> tuple[Dataset, dict[str, int]]:
    """Merge unambiguous MeSH ids into ``nodes.xrefs.mesh`` (in-memory Dataset).

    Does not rewrite hyperedges or ``node_id``. Returns ``(updated_nodes, stats)``.
    """
    targets = select_unambiguous_mesh_targets(
        index, allowed_node_types=allowed_node_types
    )
    by_node: dict[str, list[str]] = {}
    for mesh, nid in targets.items():
        by_node.setdefault(nid, []).append(normalize_xref_id("mesh", mesh))

    if not by_node:
        return nodes, {"materialized": 0, "nodes_touched": 0}

    rows = [dict(r) for r in nodes]
    touched = 0
    for row in rows:
        nid = str(row.get("node_id") or "")
        mesh_ids = by_node.get(nid)
        if not mesh_ids:
            continue
        raw_xrefs = row.get("xrefs")
        base = _coerce_xrefs(raw_xrefs)
        merged, _prov = merge_xrefs(
            base=base,
            incoming={"mesh": mesh_ids},
            incoming_source=source,
            incoming_source_version=source_version,
            evidence="unambiguous_chemical_xref_bridge",
        )
        # nodes.xrefs is JSON-string-encoded on disk (see
        # primehkg.io.hf_dataset.normalize_cell); preserve that encoding so
        # touched and untouched rows share one Arrow type when rebuilding the
        # Dataset below (Arrow rejects mixed struct/non-struct columns).
        row["xrefs"] = normalize_cell(merged) if isinstance(raw_xrefs, str) else merged
        touched += 1

    from datasets import Dataset as HFDataset

    return HFDataset.from_list(rows), {
        "materialized": len(targets),
        "nodes_touched": touched,
    }


def materialize_unambiguous_mesh_xrefs_on_disk(
    core_dir: str | Path,
    index: ChemicalXrefIndex,
    *,
    allowed_node_types: Collection[str] | None = ("drug", "chebi"),
) -> dict[str, int]:
    """Load core nodes, materialize, rewrite the ``nodes`` split in place.

    HuggingFace ``DatasetDict.save_to_disk`` refuses to overwrite the same path
    that was loaded. Only the ``nodes/`` subdirectory is replaced; ``sidecars/``,
    manifests, and other splits are left untouched.
    """
    import shutil
    import tempfile

    from datasets import DatasetDict, load_from_disk

    root = Path(core_dir)
    loaded = load_from_disk(str(root))
    if not isinstance(loaded, DatasetDict):
        msg = f"expected DatasetDict under {root}"
        raise TypeError(msg)
    nodes = loaded["nodes"]
    updated, stats = materialize_unambiguous_mesh_xrefs(
        nodes, index, allowed_node_types=allowed_node_types
    )
    if stats["nodes_touched"] == 0:
        return stats

    nodes_dest = root / "nodes"
    with tempfile.TemporaryDirectory(prefix="primehkg-nodes-mat-") as td:
        staging = Path(td) / "nodes"
        updated.save_to_disk(str(staging))
        bak = root / "nodes.__materialize_bak__"
        if bak.exists():
            shutil.rmtree(bak)
        nodes_dest.rename(bak)
        try:
            shutil.move(str(staging), str(nodes_dest))
        except Exception:
            if nodes_dest.exists():
                shutil.rmtree(nodes_dest, ignore_errors=True)
            bak.rename(nodes_dest)
            raise
        shutil.rmtree(bak, ignore_errors=True)
    return stats
