"""Compose / write / load the ``chemical_xref_v1`` sidecar."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from primehkg.io.build_artifacts import (
    CHEMICAL_XREF_FILE_VERSION,
    CHEMICAL_XREF_MAPPINGS_FILENAME,
    CHEMICAL_XREF_META_FILENAME,
    CHEMICAL_XREF_SCHEMA_VERSION,
    CHEMICAL_XREF_SIDECAR_NAME,
    build_snapshot_sidecar_dir,
    dataset_mirror_sidecar_dir,
    relative_snapshot_sidecar_path,
    resolve_artifacts_root,
)
from primehkg.io.build_manifest import merge_manifest_sidecars, sha256_file
from primehkg.io.curated_node_ids import curated_chebi_node_id
from primehkg.io.curated_sources import (
    index_ctd_mesh_pubchem_v2,
    index_unichem_pubchem_xref_v2,
    load_ctd_mesh_pubchem_v2,
    load_unichem_pubchem_xref_v2,
)
from primehkg.linking.pipeline import node_type_of

_MAPPING_COLUMNS = [
    "external_namespace",
    "external_id",
    "node_id",
    "node_type",
    "mapping_path",
    "mapping_relation",
    "mapping_source",
    "mapping_source_version",
    "is_live_node",
    "ambiguity_count",
]


class ChemicalXrefFingerprintError(ValueError):
    """Raised when chemical xref sidecar fingerprint checks fail."""


class ChemicalXrefFingerprintMissingError(ChemicalXrefFingerprintError):
    """Raised when a sidecar is missing ``inputs_fingerprint``."""

    def __init__(self) -> None:
        super().__init__("chemical_xref_v1 missing inputs_fingerprint")


class ChemicalXrefFingerprintMismatchError(ChemicalXrefFingerprintError):
    """Raised when sidecar fingerprint does not match the expected value."""

    def __init__(self) -> None:
        super().__init__("chemical_xref_v1 inputs_fingerprint mismatch")


def _terminal_node_id(namespace: str, external_id: str) -> str | None:
    if namespace == "drugbank":
        return f"primekg:drug:{external_id}"
    if namespace == "chebi":
        return curated_chebi_node_id(external_id)
    return None


def _source_stamp(rows: list[dict[str, str]], *, default: str) -> tuple[str, str]:
    if not rows:
        return default, "unknown"
    source = (rows[0].get("source") or default).strip() or default
    version = (rows[0].get("source_version") or "unknown").strip() or "unknown"
    return source, version


def compose_chemical_xref_mapping_rows(
    *,
    mesh_rows: list[dict[str, str]],
    unichem_rows: list[dict[str, str]],
    live_node_ids: set[str],
) -> list[dict[str, Any]]:
    """Expand lossless v2 tables into flat mapping rows (no first-win selection)."""
    mesh_to_cids = index_ctd_mesh_pubchem_v2(mesh_rows)
    cid_xrefs = index_unichem_pubchem_xref_v2(unichem_rows)
    ctd_source, ctd_version = _source_stamp(mesh_rows, default="CTD")
    uc_source, uc_version = _source_stamp(unichem_rows, default="UniChem")

    raw: list[dict[str, Any]] = []

    def _add(
        *,
        external_namespace: str,
        external_id: str,
        node_id: str,
        mapping_path: str,
        mapping_relation: str,
        mapping_source: str,
        mapping_source_version: str,
    ) -> None:
        ntype = node_type_of(node_id) or ""
        raw.append(
            {
                "external_namespace": external_namespace,
                "external_id": external_id,
                "node_id": node_id,
                "node_type": ntype,
                "mapping_path": mapping_path,
                "mapping_relation": mapping_relation,
                "mapping_source": mapping_source,
                "mapping_source_version": mapping_source_version,
                "is_live_node": node_id in live_node_ids,
            }
        )

    for cid, pairs in cid_xrefs.items():
        for namespace, external_id in pairs:
            nid = _terminal_node_id(namespace, external_id)
            if nid is None:
                continue
            relation = (
                "unichem_pubchem_to_drugbank"
                if namespace == "drugbank"
                else "unichem_pubchem_to_chebi"
            )
            _add(
                external_namespace="pubchem",
                external_id=cid,
                node_id=nid,
                mapping_path=f"pubchem:{cid}->{namespace}:{external_id}->{nid}",
                mapping_relation=relation,
                mapping_source=uc_source,
                mapping_source_version=uc_version,
            )
            _add(
                external_namespace=namespace,
                external_id=external_id,
                node_id=nid,
                mapping_path=f"{namespace}:{external_id}->{nid}",
                mapping_relation="direct_xref",
                mapping_source=uc_source,
                mapping_source_version=uc_version,
            )

    for mesh, cids in mesh_to_cids.items():
        for cid in cids:
            for namespace, external_id in cid_xrefs.get(cid) or []:
                nid = _terminal_node_id(namespace, external_id)
                if nid is None:
                    continue
                path = f"mesh:{mesh}->pubchem:{cid}->{namespace}:{external_id}->{nid}"
                _add(
                    external_namespace="mesh",
                    external_id=mesh,
                    node_id=nid,
                    mapping_path=path,
                    mapping_relation="composed_bridge",
                    mapping_source=f"{ctd_source}+{uc_source}",
                    mapping_source_version=f"{ctd_version}|{uc_version}",
                )

    # Dedup exact mapping rows (same external key + node + path + relation).
    dedup_keys: set[tuple[str, ...]] = set()
    unique: list[dict[str, Any]] = []
    for row in raw:
        key = (
            str(row["external_namespace"]),
            str(row["external_id"]),
            str(row["node_id"]),
            str(row["mapping_path"]),
            str(row["mapping_relation"]),
        )
        if key in dedup_keys:
            continue
        dedup_keys.add(key)
        unique.append(row)

    live_by_ext: dict[tuple[str, str], set[str]] = {}
    for row in unique:
        if row["is_live_node"]:
            live_by_ext.setdefault(
                (str(row["external_namespace"]), str(row["external_id"])), set()
            ).add(str(row["node_id"]))

    for row in unique:
        key = (str(row["external_namespace"]), str(row["external_id"]))
        row["ambiguity_count"] = len(live_by_ext.get(key, set()))

    unique.sort(
        key=lambda r: (
            str(r["external_namespace"]),
            str(r["external_id"]),
            str(r["node_id"]),
            str(r["mapping_path"]),
            str(r["mapping_relation"]),
        )
    )
    return unique


def write_chemical_xref_sidecar(
    out_dir: str | Path,
    *,
    rows: list[dict[str, Any]],
    inputs_fingerprint: str,
    source_digests: dict[str, str],
) -> Path:
    """Write ``meta.json`` + ``mappings.parquet`` under ``out_dir``."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    live_terminal = sum(1 for r in rows if r.get("is_live_node"))
    ambiguous_ext = {
        (r["external_namespace"], r["external_id"])
        for r in rows
        if int(r.get("ambiguity_count") or 0) > 1
    }
    meta = {
        "file_version": CHEMICAL_XREF_FILE_VERSION,
        "inputs_fingerprint": str(inputs_fingerprint),
        "schema_version": CHEMICAL_XREF_SCHEMA_VERSION,
        "source_digests": dict(sorted(source_digests.items())),
        "counts": {
            "rows": len(rows),
            "external_ids": len(
                {(r["external_namespace"], r["external_id"]) for r in rows}
            ),
            "live_terminal_rows": live_terminal,
            "ambiguous_external_ids": len(ambiguous_ext),
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    (dest / CHEMICAL_XREF_META_FILENAME).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    schema = pa.schema(
        [
            ("external_namespace", pa.string()),
            ("external_id", pa.string()),
            ("node_id", pa.string()),
            ("node_type", pa.string()),
            ("mapping_path", pa.string()),
            ("mapping_relation", pa.string()),
            ("mapping_source", pa.string()),
            ("mapping_source_version", pa.string()),
            ("is_live_node", pa.bool_()),
            ("ambiguity_count", pa.int32()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()
    pq.write_table(table, dest / CHEMICAL_XREF_MAPPINGS_FILENAME)
    return dest


def load_chemical_xref_sidecar(
    sidecar_dir: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load meta + mapping rows; raise on fingerprint mismatch when expected."""
    root = Path(sidecar_dir)
    meta_path = root / CHEMICAL_XREF_META_FILENAME
    map_path = root / CHEMICAL_XREF_MAPPINGS_FILENAME
    if not meta_path.is_file() or not map_path.is_file():
        msg = f"chemical_xref_v1 incomplete under {root}"
        raise FileNotFoundError(msg)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    loaded_fp = meta.get("inputs_fingerprint")
    if expected_fingerprint is not None:
        if loaded_fp is None:
            raise ChemicalXrefFingerprintMissingError
        if str(loaded_fp) != str(expected_fingerprint):
            raise ChemicalXrefFingerprintMismatchError
    table = pq.read_table(map_path)
    rows = table.to_pylist()
    return meta, rows


def build_chemical_xref_sidecar_from_sources(
    *,
    ctd_mesh_pubchem_v2: str | Path,
    unichem_pubchem_xref_v2: str | Path,
    live_node_ids: set[str],
    inputs_fingerprint: str,
    out_dir: str | Path,
) -> Path:
    """Compose from v2 TSVs and write a sidecar directory."""
    mesh_path = Path(ctd_mesh_pubchem_v2)
    unichem_path = Path(unichem_pubchem_xref_v2)
    mesh_rows = load_ctd_mesh_pubchem_v2(mesh_path)
    unichem_rows = load_unichem_pubchem_xref_v2(unichem_path)
    rows = compose_chemical_xref_mapping_rows(
        mesh_rows=mesh_rows,
        unichem_rows=unichem_rows,
        live_node_ids=live_node_ids,
    )
    digests = {
        "ctd_mesh_pubchem_v2": sha256_file(mesh_path),
        "unichem_pubchem_xref_v2": sha256_file(unichem_path),
    }
    return write_chemical_xref_sidecar(
        out_dir,
        rows=rows,
        inputs_fingerprint=inputs_fingerprint,
        source_digests=digests,
    )


def emit_chemical_xref_sidecar_for_build(
    *,
    ctd_mesh_pubchem_v2: str | Path,
    unichem_pubchem_xref_v2: str | Path,
    live_node_ids: set[str],
    inputs_fingerprint: str,
    out_dir: str | Path,
    artifacts_root: Path | None = None,
    merge_manifest: bool = True,
) -> dict[str, str]:
    """Write canonical + dataset-mirror sidecar; optionally patch build_manifest."""
    canonical = build_snapshot_sidecar_dir(
        inputs_fingerprint,
        CHEMICAL_XREF_SIDECAR_NAME,
        artifacts_root=artifacts_root,
    )
    build_chemical_xref_sidecar_from_sources(
        ctd_mesh_pubchem_v2=ctd_mesh_pubchem_v2,
        unichem_pubchem_xref_v2=unichem_pubchem_xref_v2,
        live_node_ids=live_node_ids,
        inputs_fingerprint=inputs_fingerprint,
        out_dir=canonical,
    )
    out = Path(out_dir)
    local = out / dataset_mirror_sidecar_dir(CHEMICAL_XREF_SIDECAR_NAME)
    if local.resolve() != canonical.resolve():
        if local.exists():
            shutil.rmtree(local)
        shutil.copytree(canonical, local)
    dataset_rel = str(dataset_mirror_sidecar_dir(CHEMICAL_XREF_SIDECAR_NAME))
    if merge_manifest:
        merge_manifest_sidecars(out, {CHEMICAL_XREF_SIDECAR_NAME: dataset_rel})
    return {
        "artifacts": relative_snapshot_sidecar_path(
            inputs_fingerprint,
            CHEMICAL_XREF_SIDECAR_NAME,
            artifacts_root=resolve_artifacts_root(artifacts_root),
        ),
        "dataset_dir": dataset_rel,
    }
