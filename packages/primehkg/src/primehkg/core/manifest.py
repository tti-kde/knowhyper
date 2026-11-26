"""Read and interpret ``build_manifest.json`` from a PrimeHKG build directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, cast

from primehkg.core.exceptions import CoreDatasetLoadError


class BuildManifest(TypedDict, total=False):
    """Shape of ``build_manifest.json`` (subset used by consumers)."""

    schema_version: str
    inputs_fingerprint: str
    builders: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    outputs: dict[str, int]
    sidecars: dict[str, Any]
    build_profile: str


def read_build_manifest(root: str | Path) -> BuildManifest:
    """Load ``build_manifest.json`` from a dataset root directory."""
    path = Path(root) / "build_manifest.json"
    if not path.is_file():
        msg = f"Missing build_manifest.json under {root!r}"
        raise CoreDatasetLoadError(msg)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "Invalid build_manifest.json: root must be an object"
        raise CoreDatasetLoadError(msg)
    fp = raw.get("inputs_fingerprint")
    if not isinstance(fp, str) or not fp.strip():
        msg = "Invalid build_manifest.json: missing inputs_fingerprint"
        raise CoreDatasetLoadError(msg)
    return raw  # type: ignore[return-value]


def is_enriched(manifest: BuildManifest) -> bool:
    """Return True when manifest records an enrichment profile builder."""
    for entry in manifest.get("builders") or []:
        if isinstance(entry, dict) and entry.get("name") == "enrichment_profile":
            return True
    return False


def _node_id_local_segment(node_id: str) -> str:
    return node_id.rsplit(":", 1)[-1]


def nodes_have_linking_fields(nodes: object, *, max_rows: int = 256) -> bool:
    """Return True when ``nodes`` rows carry xref/label data usable for linking.

    Used as a Hub fallback when ``build_manifest.json`` is missing or records a
    minimal build while published ``nodes`` still contain enrichment fields.
    """
    from collections.abc import Iterable

    if not isinstance(nodes, Iterable) or isinstance(nodes, (str, bytes, dict)):
        return False
    for i, row in enumerate(nodes):
        if i >= max_rows:
            break
        if not isinstance(row, dict):
            continue
        row_dict = cast("dict[str, object]", row)
        xrefs = row_dict.get("xrefs")
        if isinstance(xrefs, dict) and any(
            isinstance(ids, list) and ids for ids in xrefs.values()
        ):
            return True
        node_id = row_dict.get("node_id")
        name = row_dict.get("name")
        if isinstance(node_id, str) and isinstance(name, str):
            local = _node_id_local_segment(node_id.strip())
            if name.strip() and name.strip() != local:
                return True
        syns = row_dict.get("synonyms")
        if isinstance(syns, list) and any(
            isinstance(syn, str) and syn.strip() for syn in syns
        ):
            return True
    return False
