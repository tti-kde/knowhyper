from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

VIEW_VERSION = "entity_view_v1"


def build_entity_view_manifest(
    *,
    source_inputs_fingerprint: str,
    source_core_dir: str,
    dense_modalities: Iterable[str],
    tables: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Assemble the entity-view sidecar manifest (provenance + counts).

    Distinct from the core ``build_manifest.json``: a small bespoke document that records
    which core build this view derives from, the view version, the dense modality set, and
    per-table row counts.
    """
    return {
        "view_version": VIEW_VERSION,
        "source_inputs_fingerprint": source_inputs_fingerprint,
        "source_core_dir": source_core_dir,
        "dense_modalities": list(dense_modalities),
        "counts": {name: len(rows) for name, rows in tables.items()},
    }


def dumps_manifest(manifest: Mapping[str, Any]) -> str:
    """Serialize a manifest to deterministic compact sorted JSON (trailing newline)."""
    return (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
