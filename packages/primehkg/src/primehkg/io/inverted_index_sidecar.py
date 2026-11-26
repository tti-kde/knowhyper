from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from primehkg.io.build_artifacts import (
    INVERTED_INDEX_JSON_FILENAME,
    INVERTED_INDEX_SIDECAR_NAME,
    build_snapshot_sidecar_dir,
    dataset_mirror_sidecar_dir,
)
from primehkg.retrieval.index import build_inverted_index
from primehkg.retrieval.index_io import save_inverted_index_json

if TYPE_CHECKING:
    from collections.abc import Mapping


def dataset_relative_inverted_index_json_path() -> Path:
    """Relative path to inverted-index JSON under ``primehkg-build --out-dir``."""
    return dataset_mirror_sidecar_dir(INVERTED_INDEX_SIDECAR_NAME) / (
        INVERTED_INDEX_JSON_FILENAME
    )


def write_inverted_index_sidecar_files(
    inverted_index: Mapping[str, list[str]],
    *,
    inputs_fingerprint: str,
    out_dir: str | Path,
    artifacts_root: Path | None = None,
) -> str:
    """Write inverted-index JSON to artifacts snapshot + dataset mirror.

    Returns the manifest-relative path string for ``sidecars.inverted_index``.
    """
    canonical_dir = build_snapshot_sidecar_dir(
        inputs_fingerprint,
        INVERTED_INDEX_SIDECAR_NAME,
        artifacts_root=artifacts_root,
    )
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_file = canonical_dir / INVERTED_INDEX_JSON_FILENAME
    save_inverted_index_json(
        canonical_file,
        inverted_index,
        inputs_fingerprint=inputs_fingerprint,
    )

    out = Path(out_dir)
    local_file = out / dataset_relative_inverted_index_json_path()
    local_file.parent.mkdir(parents=True, exist_ok=True)
    if local_file.resolve() != canonical_file.resolve():
        shutil.copy2(canonical_file, local_file)

    return str(dataset_relative_inverted_index_json_path())


def emit_inverted_index_sidecar_for_build(
    hyperedges: list[dict[str, Any]],
    *,
    inputs_fingerprint: str,
    out_dir: str | Path,
    artifacts_root: Path | None = None,
) -> str:
    """Build inverted index from merged hyperedges and write sidecar files."""
    inv = build_inverted_index(hyperedges)
    return write_inverted_index_sidecar_files(
        inv,
        inputs_fingerprint=inputs_fingerprint,
        out_dir=out_dir,
        artifacts_root=artifacts_root,
    )
