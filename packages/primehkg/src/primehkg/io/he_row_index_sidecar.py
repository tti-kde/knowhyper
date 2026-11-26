from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from primehkg.io.build_artifacts import (
    HE_ROW_INDEX_JSON_FILENAME,
    HE_ROW_INDEX_SIDECAR_NAME,
    build_snapshot_sidecar_dir,
    dataset_mirror_sidecar_dir,
)
from primehkg.retrieval.he_row_index import build_he_row_index
from primehkg.retrieval.he_row_index_io import save_he_row_index_json

if TYPE_CHECKING:
    from collections.abc import Mapping


def dataset_relative_he_row_index_json_path() -> Path:
    """Relative path to he-row index JSON under ``primehkg-build --out-dir``."""
    return dataset_mirror_sidecar_dir(HE_ROW_INDEX_SIDECAR_NAME) / (
        HE_ROW_INDEX_JSON_FILENAME
    )


def write_he_row_index_sidecar_files(
    he_row_index: Mapping[str, int],
    *,
    inputs_fingerprint: str,
    out_dir: str | Path,
    artifacts_root: Path | None = None,
) -> str:
    """Write he-row index JSON to artifacts snapshot + dataset mirror.

    Returns the manifest-relative path string for ``sidecars.he_row_index``.
    """
    canonical_dir = build_snapshot_sidecar_dir(
        inputs_fingerprint,
        HE_ROW_INDEX_SIDECAR_NAME,
        artifacts_root=artifacts_root,
    )
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_file = canonical_dir / HE_ROW_INDEX_JSON_FILENAME
    save_he_row_index_json(
        canonical_file,
        he_row_index,
        inputs_fingerprint=inputs_fingerprint,
    )

    out = Path(out_dir)
    local_file = out / dataset_relative_he_row_index_json_path()
    local_file.parent.mkdir(parents=True, exist_ok=True)
    if local_file.resolve() != canonical_file.resolve():
        shutil.copy2(canonical_file, local_file)

    return str(dataset_relative_he_row_index_json_path())


def emit_he_row_index_sidecar_for_build(
    hyperedges: list[dict[str, Any]],
    *,
    inputs_fingerprint: str,
    out_dir: str | Path,
    artifacts_root: Path | None = None,
) -> str:
    """Build he-row index from merged hyperedges and write sidecar files."""
    idx = build_he_row_index(hyperedges)
    return write_he_row_index_sidecar_files(
        idx,
        inputs_fingerprint=inputs_fingerprint,
        out_dir=out_dir,
        artifacts_root=artifacts_root,
    )
