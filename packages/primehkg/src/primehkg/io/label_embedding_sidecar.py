from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from primehkg.io.build_artifacts import (
    LABEL_EMBEDDING_SIDECAR_NAME,
    build_snapshot_sidecar_dir,
    dataset_mirror_sidecar_dir,
)
from primehkg.linking.embedding_index_io import save_label_embedding_sidecar
from primehkg.linking.surfaces import collect_label_surfaces

if TYPE_CHECKING:
    from collections.abc import Callable


def dataset_relative_label_embedding_sidecar_dir() -> Path:
    """Relative path to label-embedding sidecar under ``primehkg-build --out-dir``."""
    return dataset_mirror_sidecar_dir(LABEL_EMBEDDING_SIDECAR_NAME)


def write_label_embedding_sidecar_files(
    *,
    surfaces: list[dict[str, str]],
    embeddings: Any,
    inputs_fingerprint: str,
    model_id: str,
    out_dir: str | Path,
    artifacts_root: Path | None = None,
) -> str:
    """Write label-embedding sidecar to artifacts snapshot + dataset mirror.

    Returns the manifest-relative path string for ``sidecars.label_embedding_v1``.
    """
    canonical_dir = build_snapshot_sidecar_dir(
        inputs_fingerprint,
        LABEL_EMBEDDING_SIDECAR_NAME,
        artifacts_root=artifacts_root,
    )
    save_label_embedding_sidecar(
        canonical_dir,
        surfaces=surfaces,
        embeddings=embeddings,
        inputs_fingerprint=inputs_fingerprint,
        model_id=model_id,
    )

    out = Path(out_dir)
    local_dir = out / dataset_relative_label_embedding_sidecar_dir()
    local_dir.mkdir(parents=True, exist_ok=True)
    if local_dir.resolve() != canonical_dir.resolve():
        for name in ("meta.json", "surfaces.jsonl", "embeddings.f32.npy"):
            src = canonical_dir / name
            if src.is_file():
                shutil.copy2(src, local_dir / name)

    return str(dataset_relative_label_embedding_sidecar_dir())


def emit_label_embedding_sidecar_for_build(
    nodes: list[dict[str, Any]],
    *,
    inputs_fingerprint: str,
    out_dir: str | Path,
    encode_fn: Callable[[list[str]], Any],
    model_id: str,
    artifacts_root: Path | None = None,
) -> str:
    """Collect surfaces, encode with ``encode_fn``, and write sidecar files."""
    surfaces = collect_label_surfaces(nodes)
    if not surfaces:
        import numpy as np

        embeddings = np.zeros((0, 0), dtype=np.float32)
    else:
        texts = [row["surface"] for row in surfaces]
        embeddings = encode_fn(texts)
    return write_label_embedding_sidecar_files(
        surfaces=surfaces,
        embeddings=embeddings,
        inputs_fingerprint=inputs_fingerprint,
        model_id=model_id,
        out_dir=out_dir,
        artifacts_root=artifacts_root,
    )
