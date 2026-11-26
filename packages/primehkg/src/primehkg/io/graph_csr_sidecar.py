from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from primehkg.io.build_artifacts import (
    GRAPH_CSR_SIDECAR_NAME,
    build_snapshot_sidecar_dir,
    dataset_mirror_sidecar_dir,
    relative_snapshot_sidecar_path,
    resolve_artifacts_root,
)
from primehkg.retrieval.graph_csr.build import build_graph_csr_from_hyperedges
from primehkg.retrieval.graph_csr.io import write_graph_csr_sidecar


def canonical_graph_csr_sidecar_path(
    inputs_fingerprint: str,
    *,
    artifacts_root: Path | None = None,
) -> Path:
    """Return ``artifacts/builds/<fp>/sidecars/graph_csr_v1`` (design layout)."""
    return build_snapshot_sidecar_dir(
        inputs_fingerprint,
        GRAPH_CSR_SIDECAR_NAME,
        artifacts_root=artifacts_root,
    )


def dataset_relative_graph_csr_sidecar_path() -> Path:
    """Sidecar path colocated with a ``primehkg-build`` output directory."""
    return dataset_mirror_sidecar_dir(GRAPH_CSR_SIDECAR_NAME)


def emit_graph_csr_sidecar_for_build(
    hyperedges: list[dict[str, Any]],
    *,
    inputs_fingerprint: str,
    out_dir: str | Path,
    artifacts_root: Path | None = None,
) -> dict[str, str]:
    """Build CSR from hyperedges and write the canonical artifacts sidecar (+ local mirror).

    Returns relative path strings recorded in ``build_manifest.json`` → ``sidecars``.
    """
    g = build_graph_csr_from_hyperedges(
        hyperedges, inputs_fingerprint=str(inputs_fingerprint)
    )
    canonical = canonical_graph_csr_sidecar_path(
        inputs_fingerprint, artifacts_root=artifacts_root
    )
    canonical.mkdir(parents=True, exist_ok=True)
    write_graph_csr_sidecar(canonical, g)

    out = Path(out_dir)
    local = out / dataset_relative_graph_csr_sidecar_path()
    if local.resolve() != canonical.resolve():
        if local.exists():
            shutil.rmtree(local)
        shutil.copytree(canonical, local)

    return {
        "artifacts": relative_snapshot_sidecar_path(
            inputs_fingerprint,
            GRAPH_CSR_SIDECAR_NAME,
            artifacts_root=resolve_artifacts_root(artifacts_root),
        ),
        "dataset_dir": str(dataset_relative_graph_csr_sidecar_path()),
    }
