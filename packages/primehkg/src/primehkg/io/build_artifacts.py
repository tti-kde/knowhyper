from __future__ import annotations

from pathlib import Path

DEFAULT_ARTIFACTS_ROOT = Path("artifacts")
BUILDS_DIR_NAME = "builds"
SIDECARS_DIR_NAME = "sidecars"
DATASETDICT_DIR_NAME = "datasetdict"
BUILDER_DIR_NAME = "builder"

BUILD_MANIFEST_FILE = "build_manifest.json"
BUILD_REPORT_FILE = "build_report.json"
SNAPSHOT_META_FILE = "snapshot_meta.json"

GRAPH_CSR_SIDECAR_NAME = "graph_csr_v1"
INVERTED_INDEX_SIDECAR_NAME = "inverted_index_v1"
INVERTED_INDEX_JSON_FILENAME = "inverted_index.json"
HE_ROW_INDEX_SIDECAR_NAME = "he_row_index_v1"
HE_ROW_INDEX_JSON_FILENAME = "he_row_index.json"
LABEL_EMBEDDING_SIDECAR_NAME = "label_embedding_v1"
LABEL_EMBEDDING_SURFACES_FILENAME = "surfaces.jsonl"
LABEL_EMBEDDING_EMBEDDINGS_FILENAME = "embeddings.f32.npy"
LABEL_EMBEDDING_META_FILENAME = "meta.json"
CHEMICAL_XREF_SIDECAR_NAME = "chemical_xref_v1"
CHEMICAL_XREF_META_FILENAME = "meta.json"
CHEMICAL_XREF_MAPPINGS_FILENAME = "mappings.parquet"
CHEMICAL_XREF_FILE_VERSION = "chemical_xref_v1"
CHEMICAL_XREF_SCHEMA_VERSION = "chemical_xref_mappings_v1"


def resolve_artifacts_root(artifacts_root: Path | str | None = None) -> Path:
    """Return the git-free artifacts root (default ``artifacts/``)."""
    if artifacts_root is None:
        return DEFAULT_ARTIFACTS_ROOT
    return Path(artifacts_root)


def build_snapshot_dir(
    inputs_fingerprint: str,
    *,
    artifacts_root: Path | str | None = None,
) -> Path:
    """Canonical fingerprint directory: ``<artifacts_root>/builds/<inputs_fingerprint>/``."""
    root = resolve_artifacts_root(artifacts_root)
    return root / BUILDS_DIR_NAME / str(inputs_fingerprint)


def build_snapshot_sidecar_dir(
    inputs_fingerprint: str,
    sidecar_name: str,
    *,
    artifacts_root: Path | str | None = None,
) -> Path:
    """Sidecar under a fingerprint snapshot: ``.../builds/<fp>/sidecars/<name>/``."""
    return (
        build_snapshot_dir(inputs_fingerprint, artifacts_root=artifacts_root)
        / SIDECARS_DIR_NAME
        / sidecar_name
    )


def build_snapshot_datasetdict_dir(
    inputs_fingerprint: str,
    *,
    artifacts_root: Path | str | None = None,
) -> Path:
    """Return optional full DatasetDict copy path ``.../builds/<fp>/datasetdict/``."""
    return (
        build_snapshot_dir(inputs_fingerprint, artifacts_root=artifacts_root)
        / DATASETDICT_DIR_NAME
    )


def dataset_mirror_sidecar_dir(sidecar_name: str) -> Path:
    """Sidecar mirror colocated with ``primehkg-build --out-dir``."""
    return Path(SIDECARS_DIR_NAME) / sidecar_name


def relative_snapshot_sidecar_path(
    inputs_fingerprint: str,
    sidecar_name: str,
    *,
    artifacts_root: Path | str | None = None,
) -> str:
    """Relative path string for ``build_manifest.json`` sidecar entries."""
    return str(
        build_snapshot_sidecar_dir(
            inputs_fingerprint,
            sidecar_name,
            artifacts_root=artifacts_root,
        )
    )
