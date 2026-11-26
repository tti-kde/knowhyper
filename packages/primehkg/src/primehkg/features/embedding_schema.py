from __future__ import annotations

from typing import Any, Literal

NODE_EMBEDDING_CONFIG_NAME = "node_features_embedding"
HE_EMBEDDING_CONFIG_NAME = "he_features_embedding"
EMBEDDING_CONFIG_NAME = NODE_EMBEDDING_CONFIG_NAME
EMBEDDING_DTYPE_V1 = "float32"

NODE_EMBEDDING_PK_COLUMNS = (
    "node_id",
    "input_kind",
    "source_modality",
    "model_id",
    "preprocess_version",
)
HE_EMBEDDING_PK_COLUMNS = (
    "he_id",
    "input_kind",
    "source_template_id",
    "model_id",
    "preprocess_version",
)
EMBEDDING_PK_COLUMNS = ("node_id", "source_modality", "model_id")

EmbeddingTarget = Literal["node", "hyperedge"]


def _validate_target(target: str) -> None:
    if target not in ("node", "hyperedge"):
        msg = f"unsupported embedding target {target!r}"
        raise ValueError(msg)


def _required_columns(target: EmbeddingTarget) -> tuple[str, ...]:
    _validate_target(target)
    common = (
        "input_kind",
        "model_id",
        "encoder_family",
        "encoder_revision",
        "preprocess_version",
        "embedding_dim",
        "vector",
        "dtype",
        "input_hash",
        "source",
        "source_version",
    )
    if target == "node":
        return ("node_id", "source_modality", *common)
    return ("he_id", "source_template_id", *common)


def embedding_row_sort_key(
    row: dict[str, Any],
    *,
    target: EmbeddingTarget = "node",
) -> tuple[str, str, str, str, str]:
    _validate_target(target)
    if target == "node":
        return (
            str(row["node_id"]),
            str(row["input_kind"]),
            str(row["source_modality"]),
            str(row["model_id"]),
            str(row["preprocess_version"]),
        )
    return (
        str(row["he_id"]),
        str(row["input_kind"]),
        str(row["source_template_id"]),
        str(row["model_id"]),
        str(row["preprocess_version"]),
    )


def validate_embedding_row(
    row: dict[str, Any],
    *,
    target: EmbeddingTarget = "node",
) -> None:
    for key in _required_columns(target):
        if key not in row:
            msg = f"missing required key {key!r}"
            raise ValueError(msg)
    if row["dtype"] != EMBEDDING_DTYPE_V1:
        msg = f"unsupported dtype {row['dtype']!r}"
        raise ValueError(msg)
    dim = int(row["embedding_dim"])
    vec = row["vector"]
    if not isinstance(vec, list) or len(vec) != dim:
        msg = "embedding_dim does not match len(vector)"
        raise ValueError(msg)
    if (
        not isinstance(row["input_hash"], str)
        or len(row["input_hash"]) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in row["input_hash"])
    ):
        msg = "input_hash must be a 64-character sha256 hex string"
        raise ValueError(msg)
    for x in vec:
        if not isinstance(x, (int, float)):
            msg = "vector elements must be numeric"
            raise TypeError(msg)
