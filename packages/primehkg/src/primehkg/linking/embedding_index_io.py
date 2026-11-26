"""Load and save label-embedding sidecar files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABEL_EMBEDDING_FILE_VERSION = "1"


@dataclass(frozen=True)
class LabelEmbeddingSidecar:
    surfaces: list[dict[str, str]]
    embeddings: Any
    meta: dict[str, Any]


def save_label_embedding_sidecar(
    sidecar_dir: str | Path,
    *,
    surfaces: list[dict[str, str]],
    embeddings: Any,
    inputs_fingerprint: str,
    model_id: str,
    pooling: str = "cls",
) -> None:
    import numpy as np

    root = Path(sidecar_dir)
    root.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        msg = f"embeddings must be 2-D, got shape {arr.shape}"
        raise ValueError(msg)
    if len(surfaces) != arr.shape[0]:
        msg = f"surface row count {len(surfaces)} != embedding rows {arr.shape[0]}"
        raise ValueError(msg)

    meta = {
        "file_version": LABEL_EMBEDDING_FILE_VERSION,
        "inputs_fingerprint": inputs_fingerprint,
        "model_id": model_id,
        "pooling": pooling,
        "dim": int(arr.shape[1]),
        "num_vectors": int(arr.shape[0]),
        "dtype": "float32",
    }
    (root / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    surfaces_path = root / "surfaces.jsonl"
    with surfaces_path.open("w", encoding="utf-8") as fh:
        for row in surfaces:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    np.save(root / "embeddings.f32.npy", arr)


def load_label_embedding_sidecar(
    sidecar_dir: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> LabelEmbeddingSidecar:
    import numpy as np

    root = Path(sidecar_dir)
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        msg = f"missing label embedding meta: {meta_path}"
        raise FileNotFoundError(msg)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("file_version") != LABEL_EMBEDDING_FILE_VERSION:
        msg = f"unsupported label embedding file_version: {meta.get('file_version')}"
        raise ValueError(msg)

    fp = meta.get("inputs_fingerprint")
    if expected_fingerprint is not None and fp != expected_fingerprint:
        msg = (
            f"label embedding fingerprint mismatch: expected "
            f"{expected_fingerprint!r}, got {fp!r}"
        )
        raise ValueError(msg)

    surfaces: list[dict[str, str]] = []
    surfaces_path = root / "surfaces.jsonl"
    with surfaces_path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            surfaces.append(
                {
                    "node_id": str(row["node_id"]),
                    "field": str(row["field"]),
                    "surface": str(row["surface"]),
                }
            )

    emb_path = root / "embeddings.f32.npy"
    embeddings = np.load(emb_path, mmap_mode="r")
    if len(surfaces) != embeddings.shape[0]:
        msg = (
            f"surface row count {len(surfaces)} != embedding rows {embeddings.shape[0]}"
        )
        raise ValueError(msg)

    return LabelEmbeddingSidecar(
        surfaces=surfaces,
        embeddings=embeddings,
        meta=meta,
    )
