from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def write_vector_sidecar(
    *,
    out_dir: str | Path,
    ids: list[str],
    embeddings: np.ndarray,
    file_version: str,
    inputs_fingerprint: str,
    model_id: str,
    source_template_id: str,
    ann_backend: str = "brute_force",
    ann_params: dict[str, Any] | None = None,
) -> Path:
    """Write a vector sidecar (meta.json + vocab.txt + embeddings.f32.npy)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        msg = f"embeddings must be 2-D, got shape {emb.shape}"
        raise ValueError(msg)
    if emb.shape[0] != len(ids):
        msg = f"ids/embeddings row count mismatch: {len(ids)} vs {emb.shape[0]}"
        raise ValueError(msg)

    meta: dict[str, Any] = {
        "file_version": file_version,
        "inputs_fingerprint": inputs_fingerprint,
        "model_id": model_id,
        "source_template_id": source_template_id,
        "embedding_dim": int(emb.shape[1]),
        "num_vectors": len(ids),
        "dtype": "float32",
        "ann_backend": ann_backend,
        "ann_params": ann_params or {},
        "created_at": datetime.now(UTC).isoformat(),
    }

    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "vocab.txt").write_text(
        "\n".join(ids) + ("\n" if ids else ""), encoding="utf-8"
    )
    np.save(out / "embeddings.f32.npy", emb)

    return out


def load_vector_sidecar_meta(sidecar_dir: str | Path) -> dict[str, Any]:
    """Load and return meta.json from a vector sidecar directory."""
    path = Path(sidecar_dir) / "meta.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)
