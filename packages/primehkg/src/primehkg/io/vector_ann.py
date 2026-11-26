from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def build_hnsw_ann_index(
    *,
    sidecar_dir: str | Path,
    embeddings: np.ndarray,
    m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 64,
) -> dict[str, Any]:
    """Build and persist an hnswlib cosine index under ``ann_index/``."""
    import hnswlib  # ty: ignore[unresolved-import]

    out = Path(sidecar_dir)
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        msg = f"embeddings must be 2-D, got shape {emb.shape}"
        raise ValueError(msg)

    n, dim = emb.shape
    if n == 0:
        msg = "cannot build HNSW index for zero vectors"
        raise ValueError(msg)

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
    index.add_items(emb, np.arange(n, dtype=np.int64))
    index.set_ef(ef_search)

    ann_dir = out / "ann_index"
    ann_dir.mkdir(parents=True, exist_ok=True)
    index_file = "hnsw.bin"
    index.save_index(str(ann_dir / index_file))

    ann_meta: dict[str, Any] = {
        "ann_backend": "hnswlib",
        "index_file": index_file,
        "M": int(m),
        "ef_construction": int(ef_construction),
        "ef_search": int(ef_search),
        "num_vectors": int(n),
        "embedding_dim": int(dim),
    }
    (ann_dir / "meta.json").write_text(
        json.dumps(ann_meta, indent=2),
        encoding="utf-8",
    )
    return ann_meta
