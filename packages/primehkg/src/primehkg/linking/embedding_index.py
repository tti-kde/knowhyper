"""In-memory label embedding index for dense linking."""

from __future__ import annotations

from typing import Any


class LabelEmbeddingIndex:
    """Brute-force cosine similarity search over label embedding rows."""

    def __init__(
        self,
        *,
        surfaces: list[dict[str, str]],
        embeddings: Any,
    ) -> None:
        import numpy as np

        self.surfaces = surfaces
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            msg = f"embeddings must be 2-D, got shape {emb.shape}"
            raise ValueError(msg)
        if len(surfaces) != emb.shape[0]:
            msg = f"surface row count {len(surfaces)} != embedding rows {emb.shape[0]}"
            raise ValueError(msg)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        self._embeddings = emb / norms

    def cosine_topk(
        self,
        query_vec: Any,
        k: int,
    ) -> list[tuple[int, float]]:
        import numpy as np

        if k <= 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            return []
        q = q / q_norm

        scores = self._embeddings @ q
        n = scores.shape[0]
        kk = min(k, n)
        if kk == 0:
            return []

        if kk == n:
            idx = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, kk - 1)[:kk]
            idx = part[np.argsort(-scores[part])]

        return [(int(i), float(scores[i])) for i in idx]
