from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_TOPK_CHUNK_SIZE = 8192


class VectorIndex:
    """mmap-backed vector store with chunked brute-force or optional HNSW search."""

    def __init__(self, sidecar_dir: str | Path) -> None:
        self._dir = Path(sidecar_dir)
        meta_path = self._dir / "meta.json"
        with meta_path.open(encoding="utf-8") as f:
            self._meta: dict[str, Any] = json.load(f)

        vocab_path = self._dir / "vocab.txt"
        self._ids = vocab_path.read_text(encoding="utf-8").strip().split("\n")
        if self._ids == [""]:
            self._ids = []
        self._id_to_row: dict[str, int] = {id_: i for i, id_ in enumerate(self._ids)}

        emb_path = self._dir / "embeddings.f32.npy"
        self._embeddings = np.load(emb_path, mmap_mode="r")

        if self._embeddings.shape[0] != len(self._ids):
            msg = (
                f"vocab has {len(self._ids)} entries but embeddings has "
                f"{self._embeddings.shape[0]} rows"
            )
            raise ValueError(msg)

        self._hnsw: Any | None = None
        self._hnsw_ef_search = int(
            self._meta.get("ann_params", {}).get("ef_search", 64)
        )
        if self.ann_backend == "hnswlib":
            self._try_load_hnsw()

    def _try_load_hnsw(self) -> None:
        ann_dir = self._dir / "ann_index"
        ann_meta_path = ann_dir / "meta.json"
        if not ann_meta_path.is_file():
            if self.ann_backend == "hnswlib":
                import warnings

                warnings.warn(
                    f"HNSW ann_index missing under {ann_dir}; using brute-force topk",
                    stacklevel=2,
                )
            return
        with ann_meta_path.open(encoding="utf-8") as f:
            ann_meta = json.load(f)
        index_file = str(ann_meta.get("index_file", "hnsw.bin"))
        index_path = ann_dir / index_file
        if not index_path.is_file():
            if self.ann_backend == "hnswlib":
                import warnings

                warnings.warn(
                    f"HNSW ann_index missing under {ann_dir}; using brute-force topk",
                    stacklevel=2,
                )
            return

        try:
            import hnswlib  # ty: ignore[unresolved-import]
        except ImportError as exc:
            msg = (
                "vector sidecar ann_backend='hnswlib' but hnswlib is not installed; "
                "run `uv sync --group vector`"
            )
            raise ImportError(msg) from exc

        ef = int(ann_meta.get("ef_search", self._hnsw_ef_search))
        try:
            index = hnswlib.Index(space="cosine", dim=self.embedding_dim)
            index.load_index(str(index_path), max_elements=self.num_vectors)
            index.set_ef(ef)
        except Exception as exc:
            import warnings

            warnings.warn(
                f"Failed to load HNSW index from {index_path} ({exc}); "
                "using brute-force topk",
                stacklevel=2,
            )
            return

        self._hnsw = index
        self._hnsw_ef_search = ef

    @property
    def model_id(self) -> str:
        return str(self._meta["model_id"])

    @property
    def embedding_dim(self) -> int:
        return int(self._meta["embedding_dim"])

    @property
    def num_vectors(self) -> int:
        return len(self._ids)

    @property
    def inputs_fingerprint(self) -> str:
        return str(self._meta.get("inputs_fingerprint", ""))

    @property
    def ann_backend(self) -> str:
        return str(self._meta.get("ann_backend", "brute_force"))

    def lookup(self, id_: str) -> np.ndarray | None:
        row = self._id_to_row.get(id_)
        if row is None:
            return None
        return np.array(self._embeddings[row], dtype=np.float32)

    def topk(
        self,
        query_vec: np.ndarray,
        k: int,
        *,
        chunk_size: int = _DEFAULT_TOPK_CHUNK_SIZE,
        ef_search: int | None = None,
    ) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            return []
        q = q / q_norm

        n = self._embeddings.shape[0]
        kk = min(k, n)
        if kk == 0:
            return []

        if self._hnsw is not None:
            return self._topk_hnsw(q, kk, ef_search=ef_search)

        best_idx = np.full(kk, -1, dtype=np.int64)
        best_scores = np.full(kk, -np.inf, dtype=np.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = np.array(self._embeddings[start:end], dtype=np.float32)
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            scores = (chunk / norms) @ q

            for local_i, score in enumerate(scores):
                if score <= best_scores[-1]:
                    continue
                insert_at = int(np.searchsorted(-best_scores, -score))
                best_scores[insert_at + 1 :] = best_scores[insert_at:-1]
                best_scores[insert_at] = score
                best_idx[insert_at + 1 :] = best_idx[insert_at:-1]
                best_idx[insert_at] = start + local_i

        order = np.argsort(-best_scores)
        return [(self._ids[int(best_idx[i])], float(best_scores[i])) for i in order]

    def _topk_hnsw(
        self,
        query: np.ndarray,
        k: int,
        *,
        ef_search: int | None = None,
    ) -> list[tuple[str, float]]:
        hnsw = self._hnsw
        if hnsw is None:
            msg = "HNSW index is not loaded"
            raise RuntimeError(msg)
        ef = int(ef_search) if ef_search is not None else self._hnsw_ef_search
        hnsw.set_ef(ef)
        labels, distances = hnsw.knn_query(
            query.reshape(1, -1),
            k=k,
        )
        row = labels[0]
        dist_row = distances[0]
        scores = 1.0 - dist_row
        order = np.argsort(-scores)
        return [
            (self._ids[int(row[i])], float(scores[i]))
            for i in order
            if int(row[i]) >= 0
        ]
