from __future__ import annotations

import hashlib
import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

_MAP_SIZE_DEFAULT = (
    1 << 40
)  # 1 TiB virtual; sparse on ext4/xfs, may over-allocate on NFS
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingCacheKey:
    """Stable cache key for one normalized input and encoder configuration."""

    input_kind: str
    input_hash: str
    model_id: str
    encoder_revision: str
    preprocess_version: str
    embedding_dim: int


class LocalEmbeddingCache:
    """LMDB cache for one-vector-per-key float32 embeddings (batched get/put)."""

    def __init__(self, root: str | Path, *, map_size: int = _MAP_SIZE_DEFAULT) -> None:
        """Open (or create) the LMDB environment rooted at `root`."""
        try:
            lmdb = importlib.import_module("lmdb")
        except ImportError as exc:
            msg = "LocalEmbeddingCache requires `uv sync --group embed` (lmdb)"
            raise ImportError(msg) from exc
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._env = lmdb.open(
            str(self.root), map_size=map_size, subdir=True, max_dbs=1, lock=True
        )

    def close(self) -> None:
        """Close the LMDB environment."""
        self._env.close()

    def __enter__(self) -> Self:
        """Support use as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the cache on context-manager exit."""
        self.close()

    @staticmethod
    def _key_bytes(key: EmbeddingCacheKey) -> bytes:
        digest = "\n".join(
            [
                key.input_kind,
                key.input_hash,
                key.model_id,
                key.encoder_revision,
                key.preprocess_version,
                str(key.embedding_dim),
            ]
        )
        return hashlib.sha256(digest.encode("utf-8")).digest()

    @staticmethod
    def _decode(buf: bytes | None, dim: int) -> list[float] | None:
        if buf is None:
            return None
        if len(buf) != dim * 4:
            _log.warning(
                "embedding cache: corrupt entry (%d bytes, expected %d) — treating as miss",
                len(buf),
                dim * 4,
            )
            return None
        arr = np.frombuffer(buf, dtype=np.float32)
        return arr.tolist()

    @staticmethod
    def _decode_many(bufs: list[bytes | None], dim: int) -> list[list[float] | None]:
        expected = dim * 4
        # Fast path: collect indices of well-formed buffers, decode them in one numpy pass.
        good_indices: list[int] = []
        good_bufs: list[bytes] = []
        result: list[list[float] | None] = [None] * len(bufs)
        for i, buf in enumerate(bufs):
            if buf is None:
                continue
            if len(buf) != expected:
                _log.warning(
                    "embedding cache: corrupt entry (%d bytes, expected %d) — treating as miss",
                    len(buf),
                    expected,
                )
                continue
            good_indices.append(i)
            good_bufs.append(buf)
        if good_bufs:
            arr = np.frombuffer(b"".join(good_bufs), dtype=np.float32).reshape(
                len(good_bufs), dim
            )
            decoded = arr.tolist()
            for slot, vec in zip(good_indices, decoded, strict=True):
                result[slot] = vec
        return result

    @staticmethod
    def _encode_vector(vector: list[float], dim: int) -> bytes:
        if len(vector) != dim:
            msg = f"vector length {len(vector)} != embedding_dim {dim}"
            raise ValueError(msg)
        return np.asarray(vector, dtype=np.float32).tobytes()

    def get(self, key: EmbeddingCacheKey) -> list[float] | None:
        """Return a cached vector for `key`, or None when absent/corrupt."""
        with self._env.begin(write=False) as txn:
            buf = txn.get(self._key_bytes(key))
        return self._decode(buf, key.embedding_dim)

    def put(self, key: EmbeddingCacheKey, vector: list[float]) -> None:
        """Store `vector` for `key` as float32 bytes."""
        payload = self._encode_vector(vector, key.embedding_dim)
        with self._env.begin(write=True) as txn:
            txn.put(self._key_bytes(key), payload)

    def get_many(self, keys: list[EmbeddingCacheKey]) -> list[list[float] | None]:
        """Batch lookup in one read transaction; order matches `keys`."""
        if not keys:
            return []
        with self._env.begin(write=False) as txn:
            bufs = [txn.get(self._key_bytes(key)) for key in keys]
        # all keys in a get_many call share the same embedding_dim in practice, but decode
        # per distinct dim to stay correct if they ever differ.
        dims = {key.embedding_dim for key in keys}
        if len(dims) == 1:
            return self._decode_many(bufs, keys[0].embedding_dim)
        # rare mixed-dim path: fall back to per-key decode (byte-identical)
        return [
            self._decode(buf, key.embedding_dim)
            for buf, key in zip(bufs, keys, strict=True)
        ]

    def put_many(self, items: list[tuple[EmbeddingCacheKey, list[float]]]) -> None:
        """Batch store in one write transaction."""
        encoded = [
            (self._key_bytes(key), self._encode_vector(vector, key.embedding_dim))
            for key, vector in items
        ]
        with self._env.begin(write=True) as txn:
            for kb, payload in encoded:
                txn.put(kb, payload)
