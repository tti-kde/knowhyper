from __future__ import annotations

import hashlib


def _hash_to_unit_float(source_text_hash: str, index: int) -> float:
    """Map ``sha256(source_text_hash + index)`` to a float in ``[0, 1)``."""
    payload = f"{source_text_hash}{index}".encode()
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:4], "big")
    return value / (2**32)


class StubTextEncoder:
    """Deterministic stub encoder for tests and CI.

    For each dimension index ``i`` in ``[0, embedding_dim)``, compute
    ``sha256(f"{source_text_hash}{i}")`` and map the first four digest bytes to
    a float32-like value in ``[0, 1)`` via ``uint32 / 2**32``. The ``text``
    argument is ignored; callers should pass a precomputed ``source_text_hash``.
    """

    def __init__(self, embedding_dim: int, model_id: str = "stub:v1") -> None:
        """Create a stub encoder with fixed output dimension."""
        self._embedding_dim = embedding_dim
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        """Return the configured model identifier."""
        return self._model_id

    @property
    def embedding_dim(self) -> int:
        """Return the configured embedding dimension."""
        return self._embedding_dim

    def encode_one(self, text: str, *, source_text_hash: str) -> list[float]:
        """Return a deterministic hash-derived vector for ``source_text_hash``."""
        del text
        return [
            _hash_to_unit_float(source_text_hash, index)
            for index in range(self._embedding_dim)
        ]

    def encode_batch(
        self,
        texts: list[str],
        *,
        source_text_hashes: list[str],
    ) -> list[list[float]]:
        """Encode multiple texts using ``encode_one`` per row."""
        if len(texts) != len(source_text_hashes):
            msg = "texts and source_text_hashes must have the same length"
            raise ValueError(msg)
        return [
            self.encode_one(text, source_text_hash=source_hash)
            for text, source_hash in zip(texts, source_text_hashes, strict=True)
        ]
