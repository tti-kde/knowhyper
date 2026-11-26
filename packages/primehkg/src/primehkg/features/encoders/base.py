from __future__ import annotations

from typing import Protocol


class TextEncoder(Protocol):
    """Pluggable text encoder for offline embedding builds."""

    @property
    def model_id(self) -> str:
        """Stable encoder + preprocessing identifier."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Output vector length."""
        ...

    def encode_one(self, text: str, *, source_text_hash: str) -> list[float]:
        """Encode a single text into an embedding vector."""
        ...

    def encode_batch(
        self,
        texts: list[str],
        *,
        source_text_hashes: list[str],
    ) -> list[list[float]]:
        """Encode multiple texts (optional batch path)."""
        ...
