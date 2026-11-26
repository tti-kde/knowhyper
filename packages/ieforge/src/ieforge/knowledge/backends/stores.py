"""In-memory stores for dependency-light knowledge fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


class InMemoryDocumentStore:
    """Simple in-memory text and metadata store keyed by document id."""

    def __init__(self) -> None:
        """Create an empty document store."""
        self._texts: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def add(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add or replace a document."""
        self._texts[doc_id] = text
        self._metadata[doc_id] = dict(metadata) if metadata else {}

    def get(self, doc_id: str) -> str:
        """Return document text by id."""
        return self._texts[doc_id]

    def get_metadata(self, doc_id: str) -> dict[str, Any]:
        """Return a copy of document metadata by id."""
        return dict(self._metadata.get(doc_id, {}))

    def ids(self) -> Iterable[str]:
        """Return document ids in insertion order."""
        return list(self._texts.keys())

    def __len__(self) -> int:
        """Return the number of stored documents."""
        return len(self._texts)


class NumpyVectorStore:
    """In-memory cosine-search vector store backed by NumPy."""

    def __init__(self, dim: int) -> None:
        """Create an empty vector store for vectors of a fixed dimension."""
        if dim <= 0:
            msg = "dim must be > 0"
            raise ValueError(msg)
        self._dim = dim
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    @property
    def dim(self) -> int:
        """Stored vector dimension."""
        return self._dim

    def add(self, doc_id: str, vector: np.ndarray) -> None:
        """Add a normalized vector for a document id."""
        if vector.shape != (self._dim,):
            msg = f"vector dim {vector.shape} != store dim ({self._dim},)"
            raise ValueError(msg)
        normalized = _l2_normalize(vector.astype(np.float32))
        self._matrix = (
            normalized[None, :]
            if self._matrix is None
            else np.vstack([self._matrix, normalized[None, :]])
        )
        self._ids.append(doc_id)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Return top document ids and scores for a query vector."""
        if self._matrix is None or k <= 0:
            return []
        q = _l2_normalize(query.astype(np.float32))
        scores = self._matrix @ q
        top = min(k, len(self._ids))
        ordered = sorted(range(len(self._ids)), key=lambda i: (-float(scores[i]), i))
        return [(self._ids[i], float(scores[i])) for i in ordered[:top]]

    def __len__(self) -> int:
        """Return the number of stored vectors."""
        return len(self._ids)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm
