"""Corpus sources, builders, and deterministic embedders."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ieforge.knowledge.backends.stores import InMemoryDocumentStore, NumpyVectorStore
from ieforge.knowledge.core.base import (
    IEKnowledgeBuilder,
    IEKnowledgeEmbedder,
    IEKnowledgeSource,
)
from ieforge.knowledge.core.schema import BuiltKnowledge, KnowledgeArtifact
from ieforge.registry import register

if TYPE_CHECKING:
    from ieforge.knowledge.core.protocols import Embedder

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_corpus.jsonl"


@register("knowledge_source", name="jsonl_corpus")
class JSONLCorpusSource(IEKnowledgeSource):
    """Loads a JSONL corpus with required id and text fields."""

    def __init__(self, path: str | Path) -> None:
        """Create a source for a JSONL corpus path."""
        self._path = Path(path)

    def load(self) -> KnowledgeArtifact:
        """Load corpus rows into a knowledge artifact."""
        docs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        with self._path.open(encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    msg = f"corpus row {line_no} in {self._path} must be a JSON object"
                    raise TypeError(msg)
                if "id" not in row or "text" not in row:
                    msg = (
                        f"corpus row {line_no} in {self._path} missing required "
                        "'id' or 'text' field"
                    )
                    raise ValueError(msg)
                doc_id = row["id"]
                if doc_id in seen_ids:
                    msg = (
                        f"corpus row {line_no} in {self._path} "
                        f"has duplicate id {doc_id!r}"
                    )
                    raise ValueError(msg)
                seen_ids.add(doc_id)
                docs.append(row)
        return KnowledgeArtifact(
            payload=docs,
            metadata={"source": "jsonl", "path": str(self._path)},
        )


@register("knowledge_source", name="fixture_corpus")
class FixtureCorpusSource(IEKnowledgeSource):
    """Loads the packaged tiny corpus fixture unless a path is provided."""

    def __init__(self, path: str | Path | None = None) -> None:
        """Create a fixture-backed corpus source."""
        self._path = Path(path) if path is not None else _FIXTURE_PATH

    def load(self) -> KnowledgeArtifact:
        """Load the configured fixture corpus."""
        return JSONLCorpusSource(self._path).load()


@register("knowledge_embedder", name="hashing")
class HashingEmbedder(IEKnowledgeEmbedder):
    """Deterministic bag-of-words hashing embedder for fixtures and tests."""

    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        """Create a hashing embedder with a fixed output dimension."""
        if dim <= 0:
            msg = "dim must be > 0"
            raise ValueError(msg)
        self._dim = dim
        self._seed = seed

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    @property
    def identity(self) -> str:
        """Stable identity for this embedder configuration."""
        return f"hashing-bow-v1-d{self._dim}-seed{self._seed}"

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into a dense float32 matrix."""
        matrix = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                matrix[row, _hash_token(token, self._seed) % self._dim] += 1.0
        return matrix


@register("knowledge_builder", name="corpus")
class CorpusBuilder(IEKnowledgeBuilder):
    """Builds document and vector stores from corpus artifacts."""

    def __init__(self, embedder: Embedder) -> None:
        """Create a builder using the provided text embedder."""
        self._embedder = embedder

    @property
    def identity(self) -> str:
        """Stable identity for this corpus builder configuration."""
        return f"corpus:{self._embedder.identity}"

    def build(self, artifact: KnowledgeArtifact) -> BuiltKnowledge:
        """Build document and vector stores from an artifact."""
        rows = artifact.payload
        _validate_unique_ids(rows)
        doc_store = InMemoryDocumentStore()
        vec_store = NumpyVectorStore(dim=self._embedder.dim)
        texts = [row["text"] for row in rows]
        embeddings = self._embedder.encode(texts) if texts else None
        for i, row in enumerate(rows):
            metadata = {k: v for k, v in row.items() if k not in {"id", "text"}}
            doc_store.add(row["id"], row["text"], metadata=metadata)
            if embeddings is not None:
                vec_store.add(row["id"], embeddings[i])
        return BuiltKnowledge(
            handles={"documents": doc_store, "vectors": vec_store},
            metadata={"source": artifact.metadata, "builder": self.identity},
        )


def _hash_token(token: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{token}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _validate_unique_ids(rows: list[dict[str, Any]]) -> None:
    seen_ids: set[str] = set()
    for row in rows:
        doc_id = row["id"]
        if doc_id in seen_ids:
            msg = f"corpus artifact has duplicate id {doc_id!r}"
            raise ValueError(msg)
        seen_ids.add(doc_id)
