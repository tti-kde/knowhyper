from __future__ import annotations

import importlib

_MAX_LENGTH = 512
"""Maximum input sequence length for tokenization (``max_length=512``)."""


def _load_sentence_transformer(model_name: str):
    try:
        st_mod = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        msg = (
            "sentence-transformers is not installed. "
            "Install with: uv sync --group embed"
        )
        raise ImportError(msg) from exc
    sentence_transformer_cls = st_mod.SentenceTransformer
    model = sentence_transformer_cls(model_name)
    model.max_seq_length = _MAX_LENGTH
    return model


class SentenceTransformersEncoder:
    """Real text encoder using Hugging Face sentence-transformers.

    Tokenization uses ``max_length=512``. Pass a stable ``model_id`` (typically
    including preprocessing hints) for manifest rows; ``model_name`` is the
    Hugging Face hub id passed to ``SentenceTransformer``.
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_id: str,
        encode_batch_size: int = 32,
    ) -> None:
        """Load ``model_name`` and derive ``embedding_dim`` from the model."""
        if encode_batch_size < 1:
            msg = f"encode_batch_size must be >= 1, got {encode_batch_size}"
            raise ValueError(msg)
        self._model_name = model_name
        self._model_id = model_id
        self._encode_batch_size = encode_batch_size
        self._model = _load_sentence_transformer(model_name)
        dim = self._model.get_embedding_dimension()
        if dim is None:
            msg = f"model {model_name!r} did not expose embedding dimension"
            raise ValueError(msg)
        self._embedding_dim = int(dim)

    @property
    def model_id(self) -> str:
        """Return the configured model identifier."""
        return self._model_id

    @property
    def embedding_dim(self) -> int:
        """Return the model output dimension."""
        return self._embedding_dim

    def encode_one(self, text: str, *, source_text_hash: str) -> list[float]:
        """Encode a single text via ``encode_batch`` of size 1."""
        return self.encode_batch([text], source_text_hashes=[source_text_hash])[0]

    def encode_batch(
        self,
        texts: list[str],
        *,
        source_text_hashes: list[str],
    ) -> list[list[float]]:
        """Encode multiple texts with ``max_length=512`` truncation."""
        if len(texts) != len(source_text_hashes):
            msg = "texts and source_text_hashes must have the same length"
            raise ValueError(msg)
        if not texts:
            return []
        del source_text_hashes
        embeddings = self._model.encode(
            texts,
            batch_size=self._encode_batch_size,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in embeddings]
