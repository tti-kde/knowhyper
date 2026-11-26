from __future__ import annotations

import importlib
from typing import Any


def _bounded_model_max_length(*, model: Any, tokenizer: Any) -> int | None:
    values = (
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    )
    candidates = [
        value for value in values if isinstance(value, int) and 0 < value < 1_000_000
    ]
    if not candidates:
        return None
    return min(candidates)


class TransformersMeanPoolingEncoder:
    """Mean-pooling encoder for generic Hugging Face Transformer models."""

    def __init__(
        self,
        *,
        model_name: str,
        model_id: str,
        encode_batch_size: int = 8,
        trust_remote_code: bool = False,
        device: str | None = None,
    ) -> None:
        """Load tokenizer/model and configure batch encoding."""
        try:
            torch = importlib.import_module("torch")
            transformers: Any = importlib.import_module("transformers")
        except ImportError as exc:
            msg = "transformers encoder requires `uv sync --group embed`"
            raise ImportError(msg) from exc
        auto_tokenizer: Any = transformers.AutoTokenizer
        auto_model: Any = transformers.AutoModel
        self._torch = torch
        self._tokenizer: Any = auto_tokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self._model: Any = auto_model.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        self._max_length = _bounded_model_max_length(
            model=self._model,
            tokenizer=self._tokenizer,
        )
        self._model_id = model_id
        self._encode_batch_size = encode_batch_size
        self._embedding_dim = int(self._model.config.hidden_size)

    @property
    def model_id(self) -> str:
        """Stable encoder identifier recorded on rows."""
        return self._model_id

    @property
    def embedding_dim(self) -> int:
        """Output vector length."""
        return self._embedding_dim

    def encode_one(self, text: str, *, source_text_hash: str) -> list[float]:
        """Encode one string into a mean-pooled vector."""
        del source_text_hash
        return self.encode_batch([text], source_text_hashes=[""])[0]

    def encode_batch(
        self,
        texts: list[str],
        *,
        source_text_hashes: list[str],
    ) -> list[list[float]]:
        """Encode strings in batches using attention-mask mean pooling."""
        del source_text_hashes
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._encode_batch_size):
            chunk = texts[start : start + self._encode_batch_size]
            tokenize_kwargs: dict[str, Any] = {
                "padding": True,
                "truncation": True,
                "return_tensors": "pt",
            }
            if self._max_length is not None:
                tokenize_kwargs["max_length"] = self._max_length
            encoded: Any = self._tokenizer(chunk, **tokenize_kwargs)
            encoded = encoded.to(self._device)
            with self._torch.no_grad():
                outputs = self._model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            vectors.extend(pooled.cpu().to(dtype=self._torch.float32).tolist())
        return vectors
