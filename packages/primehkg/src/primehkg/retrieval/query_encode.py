"""Optional runtime query encoders for vector global candidates."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _load_sentence_transformer(model_name: str, *, encode_batch_size: int = 32):
    from primehkg.features.encoders.sentence_transformers_encoder import (
        SentenceTransformersEncoder,
    )

    return SentenceTransformersEncoder(
        model_name=model_name,
        model_id=f"{model_name}:mean_pool:max_length_512",
        encode_batch_size=encode_batch_size,
    )


def make_sentence_transformers_encode_fn(
    model_name: str,
    *,
    encode_batch_size: int = 32,
) -> Callable[[str], np.ndarray]:
    enc = _load_sentence_transformer(
        model_name,
        encode_batch_size=encode_batch_size,
    )

    def _encode(text: str) -> np.ndarray:
        vec = enc.encode_one(text, source_text_hash="query")
        return np.asarray(vec, dtype=np.float32)

    return _encode


def resolve_encode_fn(
    *,
    encode_fn: Callable[[str], np.ndarray] | None,
    query_encode_model: str | None,
    encode_batch_size: int = 32,
) -> Callable[[str], np.ndarray] | None:
    if encode_fn is not None:
        return encode_fn
    if query_encode_model is None:
        return None
    return make_sentence_transformers_encode_fn(
        query_encode_model,
        encode_batch_size=encode_batch_size,
    )
