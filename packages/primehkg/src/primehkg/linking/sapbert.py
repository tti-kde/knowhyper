"""SapBERT text encoding for dense label linking."""

from __future__ import annotations

import importlib
from typing import Any

DEFAULT_SAPBERT_MODEL_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
_EMBED_GROUP_HINT = "uv sync --group embed"


def _require_transformers() -> Any:
    try:
        torch = importlib.import_module("torch")
        transformers: Any = importlib.import_module("transformers")
    except ImportError as exc:
        msg = (
            "SapBERT encoding requires optional embed dependencies; "
            f"run {_EMBED_GROUP_HINT}"
        )
        raise ImportError(msg) from exc

    return (
        torch,
        transformers.AutoModel,
        transformers.AutoTokenizer,
    )


def encode_texts_sapbert(
    texts: list[str],
    *,
    model_id: str = DEFAULT_SAPBERT_MODEL_ID,
    batch_size: int = 64,
    device: str = "auto",
) -> Any:
    """Encode texts with SapBERT [CLS] pooling; rows are L2-normalized.

    ``device`` selects the torch device ("auto" picks cuda when available else cpu).
    """
    import numpy as np

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    torch_mod, auto_model_cls, auto_tokenizer_cls = _require_transformers()

    if device == "auto":
        device = "cuda" if torch_mod.cuda.is_available() else "cpu"

    tokenizer = auto_tokenizer_cls.from_pretrained(model_id)
    model = auto_model_cls.from_pretrained(model_id).to(device)
    model.eval()

    rows: list[Any] = []
    with torch_mod.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            outputs = model(**encoded)
            cls = outputs.last_hidden_state[:, 0, :]
            norms = cls.norm(dim=1, keepdim=True).clamp(min=1e-12)
            cls = cls / norms
            rows.append(cls.cpu().numpy())

    return np.vstack(rows).astype(np.float32)
