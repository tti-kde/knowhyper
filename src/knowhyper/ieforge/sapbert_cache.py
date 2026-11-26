"""Process-wide cached SapBERT encoder for dense entity linking.

primehkg's ``encode_texts_sapbert`` loads the SapBERT model and tokenizer with
``from_pretrained`` on EVERY call (see ``primehkg.linking.sapbert``). Because the
dense resolver encodes one query per ``link()``, that reloads the whole ~109M
model from disk per mention — the repeated ``Loading weights: 199/199`` seen in
training logs, which pins linking at ~2 s/mention and idles the GPU.

This module provides a drop-in ``cached_encode_texts_sapbert`` that memoizes the
``(tokenizer, model)`` pair by ``(model_id, device)`` so the model loads once per
process and is reused. :func:`install_sapbert_cache` monkeypatches primehkg's
module-level symbol so the dense resolver picks it up transparently.

This is a version-absorption shim: it reimplements the encode loop to match
primehkg's published behavior ([CLS] pooling, L2-normalized float32 rows) at the
pinned rev. If upstream changes that behavior, the parity is asserted by the
real-data linking tests, not silently.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SAPBERT_MODEL_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# Mutable flag (dict, like _CACHE) tracking whether we've warned that linking is
# running on CPU inside a worker — so a cache-miss storm is visible without a line
# per mention, and without a rebinding `global` statement.
_WARN_STATE: dict[str, bool] = {"worker_cpu": False}

# (model_id, device) -> (tokenizer, model). Process-wide, intentionally unbounded:
# only a couple of (model, device) pairs are ever used in a run.
_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def clear_sapbert_cache() -> None:
    """Drop all cached encoders (used by tests for a cold start)."""
    _CACHE.clear()
    _WARN_STATE["worker_cpu"] = False


def _resolve_auto_device(torch: Any) -> str:
    """Resolve ``device="auto"`` to a device safe for the current process.

    Prefers CUDA when available, but forces CPU inside a forked DataLoader
    worker: CUDA is *visible* in the fork (``is_available()`` is True) yet
    ``.to("cuda")`` there raises "Cannot re-initialize CUDA in forked
    subprocess". This lets a stray cache miss link on CPU during
    ``num_workers>0`` training instead of crashing the run.

    Warns once if it falls back to CPU while CUDA is visible: with a complete
    sub-HG cache, training links zero times, so seeing this means the cache is
    missing for some instances — a "training is just slow" symptom that should be
    traceable to its cause rather than silent.
    """
    if torch.utils.data.get_worker_info() is not None:
        if not _WARN_STATE["worker_cpu"] and torch.cuda.is_available():
            logger.warning(
                "SapBERT linking on CPU inside a DataLoader worker (CUDA "
                "unusable after fork). Expected zero linking with a warm sub-HG "
                "cache — a stream of these means the cache is missing for many "
                "instances and training will be slow."
            )
            _WARN_STATE["worker_cpu"] = True
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(model_id: str, device: str) -> tuple[Any, Any]:
    """Return a cached ``(tokenizer, model)`` for ``(model_id, device)``."""
    key = (model_id, device)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    _CACHE[key] = (tokenizer, model)
    return tokenizer, model


def cached_encode_texts_sapbert(
    texts: list[str],
    *,
    model_id: str = DEFAULT_SAPBERT_MODEL_ID,
    batch_size: int = 64,
    device: str = "auto",
) -> Any:
    """Encode texts with a CACHED SapBERT; matches primehkg's encode contract.

    [CLS] pooling, L2-normalized float32 rows. The model/tokenizer load once per
    ``(model_id, device)`` and are reused across calls.
    """
    import numpy as np

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    torch = importlib.import_module("torch")
    if device == "auto":
        device = _resolve_auto_device(torch)

    tokenizer, model = _load(model_id, device)

    rows: list[Any] = []
    with torch.no_grad():
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


def install_sapbert_cache() -> None:
    """Patch primehkg's ``encode_texts_sapbert`` to the cached implementation.

    Idempotent: re-installing is a no-op once primehkg already points at the
    cached function. Imports primehkg lazily so this module is import-safe even
    when the dense linking extras are absent.
    """
    sapbert = importlib.import_module("primehkg.linking.sapbert")
    resolver = importlib.import_module("primehkg.linking.sapbert_resolver")
    if getattr(sapbert, "encode_texts_sapbert", None) is cached_encode_texts_sapbert:
        return
    # setattr (not direct assignment) so the type checker doesn't read this as an
    # implicit shadowing of the module-level function symbol.
    setattr(sapbert, "encode_texts_sapbert", cached_encode_texts_sapbert)  # noqa: B010
    # The resolver imports the symbol by name at module load, so rebind it there too.
    if hasattr(resolver, "encode_texts_sapbert"):
        setattr(resolver, "encode_texts_sapbert", cached_encode_texts_sapbert)  # noqa: B010
