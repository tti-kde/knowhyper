from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from primehkg.features.encoders.base import TextEncoder

# Per-worker-process encoder registry (populated by _worker_init in each spawned process).
_WORKER_ENCODERS: dict[str, TextEncoder] = {}


@dataclass(frozen=True)
class EncoderSpec:
    """Spec for one encoder family, used to build the encoder inside a worker."""

    family: str
    model_name: str
    model_id: str
    encode_batch_size: int
    trust_remote_code: bool


def _build_encoder(spec: EncoderSpec, device: str, encoder_kind: str) -> TextEncoder:
    if encoder_kind == "stub":
        from primehkg.features.encoders.stub import StubTextEncoder

        return StubTextEncoder(embedding_dim=8, model_id=spec.model_id)
    from primehkg.features.encoders.transformers_pooling_encoder import (
        TransformersMeanPoolingEncoder,
    )

    return TransformersMeanPoolingEncoder(
        model_name=spec.model_name,
        model_id=spec.model_id,
        encode_batch_size=spec.encode_batch_size,
        trust_remote_code=spec.trust_remote_code,
        device=device,
    )


def _worker_init(device: str, specs: list[EncoderSpec], encoder_kind: str) -> None:
    for spec in specs:
        _WORKER_ENCODERS[spec.family] = _build_encoder(spec, device, encoder_kind)


def _worker_encode(
    family: str, indexed: list[tuple[int, str, str]]
) -> list[tuple[int, list[float]]]:
    enc = _WORKER_ENCODERS[family]
    texts = [t for _, t, _ in indexed]
    hashes = [h for _, _, h in indexed]
    vecs = enc.encode_batch(texts, source_text_hashes=hashes)
    return [(i, v) for (i, _, _), v in zip(indexed, vecs, strict=True)]


class ParallelEncoderPool:
    """N single-worker process pools (one per device); merges results by index."""

    def __init__(
        self,
        *,
        devices: list[str],
        specs: list[EncoderSpec],
        encoder_kind: str = "transformers",
    ) -> None:
        """Start one single-worker spawn pool per device, each loading its encoder."""
        if not devices:
            msg = "devices must be non-empty"
            raise ValueError(msg)
        self._devices = devices
        ctx = mp.get_context("spawn")
        self._pools: list[ProcessPoolExecutor] = [
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(device, specs, encoder_kind),
            )
            for device in devices
        ]

    def encode_misses(
        self, family: str, indexed: list[tuple[int, str, str]]
    ) -> dict[int, list[float]]:
        """Shard `indexed` (index, text, source_text_hash) across per-device pools; merge by index."""
        if not indexed:
            return {}
        n = len(self._pools)
        shards: list[list[tuple[int, str, str]]] = [[] for _ in range(n)]
        for j, item in enumerate(indexed):
            shards[j % n].append(item)
        futures = [
            self._pools[k].submit(_worker_encode, family, shard)
            for k, shard in enumerate(shards)
            if shard
        ]
        # fut.result() re-raises any worker exception; merge by index.
        return {index: vector for fut in futures for index, vector in fut.result()}

    def close(self) -> None:
        """Shut down every worker pool, cancelling any pending work."""
        for pool in self._pools:
            pool.shutdown(wait=True, cancel_futures=True)
