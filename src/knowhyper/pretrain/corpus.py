"""Build a stream of Incidence batches from cached sub-hypergraphs for pre-training."""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from knowhyper.encoders.allset import Incidence
    from knowhyper.pipeline.entity_view_augmenter import EntityViewAugmenter
    from knowhyper.pipeline.protocols import IncidenceEncoder
    from knowhyper.pipeline.types import SubHypergraph


def batched(
    iterator: Iterator[Incidence], batch_size: int
) -> Iterator[list[Incidence]]:
    """Group an Incidence iterator into lists of length ``batch_size`` (last may be short)."""
    if batch_size < 1:
        msg = "batch_size must be >= 1"
        raise ValueError(msg)
    while True:
        chunk = list(islice(iterator, batch_size))
        if not chunk:
            return
        yield chunk


def cached_subhypergraphs(cache_dir: str) -> Iterator[SubHypergraph]:
    """Yield SubHypergraphs from a populated knowledge cache dir (skips manifest.json).

    Reads the per-instance JSON files written by knowhyper.cache.store; no live
    retrieval or fingerprint matching needed.
    """
    import json
    from pathlib import Path

    from knowhyper.cache.store import subhypergraph_from_dict

    for p in sorted(Path(cache_dir).rglob("*.json")):
        if p.name == "manifest.json":
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        yield subhypergraph_from_dict(data)


def augment_subhypergraphs(
    subhgs: Iterator[SubHypergraph], augmenter: EntityViewAugmenter
) -> Iterator[SubHypergraph]:
    """Yield each SubHypergraph enriched with entity_attribute edges/attr nodes."""
    for sub in subhgs:
        yield augmenter.augment(sub)


def incidence_cache_ready(cache_dir: str) -> bool:
    """Return whether ``cache_dir`` holds a completed incidence cache (DONE marker)."""
    from pathlib import Path

    return (Path(cache_dir) / "DONE").exists()


def build_incidence_cache(
    subhgs: Iterator[SubHypergraph],
    encoder: IncidenceEncoder,
    cache_dir: str,
    *,
    chunk_size: int = 500,
) -> None:
    """Precompute ``encoder.build_incidence`` for every sub-HG, saved in chunks.

    Writes ``chunk-00000.pt`` … each holding a list of up to ``chunk_size``
    ``Incidence`` objects (in stream order), then a ``DONE`` marker. This is a
    deterministic function of (sub-HG stream, encoder feature store), so callers can
    reuse it across epochs/runs instead of rebuilding incidences every epoch.
    """
    import dataclasses
    from pathlib import Path

    import torch

    def _to_cpu(inc: Incidence) -> Incidence:
        # build_incidence emits tensors on the encoder's device (cuda in real runs);
        # persist on CPU so the cache never pins GPU memory or bakes in a device.
        fields = {}
        for f in dataclasses.fields(inc):
            v = getattr(inc, f.name)
            fields[f.name] = v.cpu() if isinstance(v, torch.Tensor) else v
        return dataclasses.replace(inc, **fields)

    out = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    chunk: list[Incidence] = []
    idx = 0
    for sub in subhgs:
        chunk.append(_to_cpu(encoder.build_incidence(sub)))
        if len(chunk) >= chunk_size:
            torch.save(chunk, out / f"chunk-{idx:05d}.pt")
            idx += 1
            chunk = []
    if chunk:
        torch.save(chunk, out / f"chunk-{idx:05d}.pt")
    (out / "DONE").write_text("ok", encoding="utf-8")


def cached_incidences(cache_dir: str) -> Iterator[Incidence]:
    """Yield ``Incidence`` objects from a cache built by ``build_incidence_cache``."""
    from pathlib import Path

    import torch

    for p in sorted(Path(cache_dir).glob("chunk-*.pt")):
        chunk: list[Incidence] = torch.load(p, weights_only=False)
        yield from chunk
