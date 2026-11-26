from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import Dataset, DatasetDict

from primehkg.features.embedding_cache import EmbeddingCacheKey, LocalEmbeddingCache
from primehkg.features.embedding_hash import source_text_hash
from primehkg.features.embedding_inputs import (
    EmbeddingInput,
    input_snapshot_hash5,
)
from primehkg.features.embedding_schema import (
    EMBEDDING_CONFIG_NAME,
    EMBEDDING_DTYPE_V1,
    HE_EMBEDDING_CONFIG_NAME,
    NODE_EMBEDDING_CONFIG_NAME,
    embedding_row_sort_key,
    validate_embedding_row,
)
from primehkg.features.embedding_spill import SortedSpillWriter, merge_sorted_shards
from primehkg.features.embedding_text_sources import TextSource, collect_text_sources
from primehkg.io.embed_manifest import compute_text_snapshot_hash, write_embed_manifest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from primehkg.features.encoders.base import TextEncoder

DEFAULT_EMBED_BATCH_SIZE = 512


def build_embedding_rows(
    *,
    sources: list[TextSource],
    encoder: TextEncoder,
    model_id: str,
    source_modalities: frozenset[str] | None = None,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Encode text sources into validated embedding rows sorted by primary key."""
    if batch_size < 1:
        msg = f"batch_size must be >= 1, got {batch_size}"
        raise ValueError(msg)

    filtered = sources
    if source_modalities is not None:
        filtered = [s for s in sources if s.source_modality in source_modalities]

    rows: list[dict[str, Any]] = []
    for start in range(0, len(filtered), batch_size):
        chunk = filtered[start : start + batch_size]
        texts = [src.text for src in chunk]
        text_hashes = [source_text_hash(text) for text in texts]
        vectors = encoder.encode_batch(texts, source_text_hashes=text_hashes)
        for src, text_hash, vector in zip(chunk, text_hashes, vectors, strict=True):
            row: dict[str, Any] = {
                "node_id": src.node_id,
                "input_kind": "text",
                "source_modality": src.source_modality,
                "model_id": model_id,
                "encoder_family": "biomedical_text",
                "preprocess_version": "text_nfkc_strip_v1",
                "embedding_dim": encoder.embedding_dim,
                "vector": vector,
                "dtype": EMBEDDING_DTYPE_V1,
                "source_text_hash": text_hash,
                "input_hash": text_hash,
                "source": src.source,
                "source_version": src.source_version,
                "encoder_revision": "",
            }
            validate_embedding_row(row)
            rows.append(row)

    rows.sort(key=embedding_row_sort_key)
    return rows


def build_and_write_embedding_dataset(
    *,
    core_dir: Path | str,
    text_dataset_dir: Path | str | None,
    encoder: TextEncoder,
    out_dir: Path | str,
    source_modalities: frozenset[str] | None = None,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> None:
    """Build embedding rows from core + text inputs and write DatasetDict + manifest."""
    core_path = Path(core_dir)
    manifest_path = core_path / "build_manifest.json"
    if not manifest_path.is_file():
        msg = f"missing build_manifest.json at {core_path!r}"
        raise FileNotFoundError(msg)

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_data, dict):
        msg = f"invalid build_manifest.json at {manifest_path!r}"
        raise TypeError(msg)
    core_fp = manifest_data.get("inputs_fingerprint")
    if not isinstance(core_fp, str) or not core_fp:
        msg = f"missing inputs_fingerprint in {manifest_path!r}"
        raise ValueError(msg)

    sources = collect_text_sources(
        core_dir=core_path, text_dataset_dir=text_dataset_dir
    )
    rows = build_embedding_rows(
        sources=sources,
        encoder=encoder,
        model_id=encoder.model_id,
        source_modalities=source_modalities,
        batch_size=batch_size,
    )
    if not rows:
        msg = "no embedding rows produced (empty sources or modality filter)"
        raise ValueError(msg)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    DatasetDict({EMBEDDING_CONFIG_NAME: Dataset.from_list(rows)}).save_to_disk(
        str(out_path)
    )

    used_modalities = sorted({row["source_modality"] for row in rows})
    embed_manifest: dict[str, Any] = {
        "manifest_version": 1,
        "artifact": EMBEDDING_CONFIG_NAME,
        "core_inputs_fingerprint": core_fp,
        "text_snapshot_hash": compute_text_snapshot_hash(
            [
                s
                for s in sources
                if source_modalities is None or s.source_modality in source_modalities
            ]
        ),
        "model_id": encoder.model_id,
        "encoder_revision": "",
        "source_modalities": used_modalities,
        "row_count": len(rows),
        "embedding_dim": encoder.embedding_dim,
        "encode_batch_size": batch_size,
        "created_at": datetime.now(tz=UTC).isoformat(),
    }
    write_embed_manifest(out_path, manifest=embed_manifest)


def _encoder_family_for_input_kind(input_kind: str) -> str:
    return {
        "text": "biomedical_text",
        "hyperedge_text": "biomedical_text",
        "protein_sequence": "protein_sequence",
        "nucleotide_sequence": "nucleotide_sequence",
        "smiles": "molecular_sequence",
        "property_vector": "property_numeric",
    }[input_kind]


def _row_from_input(
    item: EmbeddingInput,
    *,
    vector: list[float],
    model_id: str,
    encoder_family: str,
    encoder_revision: str,
    preprocess_version: str,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "input_kind": item.input_kind,
        "model_id": model_id,
        "encoder_family": encoder_family,
        "encoder_revision": encoder_revision,
        "preprocess_version": preprocess_version,
        "embedding_dim": len(vector),
        "vector": vector,
        "dtype": EMBEDDING_DTYPE_V1,
        "input_hash": item.input_hash,
        "source_text_hash": item.input_hash,
        "source": item.source,
        "source_version": item.source_version,
        "license_tag": item.license_tag,
    }
    if item.target == "node":
        base["node_id"] = item.record_id
        base["source_modality"] = item.source_modality
    else:
        base["he_id"] = item.record_id
        base["source_template_id"] = item.source_modality
    return base


def _encode_chunk(
    chunk: list[EmbeddingInput],
    *,
    encoders: dict[str, TextEncoder],
    cache: LocalEmbeddingCache | None,
    pool: object | None,
    preprocess_version: str,
    encoder_revision: str,
    counters: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    keys: list[EmbeddingCacheKey] = []
    families: list[str] = []
    for item in chunk:
        family = _encoder_family_for_input_kind(item.input_kind)
        families.append(family)
        encoder = encoders[family]
        keys.append(
            EmbeddingCacheKey(
                input_kind=item.input_kind,
                input_hash=item.input_hash,
                model_id=encoder.model_id,
                encoder_revision=encoder_revision,
                preprocess_version=preprocess_version,
                embedding_dim=encoder.embedding_dim,
            )
        )
    cached = cache.get_many(keys) if cache is not None else [None] * len(chunk)

    rows_by_index: list[dict[str, Any] | None] = [None] * len(chunk)
    misses_by_family: dict[str, list[int]] = {}
    for i, item in enumerate(chunk):
        family = families[i]
        encoder = encoders[family]
        counter = counters.setdefault(
            f"{item.target}:{item.input_kind}:{family}:{encoder.model_id}",
            {"row_count": 0, "cache_hits": 0, "cache_misses": 0},
        )
        counter["row_count"] += 1
        vec = cached[i]
        if vec is not None:
            counter["cache_hits"] += 1
            rows_by_index[i] = _build_row(
                item, vec, encoder, family, encoder_revision, preprocess_version
            )
        else:
            counter["cache_misses"] += 1
            misses_by_family.setdefault(family, []).append(i)

    new_items: list[tuple[EmbeddingCacheKey, list[float]]] = []
    for family, idxs in misses_by_family.items():
        encoder = encoders[family]
        indexed = [(i, chunk[i].text, chunk[i].input_hash) for i in idxs]
        if pool is not None:
            idx_to_vec = pool.encode_misses(family, indexed)  # ty: ignore[unresolved-attribute]  # ParallelEncoderPool (EB-T4)
        else:
            texts = [t for _, t, _ in indexed]
            vecs = encoder.encode_batch(
                texts, source_text_hashes=[chunk[i].input_hash for i in idxs]
            )
            idx_to_vec = {i: v for (i, _, _), v in zip(indexed, vecs, strict=True)}
        for i in idxs:
            vec = idx_to_vec[i]
            rows_by_index[i] = _build_row(
                chunk[i], vec, encoder, family, encoder_revision, preprocess_version
            )
            new_items.append((keys[i], vec))
    if cache is not None and new_items:
        cache.put_many(new_items)
    return [r for r in rows_by_index if r is not None]


def _build_row(
    item: EmbeddingInput,
    vec: list[float],
    encoder: TextEncoder,
    family: str,
    encoder_revision: str,
    preprocess_version: str,
) -> dict[str, Any]:
    row = _row_from_input(
        item,
        vector=vec,
        model_id=encoder.model_id,
        encoder_family=family,
        encoder_revision=encoder_revision,
        preprocess_version=preprocess_version,
    )
    validate_embedding_row(row, target=item.target)
    return row


def _optional_manifest_inputs_fingerprint(path: Path | str | None) -> str | None:
    if path is None:
        return None
    manifest_path = Path(path) / "build_manifest.json"
    if not manifest_path.is_file():
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    fingerprint = data.get("inputs_fingerprint")
    if isinstance(fingerprint, str) and fingerprint:
        return fingerprint
    return None


_SPILL_SENTINEL = object()


def _prefetch_chunks(
    producer: Iterator[list[EmbeddingInput]],
    out_queue: queue.Queue[Any],
    sentinel: object,
) -> None:
    """Run the chunk producer on a background thread, feeding the main encode loop."""
    try:
        for chunk in producer:
            out_queue.put(chunk)
    except BaseException as exc:  # re-raised on the consumer thread
        out_queue.put(("__ERROR__", exc))
    finally:
        out_queue.put(sentinel)


def _snap5(item: EmbeddingInput) -> tuple[str, str, str, str, str]:
    """Reduce an EmbeddingInput to the 5-tuple used for the manifest snapshot hash."""
    return (
        item.record_id,
        item.target,
        item.input_kind,
        item.source_modality,
        item.input_hash,
    )


def build_and_write_multimodal_embeddings(
    *,
    core_dir: Path | str,
    features_dir: Path | str | None,
    he_text_dir: Path | str | None,
    out_dir: Path | str,
    cache_dir: Path | str | None,
    encoders: dict[str, TextEncoder],
    node_input_kinds: frozenset[str] | None,
    targets: frozenset[str],
    preprocess_version: str,
    encoder_revision: str = "",
    chunk_size: int = 100_000,
    pool: object | None = None,  # ParallelEncoderPool (EB-T4)
    cache_map_size: int = 1 << 40,
) -> None:
    """Build multimodal node/HE embedding tables and write a lineage manifest."""
    core_path = Path(core_dir)
    manifest_path = core_path / "build_manifest.json"
    if not manifest_path.is_file():
        msg = f"missing build_manifest.json at {core_path!r}"
        raise FileNotFoundError(msg)
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_data, dict):
        msg = f"invalid build_manifest.json at {manifest_path!r}"
        raise TypeError(msg)
    core_fp = manifest_data.get("inputs_fingerprint")
    if not isinstance(core_fp, str) or not core_fp:
        msg = f"missing inputs_fingerprint in {manifest_path!r}"
        raise ValueError(msg)

    from primehkg.features.embedding_inputs import iter_input_chunks

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    spill_dir = out_path / "_spill"
    node_writer = SortedSpillWriter(spill_dir, "node")
    he_writer = SortedSpillWriter(spill_dir, "hyperedge")
    counters: dict[str, dict[str, int]] = {}
    snap_node: list[tuple[str, str, str, str, str]] = []
    snap_he: list[tuple[str, str, str, str, str]] = []
    cache = (
        LocalEmbeddingCache(cache_dir, map_size=cache_map_size)
        if cache_dir is not None
        else None
    )
    chunk_queue: queue.Queue[Any] = queue.Queue(maxsize=2)
    producer = iter_input_chunks(
        targets=targets,
        core_dir=core_path,
        features_dir=features_dir,
        he_text_dir=he_text_dir,
        node_input_kinds=node_input_kinds,
        chunk_size=chunk_size,
    )
    prefetch = threading.Thread(
        target=_prefetch_chunks,
        args=(producer, chunk_queue, _SPILL_SENTINEL),
        daemon=True,
    )
    prefetch.start()
    try:
        while True:
            item = chunk_queue.get()
            if item is _SPILL_SENTINEL:
                break
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__ERROR__":
                raise item[1]
            chunk = item
            rows = _encode_chunk(
                chunk,
                encoders=encoders,
                cache=cache,
                pool=pool,
                preprocess_version=preprocess_version,
                encoder_revision=encoder_revision,
                counters=counters,
            )
            node_writer.write_chunk([row for row in rows if "node_id" in row])
            he_writer.write_chunk([row for row in rows if "he_id" in row])
            for source_item in chunk:
                if source_item.target == "node":
                    snap_node.append(_snap5(source_item))
                else:
                    snap_he.append(_snap5(source_item))
    finally:
        # Drain the queue so a producer parked on a full Queue(maxsize=2) can finish and
        # exit; otherwise prefetch.join() would hang forever if the consumer raised
        # mid-encode (daemon=True does not prevent a hang on an explicit join()).
        while prefetch.is_alive():
            try:
                chunk_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.001)
        prefetch.join()
        if cache is not None:
            cache.close()

    try:
        splits: dict[str, Dataset] = {}
        if node_writer.row_count and node_writer.schema is not None:
            node_pq = spill_dir / "node_merged.parquet"
            merge_sorted_shards(
                node_writer.shard_paths(),
                target="node",
                out_parquet=node_pq,
                schema=node_writer.schema,
            )
            splits[NODE_EMBEDDING_CONFIG_NAME] = Dataset.from_parquet(str(node_pq))
        if he_writer.row_count and he_writer.schema is not None:
            he_pq = spill_dir / "he_merged.parquet"
            merge_sorted_shards(
                he_writer.shard_paths(),
                target="hyperedge",
                out_parquet=he_pq,
                schema=he_writer.schema,
            )
            splits[HE_EMBEDDING_CONFIG_NAME] = Dataset.from_parquet(str(he_pq))
        if not splits:
            msg = "no embedding rows produced"
            raise ValueError(msg)

        output_dd = DatasetDict()
        for split_name, split_dataset in splits.items():
            output_dd[split_name] = split_dataset
        output_dd.save_to_disk(str(out_path))
    finally:
        if spill_dir.exists():
            shutil.rmtree(spill_dir)

    snap_all = snap_node + snap_he
    encoder_runs: list[dict[str, Any]] = []
    for key, value in sorted(counters.items()):
        target, input_kind, family, model_id = key.split(":", 3)
        source_modalities = sorted(
            {t[3] for t in snap_all if t[1] == target and t[2] == input_kind}
        )
        encoder_runs.append(
            {
                "target": target,
                "input_kind": input_kind,
                "source_modalities": source_modalities,
                "encoder_family": family,
                "model_id": model_id,
                "encoder_revision": encoder_revision,
                "preprocess_version": preprocess_version,
                "embedding_dim": encoders[family].embedding_dim,
                **value,
            }
        )

    write_embed_manifest(
        out_path / "embedding_manifest.json",
        manifest={
            "manifest_version": 1,
            "artifact": "primehkg_multimodal_embeddings",
            "core_inputs_fingerprint": core_fp,
            "features_inputs_fingerprint": _optional_manifest_inputs_fingerprint(
                features_dir
            ),
            "he_text_snapshot_hash": input_snapshot_hash5(snap_he)
            if he_writer.row_count
            else None,
            "node_input_snapshot_hash": input_snapshot_hash5(snap_node)
            if node_writer.row_count
            else None,
            "outputs": {
                NODE_EMBEDDING_CONFIG_NAME: {"row_count": node_writer.row_count},
                HE_EMBEDDING_CONFIG_NAME: {"row_count": he_writer.row_count},
            },
            "encoder_runs": encoder_runs,
        },
    )
