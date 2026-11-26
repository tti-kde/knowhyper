# scripts/ner_precompute.py
"""NER sub-hypergraph cache precompute.

Usage:
    KH=$(uv run python -c "from knowhyper.ieforge._configs import CONFIGS_DIR; print(CONFIGS_DIR)")
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
    uv run python scripts/ner_precompute.py \
        --experiment khp_ner \
        --searchpath "$KH" \
        --predictions predictions/bc5cdr_chem \
        --splits train,validation

Instantiates the linker + caching retriever from the SAME Hydra config that
training uses, loads first-pass predictions from JSONL, and runs link→retrieve
per instance. Gold entities are never used as retrieval seeds. Prediction file hashes
scope the cache namespace automatically.  The cache is then ready for knowledge-enhanced
training with num_workers>0.
"""

from __future__ import annotations

import argparse
import logging
from types import SimpleNamespace
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NER KG precompute")
    p.add_argument("--experiment", default="khp_ner", help="Hydra experiment config")
    p.add_argument("--searchpath", required=True, help="knowhyper configs dir")
    p.add_argument(
        "--predictions",
        required=True,
        help="Directory containing {split}.jsonl first-pass predictions",
    )
    p.add_argument(
        "--splits", default="train,validation", help="Comma-separated splits"
    )
    p.add_argument(
        "--cache-root",
        help="Override knowledge.pipeline.injector.retriever.cache.root",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override (repeatable)",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of disjoint instance shards",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to process",
    )
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument(
        "--link-chunk-size",
        type=int,
        default=0,
        help="Batch-link this many instances at a time; 0 keeps sequential linking",
    )
    p.add_argument(
        "--link-device",
        default="auto",
        help="Device passed to batched SapBERT linking",
    )
    p.add_argument(
        "--link-encoder-batch-size",
        type=int,
        default=256,
        help="SapBERT encoder batch size within each link chunk",
    )
    args = p.parse_args()
    if args.num_shards < 1:
        p.error("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        p.error("--shard-index must satisfy 0 <= index < num_shards")
    if args.link_chunk_size < 0:
        p.error("--link-chunk-size must be >= 0")
    if args.link_encoder_batch_size < 1:
        p.error("--link-encoder-batch-size must be >= 1")
    return args


def main() -> None:
    args = _parse_args()

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from ieforge import CONFIGS_DIR as IEFORGE_CONFIGS_DIR
    from omegaconf import OmegaConf

    overrides = [
        f"experiment={args.experiment}",
        f"hydra.searchpath=[file://{args.searchpath}]",
    ]
    if args.cache_root:
        overrides.append(
            "knowledge.pipeline.injector.retriever.cache.root=" + args.cache_root
        )
    overrides.extend(args.override)
    with initialize_config_dir(version_base=None, config_dir=str(IEFORGE_CONFIGS_DIR)):
        cfg = compose(config_name="train", overrides=overrides)

    injector_cfg = cfg.knowledge.pipeline.injector
    linker = instantiate(injector_cfg.linker)
    retriever = instantiate(injector_cfg.retriever)
    top_k = int(injector_cfg.top_k)
    dataset_name = str(cfg.dataset.name)
    query_builder = instantiate(
        cfg.knowledge.pipeline.query_builder, predictions_root=args.predictions
    )

    raw_type_map = OmegaConf.select(injector_cfg, "mention_type_map", default=None)
    if raw_type_map is not None:
        if OmegaConf.is_config(raw_type_map):
            raw_type_map = OmegaConf.to_container(raw_type_map, resolve=True)
        if not isinstance(raw_type_map, dict):
            msg = "mention_type_map must resolve to a mapping"
            raise TypeError(msg)
        mention_type_map = {
            str(k): str(v) for k, v in raw_type_map.items() if v is not None
        }
    else:
        mention_type_map = None
    expand_abbreviations = bool(
        OmegaConf.select(injector_cfg, "expand_abbreviations", default=False)
    )

    from knowhyper.cache.keys import InstanceKey
    from knowhyper.ieforge.adapters import mentions_from_ner_query
    from knowhyper.ieforge.ner_predictions import prediction_scoped_instance_key

    datamodule_keys = {"batch_size", "num_workers", "name", "task_family"}
    dataset_raw = OmegaConf.to_container(cfg.dataset, resolve=True)
    if not isinstance(dataset_raw, dict):
        msg = "dataset config must resolve to a mapping"
        raise TypeError(msg)
    dataset_cfg = OmegaConf.create(
        {k: v for k, v in dataset_raw.items() if k not in datamodule_keys}
    )

    for split in args.splits.split(","):
        split = split.strip()
        logger.info("Precomputing split=%s", split)
        dataset = instantiate(dataset_cfg, split=split)
        all_examples = list(dataset)
        examples = [
            ex
            for i, ex in enumerate(all_examples)
            if i % args.num_shards == args.shard_index
        ]
        logger.info(
            "  shard %d/%d: %d/%d examples",
            args.shard_index,
            args.num_shards,
            len(examples),
            len(all_examples),
        )
        cache_hits = 0
        pending: list[tuple[InstanceKey, list[Any]]] = []

        for ex in examples:
            view = SimpleNamespace(
                dataset=dataset_name, split=split, id=ex.id, text=ex.text
            )
            query = query_builder.build(view)
            instance = prediction_scoped_instance_key(
                InstanceKey(dataset=dataset_name, split=split, instance_id=str(ex.id)),
                query.metadata.get("prediction_fingerprint"),
            )
            if retriever.get_cached(instance) is not None:
                cache_hits += 1
                continue
            mentions = mentions_from_ner_query(
                query.text,
                list(query.entities),
                list(query.metadata["entity_types"]),
                [tuple(span) for span in query.metadata["entity_spans"]],
                type_map=mention_type_map,
                expand_abbreviations=expand_abbreviations,
            )
            pending.append((instance, list(mentions)))

        if args.link_chunk_size:
            from knowhyper.ieforge.batch_linking import batch_link

            chunks = range(0, len(pending), args.link_chunk_size)
            for chunk_start in chunks:
                chunk = pending[chunk_start : chunk_start + args.link_chunk_size]
                linked_groups = batch_link(
                    linker,
                    [mentions for _, mentions in chunk],
                    device=args.link_device,
                    batch_size=args.link_encoder_batch_size,
                )
                for (instance, _), linked in zip(chunk, linked_groups, strict=True):
                    retriever.retrieve(linked, top_k=top_k, instance=instance)
                completed = min(chunk_start + len(chunk), len(pending))
                if completed % args.log_every < len(chunk) or completed == len(pending):
                    logger.info(
                        "  %s/%s: %d/%d pending",
                        dataset_name,
                        split,
                        completed,
                        len(pending),
                    )
        else:
            for i, (instance, mentions) in enumerate(pending):
                linked = linker.link(mentions)
                retriever.retrieve(linked, top_k=top_k, instance=instance)
                if (i + 1) % args.log_every == 0:
                    logger.info(
                        "  %s/%s: %d/%d pending",
                        dataset_name,
                        split,
                        i + 1,
                        len(pending),
                    )

        logger.info(
            "  Done: %d instances for %s/%s (%d cache hits)",
            len(examples),
            dataset_name,
            split,
            cache_hits,
        )

    logger.info("Precompute complete.")


if __name__ == "__main__":
    main()
