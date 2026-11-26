"""Offline sub-hypergraph cache precompute for a dataset split.

Runs link -> retrieve through a CachingSubHGRetriever so each instance's
SubHypergraph is stored under the retriever's fingerprint. Because training uses
the SAME composed config (same retriever profile + primehkg rev + linker id), the
fingerprint matches and training reads these cache entries instead of recomputing.

The reusable core is :func:`precompute_split`; a thin Hydra/CLI wrapper that
composes ``experiment=khp_re`` and loads the dataset wires it up for real runs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from knowhyper.cache.keys import InstanceKey
from knowhyper.ieforge.adapters import mentions_from_query_text

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[key]
    return getattr(obj, key)


def _get_opt(obj: Any, key: str) -> Any:
    """Read ``key`` from a dict/attr object, returning None when absent."""
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _worker_overrides(experiment: str, split: str, searchpath: str) -> list[str]:
    """Build the Hydra overrides a parallel worker recomposes config from.

    The base three (experiment, split, searchpath) plus any extra CLI overrides the
    parent captured into ``KNOWHYPER_PRECOMPUTE_OVERRIDES`` (a JSON list). Without
    that env var the result is exactly the base three, so default behavior is
    unchanged. This is what lets a flag like ``...linker.strict_type_match=true``
    reach the workers — otherwise they silently compose the wrong fingerprint.
    """
    import json
    import os

    base = [
        f"experiment={experiment}",
        f"++dataset.split={split}",
        f"hydra.searchpath=[file://{searchpath}]",
    ]
    raw = os.environ.get("KNOWHYPER_PRECOMPUTE_OVERRIDES")
    if not raw:
        return base
    extras = json.loads(raw)
    return base + [str(x) for x in extras]


def _read_mention_type_map(injector_cfg: Any) -> dict[str, str] | None:
    """Read the injector's mention_type_map as a plain dict (or None if unset).

    Must match the injector at training time so the typed link seeds — and the
    cached sub-HG — are identical (same fingerprint).
    """
    from omegaconf import OmegaConf

    raw = _get_opt(injector_cfg, "mention_type_map")
    if raw is None:
        return None
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(raw, Mapping):
        msg = f"mention_type_map must be a mapping, got {type(raw).__name__}"
        raise TypeError(msg)
    # Strip None-valued entries (a DDI experiment nulls the inherited chemprot keys
    # via Hydra dict-merge); the injector does the same, so train/precompute agree.
    return {str(k): str(v) for k, v in raw.items() if v is not None}


def _read_expand_abbreviations(injector_cfg: Any) -> bool:
    """Read the injector's expand_abbreviations flag (default False)."""
    return bool(_get_opt(injector_cfg, "expand_abbreviations"))


def _read_derive_id_namespaces(injector_cfg: Any) -> bool:
    """Read BioRED's id-namespace normalization flag (default False)."""
    return bool(_get_opt(injector_cfg, "derive_id_namespaces"))


def _read_pair_role_provenance(injector_cfg: Any) -> bool:
    """Enable head/tail roles only for an explicitly role-aware cache schema."""
    retriever_cfg = _get_opt(injector_cfg, "retriever")
    if retriever_cfg is None:
        return False
    return _get_opt(retriever_cfg, "provenance_schema") == "linked_mentions_role_v1"


def _read_surface_fallback(pipeline_cfg: Any) -> bool:
    """Read the query builder's ``surface_fallback`` flag (default False).

    Lives on ``knowledge.pipeline.query_builder``, not the injector, because at
    training time it is the builder that emits the seeds. This module bypasses the
    builder, so it has to read the same switch or the surface arm silently
    precomputes an id-only cache under a surface fingerprint.
    """
    if pipeline_cfg is None:
        return False
    qb_cfg = _get_opt(pipeline_cfg, "query_builder")
    if qb_cfg is None:
        return False
    return bool(_get_opt(qb_cfg, "surface_fallback"))


def _entity_seeds(
    entity: Any, *, prefer_text: bool = False, surface_fallback: bool = False
) -> list[str]:
    """Return the linker seed strings for one entity, like training does.

    Training links the entity's surface ``mention`` (``dimemorfan``,
    ``sigma-1 receptor``) — the ``text`` field is the masked ``@CHEMICAL$`` marker,
    which links to nothing. So prefer ``mention`` (a ``list[str] | str | None``,
    flattened and None-skipped to mirror ieforge's ``_read_entities``); fall back to
    ``text`` only when no mention is present (e.g. the legacy dict fixtures and
    datasets without the #45 mention field).

    ``surface_fallback`` (default False) mirrors
    ``BioREDQueryBuilder(surface_fallback=True)``: emit the id FIRST and then the
    surface mentions, so an id that resolves keeps its exact hit. It only has an
    effect together with ``prefer_text`` (i.e. BioRED's id-based path); without it
    the mentions are already the seeds.

    This exists because this module builds its queries itself rather than calling
    the query builder, so `surface_fallback` on the builder alone changes the cache
    FINGERPRINT while retrieving byte-identical content -- a silent no-op that
    looks like a distinct arm. Measured 2026-07-27: a full surface precompute
    reproduced the id-only empty count exactly (485/1258 both ways).
    """
    text = _get_opt(entity, "text")
    if prefer_text and text:
        if not surface_fallback:
            return [str(text)]
        mention = _get_opt(entity, "mention")
        if isinstance(mention, str):
            extra = [mention] if mention else []
        elif isinstance(mention, (list, tuple)):
            extra = [str(m) for m in mention if m]
        elif mention is not None:
            extra = [str(mention)]
        else:
            extra = []
        return [str(text), *extra]
    mention = _get_opt(entity, "mention")
    if isinstance(mention, str):
        return [mention] if mention else []
    if isinstance(mention, (list, tuple)):
        seeds = [str(m) for m in mention if m]
        if seeds:
            return seeds
    elif mention is not None:
        return [str(mention)]
    return [str(text)] if text else []


def _prepare_instance(
    ex: Any,
    *,
    dataset: str,
    split: str,
    mention_type_map: dict[str, str] | None,
    expand_abbreviations: bool,
    derive_id_namespaces: bool = False,
    pair_role_provenance: bool = False,
    surface_fallback: bool = False,
) -> tuple[InstanceKey, list[Any]]:
    """Extract the instance key and mention list from one example."""
    text = str(_get(ex, "text"))
    entities: list[str] = []
    types: list[str | None] = []
    roles: list[str | None] = []
    raw_entities = list(_get(ex, "entities"))
    raw_head = _get_opt(ex, "head_idx")
    raw_tail = _get_opt(ex, "tail_idx")
    if raw_head is not None and raw_tail is not None:
        selected_entities = [
            ("head" if pair_role_provenance else None, raw_entities[int(raw_head)]),
            ("tail" if pair_role_provenance else None, raw_entities[int(raw_tail)]),
        ]
    elif len(raw_entities) == 2:
        selected_entities = [
            ("head" if pair_role_provenance else None, raw_entities[0]),
            ("tail" if pair_role_provenance else None, raw_entities[1]),
        ]
    else:
        selected_entities = [(None, entity) for entity in raw_entities]
    for role, e in selected_entities:
        seeds = _entity_seeds(
            e, prefer_text=derive_id_namespaces, surface_fallback=surface_fallback
        )
        label = _get_opt(e, "label") if mention_type_map is not None else None
        entities.extend(seeds)
        types.extend([label] * len(seeds))
        roles.extend([role] * len(seeds))
    instance = InstanceKey(
        dataset=dataset, split=split, instance_id=str(_get(ex, "id"))
    )
    mentions = mentions_from_query_text(
        text,
        entities,
        types=types,
        type_map=mention_type_map,
        expand_abbreviations=expand_abbreviations,
        pair_roles=roles,
        derive_id_namespaces=derive_id_namespaces,
    )
    return instance, mentions


def precompute_split(
    examples: Sequence[Any],
    *,
    linker: Any,
    retriever: Any,
    dataset: str,
    split: str,
    top_k: int,
    mention_type_map: dict[str, str] | None = None,
    expand_abbreviations: bool = False,
    derive_id_namespaces: bool = False,
    pair_role_provenance: bool = False,
    surface_fallback: bool = False,
    log_every: int = 500,
) -> int:
    """Populate the cache for ``examples``; returns the number processed.

    Each example must expose ``id``, ``text`` and ``entities`` (objects/dicts with
    a ``mention`` and/or ``text`` field). Seeds come from each entity's surface
    ``mention`` (falling back to ``text``) so the cached sub-HG matches what
    training retrieves for the same instance key. ``retriever`` must be a
    CachingSubHGRetriever bound to the cache so its ``retrieve`` stores on a miss.

    ``mention_type_map`` MUST match the injector's so the typed link seeds — and
    thus the cached sub-HG — are identical to training's (same fingerprint). Each
    entity's ``label`` is attached (mapped) to every seed string it produces.
    """
    for i, ex in enumerate(examples):
        instance = InstanceKey(
            dataset=dataset, split=split, instance_id=str(_get(ex, "id"))
        )
        # Check the persisted entry before preparing/linking mentions so a
        # resumed precompute does not repeat the dominant SapBERT work.
        if retriever.get_cached(instance) is not None:
            if (i + 1) % log_every == 0:
                logger.info(
                    "precompute %s/%s: %d/%d", dataset, split, i + 1, len(examples)
                )
            continue
        instance, mentions = _prepare_instance(
            ex,
            dataset=dataset,
            split=split,
            mention_type_map=mention_type_map,
            expand_abbreviations=expand_abbreviations,
            derive_id_namespaces=derive_id_namespaces,
            pair_role_provenance=pair_role_provenance,
            surface_fallback=surface_fallback,
        )
        linked = linker.link(mentions)
        retriever.retrieve(linked, top_k=top_k, instance=instance)
        if (i + 1) % log_every == 0:
            logger.info("precompute %s/%s: %d/%d", dataset, split, i + 1, len(examples))
    return len(examples)


def precompute_split_gpu(
    examples: Sequence[Any],
    *,
    linker: Any,
    retriever: Any,
    dataset: str,
    split: str,
    top_k: int,
    mention_type_map: dict[str, str] | None = None,
    expand_abbreviations: bool = False,
    derive_id_namespaces: bool = False,
    pair_role_provenance: bool = False,
    surface_fallback: bool = False,
    device: str = "auto",
    batch_size: int = 256,
    n_workers: int = 1,
    retriever_config: Any = None,
    log_every: int = 500,
) -> int:
    """GPU batch-link cache misses, then retrieve sequentially or in processes.

    Phase 1 checks the persistent cache before preparing mentions, then batch-
    encodes only misses through SapBERT on ``device``. This preserves the GPU
    linking speedup without repeating linking when a partial run is resumed.

    Phase 2 retrieves in the parent when ``n_workers == 1``. With more workers,
    already-linked misses are sent to spawn-based CPU processes. In that mode,
    ``retriever_config`` must be the resolved Hydra-instantiable configuration
    that created ``retriever``. Spawn is mandatory because phase 1 may have
    initialized CUDA and forking afterward is unsafe.

    The final cache entries are byte-identical to ``precompute_split`` — same
    fingerprint, same instance keys, same sub-hypergraphs.
    """
    import time

    from knowhyper.ieforge.batch_linking import batch_link

    if n_workers < 1:
        msg = f"n_workers must be >= 1, got {n_workers}"
        raise ValueError(msg)
    all_instances: list[InstanceKey] = []
    all_mentions: list[list[Any]] = []

    logger.info("phase 1: checking cache for %d instances", len(examples))
    for ex in examples:
        instance = InstanceKey(
            dataset=dataset, split=split, instance_id=str(_get(ex, "id"))
        )
        if retriever.get_cached(instance) is not None:
            continue
        instance, mentions = _prepare_instance(
            ex,
            dataset=dataset,
            split=split,
            mention_type_map=mention_type_map,
            expand_abbreviations=expand_abbreviations,
            derive_id_namespaces=derive_id_namespaces,
            pair_role_provenance=pair_role_provenance,
            surface_fallback=surface_fallback,
        )
        all_instances.append(instance)
        all_mentions.append(mentions)

    if not all_instances:
        logger.info("phase 1: all %d instances already cached", len(examples))
        return len(examples)
    if n_workers > 1 and retriever_config is None:
        msg = "retriever_config is required when n_workers > 1"
        raise ValueError(msg)

    t0 = time.perf_counter()
    all_linked = batch_link(linker, all_mentions, device=device, batch_size=batch_size)
    t_link = time.perf_counter() - t0
    total_mentions = sum(len(m) for m in all_mentions)
    logger.info(
        "phase 1 done: %d mentions linked in %.1fs (%.1f ms/mention)",
        total_mentions,
        t_link,
        (t_link / max(total_mentions, 1)) * 1000,
    )

    items = list(zip(all_instances, all_linked, strict=True))
    logger.info(
        "phase 2: retrieving %d cache misses with %d worker(s)",
        len(items),
        n_workers,
    )
    t1 = time.perf_counter()
    if n_workers > 1:
        retrieve_linked_parallel(
            items,
            n_workers=n_workers,
            retriever_config=retriever_config,
            top_k=top_k,
            expected_fingerprint=str(retriever.fingerprint),
        )
    else:
        for i, (instance, linked) in enumerate(items):
            retriever.retrieve(linked, top_k=top_k, instance=instance)
            if (i + 1) % log_every == 0:
                logger.info(
                    "phase 2 %s/%s: %d/%d",
                    dataset,
                    split,
                    i + 1,
                    len(items),
                )

    t_retrieve = time.perf_counter() - t1
    logger.info(
        "phase 2 done: %d instances retrieved in %.1fs (%.1f ms/instance)",
        len(items),
        t_retrieve,
        (t_retrieve / max(len(items), 1)) * 1000,
    )
    return len(examples)


def retrieve_linked_parallel(
    items: Sequence[tuple[InstanceKey, Sequence[Any]]],
    *,
    n_workers: int,
    retriever_config: Any = None,
    top_k: int | None = None,
    expected_fingerprint: str | None = None,
    worker: Any = None,
    pool_factory: Any = None,
) -> int:
    """Retrieve pre-linked cache misses in spawn-based round-robin shards."""
    if n_workers < 1:
        msg = f"n_workers must be >= 1, got {n_workers}"
        raise ValueError(msg)
    if not items:
        return 0

    import multiprocessing

    if worker is None:
        worker = _retrieve_linked_shard
        if retriever_config is None or top_k is None or expected_fingerprint is None:
            msg = (
                "retriever_config, top_k, and expected_fingerprint are required "
                "for the default worker"
            )
            raise ValueError(msg)
    if pool_factory is None:
        from concurrent.futures import ProcessPoolExecutor

        pool_factory = ProcessPoolExecutor

    worker_count = min(n_workers, len(items))
    shards = [list(items[index::worker_count]) for index in range(worker_count)]
    payloads: Sequence[Any]
    if worker is _retrieve_linked_shard:
        from omegaconf import OmegaConf

        config_node = (
            retriever_config
            if OmegaConf.is_config(retriever_config)
            else OmegaConf.create(retriever_config)
        )
        config = OmegaConf.to_container(config_node, resolve=True)
        payloads = [(shard, config, top_k, expected_fingerprint) for shard in shards]
    else:
        payloads = shards
    context = multiprocessing.get_context("spawn")
    with pool_factory(max_workers=worker_count, mp_context=context) as pool:
        counts = list(pool.map(worker, payloads))
    return sum(counts)


def _retrieve_linked_shard(
    payload: tuple[
        Sequence[tuple[InstanceKey, Sequence[Any]]], Mapping[str, Any], int, str
    ],
) -> int:
    """Spawn-worker entrypoint for CPU retrieval of pre-linked instances."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"

    from hydra.utils import instantiate

    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass

    shard, retriever_config, top_k, expected_fingerprint = payload
    retriever = instantiate(retriever_config)
    actual_fingerprint = str(retriever.fingerprint)
    if actual_fingerprint != expected_fingerprint:
        msg = (
            "spawn worker retriever fingerprint mismatch: "
            f"expected {expected_fingerprint}, got {actual_fingerprint}"
        )
        raise RuntimeError(msg)
    for instance, linked in shard:
        retriever.retrieve(linked, top_k=top_k, instance=instance)
    return len(shard)


def precompute_split_parallel(
    *,
    n_total: int,
    n_workers: int,
    worker: Any,
    pool_factory: Any = None,
) -> int:
    """Run ``worker`` over ``n_workers`` round-robin shards; return total processed.

    Per-instance retrieval is independent and the JSON cache is concurrency-safe
    (atomic per-PID temp + ``os.replace``), so the only parallelism cost is each
    worker building its own linker/retriever (the primehkg dataset + SapBERT model
    are not picklable, so they cannot be shared across processes).

    ``worker`` is called with ``(shard_idx, n_workers, n_total)`` and must process
    the indices ``i`` where ``i % n_workers == shard_idx`` and return how many it
    processed. ``pool_factory(max_workers=...)`` is the executor context manager;
    it defaults to :class:`concurrent.futures.ProcessPoolExecutor` and is injected
    in tests. Round-robin sharding (not contiguous blocks) keeps the load balanced
    when retrieval cost correlates with position.
    """
    if n_total <= 0:
        return 0
    if pool_factory is None:
        from concurrent.futures import ProcessPoolExecutor

        pool_factory = ProcessPoolExecutor
    shards = [(w, n_workers, n_total) for w in range(n_workers)]
    with pool_factory(max_workers=n_workers) as pool:
        counts = list(pool.map(worker, shards))
    return sum(counts)


def _precompute_shard(shard: tuple[int, int, int]) -> int:
    """Worker entrypoint: build the pipeline from config and process one shard.

    Module-level (picklable) so ProcessPoolExecutor can dispatch it. Reads the
    same composed ``experiment=khp_re`` config the sequential CLI uses, so the
    caching retriever's fingerprint matches training. The split is taken from
    ``KNOWHYPER_PRECOMPUTE_SPLIT`` so the parent need not pickle config.

    Each worker forces CPU + single-threaded math: precompute is link+retrieve
    only (no encoder forward), so it needs no GPU — and letting the SapBERT linker
    load onto an already-contended GPU across N workers triggers CUDA OOM that
    kills the whole pool. On CPU, an unpinned torch worker spawns ~all-core threads,
    so N workers oversubscribe and thrash (load >> ncores); pinning to 1 thread per
    worker keeps the pool fast. These are set before torch is imported.
    """
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    shard_idx, n_shards, _n_total = shard
    split = os.environ["KNOWHYPER_PRECOMPUTE_SPLIT"]
    # Workers recompose config themselves, so the experiment (chemprot khp_re vs
    # DDI khp_ddi etc.) must reach them via env, like split does.
    experiment = os.environ.get("KNOWHYPER_PRECOMPUTE_EXPERIMENT", "khp_re")

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from ieforge import CONFIGS_DIR as IEFORGE_CONFIGS_DIR

    from knowhyper.ieforge._configs import CONFIGS_DIR

    # Belt-and-suspenders: if torch was already imported in the parent (fork), the
    # NUM_THREADS env vars above won't re-apply, so pin the thread count directly.
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass

    overrides = _worker_overrides(experiment, split, str(CONFIGS_DIR))
    with initialize_config_dir(version_base=None, config_dir=str(IEFORGE_CONFIGS_DIR)):
        cfg = compose(config_name="train", overrides=overrides)

    injector_cfg = cfg.knowledge.pipeline.injector
    linker = instantiate(injector_cfg.linker)
    retriever = instantiate(injector_cfg.retriever)
    top_k = int(injector_cfg.top_k)
    name = str(cfg.dataset.name)

    datamodule_keys = {"batch_size", "num_workers", "name", "task_family"}
    dataset_cfg = {k: v for k, v in cfg.dataset.items() if k not in datamodule_keys}
    dataset = instantiate(dataset_cfg)

    shard_examples = [ex for i, ex in enumerate(dataset) if i % n_shards == shard_idx]
    return precompute_split(
        shard_examples,
        linker=linker,
        retriever=retriever,
        dataset=name,
        split=split,
        top_k=top_k,
        mention_type_map=_read_mention_type_map(injector_cfg),
        expand_abbreviations=_read_expand_abbreviations(injector_cfg),
        derive_id_namespaces=_read_derive_id_namespaces(injector_cfg),
        pair_role_provenance=_read_pair_role_provenance(injector_cfg),
        surface_fallback=_read_surface_fallback(cfg.knowledge.pipeline),
        log_every=200,
    )


def main() -> None:
    """CLI: precompute the sub-HG cache for a dataset split.

    Composes the SAME ``experiment=khp_re`` config training uses (so the caching
    retriever's fingerprint matches), instantiates the injector's linker +
    caching retriever, loads the requested split via the dataset config, and runs
    :func:`precompute_split`. Honors ``HF_TOKEN`` from the environment for the
    private datasets. Reads overrides from the command line, e.g.::

        uv run python -m knowhyper.scripts.precompute_subhg_cache \
            knowledge.dataset_path=<snapshot> dataset.split=train

    **GPU-accelerated mode** (recommended)::

        KNOWHYPER_PRECOMPUTE_GPU=1 uv run python -m \
            knowhyper.scripts.precompute_subhg_cache dataset.split=train

    This batch-encodes cache-miss mentions through SapBERT on GPU in one pass,
    then retrieves sequentially or in spawn-based CPU workers. It is typically
    much faster than the fully sequential CPU path for large datasets.

    Set ``KNOWHYPER_PRECOMPUTE_WORKERS=N`` (N>1) to use N CPU retrieval
    processes in either GPU-linking mode or the fully CPU path.
    """
    import os
    import sys
    import time

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from ieforge import CONFIGS_DIR as IEFORGE_CONFIGS_DIR

    from knowhyper.ieforge._configs import CONFIGS_DIR

    logging.basicConfig(level=logging.INFO)
    experiment = "khp_re"
    for a in sys.argv[1:]:
        if a.startswith("experiment="):
            experiment = a.split("=", 1)[1]
    overrides = [
        f"experiment={experiment}",
        f"hydra.searchpath=[file://{CONFIGS_DIR}]",
        *sys.argv[1:],
    ]
    with initialize_config_dir(version_base=None, config_dir=str(IEFORGE_CONFIGS_DIR)):
        cfg = compose(config_name="train", overrides=overrides)

    injector_cfg = cfg.knowledge.pipeline.injector
    top_k = int(injector_cfg.top_k)
    split = str(cfg.dataset.split)
    name = str(cfg.dataset.name)
    datamodule_keys = {"batch_size", "num_workers", "name", "task_family"}
    dataset_cfg = {k: v for k, v in cfg.dataset.items() if k not in datamodule_keys}
    dataset = instantiate(dataset_cfg)
    examples = list(dataset)

    use_gpu = os.environ.get("KNOWHYPER_PRECOMPUTE_GPU", "").strip() not in ("", "0")
    n_workers = int(os.environ.get("KNOWHYPER_PRECOMPUTE_WORKERS", "1"))
    if n_workers < 1:
        msg = f"KNOWHYPER_PRECOMPUTE_WORKERS must be >= 1, got {n_workers}"
        raise ValueError(msg)

    if n_workers > 1:
        import json

        os.environ["KNOWHYPER_PRECOMPUTE_SPLIT"] = split
        os.environ["KNOWHYPER_PRECOMPUTE_EXPERIMENT"] = experiment
        extras = [
            arg
            for arg in sys.argv[1:]
            if not arg.startswith(
                ("experiment=", "dataset.split=", "+dataset.split=", "++dataset.split=")
            )
        ]
        os.environ["KNOWHYPER_PRECOMPUTE_OVERRIDES"] = json.dumps(extras)

    common_kwargs: dict[str, Any] = {
        "dataset": name,
        "split": split,
        "top_k": top_k,
        "mention_type_map": _read_mention_type_map(injector_cfg),
        "expand_abbreviations": _read_expand_abbreviations(injector_cfg),
        "derive_id_namespaces": _read_derive_id_namespaces(injector_cfg),
        "pair_role_provenance": _read_pair_role_provenance(injector_cfg),
        "surface_fallback": _read_surface_fallback(cfg.knowledge.pipeline),
    }

    t_start = time.perf_counter()

    if use_gpu:
        linker = instantiate(injector_cfg.linker)
        retriever = instantiate(injector_cfg.retriever)
        device = os.environ.get("KNOWHYPER_PRECOMPUTE_DEVICE", "auto")
        batch_size = int(os.environ.get("KNOWHYPER_PRECOMPUTE_BATCH_SIZE", "256"))
        logger.info(
            "precompute GPU mode: %d instances, device=%s, batch_size=%d",
            len(examples),
            device,
            batch_size,
        )
        n = precompute_split_gpu(
            examples,
            linker=linker,
            retriever=retriever,
            device=device,
            batch_size=batch_size,
            n_workers=n_workers,
            retriever_config=injector_cfg.retriever,
            **common_kwargs,
        )
    elif n_workers > 1:
        logger.info(
            "precompute %s/%s: %d instances across %d workers",
            name,
            split,
            len(examples),
            n_workers,
        )
        n = precompute_split_parallel(
            n_total=len(examples), n_workers=n_workers, worker=_precompute_shard
        )
    else:
        linker = instantiate(injector_cfg.linker)
        retriever = instantiate(injector_cfg.retriever)
        n = precompute_split(
            examples,
            linker=linker,
            retriever=retriever,
            **common_kwargs,
        )

    elapsed = time.perf_counter() - t_start
    logger.info(
        "precompute complete: %d instances for %s/%s in %.1fs (%.1f ms/instance)",
        n,
        name,
        split,
        elapsed,
        (elapsed / max(n, 1)) * 1000,
    )


if __name__ == "__main__":
    main()
