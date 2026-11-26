"""k-hop / CSR latency benchmarks (#135)."""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from primehkg.retrieval.graph_csr.build import build_graph_csr_from_hyperedges
from primehkg.retrieval.khop import BudgetConfig, khop_candidates_k2
from primehkg.retrieval.truncate import HashBasedTruncator

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from primehkg.retrieval.graph_csr.types import GraphCsrInMemory

KhopBenchMode = Literal["sidecar_rebuild", "in_memory_csr", "mmap_csr"]


@dataclass(frozen=True, slots=True)
class KhopLatencyTargets:
    """Advisory SLOs for k-hop candidate generation (k=2)."""

    single_query_p50_ms: float
    single_query_p95_ms: float
    concurrent_workers: int
    concurrent_wall_ms: float


# Full-scale manual targets (graph CSR design: tens-hundreds of ms per query).
FULL_SCALE_TARGETS = KhopLatencyTargets(
    single_query_p50_ms=200.0,
    single_query_p95_ms=500.0,
    concurrent_workers=100,
    concurrent_wall_ms=10_000.0,
)

# Relaxed caps for CI / toy graphs (advisory workflow).
TOY_CI_TARGETS = KhopLatencyTargets(
    single_query_p50_ms=100.0,
    single_query_p95_ms=500.0,
    concurrent_workers=10,
    concurrent_wall_ms=5_000.0,
)


def _percentile_ms(samples_ms: list[float], p: float) -> float:
    if not samples_ms:
        return 0.0
    ordered = sorted(samples_ms)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _one_khop(
    *,
    mode: KhopBenchMode,
    seed_node_ids: list[str],
    query_id: str,
    hyperedges: Sequence[Mapping[str, object]] | None,
    graph_csr: GraphCsrInMemory | None,
    budgets: BudgetConfig,
    truncator: HashBasedTruncator,
) -> list[str]:
    if mode == "sidecar_rebuild":
        if hyperedges is None:
            msg = "hyperedges required for sidecar_rebuild mode"
            raise ValueError(msg)
        return khop_candidates_k2(
            seed_node_ids=seed_node_ids,
            hyperedges=hyperedges,
            budgets=budgets,
            truncator=truncator,
            query_id=query_id,
            graph_csr=None,
        )
    if graph_csr is None:
        msg = "graph_csr required for in_memory_csr / mmap_csr modes"
        raise ValueError(msg)
    return khop_candidates_k2(
        seed_node_ids=seed_node_ids,
        hyperedges=hyperedges or [],
        budgets=budgets,
        truncator=truncator,
        query_id=query_id,
        graph_csr=graph_csr,
    )


def benchmark_khop_latency(
    *,
    mode: KhopBenchMode,
    seed_node_ids: Sequence[str],
    n_queries: int,
    hyperedges: Sequence[Mapping[str, object]] | None = None,
    graph_csr: GraphCsrInMemory | None = None,
    budgets: BudgetConfig | None = None,
    truncator: HashBasedTruncator | None = None,
) -> dict[str, float | int | str]:
    """Time sequential k-hop candidate runs; return summary stats in milliseconds."""
    b = budgets or BudgetConfig(
        max_he_level1=500, max_nodes_level2=500, max_he_level3=500
    )
    t = truncator or HashBasedTruncator()
    seeds = list(seed_node_ids)
    if not seeds:
        msg = "seed_node_ids must be non-empty"
        raise ValueError(msg)
    n = max(1, int(n_queries))
    samples: list[float] = []
    last_len = 0
    for i in range(n):
        qid = f"bench-{i}"
        seed = [seeds[i % len(seeds)]]
        t0 = time.perf_counter()
        out = _one_khop(
            mode=mode,
            seed_node_ids=seed,
            query_id=qid,
            hyperedges=hyperedges,
            graph_csr=graph_csr,
            budgets=b,
            truncator=t,
        )
        samples.append((time.perf_counter() - t0) * 1000.0)
        last_len = len(out)
    return {
        "mode": mode,
        "n_queries": n,
        "mean_ms": statistics.mean(samples),
        "p50_ms": _percentile_ms(samples, 50.0),
        "p95_ms": _percentile_ms(samples, 95.0),
        "max_ms": max(samples),
        "last_candidate_count": last_len,
    }


def benchmark_khop_concurrent(
    *,
    mode: KhopBenchMode,
    seed_node_ids: Sequence[str],
    n_queries: int,
    workers: int,
    hyperedges: Sequence[Mapping[str, object]] | None = None,
    graph_csr: GraphCsrInMemory | None = None,
    budgets: BudgetConfig | None = None,
    truncator: HashBasedTruncator | None = None,
) -> dict[str, float | int | str]:
    """Run k-hop queries concurrently; report wall-clock ms for the pool."""
    b = budgets or BudgetConfig(
        max_he_level1=500, max_nodes_level2=500, max_he_level3=500
    )
    t = truncator or HashBasedTruncator()
    seeds = list(seed_node_ids)
    n = max(1, int(n_queries))
    w = max(1, int(workers))

    def _task(i: int) -> None:
        _one_khop(
            mode=mode,
            seed_node_ids=[seeds[i % len(seeds)]],
            query_id=f"bench-conc-{i}",
            hyperedges=hyperedges,
            graph_csr=graph_csr,
            budgets=b,
            truncator=t,
        )

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=w) as pool:
        futures = [pool.submit(_task, i) for i in range(n)]
        for fut in as_completed(futures):
            fut.result()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "mode": mode,
        "n_queries": n,
        "workers": w,
        "wall_ms": wall_ms,
    }


def _close_mmap_graph_csr(graph_csr: Mapping[str, object]) -> None:
    for key in ("node_ptr", "node_adj_he", "he_ptr", "he_adj_node"):
        close = getattr(graph_csr.get(key), "close", None)
        if callable(close):
            close()


def compare_khop_modes(
    hyperedges: Sequence[Mapping[str, object]],
    *,
    mmap_sidecar_dir: str | None = None,
    seed_node_ids: Sequence[str],
    n_queries: int = 50,
    inputs_fingerprint: str = "bench-fp",
) -> dict[str, object]:
    """Run rebuild vs in-memory CSR vs mmap CSR and return a JSON-serializable report."""
    import tempfile
    from pathlib import Path

    from primehkg.retrieval.graph_csr.io import (
        load_graph_csr_sidecar,
        write_graph_csr_sidecar,
    )

    g = build_graph_csr_from_hyperedges(
        hyperedges, inputs_fingerprint=inputs_fingerprint
    )
    if mmap_sidecar_dir is not None:
        mmap_g = load_graph_csr_sidecar(
            Path(mmap_sidecar_dir),
            expected_fingerprint=inputs_fingerprint,
            strict_fingerprint=True,
        )
        try:
            return _compare_three_modes(
                hyperedges,
                seed_node_ids=seed_node_ids,
                n_queries=n_queries,
                in_memory=g,
                mmap=mmap_g,
            )
        finally:
            _close_mmap_graph_csr(mmap_g)

    with tempfile.TemporaryDirectory() as tmp:
        sidecar = Path(tmp) / "graph_csr_v1"
        sidecar.mkdir()
        write_graph_csr_sidecar(sidecar, g)
        mmap_g = load_graph_csr_sidecar(
            sidecar,
            expected_fingerprint=inputs_fingerprint,
            strict_fingerprint=True,
        )
        try:
            return _compare_three_modes(
                hyperedges,
                seed_node_ids=seed_node_ids,
                n_queries=n_queries,
                in_memory=g,
                mmap=mmap_g,
            )
        finally:
            _close_mmap_graph_csr(mmap_g)


def _compare_three_modes(
    hyperedges: Sequence[Mapping[str, object]],
    *,
    seed_node_ids: Sequence[str],
    n_queries: int,
    in_memory: GraphCsrInMemory,
    mmap: GraphCsrInMemory,
) -> dict[str, object]:
    return {
        "n_queries": n_queries,
        "sequential": {
            "sidecar_rebuild": benchmark_khop_latency(
                mode="sidecar_rebuild",
                seed_node_ids=seed_node_ids,
                n_queries=n_queries,
                hyperedges=hyperedges,
            ),
            "in_memory_csr": benchmark_khop_latency(
                mode="in_memory_csr",
                seed_node_ids=seed_node_ids,
                n_queries=n_queries,
                hyperedges=hyperedges,
                graph_csr=in_memory,
            ),
            "mmap_csr": benchmark_khop_latency(
                mode="mmap_csr",
                seed_node_ids=seed_node_ids,
                n_queries=n_queries,
                hyperedges=hyperedges,
                graph_csr=mmap,
            ),
        },
    }


def check_meets_targets(
    seq_stats: Mapping[str, float | int | str],
    *,
    targets: KhopLatencyTargets,
) -> dict[str, bool]:
    """Return pass/fail flags for advisory SLO checks."""
    p50 = float(seq_stats["p50_ms"])
    p95 = float(seq_stats["p95_ms"])
    return {
        "p50_ok": p50 <= targets.single_query_p50_ms,
        "p95_ok": p95 <= targets.single_query_p95_ms,
    }
