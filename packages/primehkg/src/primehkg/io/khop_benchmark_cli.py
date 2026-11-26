"""CLI for k-hop / CSR latency benchmarks (#135)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import typer
from typer.main import get_command

from primehkg.retrieval.benchmark_khop import (
    FULL_SCALE_TARGETS,
    TOY_CI_TARGETS,
    benchmark_khop_concurrent,
    check_meets_targets,
    compare_khop_modes,
)
from primehkg.retrieval.graph_csr.build import build_graph_csr_from_hyperedges

app = typer.Typer(
    rich_markup_mode="rich",
    help="Benchmark k-hop candidate generation (rebuild vs CSR vs mmap).",
)

HYPEREDGES_JSON_OPT = typer.Option(
    ...,
    "--hyperedges-json",
    exists=True,
    dir_okay=False,
    readable=True,
)
SIDECAR_DIR_OPT = typer.Option(
    None,
    "--sidecar-dir",
    exists=True,
    file_okay=False,
    dir_okay=True,
    help="Optional published graph_csr_v1 directory (else temp sidecar).",
)
N_QUERIES_OPT = typer.Option(50, "--n-queries", min=1)
WORKERS_OPT = typer.Option(1, "--workers", min=1)
INPUTS_FINGERPRINT_OPT = typer.Option(
    "bench-fp",
    "--inputs-fingerprint",
    help="Fingerprint recorded in CSR meta (must match sidecar when set).",
)
REPORT_JSON_OPT = typer.Option(
    None,
    "--report-json",
    dir_okay=False,
    writable=True,
)
TOY_TARGETS_OPT = typer.Option(
    False,
    "--toy-targets",
    help="Use relaxed SLO caps (CI advisory).",
)


def _read_hyperedges_json(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        msg = f"{path} must be a JSON array of hyperedge objects"
        raise typer.BadParameter(msg)
    return [cast("dict[str, Any]", x) for x in raw if isinstance(x, dict)]


@app.command()
def run_command(
    hyperedges_json: Path = HYPEREDGES_JSON_OPT,
    sidecar_dir: Path | None = SIDECAR_DIR_OPT,
    n_queries: int = N_QUERIES_OPT,
    workers: int = WORKERS_OPT,
    inputs_fingerprint: str = INPUTS_FINGERPRINT_OPT,
    report_json: Path | None = REPORT_JSON_OPT,
    toy_targets: bool = TOY_TARGETS_OPT,
) -> None:
    """Compare rebuild / in-memory CSR / mmap CSR latencies."""
    hyperedges_json = Path(hyperedges_json)
    sidecar_dir = Path(sidecar_dir) if sidecar_dir is not None else None
    report_json = Path(report_json) if report_json is not None else None
    hes = _read_hyperedges_json(hyperedges_json)
    seeds: list[str] = []
    for he in hes[:10]:
        nids = he.get("node_ids")
        if isinstance(nids, list):
            for n in nids:
                if isinstance(n, str) and n not in seeds:
                    seeds.append(n)
    if not seeds:
        msg = "No seed node ids found in hyperedges JSON"
        raise typer.BadParameter(msg)

    report: dict[str, Any] = dict(
        compare_khop_modes(
            hes,
            mmap_sidecar_dir=str(sidecar_dir) if sidecar_dir is not None else None,
            seed_node_ids=seeds[:5],
            n_queries=n_queries,
            inputs_fingerprint=inputs_fingerprint,
        )
    )
    targets = TOY_CI_TARGETS if toy_targets else FULL_SCALE_TARGETS
    seq = cast("dict[str, dict[str, float | int | str]]", report["sequential"])
    mmap_stats = seq["mmap_csr"]
    slo: dict[str, Any] = {
        "targets": {
            "single_query_p50_ms": targets.single_query_p50_ms,
            "single_query_p95_ms": targets.single_query_p95_ms,
        },
        "mmap_csr": check_meets_targets(mmap_stats, targets=targets),
    }
    report["slo"] = slo
    if workers > 1:
        from primehkg.retrieval.graph_csr.io import load_graph_csr_sidecar

        mmap_g = None
        if sidecar_dir is not None:
            mmap_g = load_graph_csr_sidecar(
                sidecar_dir,
                expected_fingerprint=inputs_fingerprint,
                strict_fingerprint=True,
            )
        else:
            mmap_g = build_graph_csr_from_hyperedges(
                hes, inputs_fingerprint=inputs_fingerprint
            )
        report["concurrent"] = benchmark_khop_concurrent(
            mode="mmap_csr" if sidecar_dir is not None else "in_memory_csr",
            seed_node_ids=seeds[:5],
            n_queries=n_queries,
            workers=workers,
            hyperedges=hes,
            graph_csr=mmap_g,
        )
        wall = float(report["concurrent"]["wall_ms"])
        slo["concurrent_wall_ok"] = wall <= targets.concurrent_wall_ms

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if report_json is not None:
        report_json.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def entrypoint(argv: list[str] | None = None) -> None:
    get_command(app)(args=list(argv) if argv is not None else None)


if __name__ == "__main__":
    entrypoint()
