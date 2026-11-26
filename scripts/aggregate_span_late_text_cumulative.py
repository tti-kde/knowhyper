#!/usr/bin/env python3
"""Aggregate fixed cumulative NER and decide the pre-registered promotion rule.

The output always reports absolute validation and test F1 for real/shuffled cumulative
arms and both completed standalone comparators. Delta statistics appear only in this audit
record, never in the paper's main table.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from knowhyper import pins
from knowhyper.span_stats import holm, paired_delta

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

DEFAULT_BACKBONES = ("pubmedbert", "biolinkbert")
DEFAULT_DATASETS = ("bc5cdr_chem", "bc2gm", "bc5cdr_disease")
ARMS = ("cumulative_real", "cumulative_shuffled", "text_only", "graph_only")
_TEST_F1 = re.compile(r"\btest/f1\b[^0-9]*([01]\.\d+)")
_CKPT = re.compile(r"epoch_(\d+)-f1_([0-9.]+)\.ckpt")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--text-cells", type=Path, required=True)
    parser.add_argument("--graph-cells", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--backbones", nargs="+", default=list(DEFAULT_BACKBONES))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    return parser.parse_args()


def _best_val(run_dir: Path) -> float | None:
    names: list[str] = []
    inventory = run_dir / "best_checkpoint_name.txt"
    if inventory.is_file():
        names.extend(inventory.read_text().split())
    checkpoints = run_dir / "checkpoints"
    if checkpoints.is_dir():
        names.extend(path.name for path in checkpoints.iterdir())
    values = [float(match.group(2)) for name in names if (match := _CKPT.search(name))]
    return max(values) if values else None


def _parse_run(run_dir: Path) -> dict[str, float] | None:
    log = run_dir / "driver.log"
    if not log.is_file():
        return None
    matches = list(_TEST_F1.finditer(log.read_text(errors="replace")))
    val = _best_val(run_dir)
    if not matches or val is None:
        return None
    return {"val": val, "test": float(matches[-1].group(1))}


def load_cumulative(
    runs_root: Path,
) -> dict[tuple[str, str, str], dict[int, dict[str, float]]]:
    found: dict[tuple[str, str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    arm_map = {
        "text_ev_warm_real": "cumulative_real",
        "text_ev_warm_shuffled": "cumulative_shuffled",
    }
    for log in sorted(runs_root.glob("*/*/*/seed_*/driver.log")):
        run_dir = log.parent
        arm = arm_map.get(run_dir.parent.name)
        if arm is None:
            continue
        backbone = run_dir.parents[2].name
        dataset = run_dir.parents[1].name
        try:
            seed = int(run_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        scores = _parse_run(run_dir)
        if scores is not None:
            found[(backbone, dataset, arm)][seed] = scores
    return found


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        msg = f"comparator cells file missing: {path}"
        raise FileNotFoundError(msg)
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            msg = f"invalid JSON at {path}:{line_number}"
            raise ValueError(msg) from error
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_comparators(
    text_path: Path, graph_path: Path
) -> dict[tuple[str, str, str], dict[int, dict[str, float]]]:
    found: dict[tuple[str, str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    for row in _load_jsonl(text_path):
        if row.get("arm") != "text_real":
            continue
        _insert_comparator(found, row, "text_only", text_path)
    for row in _load_jsonl(graph_path):
        if row.get("control") != "real":
            continue
        _insert_comparator(found, row, "graph_only", graph_path)
    return found


def _insert_comparator(
    target: dict[tuple[str, str, str], dict[int, dict[str, float]]],
    row: Mapping[str, Any],
    arm: str,
    source: Path,
) -> None:
    key = (str(row["backbone"]), str(row["dataset"]), arm)
    seed = int(row["seed"])
    scores = {"val": float(row["val"]), "test": float(row["test"])}
    previous = target[key].get(seed)
    if previous is not None and previous != scores:
        msg = f"conflicting duplicate comparator {key}/seed_{seed} in {source}"
        raise ValueError(msg)
    target[key][seed] = scores


def _summary(values: Iterable[float]) -> tuple[float, float]:
    materialized = list(values)
    return statistics.mean(materialized), statistics.stdev(materialized) if len(
        materialized
    ) > 1 else 0.0


def _contrast_rows(
    data: Mapping[tuple[str, str, str], Mapping[int, Mapping[str, float]]],
    treated_arm: str,
    control_arm: str,
    split: str,
    backbones: tuple[str, ...],
    datasets: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backbone in backbones:
        for dataset in datasets:
            treated = data.get((backbone, dataset, treated_arm), {})
            control = data.get((backbone, dataset, control_arm), {})
            shared = sorted(set(treated) & set(control))
            if len(shared) < 2:
                rows.append({"backbone": backbone, "dataset": dataset, "result": None})
                continue
            result = paired_delta(
                [treated[seed][split] for seed in shared],
                [control[seed][split] for seed in shared],
            )
            rows.append({"backbone": backbone, "dataset": dataset, "result": result})
    measured = [row for row in rows if row["result"] is not None]
    adjusted = holm([row["result"].p for row in measured])
    for row, p_adjusted in zip(measured, adjusted, strict=True):
        row["holm"] = p_adjusted
    return rows


def main() -> int:
    args = _parse_args()
    if not args.runs_root.is_dir():
        print(f"FATAL cumulative runs root missing: {args.runs_root}", file=sys.stderr)
        return 2
    try:
        data = load_cumulative(args.runs_root)
        comparators = load_comparators(args.text_cells, args.graph_cells)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"FATAL {error}", file=sys.stderr)
        return 2
    backbones = tuple(args.backbones)
    datasets = tuple(args.datasets)
    for key, seeds in comparators.items():
        data[key].update(seeds)

    expected = {
        (backbone, dataset, arm)
        for backbone in backbones
        for dataset in datasets
        for arm in ARMS
    }
    incomplete = [
        key for key in sorted(expected) if set(data.get(key, {})) != set(range(10))
    ]
    if incomplete:
        print(f"FATAL incomplete n=10 arms: {incomplete}", file=sys.stderr)
        return 2

    contrast_specs = (
        ("content_real_minus_shuffled", "cumulative_real", "cumulative_shuffled"),
        ("cumulative_minus_text_only", "cumulative_real", "text_only"),
        ("cumulative_minus_graph_only", "cumulative_real", "graph_only"),
    )
    contrasts = {
        (name, split): _contrast_rows(
            data, treated, control, split, backbones, datasets
        )
        for name, treated, control in contrast_specs
        for split in ("val", "test")
    }

    test_content = contrasts[("content_real_minus_shuffled", "test")]
    test_vs_text = contrasts[("cumulative_minus_text_only", "test")]
    test_vs_graph = contrasts[("cumulative_minus_graph_only", "test")]
    higher_than_both = False
    for backbone in backbones:
        for dataset in datasets:
            real_mean = _summary(
                score["test"]
                for score in data[(backbone, dataset, "cumulative_real")].values()
            )[0]
            text_mean = _summary(
                score["test"]
                for score in data[(backbone, dataset, "text_only")].values()
            )[0]
            graph_mean = _summary(
                score["test"]
                for score in data[(backbone, dataset, "graph_only")].values()
            )[0]
            higher_than_both |= real_mean > text_mean and real_mean > graph_mean
    content_significant = any(
        row.get("holm", 1.0) < 0.05 and row["result"].delta > 0 for row in test_content
    )
    significant_loss = any(
        row.get("holm", 1.0) < 0.05 and row["result"].delta < 0
        for row in (*test_vs_text, *test_vs_graph)
        if row["result"] is not None
    )
    promoted = higher_than_both and content_significant and not significant_loss

    lines = [
        "# METRIC: f1 (ieforge EntityF1; exact entity-level micro) | SPLITS: val best checkpoint, test that checkpoint",
        "# PROTOCOL: 10 epochs | bs=16 | seeds 0-9 | fixed #220 text + #231 ev_warm; no sweep",
        f"# CORE: PrimeHKG {pins.CORE_REVISION} | FEATURES: {pins.FEATURES_REVISION} | PACKAGE: {pins.PACKAGE_REV}",
        f"# IEFORGE: 0.1.11 | COMMIT: {args.commit} | RUN: {args.run_id}",
        "# SCOPE: backbones="
        + ",".join(backbones)
        + " | datasets="
        + ",".join(datasets),
        f"# PROMOTED: {str(promoted).lower()} | higher_than_both={higher_than_both} | content_holm={content_significant} | significant_loss={significant_loss}",
        "",
        "## absolute F1",
    ]
    cell_rows = []
    for backbone in backbones:
        for dataset in datasets:
            for arm in ARMS:
                seeds = data[(backbone, dataset, arm)]
                for split in ("val", "test"):
                    values = [seeds[seed][split] for seed in range(10)]
                    mean, sd = _summary(values)
                    lines.append(
                        f"{backbone}/{dataset}/{arm} {split}: mean {mean:.4f} sd {sd:.4f} n=10 per-seed {[round(value, 4) for value in values]}"
                    )
                cell_rows.extend(
                    {
                        "backbone": backbone,
                        "dataset": dataset,
                        "arm": arm,
                        "seed": seed,
                        **seeds[seed],
                    }
                    for seed in range(10)
                )
    lines.append("")
    for name, _, _ in contrast_specs:
        for split in ("val", "test"):
            lines.append(
                f"## {name} ({split}; Holm over {len(backbones) * len(datasets)})"
            )
            for row in contrasts[(name, split)]:
                result = row["result"]
                if result is None:
                    lines.append(f"{row['backbone']}/{row['dataset']}: INCOMPLETE")
                    continue
                verdict = "SIGNIFICANT" if row["holm"] < 0.05 else "n.s."
                lines.append(
                    f"{row['backbone']}/{row['dataset']}: delta {result.delta:+.4f} sd {result.sd:.4f} t {result.t:+.2f} p {result.p:.4g} holm {row['holm']:.4g} wins {result.wins}/{result.n} {verdict}"
                )
            lines.append("")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "aggregate.txt").write_text("\n".join(lines) + "\n")
    with (args.out / "cells.jsonl").open("w") as writer:
        for row in cell_rows:
            writer.write(json.dumps(row, sort_keys=True) + "\n")
    (args.out / "promotion.json").write_text(
        json.dumps(
            {
                "promoted": promoted,
                "higher_than_both": higher_than_both,
                "content_holm_significant": content_significant,
                "significant_loss": significant_loss,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {args.out}; promoted={promoted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
