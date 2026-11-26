#!/usr/bin/env python3
"""Aggregate completed matrix cells into a provenance-headed record.

    uv run python scripts/aggregate_matrix.py \
        --output-root "$KNOWHYPER_HOME/runs/matrix/2026-07-26" \
        > docs/reports/data/matrix_2026_07_26/aggregate.txt

Walks `cell.json` files, groups by (dataset, backbone, method), and emits the
format CLAUDE.md §Result Recording requires: a header stating metric/split/epochs,
then **val AND test** per arm with per-seed values, then paired contrasts against
each arm's declared control.

Recording both splits is not a formality. On DDI this design produced a
val-significant, test-null result four times, once with a sign flip -- a bare
number without its split is unusable.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

from knowhyper.matrix import METHODS_BY_NAME


def _load(output_root: Path) -> list[dict]:
    cells = []
    for p in output_root.rglob("cell.json"):
        try:
            cells.append(json.loads(p.read_text()))
        except Exception as exc:
            print(f"warn: unreadable {p}: {exc}", file=sys.stderr)
    return cells


def _paired(a: dict[int, float], b: dict[int, float]) -> str | None:
    """Paired t-test over seeds present in both arms."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    x = [a[s] for s in seeds]
    y = [b[s] for s in seeds]
    diff = [i - j for i, j in zip(x, y, strict=True)]
    mean = st.fmean(diff)
    sd = st.stdev(diff)
    res = stats.ttest_rel(x, y)
    wins = sum(1 for d in diff if d > 0)
    return (
        f"d {mean:+.4f} sd {sd:.4f} t {res.statistic:+.2f} (df{len(seeds) - 1}) "
        f"p {res.pvalue:.3f} wins {wins}/{len(seeds)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--run-id", default="", help="ABCI run id(s) for the header")
    args = ap.parse_args()

    cells = _load(args.output_root)
    done = [c for c in cells if c.get("status") == "completed"]
    failed = [c for c in cells if c.get("status") != "completed"]

    if not done:
        print("no completed cells found", file=sys.stderr)
        return 1

    epochs = sorted({c.get("epochs") for c in done})
    backbones = sorted({c["backbone"] for c in done})

    print("# Matrix run aggregate")
    print(
        "# METRIC: per-dataset f1 (chemprot/ddi relation_f1, biored biored_f1) "
        "| SPLIT: val = best-epoch validation (from the checkpoint filename), "
        "test = that checkpoint on test"
    )
    print(
        f"# EPOCHS: {','.join(str(e) for e in epochs)} | logger=none | "
        f"save_top_k=1 save_last=false | bs=64 num_workers=8"
    )
    print(f"# BACKBONES: {', '.join(backbones)}")
    if args.run_id:
        print(f"# RUNS: {args.run_id}")
    print(f"# CELLS: {len(done)} completed, {len(failed)} failed")
    print(
        "#\n# Comparable WITHIN a dataset. Do NOT compare across datasets "
        "(different label sets) or across headers (different epoch schedules)."
    )
    if failed:
        print("#\n# FAILED CELLS (not silently dropped):")
        for c in failed:
            print(f"#   {c.get('label', '?')}: {c.get('status')}")
    print()

    # (dataset, backbone) -> method -> seed -> (val, test)
    grouped: dict[tuple[str, str], dict[str, dict[int, tuple[float, float]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    for c in done:
        if c.get("val_f1") is None or c.get("test_f1") is None:
            continue
        # A hyperparameter-sweep arm keys by `method__variant`, not `method`.
        # Grouping sweep cells under the bare method would average every
        # multiplier into one row -- i.e. silently report the mean of the thing
        # being swept, which looks like a valid n=25 result.
        key = c["method"]
        if c.get("variant"):
            key = f"{key}__{c['variant']}"
        grouped[(c["dataset"], c["backbone"])][key][int(c["seed"])] = (
            float(c["val_f1"]),
            float(c["test_f1"]),
        )

    for (dataset, backbone), methods in sorted(grouped.items()):
        print(f"=== {dataset} / {backbone} ===")
        for method, per_seed in sorted(methods.items()):
            for idx, split in ((0, "val "), (1, "test")):
                vals = [per_seed[s][idx] for s in sorted(per_seed)]
                sd = st.stdev(vals) if len(vals) > 1 else 0.0
                pretty = ", ".join(f"{v:.4f}" for v in vals)
                print(
                    f"{method:22} {split}: mean {st.fmean(vals):.4f} sd {sd:.4f} "
                    f"n {len(vals)} per-seed [{pretty}]"
                )
        # Contrasts against each method's declared controls. The first control is
        # the pre-registered primary contrast, so mark it: R2b's headline is
        # `serialized - serialized_shuffled`, not the off_short comparison, and a
        # reader scanning the record must not pick the wrong line.
        for method in sorted(methods):
            # `edge__lrmult3` inherits `edge`'s declared controls, so a sweep arm
            # is still contrasted against `base` rather than dropped silently.
            base_method, _, variant = method.partition("__")
            spec = METHODS_BY_NAME.get(base_method)
            for rank, control in enumerate(spec.controls if spec else ()):
                # When the arm AND its control are both swept, the control's key
                # carries the variant too: `ev_warm__lrmult3`'s control `ev_cold`
                # exists only as `ev_cold__lrmult3`. Prefer the same-variant
                # control (a cross-variant contrast would confound the sweep with
                # the thing being contrasted), then fall back to the bare arm.
                resolved = None
                for candidate in (f"{control}__{variant}" if variant else "", control):
                    if candidate and candidate in methods:
                        resolved = candidate
                        break
                if resolved is None:
                    # Never drop a declared contrast in silence: `ev_warm`'s primary
                    # is another sweepable arm, so this fired for the whole ev sweep
                    # and produced per-arm rows with NO contrast lines at all.
                    print(
                        f"# WARNING: {method} declares control {control!r} but "
                        f"neither {control!r} nor "
                        f"{control + '__' + variant if variant else control!r} "
                        "is present -- contrast NOT computed"
                    )
                    continue
                control = resolved
                tag = "PRIMARY" if rank == 0 else "secondary"
                for idx, split in ((0, "val "), (1, "test")):
                    a = {s: v[idx] for s, v in methods[method].items()}
                    b = {s: v[idx] for s, v in methods[control].items()}
                    line = _paired(a, b)
                    if line:
                        print(f"[{tag}] {method} - {control} {split}: {line}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
