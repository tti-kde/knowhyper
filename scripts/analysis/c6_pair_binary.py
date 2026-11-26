#!/usr/bin/env python3
"""Pair the C6 `binary` arm against a previously recorded `ev_cold` arm.

`aggregate_matrix.py` computes a contrast only when both arms live in the SAME
output root. The 2026-08-03 C6 re-run deliberately reuses the features-0.12.15
`ev_cold` cells rather than re-running them, so the paired contrast has to be
assembled from two records. This script does that -- and refuses to do it when
the two records are not comparable.

Comparability is not assumed. The 2026-07-21 C6 result could not fill the
paper's ablation TBDs precisely because it disagreed with the table on metric
(val vs test), features revision (attributes were all-zero through 0.12.14),
and cache fingerprint. So the checks below are the point of the script, not
boilerplate:

  * seeds must MATCH as sets, not merely count 10 each -- pairing by position
    across two campaigns is how a seed-2-vs-seed-7 diff becomes a "result".
  * the control's per-seed values are read from the recorded aggregate, so the
    numbers that enter the test are the same ones the paper cites.

Usage:
  uv run python scripts/analysis/c6_pair_binary.py \
      --output-root "$KNOWHYPER_HOME/runs/matrix/c6_binary_2026_08_03" \
      --control docs/reports/data/re_ev_features0_12_15_2026_07_29/aggregate.txt
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

from scipy import stats as sps

# `chemprot_ev_cold_test: mean 0.7593 sd 0.0045 n 10 per-seed [0.7651, ...]`
_ROW = re.compile(
    r"^(?P<dataset>\w+?)_(?P<method>ev_cold)_(?P<split>val|test):.*?per-seed \[(?P<vals>[^\]]+)\]"
)


def read_control(
    path: Path, method: str = "ev_cold"
) -> dict[tuple[str, str], list[float]]:
    """Per-seed control values keyed by (dataset, split).

    The recorded aggregate lists per-seed values in seed order (seeds 0-9);
    `aggregate_matrix.py` emits them via `sorted(per_seed)`. We therefore index
    them as seed i -> value i, and the caller verifies the treated arm really
    carries seeds 0..n-1 before pairing.
    """
    row = re.compile(
        rf"^(?P<dataset>\w+?)_{re.escape(method)}_(?P<split>val|test):"
        r".*?per-seed \[(?P<vals>[^\]]+)\]"
    )
    out: dict[tuple[str, str], list[float]] = {}
    for line in path.read_text().splitlines():
        m = row.match(line.strip())
        if not m:
            continue
        vals = [float(v) for v in m.group("vals").split(",")]
        out[(m.group("dataset"), m.group("split").strip())] = vals
    return out


def read_treated(root: Path, method: str) -> dict[tuple[str, str], dict[int, float]]:
    """Per-seed treated-arm values keyed by (dataset, split) from cell.json files."""
    out: dict[tuple[str, str], dict[int, float]] = {}
    for cell_file in sorted(root.rglob("cell.json")):
        cell = json.loads(cell_file.read_text())
        if cell.get("method") != method:
            continue
        if cell.get("status") != "completed":
            continue
        if cell.get("val_f1") is None or cell.get("test_f1") is None:
            continue
        seed = int(cell["seed"])
        for split, key in (("val", "val_f1"), ("test", "test_f1")):
            out.setdefault((cell["dataset"], split), {})[seed] = float(cell[key])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--control", required=True, type=Path)
    # Defaults keep the 2026-08-03 C6 invocation reproducible verbatim; the flags
    # exist so the sibling modality controls reuse this pairing rather than getting
    # a near-copy of it, which is how two records drift apart.
    ap.add_argument("--method", default="binary")
    ap.add_argument("--control-method", default="ev_cold")
    args = ap.parse_args()

    control = read_control(args.control, args.control_method)
    treated = read_treated(args.output_root, args.method)

    if not treated:
        print(
            f"no completed `{args.method}` cells under {args.output_root}",
            file=sys.stderr,
        )
        return 1

    print(
        f"# paired contrast `{args.method}` vs `{args.control_method}`, treated from {args.output_root}"
    )
    print(f"# control `{args.control_method}` per-seed read from {args.control}")
    print(
        f"# METRIC: relation_f1 | treated={args.method} control={args.control_method}"
    )
    print("# Both arms: features 0.12.15 / core 0.12.13, shared SubHG fingerprint,")
    print("# PubMedBERT, difflr_nolog, knowledge_lr_mult=10, bs=64, 10 epochs.")
    print()

    exit_code = 0
    for dataset, split in sorted(treated):
        per_seed = treated[(dataset, split)]
        ctrl = control.get((dataset, split))
        if ctrl is None:
            print(f"# WARNING: no control row for {dataset}/{split} -- SKIPPED")
            exit_code = 1
            continue

        seeds = sorted(per_seed)
        # Pairing by position across campaigns is only valid if the treated arm
        # really carries seeds 0..n-1 and the control has the same count.
        if seeds != list(range(len(ctrl))):
            print(
                f"# WARNING: {dataset}/{split} treated seeds {seeds} do not match "
                f"control's {len(ctrl)} seeds (0..{len(ctrl) - 1}) -- NOT PAIRED"
            )
            exit_code = 1
            continue

        tvals = [per_seed[s] for s in seeds]
        diffs = [t - c for t, c in zip(tvals, ctrl)]
        res = sps.ttest_rel(tvals, ctrl)
        wil = sps.wilcoxon(tvals, ctrl).pvalue if any(diffs) else float("nan")
        mean_d = st.fmean(diffs)
        sd_d = st.stdev(diffs) if len(diffs) > 1 else 0.0
        wins = sum(d > 0 for d in diffs)
        ties = sum(d == 0 for d in diffs)

        pretty_t = ", ".join(f"{v:.4f}" for v in tvals)
        pretty_c = ", ".join(f"{v:.4f}" for v in ctrl)
        pretty_d = ", ".join(f"{d:+.4f}" for d in diffs)
        print(f"=== {dataset} / {split} (n={len(seeds)}) ===")
        print(
            f"{args.method:<8}: mean {st.fmean(tvals):.4f} sd {st.stdev(tvals):.4f} per-seed [{pretty_t}]"
        )
        print(
            f"{args.control_method:<8}: mean {st.fmean(ctrl):.4f} sd {st.stdev(ctrl):.4f} per-seed [{pretty_c}]"
        )
        print(f"per-seed diff ({args.method} - {args.control_method}): [{pretty_d}]")
        # Spell out wins/losses/ties rather than `wins {w}/{n}`. The bare form is
        # what aggregate_matrix.py emits, and there the denominator is the seed
        # COUNT -- so printing wins-over-losses in the same shape would read as
        # wins-over-total and quietly understate n.
        print(
            f"{args.method} - {args.control_method}: d {mean_d:+.4f} sd {sd_d:.4f} "
            f"t {res.statistic:+.3f} (df{len(seeds) - 1}) p {res.pvalue:.3f} "
            f"wilcoxon {wil:.3f} "
            f"wins {wins} / losses {len(seeds) - ties - wins} / ties {ties} "
            f"(n={len(seeds)})"
        )
        print()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
