#!/usr/bin/env python
"""Aggregate the #220 span Knowledge-as-text campaign into Table 6 inputs.

    uv run python scripts/aggregate_span_ner_text.py \
        --runs-root "$KNOWHYPER_HOME/runs/ner_span_text" \
        --out docs/reports/data/ner_span_text_2026_08_01 \
        --runs <abci-run-id> --commit "$(git rev-parse --short HEAD)"

Writes `aggregate.txt` (provenance header + per-arm lines + the three Holm-corrected
contrast families) and `cells.jsonl` (one row per cell).

Two parsing rules are inherited from the proven #218/#202 aggregators rather than
re-derived, because both were bugs once:

* test comes from the EXACT key `test/f1` in driver.log. `EntityF1` is the metric
  CLASS and `f1` the logged key; a loose `*f1` pattern also matches `test/macro_f1`
  and, being last-wins, silently swaps the recorded number.
* validation comes from the CHECKPOINT FILENAME, not the log. The log holds one val
  line per epoch and the last is the FINAL epoch, while `save_top_k=1` keeps the
  BEST -- reading the log reports a drifted score under a best-epoch header.

A cell with no test score or no recoverable best-epoch val is REJECTED, not admitted
as val-only: that would silently shrink the paired n, and walltime kills land on the
slowest arm first -- the arm most worth measuring.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from knowhyper.span_text_campaign import ARMS
from knowhyper.span_text_stats import holm, paired_delta, text_contrast_families

_TEST_F1 = re.compile(r"\btest/f1\b[^0-9]*([01]\.\d+)")
_CKPT = re.compile(r"epoch_(\d+)-f1_([0-9.]+)\.ckpt")


def _best_val(run_dir: Path) -> float | None:
    """Best-epoch validation f1, from the checkpoint filename inventory."""
    names: list[str] = []
    inventory = run_dir / "best_checkpoint_name.txt"
    if inventory.is_file():
        names.extend(inventory.read_text().split())
    checkpoints = run_dir / "checkpoints"
    if checkpoints.is_dir():
        names.extend(path.name for path in checkpoints.iterdir())

    best: float | None = None
    for name in names:
        match = _CKPT.search(name)
        if match is None:
            continue
        value = float(match.group(2))
        best = value if best is None else max(best, value)
    return best


def parse_cell(run_dir: Path) -> dict[str, float] | None:
    """Return {val, test, degenerate} for a finished cell, or None if unusable.

    `degenerate` marks a cell whose best-epoch validation f1 is exactly 0 — the
    span head collapsed to predicting nothing (focal loss is minimised that way
    on a sparse label space), `EarlyStopping` then ended the run and
    `save_top_k=1` kept the epoch-0 checkpoint. Such a cell exits 0 and the
    driver records `OK`, so nothing upstream flags it.

    It is REPORTED, not silently dropped: excluding it quietly would hide the
    very defect it evidences, and averaging it in would let a 0.0000 drag an
    arm's mean without explanation. Phase 2 (n=3) had 4 of 36 text cells here.
    """
    log = run_dir / "driver.log"
    if not log.is_file():
        return None
    test_match = None
    for test_match in _TEST_F1.finditer(log.read_text(errors="replace")):  # noqa: B007
        pass
    if test_match is None:
        return None
    best_val = _best_val(run_dir)
    if best_val is None:
        return None
    return {
        "val": best_val,
        "test": float(test_match.group(1)),
        "degenerate": float(best_val == 0.0),
    }


def collect(
    *runs_roots: Path,
) -> dict[tuple[str, str, str], dict[int, dict[str, float]]]:
    """Map (backbone, dataset, arm) -> seed -> {val, test} across one or more roots.

    Several roots are needed when a phase is re-run in part: #220 Phase 2b
    re-ran only the 36 text cells (into `..._phase2b`) after the trainer fix,
    while the 18 `base` cells stayed correct in the original `..._phase2` tree.

    LATER ROOTS WIN on a duplicate (backbone, dataset, arm, seed), so pass the
    re-run root last. A silent first-wins would have kept the superseded numbers,
    which is exactly the failure the re-run existed to undo.
    """
    found: dict[tuple[str, str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    for runs_root in runs_roots:
        for log in sorted(runs_root.glob("*/*/*/seed_*/driver.log")):
            seed_dir = log.parent
            arm_dir = seed_dir.parent
            dataset_dir = arm_dir.parent
            backbone_dir = dataset_dir.parent
            if arm_dir.name not in ARMS:
                continue
            try:
                seed = int(seed_dir.name.removeprefix("seed_"))
            except ValueError:
                continue
            parsed = parse_cell(seed_dir)
            if parsed is None:
                continue
            found[(backbone_dir.name, dataset_dir.name, arm_dir.name)][seed] = parsed
    return found


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        required=True,
        type=Path,
        nargs="+",
        help=(
            "one or more run roots; LATER roots win on a duplicate cell, so pass "
            "a re-run root last (Phase 2b's text cells supersede Phase 2's)"
        ),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--runs", default="<unrecorded>")
    parser.add_argument("--commit", default="<unrecorded>")
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["pubmedbert", "biolinkbert"],
        help="backbones included in each Holm family",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bc5cdr_chem", "bc2gm", "bc5cdr_disease"],
        help="datasets included in each Holm family",
    )
    args = parser.parse_args()

    found = collect(*args.runs_root)
    if not found:
        print(f"no usable cells under {args.runs_root}")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    with (args.out / "cells.jsonl").open("w") as handle:
        for (backbone, dataset, arm), seeds in sorted(found.items()):
            for seed, values in sorted(seeds.items()):
                handle.write(
                    json.dumps(
                        {
                            "backbone": backbone,
                            "dataset": dataset,
                            "arm": arm,
                            "seed": seed,
                            "val": values["val"],
                            "test": values["test"],
                            "degenerate": bool(values.get("degenerate")),
                            "commit": args.commit,
                        }
                    )
                    + "\n"
                )

    lines: list[str] = [
        "# METRIC: f1 (ieforge EntityF1: exact entity-level micro) | SPLIT: val = "
        "best-epoch validation from the checkpoint filename, test = that checkpoint "
        "on test",
        # bs=16, NOT #218's 64: the span arms pin it (da98366) because
        # `_run_defaults`' 64 is an RE value that undertrains them. The header is
        # the only thing a later reader has to tell comparable runs from
        # non-comparable ones, so this must state the protocol that actually ran.
        # benchmark_nolog on EVERY arm: all three are plain HFSpanClassifiers and
        # difflr would put their whole backbone in a 10x group.
        "# EPOCHS: 10 | logger=none | bs=16 | TRAINER: benchmark_nolog (all arms, "
        "no differential LR: the text route has no knowledge branch)",
        "# SPAN: start_end max_span_width=12 | LOSS: focal | DECODER: max_total_score",
        "# PROPOSALS: backbone-specific OOF lattices, shared within each cell | "
        "RETRIEVAL: config-resolved prediction-scoped SubHG; exact fingerprints are "
        "recorded in each run's .hydra config and the preflight cache-scope record",
        "# PAYLOAD: max_targets=8 max_knowledge_tokens=256 min_source_tokens=256",
        f"# IEFORGE: 0.1.11 | COMMIT: {args.commit} | RUNS: {args.runs}",
        "# HEADLINE SPLIT: test (validation selects the checkpoint)",
        "# BASE IS MATCHED, NOT BORROWED: #218's shared_zero carries ~11.8M fusion "
        "params in a 10x-LR group (Phase 0.3), so it is not a comparator for these "
        "arms.",
        "",
    ]

    # BOTH splits, always -- not just the one `--split` selects for the contrasts.
    # "Record val AND test. They disagree in magnitude and significance ... A bare
    # number is unusable." (CLAUDE.md, Result Recording Convention). Emitting only
    # the selected split would force the operator to run this twice and merge the
    # outputs by hand, and a file that silently holds one split is exactly the
    # "bare number" the convention forbids. `--split` still chooses which split the
    # paired contrasts below are computed on.
    for (backbone, dataset, arm), seeds in sorted(found.items()):
        for split in ("val", "test"):
            values = [seeds[s][split] for s in sorted(seeds) if split in seeds[s]]
            if not values:
                continue
            lines.append(
                f"{backbone}/{dataset}/{arm} {split}: mean {_mean(values):.4f} "
                f"sd {_sd(values):.4f} n={len(values)} "
                f"per-seed [{', '.join(f'{v:.4f}' for v in values)}]"
            )
        # Name the collapsed seeds on their own line. A 0.0000 inside `per-seed`
        # is easy to read as a bad run rather than a model that predicted
        # nothing, and it drags the mean without saying so.
        degenerate = [s for s in sorted(seeds) if seeds[s].get("degenerate")]
        if degenerate:
            lines.append(
                f"{backbone}/{dataset}/{arm} DEGENERATE: seeds {degenerate} "
                f"({len(degenerate)}/{len(seeds)}) collapsed to the empty "
                "prediction (best-epoch val f1 = 0); their 0.0000 IS included "
                "in the mean above — treat this arm as unmeasured"
            )
    lines.append("")

    families = text_contrast_families(
        backbones=tuple(args.backbones), datasets=tuple(args.datasets)
    )
    lines.insert(
        7,
        "# CONTRAST SCOPE: backbones="
        + ",".join(args.backbones)
        + " | datasets="
        + ",".join(args.datasets),
    )
    for family_name, contrasts in families.items():
        rows: list[tuple[str, object]] = []
        p_values: list[float] = []
        incomplete: list[str] = []
        for contrast in contrasts:
            treated = found.get(
                (contrast.backbone, contrast.dataset, contrast.treated), {}
            )
            control = found.get(
                (contrast.backbone, contrast.dataset, contrast.control), {}
            )
            shared = sorted(set(treated) & set(control))
            if len(shared) < 2:
                incomplete.append(contrast.label)
                continue
            result = paired_delta(
                [treated[s][args.split] for s in shared],
                [control[s][args.split] for s in shared],
            )
            rows.append((contrast.label, result))
            p_values.append(result.p)

        # Holm WITHIN the family. Incomplete contrasts are excluded from the family
        # SIZE as well as the listing -- counting them would make every surviving
        # contrast look more significant than the protocol allows.
        adjusted = holm(p_values)
        marker = " (PRIMARY)" if family_name == "text_real_minus_text_sham" else ""
        lines.append(f"== {family_name}{marker} — Holm over {len(rows)} contrasts ==")
        for (label, result), p_adj in zip(rows, adjusted, strict=True):
            lines.append(
                f"{label}: Δ {result.delta:+.4f} sd {result.sd:.4f} "
                f"t {result.t:+.2f} p {result.p:.4f} p_holm {p_adj:.4f} "
                f"wins {result.wins}/{result.n}"
            )
        lines.extend(
            f"{label}: INCOMPLETE (<2 paired seeds; excluded from Holm)"
            for label in incomplete
        )
        if incomplete:
            lines.append(
                f"# NOTE {len(incomplete)} contrast(s) excluded from this family's "
                "Holm size; the family is not complete as pre-registered."
            )
        lines.append("")

    lines.append(
        "# A non-significant contrast is NOT a negative result. Δ, sd, t, p and "
        "wins are recorded either way."
    )

    (args.out / "aggregate.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out / 'aggregate.txt'}")
    print(f"wrote {args.out / 'cells.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
