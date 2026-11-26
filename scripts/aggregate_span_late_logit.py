#!/usr/bin/env python3
"""Aggregate the #231 Variant 1 campaign into a provenance-headed record.

Derived from the #218 aggregator: the parsing, the header-from-cells rule, the collapsed-seed
median/MAD guard and the Holm families were each fixed in response to a measured failure.

ONE thing inverts, and it matters: `residual_always_zero` is #218's STOP condition (a dead
knowledge branch) but #231's CONTRACT -- Variant 1 never touches the span representation, so
the accumulator reports it True on every real arm. Reusing #218's `tripped_verdicts` here
would flag all 54 cells as failed. The real silent failure for a late-logit variant is a dead
`logit_delta`: the frozen baseline is returned unchanged while the run reports a plausible F1.


    uv run python scripts/aggregate_span_late_logit.py \
        --runs-root "$KNOWHYPER_HOME/runs/ner/span_late_logit/variant1_phase3" \
        --out docs/reports/data/ner_span_late_logit_2026_08_02

Reads each cell's `driver.log`, emits `aggregate.txt` + `cells.jsonl`, and applies
Holm correction WITHIN each of the three pre-registered families.

Records val AND test for every arm. They disagree in magnitude and significance,
so a bare number is unusable; headline claims use test because validation selected
the checkpoint. A non-significant contrast is recorded as non-significant, never as
"no effect".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from knowhyper import pins
from knowhyper.span_stats import contrast_families, holm, paired_delta

_CORE_DATASETS = frozenset({"bc5cdr_chem", "bc2gm", "bc5cdr_disease"})
_EXTENSION_DATASETS = frozenset({"jnlpba", "ncbi_disease"})

# Header settings are READ FROM THE CELLS, never hardcoded. This aggregator was seeded
# from the RE campaign's and inherited its `bs=64` literal while every span cell ran at
# 16 -- and "Never compare across headers" is a rule that ASSUMES the header is true, so
# a wrong one silently licenses comparison against the retracted bs=64 tree. Anything
# that must be identical across arms is measured from `.hydra/config.yaml`, not retyped.
_SETTING_PATTERNS = {
    "batch_size": re.compile(r"^\s*batch_size:\s*(\d+)\s*$", re.MULTILINE),
    "max_epochs": re.compile(r"^\s*max_epochs:\s*(\d+)\s*$", re.MULTILINE),
    "knowledge_lr_mult": re.compile(
        r"^\s*knowledge_lr_mult:\s*(\d+(?:\.\d+)?)\s*$", re.MULTILINE
    ),
}


# A collapsed seed must be FLAGGED, not averaged in. Found in the n=10 record:
# `biolinkbert/bc2gm/shared_zero/seed_7` finished at test 0.5775 against nine siblings at
# ~0.838 -- one bad cell in 420. The per-seed list made it visible but nothing flagged it,
# and the arm mean fell 0.8378 -> 0.8117 with sd inflating 0.0024 -> 0.0823.
#
# `shared_zero` is the zero control for EVERY condition of that backbone x dataset, so one
# failed seed corrupts six paired contrasts, and a depressed control pushes `real - zero`
# UPWARD -- the direction that manufactures a false positive. Same shape as the retracted
# bs=64 run, where the controls fell further than the real arm.
#
# Median/MAD rather than mean/sd: the statistic must not be dragged by the point it is
# meant to catch. Relative rather than absolute, because a healthy F1 is ~0.84 on bc2gm
# and ~0.93 on chem, so any fixed cutoff either misses failures or condemns good arms.
# Threshold picked from the MEASURED separation, not convention. Recomputed from the
# 9-seed arm this incident refers to, pinned in `tests/test_span_ner_knowledge_outlier.py`
# (bc2gm shared_zero): median 0.8381, MAD 0.00160; the collapsed seed (0.5775) sits at
# **162.9x** MAD and the worst healthy seed (0.8332) at **3.06x**. Two orders of magnitude
# apart, so 20 sits far from both edges.
#
# An earlier version of this comment said "MAD 0.00070, worst healthy seed ... 7x MAD".
# Those two numbers are not reproducible from this arm -- 0.00070 is not its MAD, and 7x
# only follows from that wrong MAD -- while the 162.9x in the same sentence does reproduce,
# so the pair was mixed in from elsewhere. The chosen constant is unaffected (20 is still
# well above 3.06 and well below 162.9), but a justification that cannot be recomputed is
# how a threshold later gets "corrected" toward a number nobody measured.
#
# A conventional 3x or 5x MAD would flag healthy seeds every run, and a gate that cries
# wolf stops being read -- that already happened once here with the zero-arm residual
# verdicts.
_OUTLIER_MAD_MULTIPLE = 20.0
_MIN_SEEDS_FOR_OUTLIERS = 4


def seed_outliers(values: list[float]) -> list[tuple[int, float]]:
    """Indices and values of seeds that sit implausibly far from the arm's median.

    Returns both directions: an implausibly GOOD seed is equally a symptom (a mislabelled
    arm, a leaked split) and looking only downward would miss it.
    """
    if len(values) < _MIN_SEEDS_FOR_OUTLIERS:
        # With 2-3 seeds the spread is not estimable; refusing to guess beats a false flag.
        return []
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    deviations = sorted(abs(v - median) for v in values)
    mid = len(deviations) // 2
    mad = (
        deviations[mid]
        if len(deviations) % 2
        else (deviations[mid - 1] + deviations[mid]) / 2
    )
    if mad <= 0:
        # Identical (or near-identical) seeds: no spread, so nothing is an outlier.
        return []
    limit = _OUTLIER_MAD_MULTIPLE * mad
    return [
        (index, value)
        for index, value in enumerate(values)
        if abs(value - median) > limit
    ]


def protocol_settings(runs_root: Path) -> dict[str, Any]:
    """Read the settings the cells actually composed, across every cell in the tree.

    A disagreement is reported as `MIXED{...}` rather than resolved: two batch sizes in
    one tree means the arms are not comparable, and picking the first would produce a
    header true of some cells and false of others -- the same defect as the literal,
    only harder to notice.
    """
    found: dict[str, set[str]] = {name: set() for name in _SETTING_PATTERNS}
    for config in sorted(runs_root.glob("*/*/*/*/.hydra/config.yaml")):
        text = config.read_text()
        for name, pattern in _SETTING_PATTERNS.items():
            match = pattern.search(text)
            if match is not None:
                found[name].add(match.group(1))

    settings: dict[str, Any] = {}
    for name, values in found.items():
        if not values:
            settings[name] = "UNKNOWN"
        elif len(values) == 1:
            raw = next(iter(values))
            settings[name] = int(float(raw)) if float(raw).is_integer() else float(raw)
        else:
            ordered = sorted(values, key=float)
            settings[name] = "MIXED{" + ",".join(ordered) + "}"
    return settings


def header_lines(
    *,
    settings: dict[str, Any],
    commit: str,
    runs: str,
    split: str,
    baseline_source: str = "ner_span_text_phase3 (seeds 3-9, patience=100)",
) -> list[str]:
    """Render the provenance header from measured settings."""
    return [
        "# METRIC: f1 (ieforge EntityF1: exact entity-level micro) | "
        "SPLIT: val = best-epoch validation from the checkpoint filename, "
        "test = that checkpoint on test",
        # `knowledge_lr_mult` is READ from each cell's `.hydra`, never asserted. This line
        # previously said `frozen inner (no knowledge_lr_mult)` on the grounds that "#231 does
        # not pass it" -- true when written, false once the driver started passing
        # `trainer.callbacks.3.knowledge_lr_mult=$LR_MULT` on every cell. Denying a multiplier
        # the run DID use is the same `bs=64` header failure as stating one it did not, so the
        # value comes from `protocol_settings()`, which surfaces `MIXED{...}` when cells
        # disagree rather than picking one.
        #
        # Note what it means here: with the inner frozen there is ONE param group, so this
        # scales the only LR rather than setting a trunk-vs-branch ratio. It is NOT comparable
        # to CLAUDE.md's RE table. See docs/EXPERIMENTS.md §3.8.
        f"# EPOCHS: {settings['max_epochs']} | TRAINER: difflr_benchmark_nolog | "
        f"logger=none | bs={settings['batch_size']} | frozen inner (one param group) | "
        # `.get` with an explicit UNKNOWN rather than a KeyError or a silent omission: a header
        # that quietly drops the field reads as "not applicable", which is the claim that was
        # wrong before. If no cell recorded a multiplier, say so.
        f"knowledge_lr_mult={settings.get('knowledge_lr_mult', 'UNKNOWN')}",
        "# VARIANT: #231 Variant 1, baseline-preserving late-logit verifier. The frozen "
        f"span baseline comes from `{baseline_source}`.",
        "# SPAN: start_end max_span_width=12 | LOSS: focal | DECODER: max_total_score",
        # Read from `knowhyper.pins`, never written as literals. These were literals until
        # a review caught it: the same failure mode as the `bs=64` header this module
        # already fixes -- correct today, silently stale the next time PrimeHKG is bumped,
        # with nothing failing loudly. `pins` reads the values from the pinned config.
        f"# CORE: PrimeHKG {pins.CORE_REVISION} | FEATURES: {pins.FEATURES_REVISION} | "
        f"PACKAGE: primehkg=={pins.PACKAGE_REV.lstrip('v')} ({pins.PACKAGE_REV})",
        f"# IEFORGE: 0.1.11 | COMMIT: {commit or 'TBD'} | RUNS: {runs or 'TBD'}",
        f"# HEADLINE SPLIT: {split} (validation selects the checkpoint)",
        "",
    ]


def baseline_source_for(datasets: list[str], explicit: str | None = None) -> str:
    """Name the actual frozen first-pass tree or reject an ambiguous scope."""
    if explicit:
        return explicit
    selected = set(datasets)
    if selected and selected <= _CORE_DATASETS:
        return "ner_span_text_phase3 (seeds 3-9, patience=100)"
    if selected and selected <= _EXTENSION_DATASETS:
        return "ner_span_text_extension (seeds 0-9, patience=100)"
    msg = (
        "mixed or unknown dataset scope cannot infer one frozen baseline tree; "
        "pass --baseline-source explicitly"
    )
    raise ValueError(msg)


# The ONLY key parsed from driver.log. ieforge's `monitor: val/f1` means it logs
# `val/f1` and `test/f1` -- NOT `test/entity_f1` (EntityF1 is the metric CLASS; `f1`
# is the key). A loose `[a-z_]*f1` pattern also matches `test/macro_f1` and, being
# last-wins, would silently swap the recorded number if ieforge ever logs a second f1
# key. Note validation is NOT read from the log at all (see _CKPT below), so no
# `val/f1` pattern exists here.
_TEST_F1 = re.compile(r"\btest/f1\b[^0-9]*([01]\.\d+)")
# Best-epoch validation comes from the checkpoint FILENAME, not driver.log: the log
# holds one val line per epoch and the last one is the FINAL epoch, while
# save_top_k=1 keeps the BEST. Reading the log would report a drifted final-epoch
# score under a header that claims best-epoch.
_CKPT = re.compile(r"epoch_(\d+)-f1_([0-9.]+)\.ckpt")


def parse_cell(run_dir: Path) -> dict[str, float] | None:
    """Return {val, test} for a finished cell, or None if it is not usable.

    A cell is rejected when it has no `test/f1` (walltime kill, crash, or a run
    still in flight) or no recoverable best-epoch validation. Admitting such a cell
    as a val-only arm silently shrinks the paired n with no way to tell a killed
    cell from an unsubmitted one -- and walltime kills land on the slowest arm
    first, i.e. the arm most worth measuring.
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
        # Phase 2's knowledge-branch record, written in-job by
        # SpanDiagnosticsCallback. None for arms with no knowledge branch -- absent is
        # not the same as measured-and-zero.
        "diagnostics": _diagnostics(run_dir),
    }


def _diagnostics(run_dir: Path) -> dict[str, Any] | None:
    """Read the in-job knowledge diagnostics, or None if this arm has no branch."""
    path = run_dir / "knowledge_diagnostics.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def tripped_verdicts(
    data: dict[tuple[str, str, str, str], dict[int, dict[str, Any]]],
) -> list[tuple[str, list[str]]]:
    """Cells whose diagnostics tripped a Phase 2 stop condition.

    Returned so the aggregate can lead with them: a dead branch and a genuine null
    produce the same F1, so a tripped verdict has to be read BEFORE any contrast.
    """
    tripped: list[tuple[str, list[str]]] = []
    for (backbone, dataset, condition, control), seeds in sorted(data.items()):
        for seed, scores in sorted(seeds.items()):
            record = scores.get("diagnostics")
            if not isinstance(record, dict):
                continue
            verdicts = record.get("verdicts") or {}
            delta = record.get("logit_delta") or {}
            delta_max = delta.get("max")
            # `residual_always_zero` is EXPECTED on every #231 arm and is never reported.
            # #218 already paid for the opposite mistake: the zero arm's correct behaviour was
            # reported as 16 failures, and a gate that cries wolf stops being read.
            names = [
                name
                for name, hit in sorted(verdicts.items())
                if hit and name != "residual_always_zero"
            ]
            if control == "zero":
                # The zero arm must return EXACT base logits, so a moved delta is the failure.
                if delta_max not in (None, 0, 0.0):
                    names.append("zero_arm_delta_moved")
            elif delta_max in (0, 0.0):
                # A real/shuffled arm whose delta never moved returned the frozen baseline.
                names.append("late_delta_always_zero")
            if names:
                label = f"{backbone}/{dataset}/{condition}_{control}/seed_{seed}"
                tripped.append((label, sorted(set(names))))
    return tripped


def _best_val(run_dir: Path) -> float | None:
    """Best-epoch validation f1, read from the checkpoint filename inventory."""
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


def collect(
    runs_root: Path,
) -> dict[tuple[str, str, str, str], dict[int, dict[str, float]]]:
    """Map (backbone, dataset, condition, control) -> seed -> {val, test}."""
    found: dict[tuple[str, str, str, str], dict[int, dict[str, float]]] = defaultdict(
        dict
    )
    for log in sorted(runs_root.glob("*/*/*/seed_*/driver.log")):
        seed_dir = log.parent
        arm_dir = seed_dir.parent
        dataset_dir = arm_dir.parent
        backbone_dir = dataset_dir.parent
        if "_" not in arm_dir.name:
            continue
        # LR-sweep cells live in the SAME output root as the primary campaign, as
        # `<condition>_<control>_mult<N>`. `rpartition("_")` would turn `edge_real_mult15`
        # into condition="edge_real" / control="mult15" -- neither real -- so those cells
        # would match no contrast and vanish from the BODY while `protocol_settings()` still
        # folded their `.hydra` into the HEADER. That yields a header describing a superset of
        # the cells the contrasts use, with nothing failing. `span_late_logit_lrsweep.py`
        # excludes them for the same reason ("a sweep cell must never be mistaken for it").
        if "_mult" in arm_dir.name:
            continue
        condition, _, control = arm_dir.name.rpartition("_")
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        scores = parse_cell(seed_dir)
        if scores is None:
            continue
        key = (backbone_dir.name, dataset_dir.name, condition, control)
        found[key][seed] = scores
    return found


def _arm(
    data: dict[tuple[str, str, str, str], dict[int, dict[str, float]]],
    backbone: str,
    dataset: str,
    condition: str,
    control: str,
) -> dict[int, dict[str, float]]:
    """Fetch an arm, falling back to the shared zero for a zero control."""
    key = (backbone, dataset, condition, control)
    if key in data:
        return data[key]
    if control == "zero":
        return data.get((backbone, dataset, "shared", "zero"), {})
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--runs", default="", help="ABCI run ids for the header")
    ap.add_argument("--commit", default="", help="commit every cell ran at")
    ap.add_argument(
        "--backbones",
        nargs="+",
        default=["pubmedbert", "biolinkbert"],
        help="backbones included in each Holm family",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["bc5cdr_chem", "bc2gm", "bc5cdr_disease"],
        help="datasets included in each Holm family",
    )
    ap.add_argument(
        "--baseline-source",
        default=None,
        help="frozen first-pass run tree named in the provenance header",
    )
    args = ap.parse_args()

    try:
        baseline_source = baseline_source_for(args.datasets, args.baseline_source)
    except ValueError as error:
        print(f"FATAL {error}", file=sys.stderr)
        return 2

    if not args.runs_root.is_dir():
        print(f"FATAL runs root does not exist: {args.runs_root}", file=sys.stderr)
        return 2

    data = collect(args.runs_root)
    if not data:
        print(f"FATAL no cells with scores under {args.runs_root}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    cells_path = args.out / "cells.jsonl"
    with cells_path.open("w") as writer:
        for (backbone, dataset, condition, control), seeds in sorted(data.items()):
            for seed, scores in sorted(seeds.items()):
                writer.write(
                    json.dumps(
                        {
                            "backbone": backbone,
                            "dataset": dataset,
                            "condition": condition,
                            "control": control,
                            "seed": seed,
                            "val": scores.get("val"),
                            "test": scores.get("test"),
                            "commit": args.commit or None,
                        }
                    )
                    + "\n"
                )

    lines: list[str] = header_lines(
        settings=protocol_settings(args.runs_root),
        commit=args.commit,
        runs=args.runs,
        split=args.split,
        baseline_source=baseline_source,
    )

    # LEAD with any tripped Phase 2 stop condition. A dead knowledge branch and a
    # genuine null give the same F1, so a contrast computed over a cell whose residual
    # never moved is not a measurement of knowledge content at all.
    tripped = tripped_verdicts(data)
    if tripped:
        lines.append(
            f"## STOP CONDITIONS TRIPPED in {len(tripped)} cell(s) -- read before any "
            "contrast below"
        )
        for label, names in tripped:
            lines.append(f"  {label}: {', '.join(names)}")
        lines.append(
            "# A tripped verdict is an IMPLEMENTATION reading, not a result: fix and "
            "re-run the same seeds (issue #218 Phase 2)."
        )
        lines.append("")
    else:
        lines.append("## stop conditions: none tripped")
        cells_with = sum(
            1
            for seeds in data.values()
            for scores in seeds.values()
            if scores.get("diagnostics") is not None
        )
        total = sum(len(seeds) for seeds in data.values())
        lines.append(
            f"# knowledge diagnostics present for {cells_with} of {total} cells"
        )
        lines.append("")

    # Collapsed seeds go ABOVE the numbers, like the stop conditions: a contrast computed
    # against an arm holding a failed run is not a measurement, and a control arm is the
    # worst place for it (one bad `shared_zero` seed corrupts every condition of that
    # backbone x dataset, in the direction that INFLATES `real - zero`).
    flagged: list[str] = []
    for (backbone, dataset, condition, control), seeds in sorted(data.items()):
        ordered = sorted(seeds.items())
        for split in ("val", "test"):
            values = [scores[split] for _, scores in ordered if split in scores]
            if not values:
                continue
            for position, value in seed_outliers(values):
                seed = ordered[position][0]
                flagged.append(
                    f"  {backbone}/{dataset}/{condition}_{control} seed_{seed} "
                    f"{split}={value:.4f} -- far from the arm median"
                )
    if flagged:
        lines.append(f"## COLLAPSED SEEDS: {len(flagged)} -- read before any contrast")
        lines += flagged
        lines.append(
            "# These are training failures, not results. Re-run the seed, or state the "
            "exclusion explicitly; do NOT average them in silently."
        )
        lines.append("")

    for (backbone, dataset, condition, control), seeds in sorted(data.items()):
        for split in ("val", "test"):
            values = [
                scores[split] for _, scores in sorted(seeds.items()) if split in scores
            ]
            if not values:
                continue
            mean = sum(values) / len(values)
            sd = (
                (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
                if len(values) > 1
                else 0.0
            )
            arm = f"{backbone}/{dataset}/{condition}_{control}"
            lines.append(
                f"{arm} {split}: mean {mean:.4f} sd {sd:.4f} n={len(values)} "
                f"per-seed {[round(v, 4) for v in values]}"
            )
    lines.append("")

    # Three families, Holm-corrected independently. Pooling them would change the
    # verdicts, so the split is part of the protocol.
    families = contrast_families(
        backbones=tuple(args.backbones), datasets=tuple(args.datasets)
    )
    for family, contrasts in families.items():
        rows: list[tuple[str, Any]] = []
        for contrast in contrasts:
            treated_control = _resolve_pair(contrast)
            if treated_control is None:
                continue
            (t_cond, t_ctl), (c_cond, c_ctl) = treated_control
            treated = _arm(data, contrast.backbone, contrast.dataset, t_cond, t_ctl)
            control = _arm(data, contrast.backbone, contrast.dataset, c_cond, c_ctl)
            shared = sorted(set(treated) & set(control))
            paired = [
                (treated[s][args.split], control[s][args.split])
                for s in shared
                if args.split in treated[s] and args.split in control[s]
            ]
            if len(paired) < 2:
                rows.append((contrast.label, None))
                continue
            rows.append(
                (
                    contrast.label,
                    paired_delta([p[0] for p in paired], [p[1] for p in paired]),
                )
            )

        measured = [(label, result) for label, result in rows if result is not None]
        adjusted = holm([result.p for _, result in measured])
        lines.append(
            f"## family {family} (Holm over {len(measured)} measured contrasts)"
        )
        for (label, result), p_adj in zip(measured, adjusted, strict=True):
            verdict = "SIGNIFICANT" if p_adj < 0.05 else "n.s."
            lines.append(
                f"{label}: Δ {result.delta:+.4f} sd {result.sd:.4f} "
                f"t {result.t:+.2f} (df{result.n - 1}) p {result.p:.4f} "
                f"holm {p_adj:.4f} wins {result.wins}/{result.n} {verdict}"
            )
        for label, result in rows:
            if result is None:
                lines.append(f"{label}: INCOMPLETE (fewer than 2 paired seeds)")
        if len(measured) < len(contrasts):
            lines.append(
                f"# NOTE {len(contrasts) - len(measured)} of {len(contrasts)} "
                "contrasts in this family are incomplete and were EXCLUDED from the "
                "Holm family size above."
            )
        lines.append("")

    (args.out / "aggregate.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out / 'aggregate.txt'}")
    print(f"wrote {cells_path}")
    return 0


def _resolve_pair(contrast: Any) -> tuple[tuple[str, str], tuple[str, str]] | None:
    """Return ((treated condition, control), (control condition, control))."""
    if contrast.kind == "real_minus_shuffled":
        return (contrast.condition, "real"), (contrast.condition, "shuffled")
    if contrast.kind == "real_minus_zero":
        return (contrast.condition, "real"), (contrast.condition, "zero")
    if contrast.kind == "attributes_minus_structure":
        return ("ev", "real"), ("edge", "real")
    if contrast.kind == "pretraining_minus_attributes":
        return ("ev_warm", "real"), ("ev", "real")
    return None


if __name__ == "__main__":
    raise SystemExit(main())
