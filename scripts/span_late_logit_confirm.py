#!/usr/bin/env python3
"""#231 Phase 3: the confirmatory record. Test is read here for the first time.

    uv run python scripts/span_late_logit_confirm.py \
        --runs-root "$KNOWHYPER_HOME/runs/ner/span_late_logit/variant1_phase3" \
        --extra-runs-root "$KNOWHYPER_HOME/runs/ner/span_late_logit/variant1_phase2" \
        --baseline-runs-root "$KNOWHYPER_HOME/runs" \
        --out docs/reports/data/ner_span_late_logit_2026_08_02/phase3_confirm

The success criteria are #231's, and they are written here BEFORE the numbers arrive:

* at least one dataset where `real - shuffled` AND `real - zero` both clear Holm and are
  positive
* on that dataset, wins >= 70% of seeds -- #231's ">= 7/10", computed from each contrast's
  own n by `win_floor()` (7/10 at n=10, 5/7 at the baseline-bounded n=7)
* no Holm-significant HARM on another dataset
* the aggregate agrees with the evidenced-candidate raw counts
* Chem does not reproduce #218's cross-attention recall harm

Two families, Holm-corrected within each: `real_minus_shuffled` (6 contrasts) and
`real_minus_zero` (6). Six because 2 backbones x 3 datasets, one condition (EV-cold).

`real - zero` reads the zero arm from the frozen baseline record (#231 decision (B)) and every
such row is flagged `trained=False`. IMPORTANT for interpretation: Phase 2 measured the
shuffled arm converging to EXACTLY the frozen baseline in 6/6 cells, so `shuffled` and `zero`
are empirically the same number and the two families are NOT independent evidence. That is
recorded in the output rather than left for a reader to notice.

n is READ FROM THE CELLS, never assumed. The first run of this script was n=7 because the
frozen baselines existed for seeds 3-9 only (0-2 were the `_phase2` protocol, excluded by
decision (a)); the top-up then produced seeds 0-2 under `_phase3` and Phase 3b the matching
arm cells, so passing all three roots gives n=10. The emitted header states the n it actually
found, and `win_floor()` scales the seed-win criterion to it -- so neither record can be
misread as the other, and moving to n=10 cannot silently relax the bar.

**The n=10 record is a POST-HOC EXTENSION and is labelled as such.** #231 pre-registered
n=10, but the decision to add seeds was taken after seeing the n=7 test numbers (5/6 positive,
0/6 clearing Holm), which is the "choose n while looking at test" the issue warns against.
Both records are kept; the n=10 one does not replace the n=7 one.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path
from typing import Any

from knowhyper.span_stats import holm, paired_delta
from knowhyper.span_zero_arm import read_zero_arm

_TEST_F1 = re.compile(r"\btest/f1\b[^0-9]*([01]\.\d+)")
_TEST_P = re.compile(r"\btest/precision\b[^0-9]*([01]\.\d+)")
_TEST_R = re.compile(r"\btest/recall\b[^0-9]*([01]\.\d+)")
_CKPT = re.compile(r"epoch_(\d+)-f1_([0-9.]+)\.ckpt")

_DEGENERATE = 0.50
# #231's seed-win criterion, as a FRACTION. It used to be the absolute `_WIN_FLOOR = 5`,
# correct only for the n=7 the frozen baselines then bounded us to (5/7 = 71%). Once the
# baseline top-up lifted n to 10 the same constant would have meant 5/10 -- a **50%** bar --
# loosening the pre-registered criterion at exactly the moment the record became the headline
# one, and the verdict line's literal `/7` would have misstated the denominator too. Same
# shape as the aggregate-header incident: a number correct under one protocol, carried into
# the next unchanged, with nothing failing.
_WIN_FRACTION = 0.70


def win_floor(n: int) -> int:
    """Smallest win count that meets #231's 70% floor at sample size `n`.

    Rounds UP: at n=10 this is 7 (#231's ">= 7/10" verbatim), at n=7 it is 5, preserving the
    value the n=7 record was judged against.
    """
    return -((-n * 7) // 10)  # ceil(0.7 * n) without float rounding surprises


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--extra-runs-root", type=Path, action="append", default=[])
    parser.add_argument("--baseline-runs-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runs", default="")
    parser.add_argument("--commit", default="")
    # Phase 4 writes edge_* and ev_warm_* into ONE root. `collect()` filters to a single
    # condition (see its docstring), so without this flag the script could only ever emit the
    # EV record -- and pointing it at the Phase 4 root would produce an EMPTY table with no
    # error, which is worse than failing.
    parser.add_argument(
        "--condition",
        default="ev",
        choices=("ev", "edge", "ev_warm"),
        help="which condition's arms to read; the record names it in the header",
    )
    return parser.parse_args()


def _last(pattern: re.Pattern[str], text: str) -> float | None:
    found = pattern.findall(text)
    return float(found[-1]) if found else None


def read_cell(run_dir: Path) -> dict[str, Any] | None:
    log = run_dir / "driver.log"
    if not log.is_file():
        return None
    text = log.read_text(errors="replace")
    test = _last(_TEST_F1, text)
    if test is None:
        # A cell with no test score is unfinished; admitting it would silently shrink n.
        return None
    best_val = None
    for candidate in sorted((run_dir / "checkpoints").glob("*.ckpt")):
        match = _CKPT.search(candidate.name)
        if match:
            best_val = float(match.group(2))
            break
    diagnostics = None
    diag = run_dir / "knowledge_diagnostics.json"
    if diag.is_file():
        try:
            diagnostics = json.loads(diag.read_text())
        except json.JSONDecodeError:
            diagnostics = None
    return {
        "val": best_val,
        "test": test,
        "test_precision": _last(_TEST_P, text),
        "test_recall": _last(_TEST_R, text),
        "diagnostics": diagnostics,
    }


def collect(
    roots: list[Path], *, condition: str = "ev"
) -> dict[tuple[str, str, str], dict[int, dict[str, Any]]]:
    """Merge cells from several roots.

    Phase 2's 12 cells are confirmatory too -- the architecture was frozen before they ran --
    so they are merged rather than re-run. A cell present in two roots keeps the FIRST one
    seen, and the caller orders roots so the primary root wins.

    The key carries `control` but NOT `condition`, because #231 is EV-cold only. That makes
    the key ambiguous the moment a second condition exists: `ev_real`, `edge_real` and
    `ev_warm_real` all `rpartition("_")` to `control='real'`, so they would land in ONE
    bucket holding seeds from three architectures, with `setdefault` keeping whichever root
    was scanned first. Nothing in the output would show it -- the record would report a
    plausible mean over 10 seeds.

    So arms whose condition is not `condition` are SKIPPED rather than merged. Phase 4 must
    either call this once per condition or widen the key; a wrong number is worse than a
    missing one. This also excludes the LR-sweep arms (`<arm>_mult<N>`), whose cells train at
    a different learning rate and must never enter the primary record.
    """
    cells: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {}
    for root in roots:
        for run_dir in sorted(root.glob("*/*/*/seed_*")):
            backbone, dataset, arm, seed_dir = run_dir.parts[-4:]
            arm_condition, _, control = arm.rpartition("_")
            if arm_condition != condition:
                continue
            record = read_cell(run_dir)
            if record is None:
                continue
            key = (backbone, dataset, control)
            seed = int(seed_dir.removeprefix("seed_"))
            cells.setdefault(key, {}).setdefault(seed, record)
    return cells


def main() -> int:
    args = _parse_args()
    roots = [args.runs_root, *args.extra_runs_root]
    cells = collect(roots, condition=args.condition)

    backbones = ("pubmedbert", "biolinkbert")
    datasets = ("bc5cdr_chem", "bc2gm", "bc5cdr_disease")

    # The N line is DERIVED, never typed. It used to read "# N: **7**, seeds 3-9 -- NOT 10"
    # as a literal, which was true when written and would have kept asserting n=7 over an
    # n=10 table once the baseline top-up landed -- the aggregate-header failure exactly.
    # Arms that disagree are exposed as a range rather than averaged into one claim.
    # Named, never assumed: this line used to hardcode "EV-cold only", which would be FALSE
    # on a Phase 4 record whose every number came from `edge` or `ev_warm`.
    _CONDITION_BLURB = {
        "ev": "**EV-cold** (`+attributes`, `khp_span_late_ev`)",
        "edge": "**Structure only** (`edge`, `khp_span_late_edge`) -- no attribute nodes",
        "ev_warm": "**+cross-modal pretraining** (`ev_warm`, `khp_span_late_ev_warm`)",
    }
    condition_line = (
        f"# CONDITION: {_CONDITION_BLURB[args.condition]}. One condition per record; the "
        "other arms in the same root are SKIPPED, not merged."
    )

    seed_sets = {frozenset(seeds) for seeds in cells.values()}
    if len(seed_sets) == 1:
        observed = sorted(next(iter(seed_sets)))
        n_line = (
            f"# N: **{len(observed)}**, seeds "
            f"{min(observed)}-{max(observed)} (derived from the cells, not declared)"
        )
    else:
        sizes = sorted({len(s) for s in seed_sets})
        n_line = (
            f"# N: **MIXED{sizes}** -- arms do not share a seed set. Do NOT read a single n "
            "off this record; the per-arm table below is authoritative."
        )

    problems: list[str] = []
    lines = [
        "# #231 Phase 3: confirmatory record (Variant 1, frozen)",
        "",
        "# METRIC: f1 (ieforge EntityF1, exact entity-level micro) | SPLIT: val = best-epoch "
        "from the checkpoint filename, test = that checkpoint on test",
        "# VARIANT: baseline-preserving late-logit verifier. Frozen span baseline from "
        "`ner_span_text_phase3` (patience=100).",
        condition_line,
        n_line,
        "# N PROVENANCE: seeds 3-9 came from the original frozen baselines; 0-2 exist only "
        "after the `_phase3` top-up (job 2095727 + 2095732) and Phase 3b's matching arm "
        "cells. If n=10 here, this record is a POST-HOC EXTENSION of the n=7 one -- the "
        "decision to add seeds followed seeing n=7's test numbers -- and does not replace it.",
        "# ZERO ARM: read from the frozen baseline record, `trained=False` (decision B).",
        f"# COMMIT: {args.commit or 'TBD'} | RUNS: {args.runs or 'TBD'}",
        "# HEADLINE SPLIT: test (validation selected the checkpoint)",
        "",
        "**`shuffled` and `zero` are empirically the same arm.** Phase 2 measured the shuffled "
        "arm converging to exactly the frozen baseline in 6/6 cells, so the two families below "
        "are NOT independent evidence. Reported separately because #231 names both.",
        "",
    ]

    # Per-arm table.
    lines.append("## per-arm test scores")
    lines.append("")
    lines.append("| arm | n | test mean | sd | val mean |")
    lines.append("|---|---|---|---|---|")
    for backbone in backbones:
        for dataset in datasets:
            for control in ("real", "shuffled"):
                arm = cells.get((backbone, dataset, control), {})
                if not arm:
                    continue
                tests = [c["test"] for c in arm.values()]
                vals = [c["val"] for c in arm.values() if c["val"] is not None]
                sd = st.stdev(tests) if len(tests) > 1 else 0.0
                lines.append(
                    f"| {backbone}/{dataset}/{control} | {len(tests)} | "
                    f"{st.mean(tests):.4f} | {sd:.4f} | "
                    f"{st.mean(vals):.4f} |"
                    if vals
                    else f"| {backbone}/{dataset}/{control} | {len(tests)} | "
                    f"{st.mean(tests):.4f} | {sd:.4f} | - |"
                )
                for seed, cell in sorted(arm.items()):
                    if cell["test"] < _DEGENERATE:
                        problems.append(
                            f"{backbone}/{dataset}/{control}/seed_{seed}: DEGENERATE "
                            f"test {cell['test']:.4f}"
                        )
    lines.append("")

    # Two families.
    summary: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for family in ("shuffled", "zero"):
        contrasts: list[dict[str, Any]] = []
        for backbone in backbones:
            for dataset in datasets:
                real = cells.get((backbone, dataset, "real"), {})
                if not real:
                    continue
                if family == "shuffled":
                    other = cells.get((backbone, dataset, "shuffled"), {})
                    shared = sorted(set(real) & set(other))
                    baseline = {s: other[s]["test"] for s in shared}
                else:
                    shared, baseline = [], {}
                    for seed in sorted(real):
                        try:
                            row = read_zero_arm(
                                runs_root=args.baseline_runs_root,
                                backbone=backbone,
                                dataset=dataset,
                                seed=seed,
                            )
                        # `RuntimeError` covers BOTH expected data problems -- ZeroArmError
                        # and BaselineMismatchError are its subclasses -- and nothing else.
                        # This was `except (ZeroArmError, Exception)`, i.e. a bare catch-all,
                        # so a KeyError/TypeError from a future refactor would be filed as a
                        # missing seed and silently shrink n on this confirmatory contrast.
                        except RuntimeError as error:
                            problems.append(
                                f"{backbone}/{dataset}/zero/seed_{seed}: "
                                f"{str(error)[:80]}"
                            )
                            continue
                        shared.append(seed)
                        baseline[seed] = row["test"]
                if len(shared) < 2:
                    continue
                treated = [real[s]["test"] for s in shared]
                control_values = [baseline[s] for s in shared]
                stats = paired_delta(treated, control_values)
                contrasts.append(
                    {
                        "label": f"{backbone}/{dataset}",
                        "delta": stats.delta,
                        "sd": stats.sd,
                        "t": stats.t,
                        "p": stats.p,
                        "n": stats.n,
                        "wins": stats.wins,
                    }
                )
        if not contrasts:
            continue
        adjusted = holm([c["p"] for c in contrasts])
        lines.append(
            f"## family real_minus_{family} on TEST (Holm over {len(contrasts)} contrasts)"
        )
        lines.append("")
        summary[family] = {}
        for contrast, p_adj in zip(contrasts, adjusted, strict=True):
            verdict = "SIGNIFICANT" if p_adj < 0.05 else "n.s."
            lines.append(
                f"- `{contrast['label']}`: Δ {contrast['delta']:+.4f} "
                f"sd {contrast['sd']:.4f} t {contrast['t']:+.2f} "
                f"p {contrast['p']:.4f} holm {p_adj:.4f} "
                f"wins {contrast['wins']}/{contrast['n']} {verdict}"
            )
            backbone, dataset = contrast["label"].split("/")
            summary[family][(backbone, dataset)] = {**contrast, "holm": p_adj}
        lines.append("")

    # Success criteria.
    lines.append("## verdict")
    lines.append("")
    passed: list[str] = []
    harmed: list[str] = []
    # Reported rather than assumed: if the contrasts do not all share one n, the verdict line
    # says so instead of printing a single denominator that is wrong for some of them.
    floors: set[str] = set()
    for key in summary.get("shuffled", {}):
        a = summary["shuffled"][key]
        b = summary.get("zero", {}).get(key)
        if b is None:
            continue
        label = "/".join(key)
        both_sig_pos = (
            a["holm"] < 0.05 and a["delta"] > 0 and b["holm"] < 0.05 and b["delta"] > 0
        )
        # The floor is derived from EACH contrast's own n, not a shared constant: the two
        # families can differ in n (the zero arm is read per seed from the frozen record and
        # a missing baseline drops that seed), and n itself changed from 7 to 10 mid-campaign.
        if (
            both_sig_pos
            and a["wins"] >= win_floor(a["n"])
            and b["wins"] >= win_floor(b["n"])
        ):
            passed.append(label)
        if a["holm"] < 0.05 and a["delta"] < 0:
            harmed.append(label)
        floors.add(f"{win_floor(a['n'])}/{a['n']}")
    lines.append(
        f"- clears both families, positive, wins >= "
        f"{'/'.join(sorted(floors)) if floors else 'n/a'} "
        f"({_WIN_FRACTION:.0%} of seeds): {passed or 'none'}"
    )
    lines.append(f"- Holm-significant HARM on real-shuffled: {harmed or 'none'}")
    if problems:
        lines.append("")
        lines.append(f"**{len(problems)} problem(s):**")
        lines += [f"- {p}" for p in problems]
    lines.append("")
    if passed and not harmed and not problems:
        lines.append(
            f"**SUCCESS on {passed}.** #231 then allows Phase 4: the same frozen variant "
            "extended to Structure only / +pretraining."
        )
    else:
        lines.append(
            "**NOT A SUCCESS.** #231 forbids widening to edge / EV-warm in that case; the "
            "direction closes negative. The claim to record is narrow: PrimeHKG "
            "candidate-owned evidence did not improve a matched span baseline under the two "
            "span-injection variants examined -- NOT that external knowledge is impossible."
        )
    lines.append("")
    lines.append(
        "A non-significant contrast is recorded as non-significant, never as 'no effect'."
    )
    lines.append("")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "confirm.txt").write_text("\n".join(lines) + "\n")
    with (args.out / "cells.jsonl").open("w") as handle:
        for (backbone, dataset, control), seeds in sorted(cells.items()):
            for seed, cell in sorted(seeds.items()):
                handle.write(
                    json.dumps(
                        {
                            "backbone": backbone,
                            "dataset": dataset,
                            "control": control,
                            "seed": seed,
                            "val": cell["val"],
                            "test": cell["test"],
                            "test_precision": cell["test_precision"],
                            "test_recall": cell["test_recall"],
                        }
                    )
                    + "\n"
                )
    print(f"wrote {args.out / 'confirm.txt'}")
    print(f"  cells={sum(len(v) for v in cells.values())}")
    print(f"  passed={passed} harmed={harmed} problems={len(problems)}")
    return 0 if (passed and not harmed and not problems) else 1


if __name__ == "__main__":
    raise SystemExit(main())
