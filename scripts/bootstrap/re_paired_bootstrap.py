#!/usr/bin/env python3
"""Seed-level and paired test-instance uncertainty for recorded RE contrasts.

Reads per-instance test predictions (``export/predictions.jsonl``) of two
recorded arms, checks parity (same test population in every seed; recomputed
F1 equals the F1 the campaign recorded), then reports both

* the paired t statistics over training seeds, and
* a paired percentile bootstrap over test instances with the trained models
  held fixed,

as two separate intervals -- never one merged number.  Contrasts are declared
in a YAML file so the arm roots, seeds and recorded values are provenance,
not command-line memory:

    uv run python scripts/bootstrap/re_paired_bootstrap.py \\
        --contrasts scripts/bootstrap/contrasts.yaml \\
        --output-dir docs/reports/data/re_bootstrap_2026_09_17

Deterministic: the same YAML, ``--resamples`` and ``--rng-seed`` reproduce the
same intervals byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from knowhyper.analysis.bootstrap import (
    DEFAULT_PARITY_TOL,
    DEFAULT_RESAMPLES,
    DEFAULT_RNG_SEED,
    ContrastResult,
    ContrastSpec,
    holm_adjust,
    run_contrast,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ROW_FIELDS = (
    "contrast",
    "family",
    "family_size",
    "dataset",
    "arm_a",
    "arm_b",
    "n_instances",
    "n_seeds",
    "observed_f1_a",
    "observed_f1_b",
    "observed_delta",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "seed_source",
    "seed_mean_delta",
    "seed_sd",
    "seed_t",
    "seed_p",
    "seed_p_holm",
    "seed_ci_low",
    "seed_ci_high",
    "seed_wins",
    "seed_recomputed_mean_delta",
    "seed_recomputed_p",
    "parity_ok",
    "parity_mode",
    "parity_checked",
)


def _runs_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    home = os.environ.get("KNOWHYPER_HOME") or str(Path.home() / ".knowhyper")
    return Path(home) / "runs"


def _seeds(value: Any, default: list[int]) -> tuple[int, ...]:
    return tuple(int(s) for s in (value if value is not None else default))


def _recorded(arm: dict[str, Any], seeds: tuple[int, ...]) -> dict[int, float] | None:
    values = arm.get("recorded_test_f1")
    if values is None:
        return None
    if len(values) != len(seeds):
        msg = (
            f"arm {arm.get('condition')}: recorded_test_f1 has {len(values)} values "
            f"for {len(seeds)} seeds"
        )
        raise ValueError(msg)
    return {seed: float(v) for seed, v in zip(seeds, values, strict=True)}


def build_specs(
    config: dict[str, Any], runs_root: Path, only: set[str] | None
) -> tuple[list[tuple[ContrastSpec, str]], list[tuple[str, str]]]:
    """Turn the YAML document into ``(spec, family)`` pairs plus disabled rows.

    A contrast with a ``disabled: <reason>`` key is not run; it is listed in the
    header instead, so a missing row reads as a decision, not an omission.
    """
    default_seeds = [int(s) for s in config.get("defaults", {}).get("seeds", range(10))]
    arms: dict[str, dict[str, Any]] = config["arms"]
    specs: list[tuple[ContrastSpec, str]] = []
    disabled: list[tuple[str, str]] = []
    for entry in config["contrasts"]:
        name = entry["name"]
        if only is not None and name not in only:
            continue
        if entry.get("disabled"):
            disabled.append((name, str(entry["disabled"])))
            continue
        arm_a = arms[entry["a"]]
        arm_b = arms[entry["b"]]
        if arm_a["dataset"] != arm_b["dataset"]:
            msg = f"{name}: arms come from different datasets"
            raise ValueError(msg)
        seeds = _seeds(entry.get("seeds"), default_seeds)
        spec = ContrastSpec(
            name=name,
            dataset=arm_a["dataset"],
            negative_label=entry["negative_label"],
            arm_a=(arm_a["condition"], runs_root / arm_a["root"]),
            arm_b=(arm_b["condition"], runs_root / arm_b["root"]),
            seeds=seeds,
            note=entry.get("note", ""),
            recorded_a=_recorded(arm_a, seeds),
            recorded_b=_recorded(arm_b, seeds),
        )
        specs.append((spec, entry.get("family", "unspecified")))
    return specs, disabled


def declared_order(config: dict[str, Any]) -> list[tuple[str, str]]:
    """``(name, family)`` for every enabled contrast of the YAML, in file order.

    Independent of ``--only``: the rendered record is ordered by the declaration,
    so a selective re-run cannot reorder the rows it did not recompute.
    """
    return [
        (entry["name"], entry.get("family", "unspecified"))
        for entry in config["contrasts"]
        if not entry.get("disabled")
    ]


def _seed_line(
    seed: dict[str, Any], n_seeds: int, *, has_seed_test: bool = True
) -> str:
    """The numeric part of a seed-level line, or why there is none.

    ``has_seed_test`` is false when there are fewer than two seeds. Then the
    line carries NO delta and NO win count, matching what `as_row` writes to the
    TSV: the difference of two single runs is a number but not a seed-level
    statistic, and printing it here while blanking it there made the
    human-readable and machine-readable records disagree about whether the
    quantity exists.

    A block with n >= 2 whose t is undefined (constant delta) is a different
    case: the result is real, so it keeps its delta and win count and only its
    dispersion is missing.
    """
    if seed.get("mean_delta") is None:
        missing = ", ".join(seed.get("missing_recorded") or ["?"])
        return (
            "no recorded per-seed test F1 for "
            f"{missing} -- no confirmatory seed statistic exists for this contrast"
        )
    if not has_seed_test:
        return (
            f"no seed-level statistic (n_seeds={seed['n_seeds']}); "
            "the observed difference is reported at the instance level only"
        )
    if not seed.get("estimable"):
        return (
            f"d {seed['mean_delta']:+.4f} wins {seed['wins']}/{n_seeds} -- "
            f"seed-level uncertainty not estimable (n_seeds={seed['n_seeds']})"
        )
    return (
        f"d {seed['mean_delta']:+.4f} sd {seed['sd_delta']:.4f} "
        f"t {seed['t']:+.3f} (df{seed['df']}) p {seed['p']:.3f} "
        f"95% t-CI [{seed['ci_low']:+.4f}, {seed['ci_high']:+.4f}] "
        f"wins {seed['wins']}/{n_seeds} losses {seed['losses']} ties {seed['ties']}"
    )


def _mean_sd(seed: dict[str, Any], key: str) -> str:
    mean, sd = seed.get(f"mean_{key}"), seed.get(f"sd_{key}")
    mean_text = "  n/a " if mean is None else f"{mean:.4f}"
    sd_text = "  n/a " if sd is None else f"{sd:.4f}"
    return f"mean {mean_text} sd {sd_text}"


def render_block(
    result: ContrastResult,
    family: str,
    family_size: int,
    p_holm: float | None,
) -> str:
    """One human-readable block per contrast for ``aggregate.txt``.

    The **recorded** seed-level statistic is the confirmatory one and is
    labelled as the paper's; the statistic recomputed from the re-exported
    predictions follows it as a secondary line. When the seed-level test is null
    and the instance-level interval excludes zero the block says ``DISAGREE`` on
    its own line, because that pair of numbers is exactly the one a reader would
    otherwise quote as a positive result.
    """
    seed = result.seed
    recomputed = result.seed_recomputed
    boot = result.bootstrap
    lines = [
        f"=== {result.contrast}  [{family}]  {result.dataset}  "
        f"{result.condition_a} - {result.condition_b}  "
        f"n_seeds={result.n_seeds} n_instances={result.n_instances} "
        f"parity={'ok' if result.parity_ok else 'BROKEN'}"
        f" ({result.parity_mode}, {result.parity_checked} arm x seed compared)",
    ]
    if result.note:
        lines.append(f"  note: {result.note}")
    f1_a = [result.per_seed[s]["f1_a"] for s in result.seeds]
    f1_b = [result.per_seed[s]["f1_b"] for s in result.seeds]
    for condition, key, values in (
        (result.condition_a, "a", f1_a),
        (result.condition_b, "b", f1_b),
    ):
        lines.append(
            f"  {condition:<28s} recorded test: {_mean_sd(seed, key)} | recomputed "
            f"{_mean_sd(recomputed, key)} per-seed "
            f"[{', '.join(f'{v:.4f}' for v in values)}]"
        )
    # A block with fewer than two training seeds has NO seed-level test, so it
    # must not wear the confirmatory label: that is how one greedy inference run
    # came to be printed as "the paper's statistic" (#314 review F2).
    #
    # Keyed on `seed_source` (the seed count), NOT on `estimable`: the latter is
    # also false when n >= 2 but the delta is constant, and that IS a real
    # multi-seed result whose t is merely undefined. It keeps its label, its
    # printed delta and its DISAGREE handling. `as_row` keys on the same thing,
    # so aggregate.txt and bootstrap.tsv cannot disagree about whether a
    # seed-level quantity exists.
    has_seed_test = result.seed_source != "not_estimable"
    if has_seed_test:
        seed_label = f"[{result.seed_source}; CONFIRMATORY, the paper's statistic] "
    else:
        seed_label = f"[{result.seed_source}; no confirmatory seed statistic] "
    lines.append(
        f"  seed level     : {seed_label}"
        f"{_seed_line(seed, result.n_seeds, has_seed_test=has_seed_test)}"
    )
    holm = "n/a" if p_holm is None else f"{p_holm:.3f}"
    lines.append(f"    multiplicity : family {family} (k={family_size}) p_holm {holm}")
    lines.append(
        f"    seed_recomputed (secondary, from the re-exported predictions): "
        f"{_seed_line(recomputed, result.n_seeds, has_seed_test=has_seed_test)}"
    )
    lines.append(
        f"  instance level : d {boot['observed_delta']:+.4f} "
        f"95% paired bootstrap CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] "
        f"(R={boot['n_resamples']}, rng={boot['rng_seed']}, "
        f"P(d<=0)={boot['fraction_delta_le_zero']:.3f}) -- conditions on these "
        "trained models; NOT confirmatory"
    )
    excludes_zero = boot["ci_low"] > 0.0 or boot["ci_high"] < 0.0
    # DISAGREE means "a seed-level test was run, it was null, and the instance
    # interval nevertheless excludes zero". Two things that are NOT that:
    #
    # * **No test at all** (n_seeds < 2, `estimable` false). There is no null to
    #   disagree with; saying "seed-level test null" would assert a result that
    #   was never computed (#314 review F2).
    # * **Significance before multiplicity.** The header declares the confirmatory
    #   test is the HOLM-ADJUSTED one, so a contrast at raw p .03 / p_holm .09 is
    #   seed-level null and must be flagged. Gating on the raw p silently failed
    #   open in the over-claiming direction (#314 review F1). Raw p is the
    #   fallback only where no Holm value exists (a single-member family adjusts
    #   to its own raw p anyway).
    #
    # A multi-seed block whose t is undefined (constant delta) has NO p at all,
    # yet it is a real seed-level result and can sit beside an instance interval
    # that excludes zero -- so it is treated as null here rather than skipped,
    # which is the conservative direction.
    decisive_p = p_holm if p_holm is not None else seed.get("p")
    seed_null = has_seed_test and (decisive_p is None or decisive_p >= 0.05)
    if excludes_zero and seed_null:
        lines.append(
            "  DISAGREE       : seed-level test null after multiplicity, "
            "instance-level CI excludes 0. "
            "The seed-level test governs; this contrast stays null."
        )
    counts = "; ".join(
        f"s{s}: a {p['tp_a']}/{p['fp_a']}/{p['fn_a']} b {p['tp_b']}/{p['fp_b']}/{p['fn_b']}"
        for s, p in result.per_seed.items()
    )
    lines.append(f"  raw TP/FP/FN   : {counts}")
    return "\n".join(lines) + "\n"


def render_header(
    args: argparse.Namespace,
    contrasts_path: Path,
    runs_root: Path,
    digest: str,
    disabled: list[tuple[str, str]],
    families: dict[str, int],
    non_checked: list[tuple[str, str]],
    *,
    n_recomputed: int = 0,
    n_loaded: int = 0,
    stale: Sequence[str] = (),
) -> str:
    """Provenance header: what ran, on which predictions, with which knobs.

    It also states the two rules a reader needs before quoting any number here:
    which statistic is confirmatory, and what the instance-level interval does
    *not* license.

    ``n_recomputed``/``n_loaded`` say how much of the file this invocation
    actually recomputed. Under ``--only`` most rows are re-rendered from the
    per-contrast JSONs on disk, and a reader comparing two runs of this file
    needs to know which numbers the ``GENERATED`` stamp really covers.
    """
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    rendered = (
        f"# CONTRASTS RENDERED: {n_recomputed + n_loaded} ({n_recomputed} recomputed, "
        f"{n_loaded} loaded from disk)\n"
    )
    not_in_yaml = (
        "# NOT IN CURRENT YAML: "
        + "; ".join(stale)
        + " -- kept from an earlier run of this directory; the contrasts YAML no "
        "longer declares them, so they were neither recomputed nor re-checked\n"
        if stale
        else ""
    )
    skipped = "".join(f"# DISABLED: {name} -- {reason}\n" for name, reason in disabled)
    sizes = "; ".join(f"{name} = {k}" for name, k in sorted(families.items()))
    unchecked = (
        "# PARITY NOT CHECKED: "
        + "; ".join(f"{name} ({mode})" for name, mode in non_checked)
        + " -- these rows' numbers were not verified against a recorded test F1\n"
        if non_checked
        else ""
    )
    return (
        "# Paired uncertainty for recorded RE contrasts (issue #314, Agent A)\n"
        "# METRIC: exact-label micro F1, negative class excluded "
        "(== ieforge relation_f1) | SPLIT: test only, predictions of the "
        "best-validation checkpoint\n"
        "# SEED LEVEL (CONFIRMATORY): paired t over training seeds, two-sided, "
        "95% t-interval, computed from the RECORDED per-seed test F1 -- the values "
        "the paper prints (cell.json, or the published aggregate when the cell has "
        "none). A contrast is called significant only if THIS test is.\n"
        "# seed_recomputed (SECONDARY): the same statistic recomputed from the "
        "re-exported predictions. bf16-mixed inference is not bit-reproducible "
        "across processes, so it can differ -- and a few 1e-4 is enough to flip a "
        "near-tie seed and with it the win count. Reported for transparency; it is "
        "not the paper's number.\n"
        f"# INSTANCE LEVEL: paired percentile bootstrap over test instances, "
        f"R={args.resamples}, rng_seed={args.rng_seed}, trained models held fixed, "
        "same resample applied to both arms and every seed. It conditions on those "
        "ten models and so omits training-seed variance: an instance-level interval "
        "excluding zero does NOT upgrade a seed-level-null contrast and does NOT "
        "license a claim that the effect survives retraining.\n"
        "# The two intervals measure different things and are never merged.\n"
        f"# MULTIPLICITY: Holm-adjusted seed-level p within family ({sizes}); "
        "the instance-level intervals are UNADJUSTED. A single-member family "
        "adjusts to its own raw p.\n"
        f"# CONTRASTS: {contrasts_path} (sha256 {digest[:16]}) | RUNS ROOT: {runs_root}\n"
        f"# PARITY: identical instance-id set AND identical gold labels across all "
        f"seeds of both arms; recomputed per-seed F1 within {args.parity_tol} of the "
        "recorded value (cell.json or the published aggregate). parity_mode: checked "
        "= every arm x seed compared, vacuous = at least one had no recorded value, "
        "unchecked = strict parity off.\n"
        f"# GENERATED: {now} | strict_parity={not args.no_strict_parity}\n"
        f"{rendered}"
        "# BioRED is excluded: its metric (biored_f1) is not the micro F1 above.\n"
        f"{unchecked}{not_in_yaml}{skipped}\n"
    )


def compute_results(
    specs: Sequence[tuple[ContrastSpec, str]], args: argparse.Namespace
) -> list[tuple[ContrastResult, str]]:
    """Run every selected contrast and write its per-contrast JSON record."""
    results: list[tuple[ContrastResult, str]] = []
    for spec, family in specs:
        print(
            f"[{spec.name}] {spec.dataset} {spec.arm_a[0]} - {spec.arm_b[0]} ...",
            flush=True,
        )
        results.append(
            (
                run_contrast(
                    spec,
                    n_resamples=args.resamples,
                    rng_seed=args.rng_seed,
                    strict_parity=not args.no_strict_parity,
                    parity_tol=args.parity_tol,
                ),
                family,
            )
        )
    return results


def load_recorded_results(
    contrasts_dir: Path, skip: set[str]
) -> dict[str, ContrastResult]:
    """Rebuild every ``contrasts/<name>.json`` except the ones just recomputed.

    A selective run must not shrink the record: the aggregate and the TSV are
    rendered from the union of what ran now and what is already on disk. A file
    whose payload names a different contrast than its filename is a hand-edited
    record and raises rather than being merged under the wrong name.
    """
    loaded: dict[str, ContrastResult] = {}
    if not contrasts_dir.is_dir():
        return loaded
    for path in sorted(contrasts_dir.glob("*.json")):
        if path.stem in skip:
            continue
        result = ContrastResult.from_json(path.read_text())
        if result.contrast != path.stem:
            msg = f"{path}: record names contrast {result.contrast!r}"
            raise ValueError(msg)
        loaded[result.contrast] = result
    return loaded


def merge_results(
    computed: Sequence[tuple[ContrastResult, str]],
    loaded: Mapping[str, ContrastResult],
    order: Sequence[tuple[str, str]],
) -> tuple[list[tuple[ContrastResult, str]], list[str]]:
    """Order the merged records by the YAML, stale ones last.

    ``order`` is ``(name, family)`` for every enabled contrast of the YAML --
    the full declaration, not the ``--only`` selection -- so the file keeps one
    stable order whichever subset was recomputed. A loaded record whose name has
    left the YAML has no family to place it in; it is appended and named in the
    header rather than dropped, because dropping it is exactly the silent data
    loss this merge exists to prevent.
    """
    by_name = {result.contrast: result for result, _ in computed}
    families = dict(order)
    merged: list[tuple[ContrastResult, str]] = []
    for name, family in order:
        result = by_name.get(name) or loaded.get(name)
        if result is not None:
            merged.append((result, family))
    placed = {result.contrast for result, _ in merged}
    stale = [name for name in sorted(loaded) if name not in placed]
    merged.extend((loaded[name], families.get(name, "unfiled")) for name in stale)
    return merged, stale


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--contrasts", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, default=None)
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    ap.add_argument("--parity-tol", type=float, default=DEFAULT_PARITY_TOL)
    ap.add_argument("--only", nargs="*", default=None, help="contrast names")
    ap.add_argument("--no-strict-parity", action="store_true")
    args = ap.parse_args(argv)

    runs_root = _runs_root(args.runs_root)
    config = yaml.safe_load(args.contrasts.read_text())
    digest = hashlib.sha256(args.contrasts.read_bytes()).hexdigest()
    specs, disabled = build_specs(
        config, runs_root, set(args.only) if args.only else None
    )
    if not specs:
        print("no contrasts selected", file=sys.stderr)
        return 2

    out = args.output_dir
    (out / "contrasts").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.contrasts, out / "contrasts.yaml")

    computed = compute_results(specs, args)
    for result, _ in computed:
        (out / "contrasts" / f"{result.contrast}.json").write_text(result.to_json())

    # Merge with whatever else this directory already holds, then render the
    # WHOLE record. Rendering only the recomputed subset would rewrite
    # aggregate.txt and bootstrap.tsv without the other contrasts -- their JSONs
    # would survive, so the loss reads as a clean run rather than as an error.
    loaded = load_recorded_results(
        out / "contrasts", {result.contrast for result, _ in computed}
    )
    results, stale = merge_results(computed, loaded, declared_order(config))

    # Holm within each declared family, over the RECORDED seed-level p. It is
    # computed over the MERGED family, never over the `--only` subset: adjusting
    # a k=2 family as k=1 would shrink every adjusted p in the file.
    families = {f: sum(1 for _, g in results if g == f) for _, f in results}
    holm_by_name: dict[str, float | None] = {}
    for family in families:
        members = [r for r, g in results if g == family]
        adjusted = holm_adjust([r.seed["p"] for r in members])
        for result, value in zip(members, adjusted, strict=True):
            holm_by_name[result.contrast] = value

    blocks: list[str] = []
    rows: list[dict[str, Any]] = []
    non_checked: list[tuple[str, str]] = []
    for result, family in results:
        p_holm = holm_by_name[result.contrast]
        blocks.append(render_block(result, family, families[family], p_holm))
        row = result.as_row()
        row["family"] = family
        row["family_size"] = families[family]
        row["seed_p_holm"] = "" if p_holm is None else p_holm
        rows.append(row)
        if result.parity_mode != "checked":
            non_checked.append((result.contrast, result.parity_mode))
        mean = result.seed["mean_delta"]
        raw = result.seed["p"]
        print(
            f"    [{result.contrast}] seed d "
            f"{'n/a' if mean is None else f'{mean:+.4f}'} p "
            f"{'n/a' if raw is None else f'{raw:.3f}'} p_holm "
            f"{'n/a' if p_holm is None else f'{p_holm:.3f}'} | "
            f"instance CI [{result.bootstrap['ci_low']:+.4f}, "
            f"{result.bootstrap['ci_high']:+.4f}] | parity "
            f"{'ok' if result.parity_ok else 'BROKEN'} ({result.parity_mode})"
        )

    header = render_header(
        args,
        args.contrasts,
        runs_root,
        digest,
        disabled,
        families,
        non_checked,
        n_recomputed=len(computed),
        n_loaded=len(results) - len(computed),
        stale=stale,
    )
    (out / "aggregate.txt").write_text(header + "\n".join(blocks))
    with (out / "bootstrap.tsv").open("w") as fh:
        fh.write("\t".join(ROW_FIELDS) + "\n")
        for row in rows:
            fh.write(
                "\t".join(
                    f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k])
                    for k in ROW_FIELDS
                )
                + "\n"
            )
    print(f"wrote {out / 'aggregate.txt'} and {out / 'bootstrap.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
