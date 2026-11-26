#!/usr/bin/env python3
"""Aggregate per-instance attention JSONL into a cold-vs-warm comparison (#224).

`extract_attention.py` prints a per-category mean for ONE checkpoint. The question
here is different: does cross-modal pretraining change *where* the model looks?
That needs the same categories measured across seeds for both arms, with a paired
test over the seed means.

Two reporting choices, both deliberate:

* **Attention SHARE, not raw weight.** Raw per-token weight depends on how many
  knowledge tokens an instance has (softmax over a longer key sequence gives every
  token less), and the sub-hypergraph size differs per instance. A share --
  category weight divided by total knowledge weight -- is comparable across
  instances and sums to 1, so a shift between categories is readable as
  reallocation rather than as a change in overall attention magnitude.
* **Per-instance shares averaged per seed, then paired across seeds.** Pooling all
  tokens from all instances would weight large sub-hypergraphs more heavily and
  would treat tokens as independent samples when they are not; the seed is the
  unit of replication in every other contrast in this project.

Usage:
  uv run python scripts/analysis/aggregate_attention.py \
    --cold cold_s0.jsonl cold_s1.jsonl cold_s2.jsonl \
    --warm warm_s0.jsonl warm_s1.jsonl warm_s2.jsonl \
    --out docs/reports/data/re_attention_2026_08_01/shares_ddi.txt
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

#: Report attribute modalities separately from entities and hyperedges. The
#: 2026-07-13 report had to regroup `entity` and `attr:name` because they shared one
#: vector; `skip_modalities: ["name"]` means no attr:name node is built now, so the
#: categories are disjoint and no regrouping is needed. If a run ever emits
#: attr:name again, that assumption is broken -- hence the explicit check below.
_DEGENERATE = "attr:name"


def _instance_shares(row: dict) -> dict[str, float] | None:
    """Category -> fraction of this instance's total knowledge attention.

    Returns None for instances with no knowledge tokens or zero total weight:
    an empty sub-hypergraph has no allocation to report, and including it as all
    zeros would drag every category's mean towards zero in proportion to how
    often retrieval failed -- conflating "looks elsewhere" with "had nothing".
    """
    per_cat: dict[str, float] = defaultdict(float)
    total = 0.0
    for tok in row.get("tokens", ()):
        weight = float(tok.get("attn_weight", 0.0))
        per_cat[tok["category"]] += weight
        total += weight
    if total <= 0.0:
        return None
    return {cat: weight / total for cat, weight in per_cat.items()}


def _seed_shares(path: Path) -> tuple[dict[str, float], int, int]:
    """Mean per-instance share per category for one seed's JSONL."""
    sums: dict[str, float] = defaultdict(float)
    kept = 0
    skipped = 0
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            shares = _instance_shares(json.loads(line))
            if shares is None:
                skipped += 1
                continue
            kept += 1
            for cat, share in shares.items():
                sums[cat] += share
    if kept == 0:
        msg = f"{path}: no instance had any knowledge attention"
        raise ValueError(msg)
    return {cat: total / kept for cat, total in sums.items()}, kept, skipped


def _paired_t(treated: list[float], control: list[float]) -> tuple[float, float]:
    """Paired t statistic and mean delta over seeds.

    Returns nan when n < 2 (no test is possible) and a SIGNED infinity when every
    seed moved by the same amount. The sign matters: returning bare ``inf`` for a
    uniform *decrease* reports maximal evidence in the wrong direction, and a
    uniform decrease is exactly what a feature store that went dead in one arm
    produces -- which ``main`` warns about separately.
    """
    deltas = [t - c for t, c in zip(treated, control, strict=True)]
    delta = mean(deltas)
    if len(deltas) < 2:
        return float("nan"), delta
    spread = stdev(deltas)
    if spread == 0.0:
        if delta == 0.0:
            return 0.0, delta
        return float("inf") if delta > 0 else float("-inf"), delta
    return delta / (spread / len(deltas) ** 0.5), delta


def _format_t(t_stat: float) -> str:
    """Render a t statistic, making non-finite values unmistakably not numbers.

    A right-aligned lowercase ``nan``/``inf`` in a column of decimals reads as a
    rendered value rather than a sentinel, and ``inf`` in particular reads as
    maximal significance when it actually means "no variance, t is undefined".
    """
    if math.isnan(t_stat):
        return "n/a"
    if math.isinf(t_stat):
        return "+undef" if t_stat > 0 else "-undef"
    return f"{t_stat:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold", nargs="+", required=True, type=Path)
    parser.add_argument("--warm", nargs="+", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--label",
        default="",
        help="dataset name for the header; the table is dataset-agnostic but a"
        " stale name in a provenance header is how a wrong figure gets cited",
    )
    args = parser.parse_args()

    arms: dict[str, list[dict[str, float]]] = {}
    counts: dict[str, list[tuple[int, int]]] = {}
    for name, paths in (("cold", args.cold), ("warm", args.warm)):
        arms[name] = []
        counts[name] = []
        for path in paths:
            shares, kept, skipped = _seed_shares(path)
            arms[name].append(shares)
            counts[name].append((kept, skipped))

    categories = sorted({cat for seeds in arms.values() for s in seeds for cat in s})
    degenerate = [c for c in categories if c == _DEGENERATE]

    lines: list[str] = []
    what = f"{args.label} " if args.label else ""
    lines.append(f"# RE cross-attention allocation, {what}ev_cold vs ev_warm")
    lines.append("# METRIC: mean per-instance share of total knowledge attention")
    lines.append(
        "#   (per-instance shares averaged within a seed, then paired across seeds)"
    )
    lines.append(f"# SEEDS: cold n={len(args.cold)} | warm n={len(args.warm)}")
    for name in ("cold", "warm"):
        for path, (kept, skipped) in zip(
            (args.cold if name == "cold" else args.warm), counts[name], strict=True
        ):
            total = kept + skipped
            pct = 100.0 * skipped / total if total else 0.0
            lines.append(
                f"#   {name} {path.name}: {kept} kept, {skipped}/{total} "
                f"({pct:.1f}%) with no knowledge attention"
            )

    # Warnings go BEFORE the table. A reader who sees a large delta and stops
    # reading must not miss the note that says the delta is an artefact.
    warnings: list[str] = []

    # Every seed must cover the same instances, or the seed means are not
    # comparable and the paired statistic mixes different populations. A silently
    # truncated extraction is the specific way this project could manufacture a
    # one-seed "effect".
    kept_counts = {k for arm in counts.values() for k, _ in arm}
    if len(kept_counts) > 1:
        warnings.append(
            f"! Seeds do NOT cover the same number of instances: {sorted(kept_counts)}."
            " The per-seed means are not comparable and the paired t mixes"
            " populations. Suspect a truncated or partial extraction."
        )
    for name in ("cold", "warm"):
        for path, (kept, skipped) in zip(
            (args.cold if name == "cold" else args.warm), counts[name], strict=True
        ):
            total = kept + skipped
            if total and skipped / total > 0.05:
                warnings.append(
                    f"! {name} {path.name}: {100.0 * skipped / total:.1f}% of instances"
                    " had NO knowledge attention and were dropped. The remaining"
                    " sample is not random -- it is the instances retrieval"
                    " succeeded on."
                )

    if degenerate:
        warnings.append(
            "! attr:name present. Under skip_modalities=['name'] it should not"
            " exist, and historically it shared one vector with `entity`, making the"
            " two indistinguishable. Do not read entity and attr:name as separate"
            " channels."
        )

    for cat in categories:
        cold_v = [s.get(cat, 0.0) for s in arms["cold"]]
        warm_v = [s.get(cat, 0.0) for s in arms["warm"]]
        # Exactly zero in BOTH arms: structurally absent, not unattended.
        if all(v == 0.0 for v in cold_v + warm_v):
            warnings.append(
                f"! {cat} is EXACTLY zero in every seed of both arms. That is a"
                " missing/zero feature vector being masked out, not a model choice."
            )
        # Exactly zero in ONE arm only. This is the dangerous case: it renders as
        # the largest delta in the table with an undefined t, and is
        # indistinguishable from a real reallocation. The 768-dim all-zero
        # attribute vectors incident was caught only because the value was exactly
        # 0.00%, so check for exactness here rather than for smallness.
        for dead, live, label in (
            (cold_v, warm_v, "cold"),
            (warm_v, cold_v, "warm"),
        ):
            if all(v == 0.0 for v in dead) and any(v != 0.0 for v in live):
                warnings.append(
                    f"! {cat} is EXACTLY zero in every {label} seed but nonzero in the"
                    f" other arm. Suspect a DEAD FEATURE STORE in {label}, not a"
                    " shift in attention -- this renders as the table's largest"
                    " delta. Verify the feature lookup before reading the row."
                )

    lines.extend(warnings)
    if warnings:
        lines.append("")

    lines.append("")
    header = f"{'category':16s} {'cold':>9s} {'warm':>9s} {'delta':>9s} {'t':>8s}"
    lines.append(header)
    lines.append("-" * len(header))

    for cat in categories:
        cold = [s.get(cat, 0.0) for s in arms["cold"]]
        warm = [s.get(cat, 0.0) for s in arms["warm"]]
        t_stat, delta = _paired_t(warm, cold)
        lines.append(
            f"{cat:16s} {mean(cold):9.4f} {mean(warm):9.4f} {delta:+9.4f} "
            f"{_format_t(t_stat):>8s}"
        )

    lines.append("")
    lines.append(
        "Shares sum to 1 within an arm, so a positive delta in one category is"
        " reallocation away from another, not an increase in total attention."
    )
    lines.append(
        "t of 'n/a' means n<2 (no test performed); '+undef'/'-undef' mean every seed"
        " moved identically so t is undefined -- NOT maximal significance."
    )

    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
