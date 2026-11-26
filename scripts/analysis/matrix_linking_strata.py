#!/usr/bin/env python3
"""Per-linking-stratum paired analysis over a matrix run.

    uv run python scripts/analysis/matrix_linking_strata.py \
        --output-root "$KNOWHYPER_HOME/runs/matrix/2026-07-26" \
        --dataset biored --treated ev_cold --control base

Splits per-instance test predictions by how well the KG could reach each
instance, and runs the paired seed-level test *within* each stratum, so "the
knowledge does not help" is separated from "the knowledge never arrived".

Unlike `text_linking_strata.py` (which hardcodes fingerprints and reads an ABCI
path), this resolves the cache fingerprint from the arm's own config via
`scripts/resolve_cache_fingerprint.py`, so it keeps working when the retriever
profile changes or a new PrimeHKG snapshot lands.

Strata, from the cached sub-hypergraph:

  DIRECT        2+ distinct nodes linked AND a hyperedge connects them
  NO_PATH       2+ distinct nodes linked, no direct edge
  PARTIAL_LINK  exactly 1 node linked -- still receives that node's evidence
  NO_TEXT       0 nodes linked -- the only byte-identical control

Two rules this encodes, both learned from getting them wrong:

* **Report accuracy, not per-stratum F1.** Within a stratum the positive rate
  skews and the control's F1 can collapse, manufacturing a large fake delta --
  the BioRED EV report showed +0.347 even in a stratum where the KG contributes
  nothing.
* **NO_TEXT is the zero calibration.** Both arms see identical input there, so
  its measured delta is the noise floor. If a treated stratum's delta is inside
  that band, the design cannot distinguish it from zero.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

from knowhyper.matrix import METHODS_BY_NAME, Cell, MatrixError, plan

STRATA = ("DIRECT", "NO_PATH", "PARTIAL_LINK", "NO_TEXT")
REPO = Path(__file__).resolve().parents[2]


def _fingerprint(experiment: str) -> str | None:
    """Resolve an experiment's SubHG fingerprint, reusing the audited resolver."""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from resolve_cache_fingerprint import resolve  # type: ignore[import-not-found]

        fingerprint, _, _ = resolve(experiment)
    except Exception as exc:
        print(f"  ({experiment}: no cache -- {type(exc).__name__})", file=sys.stderr)
        return None
    return fingerprint


def _cache_dir(fingerprint: str, dataset: str, root: Path) -> Path | None:
    d = root / fingerprint / dataset / "test"
    if d.is_dir():
        return d
    base = root / fingerprint
    if not base.is_dir():
        return None
    subs = [p for p in base.iterdir() if p.is_dir()]
    return subs[0] / "test" if len(subs) == 1 and (subs[0] / "test").is_dir() else None


def load_strata(cache_dir: Path) -> dict[str, str]:
    """instance_id -> stratum.

    Seed ids are deduplicated first: an instance can list the SAME node twice
    (self-pairs, or two mentions normalising to one node). Those have len>=2 but
    one distinct node, so a seed-intersection test can never fire and they would
    land in NO_PATH regardless of the graph -- 358 DDI instances did.
    """
    out: dict[str, str] = {}
    for p in cache_dir.glob("*.json"):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        seeds = set(j.get("seed_node_ids") or [])
        if not seeds:
            out[p.stem] = "NO_TEXT"
            continue
        if len(seeds) == 1:
            out[p.stem] = "PARTIAL_LINK"
            continue
        hit = False
        for e in j.get("hyperedges") or []:
            nodes = set(e.get("node_ids") or e.get("nodes") or [])
            if len(seeds & nodes) >= 2:
                hit = True
                break
        out[p.stem] = "DIRECT" if hit else "NO_PATH"
    return out


def load_preds(run_dir: Path) -> dict[str, bool]:
    """instance_id -> correct?, from a cell's exported predictions."""
    preds: dict[str, bool] = {}
    for jsonl in run_dir.rglob("predictions.jsonl"):
        for line in jsonl.read_text(errors="replace").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            preds[r["instance_id"]] = r["predicted_label"] == r["gold_label"]
    return preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="pubmedbert")
    ap.add_argument("--treated", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".knowhyper/cache/subhg/v2",
    )
    args = ap.parse_args()

    spec = METHODS_BY_NAME.get(args.treated)
    try:
        treated_plan = plan(
            Cell(
                args.backbone,
                args.dataset,
                args.treated,
                0,
                # Strata come from the retrieval cache, which does not depend on
                # the warm-start weights; a placeholder satisfies the check.
                encoder="unused" if spec and spec.needs_encoder else None,
            )
        )
    except MatrixError as exc:
        print(f"cannot resolve treated arm: {exc}", file=sys.stderr)
        return 2

    fingerprint = _fingerprint(treated_plan.experiment)
    if fingerprint is None:
        print(
            "treated arm has no SubHG cache (is it a text-only baseline?); "
            "strata are undefined",
            file=sys.stderr,
        )
        return 2
    cache_dir = _cache_dir(fingerprint, args.dataset, args.cache_root)
    if cache_dir is None:
        print(f"cache {fingerprint} not built for {args.dataset}", file=sys.stderr)
        return 2

    strata = load_strata(cache_dir)
    print(f"# strata from {fingerprint} ({len(strata)} instances)")

    acc: dict[str, dict[str, list[float]]] = {
        s: {"treated": [], "control": []} for s in STRATA
    }
    counts: dict[str, int] = {}
    used = []
    for seed in range(args.seeds):
        t_dir = (
            args.output_root
            / args.dataset
            / f"{args.backbone}_{args.treated}"
            / f"seed_{seed}"
        )
        c_dir = (
            args.output_root
            / args.dataset
            / f"{args.backbone}_{args.control}"
            / f"seed_{seed}"
        )
        t, c = load_preds(t_dir), load_preds(c_dir)
        if not t or not c:
            print(f"  seed {seed}: skipped (missing export)")
            continue
        shared = set(t) & set(c)
        by: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        for i in shared:
            by[strata.get(i, "UNKNOWN")].append((t[i], c[i]))
        missing = [s for s in STRATA if not by.get(s)]
        if missing:
            print(f"  seed {seed}: skipped (no rows in {','.join(missing)})")
            continue
        used.append(seed)
        for s in STRATA:
            rows = by[s]
            counts[s] = len(rows)
            acc[s]["treated"].append(sum(1 for a, _ in rows if a) / len(rows))
            acc[s]["control"].append(sum(1 for _, b in rows if b) / len(rows))

    n = len(used)
    if n < 2:
        print("insufficient seeds with exports", file=sys.stderr)
        return 1

    total = sum(counts.values()) or 1
    print(
        f"\n=== {args.dataset} / {args.backbone}: {args.treated} - {args.control} "
        f"(seeds {n}, instances {total}) ==="
    )
    print(
        f"{'stratum':13} {'n':>6} {'share':>7} {'treated':>8} {'control':>8} "
        f"{'delta':>9} {'p':>7} {'wins':>7}"
    )
    for s in STRATA:
        t_acc, c_acc = acc[s]["treated"], acc[s]["control"]
        diff = [x - y for x, y in zip(t_acc, c_acc, strict=True)]
        p = stats.ttest_rel(t_acc, c_acc).pvalue
        wins = sum(1 for d in diff if d > 0)
        print(
            f"{s:13} {counts[s]:6d} {counts[s] / total * 100:6.1f}% "
            f"{st.fmean(t_acc):8.4f} {st.fmean(c_acc):8.4f} "
            f"{st.fmean(diff):+9.4f} {p:7.3f} {wins:4d}/{n}"
        )
    print(
        "\nNO_TEXT is the zero calibration: both arms see identical input there, "
        "so its delta is the noise floor. Treated strata inside that band are "
        "not distinguishable from zero.\n"
        "NO_PATH is the interesting stratum: a gain where the KG holds no direct "
        "edge cannot be answer-lookup."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
