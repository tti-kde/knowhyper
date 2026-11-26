#!/usr/bin/env python3
"""Run ONE matrix cell end to end: train -> val + test -> per-instance export.

    uv run python scripts/run_matrix_cell.py \
        --backbone pubmedbert --dataset biored --method ev_cold --seed 0 \
        --output-root "$KNOWHYPER_HOME/runs/matrix/2026-07-26"

Writes into `<output-root>/<dataset>/<backbone>_<method>/seed_<n>/`:

    checkpoints/epoch_*-f1_*.ckpt   best-epoch validation f1 is IN THE FILENAME
    driver.log                      training + test output (metrics live here)
    predictions.jsonl               per-instance test predictions
    cell.json                       resolved overrides, scores, status

`cell.json` is the contract the aggregator reads. It always records **val and
test**, because a val-only record is what made the paper's main table
unsubmittable: the BioRED runs passed `limit_test_batches=0` and no test number
could be recovered afterwards.

Exit code is 0 only if training, test evaluation and export all succeeded, so a
node driver can count failures without parsing logs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from knowhyper.matrix import (
    GRAFT_DIAGNOSTIC_METHODS,
    METHODS_BY_NAME,
    NER_DATASETS,
    Cell,
    MatrixError,
    plan,
)

_F1_IN_NAME = re.compile(r"f1_([0-9.]+)\.ckpt$")

# Lightning prints the test metrics as a BOX-DRAWING TABLE, not a dict:
#
#     │          test/f1          │    0.6697761416435242     │
#
# A dict-shaped regex (`'test/f1': 0.66`) matches nothing, which the cell then
# reports as "no test score" even though the run succeeded -- caught by the
# one-cell smoke test before this went out to 24 nodes. Accept both shapes: the
# table is what Lightning's own summary uses, the dict is what appears when a
# caller prints the returned metrics.
_TEST_METRIC = re.compile(r"test/(?:relation_f1|biored_f1|f1)\D{0,40}?([0-9]+\.[0-9]+)")


def _best_checkpoint(run_dir: Path) -> tuple[Path | None, float | None]:
    """Best checkpoint and its validation f1, read from the filename.

    Lightning encodes the monitored score in the name, so this survives the loss
    of any aggregate file -- that is how the paper's DDI figures were recovered
    after `ddi_g1v_logs/aggregate.txt` went missing.
    """
    best: tuple[float, Path] | None = None
    for p in (run_dir / "checkpoints").glob("epoch_*.ckpt"):
        m = _F1_IN_NAME.search(p.name)
        if not m:
            continue
        score = float(m.group(1))
        if best is None or score > best[0]:
            best = (score, p)
    return (best[1], best[0]) if best else (None, None)


def _test_score(log: Path) -> float | None:
    if not log.is_file():
        return None
    text = log.read_text(errors="replace")
    hits = _TEST_METRIC.findall(text)
    return float(hits[-1]) if hits else None


def _cell_fingerprint(dataset: str, method: str) -> str | None:
    """The SubHG cache fingerprint this cell will read, or None if unresolvable.

    Used to keep a warm-start encoder from being paired with a cache it was not
    pretrained on. Returns None rather than raising when the fingerprint cannot be
    computed (a missing optional dep, say) so a resolvable-by-name setup still
    works -- the strict/nonstrict check still applies in that case.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from resolve_cache_fingerprint import resolve

        # Read the experiment name off the registry directly. Going through
        # `plan()` is circular here: it REFUSES an ev_warm* cell without an
        # encoder, which is the very thing being resolved.
        from knowhyper.matrix import METHODS_BY_NAME

        experiment = METHODS_BY_NAME[method].experiment_for(dataset)
        if not experiment:
            return None
        return resolve(experiment)[0]
    except Exception:
        return None


def _cell_links_strictly(dataset: str, method: str) -> bool:
    """Does this cell's config set strict_type_match?

    Composed rather than inferred from the arm name. `khp_ner_ev_strict_bc2gm` is
    strict but does not end in `_strict`, and getting this wrong pairs a run with an
    encoder pretrained on a differently-linked cache -- the exact confound that made
    every earlier warm-start null.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from ieforge import CONFIGS_DIR as IEFORGE_CONFIGS_DIR

    from knowhyper.ieforge._configs import CONFIGS_DIR
    from knowhyper.matrix import METHODS_BY_NAME

    spec = METHODS_BY_NAME.get(method)
    experiment = spec.experiment_for(dataset) if spec else None
    if not experiment:
        return method.endswith("_strict")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(IEFORGE_CONFIGS_DIR)):
        cfg = compose(
            config_name="train",
            overrides=[
                f"experiment={experiment}",
                f"hydra.searchpath=[file://{CONFIGS_DIR}]",
            ],
        )
    knowledge = cfg.get("knowledge")
    if knowledge is None:
        return False
    return bool(knowledge.pipeline.injector.linker.strict_type_match)


def _resolve_encoder(spec: str | None, dataset: str, method: str) -> str | None:
    """Pick the encoder pretrained on THIS cell's SubHG cache.

    Warm-start only helped when the pretrain and end-task data matched, so one
    encoder must not be shared across differently-linked caches. Passing a
    directory lets a single sweep cover several datasets: it is searched for
    `<dir>/<dataset>/cross_modal/*<strict|nonstrict>*/encoder.pt`, keyed off
    whether the method is a strict-typing variant. A file path is returned as is.

    Fails loudly on an ambiguous or empty match -- silently training an "ev_warm"
    arm from the wrong encoder, or cold, is exactly the confound this exists to
    avoid.
    """
    if not spec:
        return None
    path = Path(spec)
    if path.is_file():
        return str(path)
    if not path.is_dir():
        msg = f"--encoder {spec!r} is neither a file nor a directory"
        raise SystemExit(msg)
    # Strictness comes from the CONFIG, not the method name. On RE the two agree --
    # `ev_warm` is non-strict and `ev_warm_strict` is strict -- so reading it off the
    # name worked. On NER it does not: every NER arm is strict (the campaign fixed one
    # linking setting so the three knowledge arms differ in one variable), yet the
    # method is still called `ev_warm`. Inferring from the name asked for a
    # `nonstrict` encoder that does not and should not exist, and all 8 cells of every
    # ev_warm subjob failed with "found 0. Available: ['strict_71b330506dd2ee18']".
    #
    # Resolve the arm and ask it, so the encoder always matches the linking the cell
    # actually trains with.
    # Read `strict_type_match` out of the COMPOSED config. Not a name test: on RE the
    # method name carries it (`ev_warm` vs `ev_warm_strict`), but the NER arms are
    # `khp_ner_ev_strict_bc2gm` -- strict, yet neither the method nor the experiment
    # name ENDS in `_strict`, so any suffix rule silently asks for the wrong encoder.
    # The config is the only source that is right for both families.
    #
    # `surface` is a THIRD value, not a third strictness. BioRED's encoders are
    # all pretrained on the surface-fallback cache, because BioRED cannot train
    # on the id-only one at all (46.2% empty sub-HGs abort validation). So an
    # `ev_warm_surface` cell needs `surface_<fingerprint>`, which the
    # strict/nonstrict binary can never produce -- it asked for `nonstrict` and
    # failed with "found 0. Available: ['surface_cb831d8592bc38aa']", losing 10
    # BioRED cells. The suffix is checked FIRST and only falls back to the
    # config test, so the NER case above (strict arms not named `_strict`)
    # keeps working.
    if method.endswith("_surface"):
        want = "surface"
    else:
        want = "strict" if _cell_links_strictly(dataset, method) else "nonstrict"
    # The directory name is `<strict|nonstrict>_<cache fingerprint>`, and BOTH
    # halves matter. Matching only the strict token would hand a run whose cache is
    # 8cc23671485b08da an encoder pretrained on 116e42e74b7255a8 -- a different
    # PrimeHKG pin entirely. Warm-start only ever helped when the pretrain and
    # end-task caches matched (DDI +0.0091 p=.002); reusing one across
    # differently-linked caches reproduced the earlier nulls, so a silent
    # cross-pin match would manufacture exactly that confound.
    fingerprint = _cell_fingerprint(dataset, method)
    hits = [
        p
        for p in sorted(path.glob(f"{dataset}/cross_modal/*/encoder.pt"))
        # "nonstrict" contains "strict", so match on the leading token.
        if p.parent.name.split("_", 1)[0] == want
        and (fingerprint is None or p.parent.name.split("_", 1)[-1] == fingerprint)
    ]
    if len(hits) != 1:
        detail = f" for cache {fingerprint}" if fingerprint else ""
        available = sorted(
            q.parent.name for q in path.glob(f"{dataset}/cross_modal/*/encoder.pt")
        )
        msg = (
            f"need exactly one {want} encoder for {dataset}{detail} under {spec}, "
            f"found {len(hits)}. Available: {available}. An encoder pretrained on "
            "a different cache must NOT be reused -- pretrain on this one first."
        )
        raise SystemExit(msg)
    return str(hits[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--encoder", help="pretrained encoder for warm-start methods")
    ap.add_argument(
        "--predictions-root",
        help=(
            "first-pass prediction dir; REQUIRED for NER knowledge arms and rejected "
            "for RE. plan() enforces both directions -- without it a NER knowledge "
            "arm resolves predictions_root=null and trains with no mentions at all, "
            "reporting a knowledge score for a model that got no knowledge."
        ),
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--no-export", action="store_true", help="skip prediction export")
    ap.add_argument(
        "--extra-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "extra Hydra override appended AFTER the method's own, so it wins. "
            "For hyperparameter sweeps over an existing method "
            "(e.g. trainer.callbacks.3.knowledge_lr_mult=3). REQUIRES --variant, "
            "because the run directory is derived from the method name and two "
            "multipliers would otherwise write to the same directory -- one live "
            "writer per run dir is a hard rule."
        ),
    )
    ap.add_argument(
        "--variant",
        help=(
            "suffix appended to the run subdirectory and label, to keep sweep "
            "arms apart (e.g. lrmult3). Mandatory with --extra-override."
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Only warm-start arms consume an encoder. Resolving one for every method made
    # `--encoder` fatal to arms that need none: a submission carrying the encoder
    # ROOT (which is how one sweep covers three datasets) failed all 8 `base` cells
    # per node with "need exactly one nonstrict encoder ... found 0" -- an accurate
    # complaint about a lookup that should never have run. `expand()` already gates
    # on this flag; the CLI path did not.
    #
    # An unknown method still reaches _resolve_encoder as before, so `plan()` keeps
    # ownership of reporting it.
    method_spec = METHODS_BY_NAME.get(args.method)
    wants_encoder = method_spec is None or method_spec.needs_encoder
    encoder = (
        _resolve_encoder(args.encoder, args.dataset, args.method)
        if wants_encoder
        else None
    )
    if args.encoder and not wants_encoder:
        print(
            f"note: {args.method} takes no encoder; ignoring --encoder",
            file=sys.stderr,
        )

    cell = Cell(
        backbone=args.backbone,
        dataset=args.dataset,
        method=args.method,
        seed=args.seed,
        encoder=encoder,
        predictions_root=args.predictions_root,
    )
    try:
        cell_plan = plan(cell)
    except MatrixError as exc:
        print(f"cannot run {cell.label}: {exc}", file=sys.stderr)
        return 2

    # A sweep arm must be distinguishable on disk AND in the aggregate, or the
    # second multiplier silently overwrites the first.
    if args.extra_override and not args.variant:
        print(
            "--extra-override requires --variant: without it every sweep arm "
            "writes to the same run directory and only the last survives",
            file=sys.stderr,
        )
        return 2

    run_dir = Path(args.output_root) / cell.run_subdir
    label = cell.label
    if args.variant:
        run_dir = (
            run_dir.parent.parent
            / f"{run_dir.parent.name}__{args.variant}"
            / run_dir.name
        )
        label = f"{label}__{args.variant}"
    hydra_args = cell_plan.hydra_args(str(run_dir), epochs=args.epochs)
    # Appended last so a sweep value overrides the method's default for the key.
    hydra_args = [*hydra_args, *args.extra_override]
    train_cmd = [
        "ieforge",
        "train",
        *hydra_args,
        "hydra.searchpath=[pkg://knowhyper.configs]",
    ]

    if args.dry_run:
        print(" \\\n  ".join(train_cmd))
        return 0

    # A run directory is IMMUTABLE once its cell finished (CLAUDE.md, Central
    # Storage). Refuse rather than retrain in place: the first line this function
    # used to write was `status: running` over the completed record, which destroys
    # the val/test scores irrecoverably -- the checkpoint filename preserves only
    # validation f1, and the test score exists nowhere else.
    #
    # This is not hypothetical. A top_k sweep declared `ev_cold` as its control, the
    # planner correctly added it, and its cells resolved to the SAME run_subdir as
    # the completed main-run arm -- so 9 finished cells (3 datasets x seeds 0-2)
    # began retraining over their own results before it was noticed. Pass
    # --variant to place a re-run beside the original instead of on top of it.
    existing = run_dir / "cell.json"
    if existing.is_file():
        try:
            prior = json.loads(existing.read_text())
        except (OSError, json.JSONDecodeError):
            prior = {}
        prior_status = str(prior.get("status", ""))
        # Statuses that hold NO usable result and so may be re-run in place.
        # `failed_train` / `failed_no_checkpoint` died before any score existed;
        # `running` means a node was killed mid-cell. Whitelisting bare "failed" was
        # wrong -- nothing writes that string, so a crashed cell became permanently
        # un-rerunnable and the text arm's retry was refused 60/60 times.
        #
        # `failed_export` is deliberately NOT here: those cells hold valid val AND
        # test and only the per-instance export failed, so overwriting them destroys
        # real measurements -- exactly what this guard exists to prevent.
        rerunnable = {
            "",
            "running",
            "failed",
            "failed_train",
            "failed_no_checkpoint",
            "failed_no_test_score",
        }
        if prior_status not in rerunnable:
            print(
                f"REFUSING to overwrite completed cell {label}: "
                f"{existing} has status={prior_status!r} "
                f"(val={prior.get('val_f1')} test={prior.get('test_f1')}). "
                "Run directories are immutable once finished -- pass --variant "
                "<name> to write a re-run alongside it, or delete the directory "
                "deliberately if the old result is genuinely obsolete.",
                file=sys.stderr,
            )
            return 1

    run_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "label": label,
        "variant": args.variant,
        "extra_overrides": list(args.extra_override),
        "backbone": cell.backbone,
        "dataset": cell.dataset,
        "method": cell.method,
        "seed": cell.seed,
        "experiment": cell_plan.experiment,
        "epochs": args.epochs,
        "encoder": cell.encoder,
        "overrides": cell_plan.overrides,
        "status": "running",
    }
    cell_json = run_dir / "cell.json"
    cell_json.write_text(json.dumps(record, indent=2) + "\n")

    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    log = run_dir / "driver.log"
    started = time.time()
    with log.open("w") as fh:
        rc = subprocess.run(  # noqa: S603 - argv built from the matrix registry
            train_cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, check=False
        ).returncode
    record["train_seconds"] = round(time.time() - started)

    ckpt, val_f1 = _best_checkpoint(run_dir)
    record["best_checkpoint"] = str(ckpt) if ckpt else None
    record["val_f1"] = val_f1
    record["test_f1"] = _test_score(log)

    if rc != 0 or ckpt is None:
        record["status"] = "failed_train" if rc != 0 else "failed_no_checkpoint"
        cell_json.write_text(json.dumps(record, indent=2) + "\n")
        print(f"FAIL {cell.label}: {record['status']}", file=sys.stderr)
        return 1

    if record["test_f1"] is None:
        # Loud, not silent: a val-only cell is exactly the defect this harness exists
        # to prevent, and it is cheap to catch here versus after 100 cells.
        record["status"] = "failed_no_test_score"
        cell_json.write_text(json.dumps(record, indent=2) + "\n")
        print(f"FAIL {cell.label}: no test score in driver.log", file=sys.stderr)
        return 1

    # `export_re_predictions.py` is RE-specific: it exports per-PAIR relation
    # predictions and passes `dataset.split=test`, a key the NER dataset configs do not
    # have (`Could not override 'dataset.split'`). NER needs span-level export, which is
    # a different artifact and a different script. Skipping is right rather than
    # failing: the cell's val and test SCORES are already recorded above and are what
    # the aggregate reads -- 60 cells were marked `failed_export` while holding
    # perfectly good numbers (e.g. bc2gm/base/seed_0 val=0.8100 test=0.8070), which
    # reads as a broken run and would have sent them to be re-trained for nothing.
    #
    # Recorded explicitly as `export_skipped_ner`, NOT silently: the per-instance
    # export is what the campaign's stratified analysis (empty vs non-empty sub-HG)
    # needs, so this is a real gap to close, not a non-issue. Tracked in the plan.
    if not args.no_export and cell.dataset in NER_DATASETS:
        record["status"] = "completed"
        record["export_dir"] = None
        record["export_skipped"] = "ner_span_export_not_implemented"
        cell_json.write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"OK   {cell.label} val={record['val_f1']} test={record['test_f1']} "
            "(export skipped: NER)"
        )
        return 0

    if not args.no_export:
        export_dir = run_dir / "export"
        export_cmd = [
            "python",
            "scripts/analysis/export_re_predictions.py",
            "--experiment",
            cell_plan.experiment,
            "--condition",
            f"{cell.backbone}_{cell.method}",
            "--seed",
            str(cell.seed),
            "--checkpoint",
            str(ckpt),
            "--split",
            "test",
            "--num-workers",
            "8",
            "--batch-size",
            "64",
            "--output-dir",
            str(export_dir),
            "--",
            *cell_plan.overrides,
            # Must be the SAME mechanism the training run used -- hardcoding
            # `backbone=` here would raise on every khp_* arm, which has no
            # /backbone in its defaults. It is a LIST because khp_* arms need
            # both the encoder and the tokenizer moved together.
            *cell_plan.backbone_overrides,
        ]
        with (run_dir / "export.log").open("w") as fh:
            erc = subprocess.run(  # noqa: S603 - argv built above
                export_cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, check=False
            ).returncode
        record["export_dir"] = str(export_dir) if erc == 0 else None
        if erc != 0:
            record["status"] = "failed_export"
            cell_json.write_text(json.dumps(record, indent=2) + "\n")
            print(f"FAIL {cell.label}: export", file=sys.stderr)
            return 1

        # GRAFT-style arms get a second, arm-specific sidecar: the per-instance gate
        # and the source-only vs final label, plus parameter/VRAM/throughput cost.
        # Without it a null cannot be read -- a gate that collapsed at init and a
        # gate that learned to ignore evidence give the same F1 and completely
        # different distributions (#189).
        #
        # A failure here is recorded but NOT fatal: val and test are already
        # measured and immutable at this point, and marking the cell failed would
        # invite a re-train that overwrites real scores to recover a diagnostic.
        if cell.method in GRAFT_DIAGNOSTIC_METHODS:
            diagnostics_dir = run_dir / "graft_diagnostics"
            diagnostics_cmd = [
                "python",
                "scripts/analysis/export_graft_diagnostics.py",
                "--experiment",
                cell_plan.experiment,
                "--condition",
                f"{cell.backbone}_{cell.method}",
                "--seed",
                str(cell.seed),
                "--checkpoint",
                str(ckpt),
                "--split",
                "test",
                "--num-workers",
                "8",
                "--batch-size",
                "64",
                "--output-dir",
                str(diagnostics_dir),
                "--",
                *cell_plan.overrides,
                # Splat: `backbone_overrides` is a LIST (a khp_* arm needs both
                # `model.inner.model_name=` and `model.tokenizer_name=`). The singular
                # `backbone_override` this line used no longer exists, so the
                # diagnostics export died with AttributeError while the sibling
                # prediction export above -- already on the plural form -- worked.
                *cell_plan.backbone_overrides,
            ]
            with (run_dir / "graft_diagnostics.log").open("w") as fh:
                drc = subprocess.run(  # noqa: S603 - argv built above
                    diagnostics_cmd,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=False,
                ).returncode
            record["graft_diagnostics_dir"] = str(diagnostics_dir) if drc == 0 else None
            if drc != 0:
                record["graft_diagnostics_error"] = "export_failed"
                print(
                    f"WARN {cell.label}: graft diagnostics export failed "
                    "(val/test scores are unaffected; see graft_diagnostics.log)",
                    file=sys.stderr,
                )

    record["status"] = "completed"
    cell_json.write_text(json.dumps(record, indent=2) + "\n")
    print(f"OK   {cell.label} val={record['val_f1']} test={record['test_f1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
