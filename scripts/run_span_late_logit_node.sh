#!/usr/bin/env bash
set -uo pipefail

# One rt_HF node's share of the #231 Variant 1 campaign (baseline-preserving late logit).
#
# Adapted from `run_span_ner_knowledge_node.sh` (#218) rather than written fresh: the
# preflight, circuit breaker, node reconciliation and exit-status handling there were each
# fixed in response to a measured failure, and re-deriving them would re-introduce those.
# The differences from #218 are exactly two:
#
#   * the model/experiment family is `khp_span_late_*` + `knowhyper_span_late_logit_*`
#   * every cell resolves and injects a FROZEN #220 baseline checkpoint, fail-closed
#
# The baseline resolution is delegated to `knowhyper.span_baseline`, which checks backbone,
# dataset, seed, batch size and the `patience=100` protocol marker. A compatible-but-wrong
# checkpoint loads cleanly, so provenance matching is not optional bookkeeping here.
#
# A separate driver on purpose: the RE / token-fusion cells go through
# run_matrix_cell.py, and span conditions are not bolted onto it.
#
# `abci sweep` passes parameters as `--KEY value` argv PAIRS with keys sorted --
# not env vars, not `--KEY=value` (abci_runner/param_exec.py:params_to_cli_args).
# Parse with a shift loop and guard with `:?` so a mis-parse dies in seconds
# instead of after ten epochs.
#
# `abci submit` passes NO parameters, so **you must pass `--index N` yourself**.
# This matters when resubmitting the casualties of an array (see CLAUDE.md on the
# 32-subjob `uv sync` race): `sweep` injects `--index` from `$PBS_ARRAY_INDEX`, a
# plain `submit` does not, and the omission is invisible in the submit JSON.
# It cost one wasted submission on 2026-08-01; the `:?` guard below caught it in
# seconds, which is the whole reason it is written that way.
#
# The two paths also differ in where logs land: `sweep` writes
# `logs/<jobid>[N].pbs1.OU|.ER`, `submit` writes `stdout.log`/`stderr.log` at the
# run-dir root. Looking for the array layout after a `submit` reads as "no logs".
#
# CELLS format: space-separated `backbone:dataset:condition:control:seed`, e.g.
#   pubmedbert:bc2gm:edge:real:0 biolinkbert:bc2gm:shared:zero:0
# Emit them with `plan_span_ner_knowledge.py --emit-cells`.

INDEX=""
NODE_LINES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --index) INDEX="$2"; shift 2 ;;
    --node-lines) NODE_LINES="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
: "${INDEX:?param index missing}"
: "${NODE_LINES:?param node-lines missing}"

# The completed #231 campaign remains the default. The cumulative wrapper sets this to
# ``cumulative`` and thereby changes only the experiment family, frozen first pass and
# output root; all battle-tested dispatch/provenance/disk guards below stay shared.
CAMPAIGN="${SPAN_LATE_CAMPAIGN:-graph}"
case "$CAMPAIGN" in
  graph|cumulative) : ;;
  *) echo "FATAL unknown SPAN_LATE_CAMPAIGN '$CAMPAIGN'" >&2; exit 2 ;;
esac

[ -s "$NODE_LINES" ] || { echo "FATAL node-lines missing/empty: $NODE_LINES" >&2; exit 1; }
total=$(grep -c . "$NODE_LINES")
if [ "$INDEX" -lt 1 ] || [ "$INDEX" -gt "$total" ]; then
  echo "FATAL index $INDEX out of range 1..$total" >&2; exit 1
fi
CELLS=$(sed -n "${INDEX}p" "$NODE_LINES")
[ -n "$CELLS" ] || { echo "FATAL empty node line at index $INDEX" >&2; exit 1; }

PH="$HOME"
# `abci sweep` has NO --pass-env, so every var the subjob needs is exported HERE.
# UV_OFFLINE is the one that bites: without it `uv sync --frozen` tries PyPI,
# which compute nodes cannot route to, and the subjob dies in ~44s.
export UV_OFFLINE=1
export KNOWHYPER_HOME="${KNOWHYPER_HOME:-$PH/.knowhyper}"
export HF_HOME="${HF_HOME:-$PH/.cache/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d "$HF_HOME" ] || { echo "FATAL HF_HOME does not exist: $HF_HOME" >&2; exit 1; }
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

EPOCHS="${EPOCHS:-10}"
# knowledge_lr_mult=10 is MEASURED -- but on the RE token-fusion architecture (260 cells
# over ChemProt/DDI), never on this span verifier. Phase 4 Case A screens it here, so it
# stays overridable while keeping 10 as the default every primary cell inherits.
#
# CAUTION -- this knob does NOT mean here what it means for RE. With `freeze_inner=true`,
# `BACKBONE_PREFIX = "model.inner."` selects exactly the parameters that were frozen and
# `split_knowledge_params` drops `requires_grad=False`, so `build_param_groups` returns ONE
# group. Measured on the instantiated `khp_span_late_ev`: 0 backbone tensors, 53 knowledge
# tensors / 11,844,866 params, 132,505,345 frozen and in no group. So `mult` scales the only
# LR (2e-5 x mult) instead of setting a trunk-vs-branch RATIO. A sweep here is a GLOBAL LR
# sweep for the late head and must be judged against its own mult=10 arm, never against
# CLAUDE.md's RE table. See docs/EXPERIMENTS.md §3.8.
#
# Passed EXPLICITLY even at the default: `.hydra/overrides.yaml` is the only durable record
# of the LR a completed cell ran at, and leaving it to the trainer config means a later edit
# to `difflr_benchmark_nolog` silently reinterprets every finished cell.
LR_MULT="${LR_MULT:-10}"
case "$LR_MULT" in
  ''|*[!0-9.]*|*.*.*) echo "FATAL LR_MULT must be a number, got '$LR_MULT'" >&2; exit 2 ;;
esac
# Optional. UNSET means "do not override", so the arms keep composing to their config
# value (-2.0). Passing a default explicitly would put an override on every primary cell
# that looks deliberate and would mask a later config change.
GATE_BIAS="${GATE_BIAS:-}"
# One full-fit smoke may inspect validation only. The fixed main matrix leaves this unset
# and records both validation and test. Accept only 0/1 so a typo cannot silently expose test.
VALIDATION_ONLY="${VALIDATION_ONLY:-0}"
case "$VALIDATION_ONLY" in
  0|1) : ;;
  *) echo "FATAL VALIDATION_ONLY must be 0 or 1, got '$VALIDATION_ONLY'" >&2; exit 2 ;;
esac
# Disk headroom, in KiB, overridable so the driver's own logic can be exercised on a
# small runner. Production defaults are 10 GiB to start and 5 GiB to keep dispatching:
# the ieforge default of 4 ckpts/cell (~1.75 GB) is what exhausts the 2 TB Lustre
# quota. CI runners have ~6 GB free, so a hardcoded 10 GiB made every executing driver
# test die at the preflight before reaching the behaviour under test -- passing locally
# and failing in CI purely on the host's free space.
MIN_FREE_KB="${MIN_FREE_KB:-10485760}"
STOP_FREE_KB="${STOP_FREE_KB:-5242880}"
if [ "$CAMPAIGN" = cumulative ]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-$KNOWHYPER_HOME/runs/ner/span_late_text_cumulative}"
  FROZEN_ARM=text_real
  COMPOSE_VERIFIER=scripts/analysis/verify_span_late_text_cumulative_compose.py
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-$KNOWHYPER_HOME/runs/ner/span_late_logit/variant1}"
  FROZEN_ARM=base
  COMPOSE_VERIFIER=scripts/analysis/verify_span_late_logit_compose.py
fi
# Optional durable checkpoint sink. The extension matrix enables this because 240
# frozen-inner cells would otherwise add ~138 GB to the 2 TB home quota at once.
# Metrics, Hydra provenance and diagnostics remain under OUTPUT_ROOT; only verified
# *.ckpt files move, through archive_completed_checkpoints.sh.
CHECKPOINT_ARCHIVE_ROOT="${CHECKPOINT_ARCHIVE_ROOT:-}"
# Where the FROZEN #220 span baselines live. This is `$KNOWHYPER_HOME/runs`, NOT
# `runs/ner` -- the tree is `runs/ner_span_text_phase3` (the resolver appends that name).
# `runs/ner/span019`, the #202 matched baseline, holds zero checkpoints (cleaned) and
# cannot serve here.
BASELINE_RUNS_ROOT="${BASELINE_RUNS_ROOT:-$KNOWHYPER_HOME/runs}"
# Scratch for per-cell stderr capture. Under the node's own run root so two concurrent
# nodes cannot collide, and so it is cleaned with the run rather than left in /tmp.
RUN_TMP="${RUN_TMP:-$OUTPUT_ROOT/.driver_tmp}"
mkdir -p "$RUN_TMP"
# Backbone-scoped prediction root. Candidate JSONL is backbone-specific: sharing
# one lattice across backbones compares architectures against different proposal
# distributions, which #218 forbids.
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-$KNOWHYPER_HOME/artifacts/predictions/span_ner}"
# Pretrain root for ev_warm, defaulted to where §0.4 writes its six encoders. A warm
# cell whose encoder is missing is still FAILED below rather than silently run cold --
# the default only removes the need for every sweep to restate the path, which was a
# gap: `abci sweep` has no env block, so an unset ENCODER_ROOT failed every warm cell
# with "refusing to run cold" and the sweep could not succeed at all.
ENCODER_ROOT="${ENCODER_ROOT:-$KNOWHYPER_HOME/artifacts/pretrain/span_ner}"
mkdir -p "$OUTPUT_ROOT"
status="$OUTPUT_ROOT/STATUS.node${INDEX}.$$"
: >"$status"

echo "=== span-knowledge node $INDEX/$total ==="
echo "campaign: $CAMPAIGN (frozen arm: $FROZEN_ARM)"
echo "cells: $CELLS"
echo "predictions root: $PREDICTIONS_ROOT"
echo "encoder root: ${ENCODER_ROOT:-<none>}"

# Preflight. A job that starts with no headroom loses every cell slowly.
if command -v lfs >/dev/null 2>&1; then
  lfs quota -h -u "$(whoami)" /home 2>/dev/null | tail -2 || true
fi
df -h "$OUTPUT_ROOT" | tail -1
free_kb=$(df -P "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')
if [ "${free_kb:-0}" -lt "$MIN_FREE_KB" ]; then
  echo "FATAL only ${free_kb} KiB free under $OUTPUT_ROOT (need ${MIN_FREE_KB} KiB); refusing to dispatch" >&2
  exit 1
fi

# Assert the pins and all 54 cells compose BEFORE spending GPU time. Catches upstream
# default drift and, critically, a wrong-directory submission: `abci submit` snapshots
# cwd, so submitting from the main repo ships a tree without this branch's model config
# and Hydra reports the experiment as unknown -- which reads like a config bug.
if ! uv run python "$COMPOSE_VERIFIER"; then
  echo "FATAL compose verification failed; refusing to dispatch" >&2
  exit 1
fi

# A breaker left by a sibling node (shared $OUTPUT_ROOT, one Lustre quota) or by an
# earlier attempt means the disk is already gone. Dispatching into it loses every
# cell slowly and reports `finished`.
if compgen -G "$OUTPUT_ROOT/.circuit_breaker.*" >/dev/null 2>&1; then
  echo "FATAL circuit breaker already present under $OUTPUT_ROOT; refusing to dispatch" >&2
  ls -1 "$OUTPUT_ROOT"/.circuit_breaker.* >&2
  exit 1
fi

declare -a pids=() labels=()
breaker_tripped=0
dispatched=0
gpu=0
# Word splitting on $CELLS is intended; pathname expansion is NOT. Without this a
# stray `*` in the node-lines file expands to repo filenames and each becomes a
# bogus "malformed spec" line.
set -f
# Count every spec on the node line up front. Counting inside the loop would stop
# at a `break`, so an early circuit-breaker exit would under-report the very cells
# it failed to dispatch.
requested=0
for _spec in $CELLS; do requested=$(( requested + 1 )); done

for spec in $CELLS; do
  IFS=: read -r backbone dataset condition control seed extra_field <<<"$spec"
  if [ -z "${seed:-}" ]; then
    echo "FAIL $spec (malformed; want backbone:dataset:condition:control:seed)" >>"$status"
    continue
  fi
  # A 6th field lands entirely in $seed with plain `read`, so a typo'd spec would
  # reach Hydra as `seed=0:JUNK` and fail only after the run dir was created.
  if [ -n "${extra_field:-}" ]; then
    echo "FAIL $spec (too many fields; want backbone:dataset:condition:control:seed)" >>"$status"
    continue
  fi
  case "$seed" in
    ''|*[!0-9]*) echo "FAIL $spec (seed '$seed' is not a non-negative integer)" >>"$status"; continue ;;
  esac
  case "$backbone" in
    pubmedbert|biolinkbert) : ;;
    *) echo "FAIL $spec (unknown backbone $backbone)" >>"$status"; continue ;;
  esac
  case "$dataset" in
    bc5cdr_chem|bc2gm|bc5cdr_disease|jnlpba|ncbi_disease) : ;;
    *) echo "FAIL $spec (unknown dataset $dataset)" >>"$status"; continue ;;
  esac
  case "$control" in
    real|shuffled) : ;;
    zero)
      # #231 decision (B): a frozen-inner zero arm IS the frozen baseline -- identical
      # weights, identical forward, and NO trainable parameter in the path. Training it is
      # impossible, not merely wasteful: `logits.requires_grad` is False, and both Phase 1
      # zero cells died with "element 0 of tensors does not require grad". #218 never hit
      # this because its inner is trainable.
      #
      # The arm's val/test are already recorded in `ner_span_text_phase3` and are read by
      # `knowhyper.span_zero_arm` at aggregate time, flagged `trained=False` so a copied
      # number can never be read as a measured one. SKIP rather than FAIL: the cell is
      # accounted for and deliberately not run.
      echo "SKIP $spec (zero arm is read from the frozen baseline record, decision B)" >>"$status"
      continue ;;
    *) echo "FAIL $spec (unknown control $control)" >>"$status"; continue ;;
  esac
  # `shared` is the collapsed zero arm: it composes the edge experiment but must
  # only ever pair with control=zero, which returns the span baseline before the
  # gate. Pairing it with real/shuffled would silently mislabel an edge arm.
  case "$condition" in
    edge)
      [ "$CAMPAIGN" = graph ] || { echo "FAIL $spec (edge is not in the fixed cumulative campaign)" >>"$status"; continue; }
      stem=khp_span_late_edge ;;
    ev)
      [ "$CAMPAIGN" = graph ] || { echo "FAIL $spec (ev is not in the fixed cumulative campaign)" >>"$status"; continue; }
      stem=khp_span_late_ev ;;
    ev_warm)
      [ "$CAMPAIGN" = graph ] || { echo "FAIL $spec (use text_ev_warm in the cumulative campaign)" >>"$status"; continue; }
      stem=khp_span_late_ev_warm ;;
    text_ev_warm)
      [ "$CAMPAIGN" = cumulative ] || { echo "FAIL $spec (text_ev_warm requires the cumulative campaign wrapper)" >>"$status"; continue; }
      stem=khp_span_late_text_ev_warm ;;
    shared)
      if [ "$control" != zero ]; then
        echo "FAIL $spec (condition 'shared' is only valid with control=zero)" >>"$status"
        continue
      fi
      stem=khp_span_late_edge ;;
    *) echo "FAIL $spec (unknown condition $condition)" >>"$status"; continue ;;
  esac
  case "$dataset" in
    bc5cdr_chem)    experiment="$stem" ;;
    bc2gm)          experiment="${stem}_bc2gm" ;;
    bc5cdr_disease) experiment="${stem}_bc5cdr_disease" ;;
    jnlpba)          experiment="${stem}_jnlpba" ;;
    ncbi_disease)   experiment="${stem}_ncbi_disease" ;;
  esac
  model="knowhyper_span_late_logit_${backbone}"

  PRED_DIR="$PREDICTIONS_ROOT/$backbone/$dataset"
  # Fail closed on missing proposals. The query builder reads `mention_source:
  # candidates`; without these files it retrieves nothing and the arm degenerates
  # into a baseline that still reports a knowledge score.
  missing=""
  for split in train validation test; do
    [ -s "$PRED_DIR/$split.jsonl" ] || missing="$missing $split"
  done
  if [ -n "$missing" ]; then
    echo "FAIL $spec (missing prediction JSONL:$missing under $PRED_DIR)" >>"$status"
    continue
  fi

  declare -a extra=()
  # No GATE_BIAS: the late head has no gate, so accepting that knob would allow a sweep
  # that changes nothing. #218 Phase 4 Case A already showed gate bias does not help there.
  if [ "$VALIDATION_ONLY" = 1 ]; then
    extra+=("trainer.limit_test_batches=0")
  fi

  # Resolve the FROZEN #220 baseline for this exact cell. Delegated to
  # `knowhyper.span_baseline` so the checks live in one tested place: backbone, dataset,
  # seed, bs=16 and the `patience=100` protocol marker that distinguishes
  # `ner_span_text_phase3` (seeds 3-9) from `_phase2` (seeds 0-2). Mixing those two inside
  # one frozen-baseline set is what #231 decision (a) avoids.
  baseline=$(uv run python -c "
import sys
from pathlib import Path
from knowhyper.span_baseline import resolve_baseline_checkpoint, BaselineMismatchError
try:
    found = resolve_baseline_checkpoint(
        runs_root=Path(sys.argv[1]), backbone=sys.argv[2], dataset=sys.argv[3],
        seed=int(sys.argv[4]), arm=sys.argv[5], compute_digest=False)
except BaselineMismatchError as error:
    print(f'MISMATCH {error}', file=sys.stderr); raise SystemExit(3)
print(found.checkpoint)
" "$BASELINE_RUNS_ROOT" "$backbone" "$dataset" "$seed" "$FROZEN_ARM" 2>"$RUN_TMP/baseline_err_$$" )
  # Capture the status IMMEDIATELY: command substitution keeps only stdout, so a
  # resolver that prints a path and THEN fails would pass an emptiness check alone.
  # A compatible-but-wrong frozen baseline loads cleanly (see this file's header),
  # which makes that a silently-wrong-number failure rather than a crash.
  baseline_rc=$?
  if [ "$baseline_rc" -ne 0 ]; then
    echo "FAIL $spec (baseline resolver exited $baseline_rc: $(tr -d '\n' <"$RUN_TMP/baseline_err_$$" | tail -c 200))" >>"$status"
    rm -f "$RUN_TMP/baseline_err_$$"
    continue
  fi
  if [ -z "$baseline" ]; then
    echo "FAIL $spec (baseline unresolved: $(tr -d '\n' <"$RUN_TMP/baseline_err_$$" | tail -c 200))" >>"$status"
    rm -f "$RUN_TMP/baseline_err_$$"
    continue
  fi
  rm -f "$RUN_TMP/baseline_err_$$"
  extra+=("model.pretrained_baseline_path=$baseline")
  if [ "$condition" = ev_warm ] || [ "$condition" = text_ev_warm ]; then
    if [ -z "$ENCODER_ROOT" ]; then
      echo "FAIL $spec (ev_warm needs ENCODER_ROOT; refusing to run cold)" >>"$status"
      continue
    fi
    # Exactly one match, per the matrix harness convention: 0 or 2 is an error,
    # never a silent pick. The encoder must be pretrained on the SAME
    # prediction-scoped SubHG cache this cell reads.
    mapfile -t found < <(find "$ENCODER_ROOT/$backbone/$dataset" -name encoder.pt 2>/dev/null | sort)
    if [ "${#found[@]}" -ne 1 ]; then
      echo "FAIL $spec (need exactly 1 encoder under $ENCODER_ROOT/$backbone/$dataset, found ${#found[@]})" >>"$status"
      continue
    fi
    encoder="${found[0]}"
    if [ ! -s "$encoder.manifest.json" ]; then
      echo "FAIL $spec (encoder has no manifest: $encoder.manifest.json)" >>"$status"
      continue
    fi
    extra+=("model.pretrained_encoder_path=$encoder")
  fi

  # A non-default LR_MULT gets its OWN arm directory. Without this, every multiplier writes
  # `<backbone>/<dataset>/<condition>_<control>/seed_<n>` and the completion check above --
  # which SKIPs any cell that already has a test score -- would silently skip the second
  # multiplier and let the aggregator read the FIRST multiplier's run twice. The sweep would
  # then report "the multiplier makes no difference": a plausible-looking wrong answer, not a
  # crash. This is the RE sweep's `--variant` bug, where every multiplier collided on one
  # directory and only the last survived.
  #
  # The DEFAULT path is deliberately unchanged (no `_mult10` suffix): the 84 completed Phase 3
  # cells live at the unsuffixed path, and suffixing it would orphan them and silently re-run
  # the campaign.
  arm_dir="${condition}_${control}"
  if [ "$LR_MULT" != "10" ]; then
    arm_dir="${arm_dir}_mult${LR_MULT}"
  fi
  RUN_DIR="$OUTPUT_ROOT/$backbone/$dataset/$arm_dir/seed_$seed"
  completion_file="$RUN_DIR/COMPLETED"
  if [ -s "$completion_file" ]; then
    if ! grep -qx "validation_only=$VALIDATION_ONLY" "$completion_file"; then
      echo "FAIL $spec (completion mode mismatch in $completion_file)" >>"$status"
      continue
    fi
    # Archiving may be enabled on a later invocation.  A completion sentinel
    # proves training succeeded; it must not suppress the requested storage move.
    if [ -n "$CHECKPOINT_ARCHIVE_ROOT" ] && ! grep -q '^OK checkpoints=' "$RUN_DIR/checkpoint_archive.log" 2>/dev/null; then
      archive_dest="$CHECKPOINT_ARCHIVE_ROOT/$backbone/$dataset/$arm_dir/seed_$seed"
      if ! bash scripts/archive_completed_checkpoints.sh \
          --source "$RUN_DIR" --destination "$archive_dest" --move \
          >"$RUN_DIR/checkpoint_archive.log" 2>&1; then
        echo "FAIL $spec (complete but checkpoint archive failed; see $RUN_DIR/checkpoint_archive.log)" >>"$status"
        continue
      fi
    fi
    echo "SKIP $spec (finished: $completion_file)" >>"$status"
    continue
  fi

  # Backward compatibility for pre-sentinel runs.  A generic `val/` match is not
  # evidence of completion because Hydra prints `monitor: val/f1` before training.
  # Require a terminal trainer record (validation smoke) or rendered test-table row
  # (main run), together with the checkpoint inventory written only after training.
  # A recorded failure OUTRANKS the fallback.  Both artifacts it keys on outlive a
  # crash after `trainer.test()`, so log text alone cannot tell "finished before
  # sentinels existed" from "died late": only a durable exit-status marker can.
  legacy_complete=0
  if [ -e "$RUN_DIR/FAILED" ]; then
    echo "FAIL $spec (previous attempt recorded rc=$(cat "$RUN_DIR/FAILED" 2>/dev/null): $RUN_DIR -- archive or remove it, then re-run)" >>"$status"
    continue
  fi
  if [ -s "$RUN_DIR/best_checkpoint_name.txt" ]; then
    if [ "$VALIDATION_ONLY" = 1 ] && \
       grep -q 'Trainer.fit.*stopped:.*max_epochs=' "$RUN_DIR/driver.log" 2>/dev/null; then
      legacy_complete=1
    elif [ "$VALIDATION_ONLY" = 0 ] && \
         grep -qE '│[[:space:]]*test/f1[[:space:]]*│' "$RUN_DIR/driver.log" 2>/dev/null; then
      legacy_complete=1
    fi
  fi
  if [ "$legacy_complete" = 1 ]; then
    if [ -n "$CHECKPOINT_ARCHIVE_ROOT" ] && ! grep -q '^OK checkpoints=' "$RUN_DIR/checkpoint_archive.log" 2>/dev/null; then
      archive_dest="$CHECKPOINT_ARCHIVE_ROOT/$backbone/$dataset/$arm_dir/seed_$seed"
      if ! bash scripts/archive_completed_checkpoints.sh \
          --source "$RUN_DIR" --destination "$archive_dest" --move \
          >"$RUN_DIR/checkpoint_archive.log" 2>&1; then
        echo "FAIL $spec (test exists but checkpoint archive failed; see $RUN_DIR/checkpoint_archive.log)" >>"$status"
        continue
      fi
    fi
    echo "SKIP $spec (finished before sentinel contract: $RUN_DIR)" >>"$status"
    continue
  fi
  if [ -e "$RUN_DIR/checkpoints" ]; then
    echo "FAIL $spec (PARTIAL run dir, no completion record: $RUN_DIR -- archive or remove it, then re-run)" >>"$status"
    continue
  fi
  mkdir -p "$RUN_DIR"

  (
    # trainer=difflr_benchmark_nolog + logger=none + the searchpath is a THREE-part
    # requirement. ieforge's `logger: null` default resolves to logger=True ==
    # CSVLogger, which aborts at a validation-epoch end with `dict contains fields
    # not in fieldnames: 'train/loss_epoch'` and leaves NO checkpoint. It is
    # deterministic given metric-key timing, so a packed node loses ALL its cells.
    # Short smokes never reach the triggering timing.
    if CUDA_VISIBLE_DEVICES="$gpu" uv run ieforge train \
        experiment="$experiment" \
        model="$model" \
        model.knowledge_control="$control" \
        trainer=difflr_benchmark_nolog \
        logger=none \
        trainer.callbacks.3.knowledge_lr_mult="$LR_MULT" \
        trainer.max_epochs="$EPOCHS" \
        seed="$seed" \
        knowledge.pipeline.query_builder.predictions_root="$PRED_DIR" \
        hydra.searchpath='[pkg://knowhyper.configs]' \
        paths.output_dir="$RUN_DIR" \
        hydra.run.dir="$RUN_DIR" \
        "${extra[@]}" >"$RUN_DIR/driver.log" 2>&1; then
      cell_rc=0
    else
      cell_rc=$?
      # POSITIVE record that this cell died.  The legacy fallback above infers
      # completion from `best_checkpoint_name.txt` + a rendered `test/f1` row, and
      # BOTH survive a crash that happens after `trainer.test()` -- so without a
      # durable failure marker a walltime-killed cell is skipped as finished on the
      # retry, which is exactly the path array casualties take.
      echo "$cell_rc" >"$RUN_DIR/FAILED"
      # Circuit breaker: a quota wall fails every remaining cell the same way, so
      # stop dispatching instead of reporting `finished` after losing them all.
      if grep -qi "Disk quota exceeded\|No space left on device" "$RUN_DIR/driver.log" 2>/dev/null; then
        echo "QUOTA" >"$OUTPUT_ROOT/.circuit_breaker.node${INDEX}"
      fi
    fi
    # Inventory the checkpoint filename IN-JOB: it is the only durable record of
    # best epoch + best val f1, and run checkpoints get cleaned later.
    ls -1 "$RUN_DIR/checkpoints" 2>/dev/null >"$RUN_DIR/best_checkpoint_name.txt"
    if [ "$cell_rc" -eq 0 ] && [ -n "$CHECKPOINT_ARCHIVE_ROOT" ]; then
      archive_dest="$CHECKPOINT_ARCHIVE_ROOT/$backbone/$dataset/$arm_dir/seed_$seed"
      if ! bash scripts/archive_completed_checkpoints.sh \
          --source "$RUN_DIR" --destination "$archive_dest" --move \
          >"$RUN_DIR/checkpoint_archive.log" 2>&1; then
        cell_rc=1
      fi
    fi
    if [ "$cell_rc" -eq 0 ]; then
      completion_tmp="$RUN_DIR/.COMPLETED.$$"
      {
        echo "validation_only=$VALIDATION_ONLY"
        echo "epochs=$EPOCHS"
        echo "abci_run_id=${ABCI_RUN_ID:-local}"
      } >"$completion_tmp"
      mv "$completion_tmp" "$completion_file"
      echo "OK   $spec" >>"$status"
    else
      echo "FAIL $spec" >>"$status"
    fi
    # EXIT WITH THE TRAINING STATUS, not the inventory's. `ls` succeeds whenever
    # Lightning already created checkpoints/ -- i.e. after every partial run -- so
    # letting it be the subshell's last command made `wait` report a crashed cell
    # as success and the whole node exit 0 while losing cells.
    exit "$cell_rc"
  ) &
  pids+=("$!")
  labels+=("$spec (gpu $gpu)")
  dispatched=$(( dispatched + 1 ))
  gpu=$(( (gpu + 1) % 8 ))
  unset extra

  # Circuit breaker, checked BEFORE dispatching the next cell. Two triggers:
  #
  #  1. any breaker file under $OUTPUT_ROOT -- this node's own (written by a cell
  #     that already died on quota) or a SIBLING node's. Cells run concurrently and
  #     take ~20 min, so this node's own breaker almost never exists yet when the
  #     loop is still dispatching; the sibling signal is what actually stops a
  #     campaign, since every node shares the one Lustre quota.
  #  2. free space measured right now. A disk filling during dispatch is the case
  #     no breaker file covers yet, and it is the case that loses every remaining
  #     cell the same way.
  if compgen -G "$OUTPUT_ROOT/.circuit_breaker.*" >/dev/null 2>&1; then
    echo "FATAL circuit breaker present (disk); stopping dispatch" >&2
    breaker_tripped=1
    break
  fi
  now_free_kb=$(df -P "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')
  if [ "${now_free_kb:-0}" -lt "$STOP_FREE_KB" ]; then
    echo "FATAL only ${now_free_kb} KiB free (need ${STOP_FREE_KB} KiB); stopping dispatch" >&2
    echo "SPACE" >"$OUTPUT_ROOT/.circuit_breaker.node${INDEX}"
    breaker_tripped=1
    break
  fi
done
set +f

rc=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "cell failed: ${labels[$i]}" >&2
    rc=1
  fi
done

# A cell the loop never dispatched is not a pass. Record it so the status file
# accounts for every spec on the node line, and fail the node.
#
# `grep -c` on a missing file prints nothing and returns non-zero, and its output
# carries a newline, so feeding it straight into $(( )) dies under `set -u`. Count
# with a helper that always echoes one bare integer.
count_lines() { grep -c "$1" "$status" 2>/dev/null | tr -dc '0-9' || true; }
skipped_lines=$(count_lines '^SKIP'); skipped_lines=${skipped_lines:-0}
failed_lines=$(count_lines '^FAIL'); failed_lines=${failed_lines:-0}
ok_lines=$(count_lines '^OK'); ok_lines=${ok_lines:-0}
# Account by STATUS LINES, not by `dispatched`. Every cell ends up with exactly one
# of OK / SKIP / FAIL: pre-flight rejections write FAIL and `continue`, and each
# dispatched cell's subshell writes OK or FAIL. Summing `dispatched + failed` instead
# double-counts a dispatched-then-failed cell, and that inflation can push `accounted`
# up to `requested` while cells the breaker never reached go unreported.
accounted=$(( ok_lines + skipped_lines + failed_lines ))
if [ "$accounted" -lt "$requested" ]; then
  echo "UNDISPATCHED $(( requested - accounted )) cells (breaker or early break)" >>"$status"
  rc=1
fi
# Any FAIL line fails the node, including a cell rejected BEFORE dispatch (missing
# proposals, no encoder, malformed spec, partial run dir). Those never enter `pids`,
# so `wait` cannot see them -- and a preflight rejection that leaves the node at
# exit 0 reads as "nothing to do" rather than "this cell did not run".
if [ "$failed_lines" -ne 0 ]; then
  rc=1
fi

echo "=== node $INDEX status ==="
sort "$status"
echo "requested: $requested  dispatched: $dispatched  OK: $ok_lines  FAIL: $failed_lines  SKIP: $skipped_lines  accounted: $accounted"
if [ "$breaker_tripped" -ne 0 ] || compgen -G "$OUTPUT_ROOT/.circuit_breaker.*" >/dev/null 2>&1; then
  echo "CIRCUIT BREAKER TRIPPED: disk exhausted, dispatch stopped early" >&2
  rc=1
fi
exit "$rc"
