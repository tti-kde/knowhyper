#!/usr/bin/env bash
set -uo pipefail

# One rt_HF node's share of the #220 span Knowledge-as-text campaign (Table 6).
#
# NEW driver on purpose. #220 forbids adding branches to the old token matrix
# driver, and `run_span_ner_knowledge_node.sh` dispatches #218's condition x control
# axis (edge/ev/ev_warm x real/shuffled/zero). This one dispatches the three-arm text
# axis, and the two differ on the ONE thing that must never be guessed: the trainer.
#
# `abci sweep` passes parameters as `--KEY value` argv PAIRS with keys sorted --
# not env vars, not `--KEY=value` (abci_runner/param_exec.py:params_to_cli_args).
# Parse with a shift loop and guard with `:?` so a mis-parse dies in seconds
# instead of after ten epochs.
#
# CELLS format: space-separated `backbone:dataset:arm:seed`, e.g.
#   pubmedbert:bc2gm:text_real:0 biolinkbert:bc2gm:base:0
# Emit them with `plan_span_ner_text.py --emit-cells`.

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

[ -s "$NODE_LINES" ] || { echo "FATAL node-lines missing/empty: $NODE_LINES" >&2; exit 1; }
total=$(grep -c . "$NODE_LINES")
if [ "$INDEX" -lt 1 ] || [ "$INDEX" -gt "$total" ]; then
  echo "FATAL index $INDEX out of range 1..$total" >&2; exit 1
fi
CELLS=$(sed -n "${INDEX}p" "$NODE_LINES")
[ -n "$CELLS" ] || { echo "FATAL empty node line at index $INDEX" >&2; exit 1; }

PH="$HOME"
# `abci sweep` has NO --pass-env, so every var the subjob needs is exported HERE.
# Do NOT reference $KNOWHYPER_HOME in a sweep's `command:` block -- it is exported
# here, not there, so at wrapper time it is unset and paths collapse to `/runs`.
export UV_OFFLINE=1
export KNOWHYPER_HOME="${KNOWHYPER_HOME:-$PH/.knowhyper}"
export HF_HOME="${HF_HOME:-$PH/.cache/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d "$HF_HOME" ] || { echo "FATAL HF_HOME does not exist: $HF_HOME" >&2; exit 1; }
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

EPOCHS="${EPOCHS:-10}"
LR_MULT="${LR_MULT:-10}"
# EarlyStopping patience, callback index 1 in both trainers. The shipped 5 is
# actively harmful here: a span head on a sparse label space can sit at
# val/f1 = 0.000 for the first epochs (focal loss is minimised by predicting
# nothing), and patience=5 then ends the run and `save_top_k=1` keeps the
# epoch-0 checkpoint -- exit 0, `OK`, and a legitimate-looking 0.0000. Phase 2
# lost 4 of 36 text cells that way and truncated 13 of 36 at 5/9 or 6/9, while
# `base` reached 9/9 on 17 of 18. Set above max_epochs so the 10 epochs the
# protocol fixes actually run; #202 also notes span models are often still
# climbing at the last epoch. Applied to EVERY arm: a stopping rule that fires
# on one arm and not another is itself a between-arm difference.
PATIENCE="${PATIENCE:-100}"
# Disk headroom in KiB, overridable so the driver's own logic can be exercised on a
# small runner. CI runners have ~6 GB free, so a hardcoded 10 GiB makes every
# executing driver test die at the preflight before reaching the behaviour under
# test -- passing locally and failing in CI purely on the host's free space.
MIN_FREE_KB="${MIN_FREE_KB:-10485760}"
STOP_FREE_KB="${STOP_FREE_KB:-5242880}"
# Scoped to #220. Sharing #218's `runs/ner/span_knowledge` tree would interleave two
# campaigns' cells under one aggregator glob.
OUTPUT_ROOT="${OUTPUT_ROOT:-$KNOWHYPER_HOME/runs/ner_span_text}"
# #218 Phase 0.2 is the canonical producer; #220 reads it and never regenerates it.
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-$KNOWHYPER_HOME/artifacts/predictions/span_ner}"
VALIDATION_ONLY="${VALIDATION_ONLY:-0}"
case "$VALIDATION_ONLY" in
  0|1) : ;;
  *) echo "FATAL VALIDATION_ONLY must be 0 or 1, got '$VALIDATION_ONLY'" >&2; exit 2 ;;
esac
mkdir -p "$OUTPUT_ROOT"
status="$OUTPUT_ROOT/STATUS.node${INDEX}.$$"
: >"$status"

echo "=== span-text node $INDEX/$total ==="
echo "cells: $CELLS"
echo "predictions root: $PREDICTIONS_ROOT"

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

# Assert every #220 cell composes BEFORE spending GPU time. Also catches a
# wrong-directory submission: `abci submit` snapshots cwd, so submitting from the
# main repo ships a tree without this branch's configs and Hydra reports the
# experiment as unknown -- which reads like a config bug rather than a wrong cwd.
if ! uv run python scripts/analysis/verify_span_text_compose.py; then
  echo "FATAL compose verification failed; refusing to dispatch" >&2
  exit 1
fi

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
# Count every spec up front: counting inside the loop would stop at a `break`, so an
# early circuit-breaker exit would under-report the very cells it failed to dispatch.
requested=0
for _spec in $CELLS; do requested=$(( requested + 1 )); done

for spec in $CELLS; do
  IFS=: read -r backbone dataset arm seed extra_field <<<"$spec"
  if [ -z "${seed:-}" ]; then
    echo "FAIL $spec (malformed; want backbone:dataset:arm:seed)" >>"$status"
    continue
  fi
  if [ -n "${extra_field:-}" ]; then
    echo "FAIL $spec (too many fields; want backbone:dataset:arm:seed)" >>"$status"
    continue
  fi
  case "$seed" in
    ''|*[!0-9]*) echo "FAIL $spec (non-integer seed $seed)" >>"$status"; continue ;;
  esac
  case "$backbone" in
    pubmedbert|biolinkbert) : ;;
    *) echo "FAIL $spec (unknown backbone $backbone)" >>"$status"; continue ;;
  esac

  # Arm decides BOTH the experiment stem and the trainer. The trainer is the part
  # that must never be inferred at the call site: `base` is a plain span classifier
  # with no `model.inner.` prefix, so difflr's BACKBONE_PREFIX matches nothing and
  # the ENTIRE pretrained backbone would train at 10x -- silently, with a plausible
  # score. See tests/test_span_ner_text_baseline_gate.py.
  # EVERY arm takes `benchmark_nolog`. The text route has NO knowledge branch --
  # `injector.encoder: null`, `integrator: null`, and all three arms instantiate
  # the same plain HFSpanClassifier -- so difflr's `model.inner.` split matches
  # nothing and puts all 130.7M backbone params in the 10x group (2e-4 on a
  # pretrained BERT). Phase 2 ran the text arms that way and they collapsed to
  # the empty prediction on some seeds; `base` at 1x never did.
  case "$arm" in
    base)      stem=span_ner_base ;;
    text_real) stem=span_ner_retrieved_text ;;
    text_sham) stem=span_ner_text_sham ;;
    *) echo "FAIL $spec (unknown arm $arm)" >>"$status"; continue ;;
  esac
  trainer=benchmark_nolog
  case "$dataset" in
    bc5cdr_chem)    experiment="$stem" ;;
    bc2gm)          experiment="${stem}_bc2gm" ;;
    bc5cdr_disease) experiment="${stem}_bc5cdr_disease" ;;
    jnlpba)          experiment="${stem}_jnlpba" ;;
    ncbi_disease)   experiment="${stem}_ncbi_disease" ;;
    *) echo "FAIL $spec (unknown dataset $dataset)" >>"$status"; continue ;;
  esac

  model="${backbone}_span_classifier"

  # Every arm reads #218's proposals: the two routes must compare on ONE proposal
  # distribution. `base` needs them too -- it is a span classifier over the same
  # candidate lattice, not a from-scratch decode.
  PRED_DIR="$PREDICTIONS_ROOT/$backbone/$dataset"
  missing=""
  for split in train validation test; do
    [ -s "$PRED_DIR/$split.jsonl" ] || missing="$missing $split"
  done
  if [ -n "$missing" ]; then
    echo "FAIL $spec (missing proposals in $PRED_DIR:$missing)" >>"$status"
    continue
  fi

  # Only the text arms carry a knowledge pipeline; passing predictions_root to
  # `base` would fail composition, since that config has no `knowledge` block.
  # No `knowledge_lr_mult` override: `benchmark_nolog` carries only
  # ModelCheckpoint and EarlyStopping, so `trainer.callbacks.3` does not exist.
  # There is nothing to give a differential LR to either -- the text route has no
  # knowledge branch, only a second input segment.
  declare -a extra=()
  if [ "$arm" != base ]; then
    extra+=( "knowledge.pipeline.query_builder.predictions_root=$PRED_DIR" )
  fi
  if [ "$VALIDATION_ONLY" = 1 ]; then
    extra+=( "trainer.limit_test_batches=0" )
  fi

  RUN_DIR="$OUTPUT_ROOT/$backbone/$dataset/$arm/seed_$seed"
  completion_file="$RUN_DIR/COMPLETED"
  # A generic `grep val/ driver.log` is NOT a completion check: Hydra prints
  # `monitor: val/f1` before training starts.  Only the success branch below writes
  # this sentinel, and its mode must match the current campaign.
  if [ -s "$completion_file" ]; then
    if ! grep -qx "validation_only=$VALIDATION_ONLY" "$completion_file"; then
      echo "FAIL $spec (completion mode mismatch in $completion_file)" >>"$status"
      continue
    fi
    echo "SKIP $spec (finished: $completion_file)" >>"$status"
    continue
  fi

  # Backward-compatible recognition for runs completed before the sentinel was
  # introduced.  Match an emitted terminal record, never a config key.  Validation
  # smokes fix patience above max_epochs, so `Trainer.fit stopped: max_epochs` plus
  # a checkpoint inventory is their terminal contract.  Full runs additionally
  # require the rendered test table row.
  # A recorded failure OUTRANKS the fallback.  Both artifacts it keys on outlive a
  # crash after `trainer.test()`, so log text alone cannot tell "finished before
  # sentinels existed" from "died late": only a durable exit-status marker can.
  #
  # KNOWN RESIDUAL HOLE, deliberately not closed here.  The marker is written by the
  # cell subshell, so a SIGKILL (walltime) leaves none, and a real completed log ends
  # mid-`Testing` progress bar with no post-test line to anchor on -- so a
  # walltime-killed cell is still indistinguishable from a pre-sentinel completion by
  # log text alone.  The fallback exists ONLY for the 216 runs under
  # ner_span_text_phase{2,2b,3} that predate the sentinel (inventory=216, COMPLETED=0),
  # and those are frozen #231 baselines.  The real fix is to backfill sentinels over
  # those trees once and then delete this branch entirely, which is a state change to
  # completed campaign data and needs an explicit decision.  Until then: anything
  # NEWLY produced by this driver gets a sentinel on success and a FAILED marker on a
  # non-zero exit, so the fallback only ever adjudicates pre-existing trees.
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
    echo "SKIP $spec (finished before sentinel contract: $RUN_DIR)" >>"$status"
    continue
  fi
  if [ -e "$RUN_DIR/checkpoints" ]; then
    echo "FAIL $spec (PARTIAL run dir, no completion record: $RUN_DIR -- archive or remove it, then re-run)" >>"$status"
    continue
  fi
  mkdir -p "$RUN_DIR"

  (
    # `logger=none` + the searchpath is mandatory alongside the trainer. ieforge's
    # `logger: null` default resolves to logger=True == CSVLogger, which aborts at a
    # validation-epoch end with `dict contains fields not in fieldnames:
    # 'train/loss_epoch'` and leaves NO checkpoint. It is deterministic given
    # metric-key timing, so a packed node loses ALL its cells; short smokes never
    # reach the triggering timing.
    if CUDA_VISIBLE_DEVICES="$gpu" uv run ieforge train \
        experiment="$experiment" \
        model="$model" \
        trainer="$trainer" \
        logger=none \
        trainer.max_epochs="$EPOCHS" \
        trainer.callbacks.1.patience="$PATIENCE" \
        trainer.callbacks.0.save_top_k=1 \
        trainer.callbacks.0.save_last=false \
        seed="$seed" \
        hydra.searchpath='[pkg://knowhyper.configs]' \
        paths.output_dir="$RUN_DIR" \
        hydra.run.dir="$RUN_DIR" \
        "${extra[@]}" >"$RUN_DIR/driver.log" 2>&1; then
      cell_rc=0
      completion_tmp="$RUN_DIR/.COMPLETED.$$"
      {
        echo "validation_only=$VALIDATION_ONLY"
        echo "epochs=$EPOCHS"
        echo "abci_run_id=${ABCI_RUN_ID:-local}"
      } >"$completion_tmp"
      mv "$completion_tmp" "$completion_file"
      echo "OK   $spec" >>"$status"
    else
      cell_rc=$?
      echo "FAIL $spec" >>"$status"
      # POSITIVE record that this cell died.  The legacy fallback below infers
      # completion from `best_checkpoint_name.txt` + a rendered `test/f1` row, and
      # BOTH survive a crash that happens after `trainer.test()` -- so without a
      # durable failure marker a walltime-killed cell is skipped as finished on the
      # retry, which is exactly the path array casualties take.
      echo "$cell_rc" >"$RUN_DIR/FAILED"
      if grep -qi "Disk quota exceeded\|No space left on device" "$RUN_DIR/driver.log" 2>/dev/null; then
        echo "QUOTA" >"$OUTPUT_ROOT/.circuit_breaker.node${INDEX}"
      fi
    fi
    # Inventory the checkpoint filename IN-JOB: it is the only durable record of
    # best epoch + best val f1, and run checkpoints get cleaned later.
    ls -1 "$RUN_DIR/checkpoints" 2>/dev/null >"$RUN_DIR/best_checkpoint_name.txt"
    # EXIT WITH THE TRAINING STATUS, not the inventory's. `ls` succeeds whenever
    # Lightning already created checkpoints/, so letting it be the subshell's last
    # command reports a crashed cell as success.
    exit "$cell_rc"
  ) &
  pids+=("$!")
  labels+=("$spec (gpu $gpu)")
  dispatched=$(( dispatched + 1 ))
  gpu=$(( (gpu + 1) % 8 ))
  unset extra

  # Circuit breaker, checked BEFORE dispatching the next cell: a sibling node's
  # breaker (every node shares the one Lustre quota) or free space measured now.
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

# A cell the loop never dispatched is not a pass. `grep -c` on a missing file prints
# nothing and returns non-zero, and its output carries a newline, so feeding it
# straight into $(( )) dies under `set -u`.
count_lines() { grep -c "$1" "$status" 2>/dev/null | tr -dc '0-9' || true; }
skipped_lines=$(count_lines '^SKIP'); skipped_lines=${skipped_lines:-0}
failed_lines=$(count_lines '^FAIL'); failed_lines=${failed_lines:-0}
ok_lines=$(count_lines '^OK'); ok_lines=${ok_lines:-0}
# Account by STATUS LINES, not by `dispatched`: summing `dispatched + failed`
# double-counts a dispatched-then-failed cell, and that inflation can hide cells the
# breaker never reached.
accounted=$(( ok_lines + skipped_lines + failed_lines ))
if [ "$accounted" -lt "$requested" ]; then
  echo "UNDISPATCHED $(( requested - accounted )) cells (breaker or early break)" >>"$status"
  rc=1
fi
# Any FAIL line fails the node, including a cell rejected BEFORE dispatch: those
# never enter `pids`, so `wait` cannot see them, and a preflight rejection that
# leaves the node at exit 0 reads as "nothing to do".
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
