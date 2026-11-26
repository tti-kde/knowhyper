#!/usr/bin/env bash
set -uo pipefail

# One rt_HF node's share of #218 §0.2: backbone-matched candidate proposals.
#
# Two stages per cell, both knowledge-FREE (no PrimeHKG, no retriever, no SubHG):
#   1. train a span model  -- OOF fold (role=train) or the full-train model
#   2. export the candidate lattice from that checkpoint via span_ner_predict.py
#
# The OOF split is the point of the whole stage: an OOF model FITS on `role=train`
# and PREDICTS its own holdout part, so train-split candidates are honest
# out-of-fold. Fitting on the holdout side instead would leak, produce better-looking
# candidates, and no metric downstream would reveal it.
#
# The #202 checkpoints are NOT reusable here: they were produced by ieforge 0.1.9 and
# `train_dataset` was not fold-partitioned for the span arm. Candidate JSONL is also
# backbone-specific -- a PubMedBERT lattice fed to the BioLinkBERT arm compares
# architectures against different proposal distributions, which #218 forbids.
#
# CELLS format: space-separated `backbone:dataset:fold`, where fold is 0..N-1 or the
# literal `full`. Emit with `plan_span_ner_knowledge.py --proposals --emit-cells`.

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
export UV_OFFLINE=1
export KNOWHYPER_HOME="${KNOWHYPER_HOME:-$PH/.knowhyper}"
export HF_HOME="${HF_HOME:-$PH/.cache/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d "$HF_HOME" ] || { echo "FATAL HF_HOME does not exist: $HF_HOME" >&2; exit 1; }
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

EPOCHS="${EPOCHS:-10}"
NUM_FOLDS="${NUM_FOLDS:-5}"
# Disk headroom, in KiB, overridable so the driver's own logic can be exercised on a
# small runner. Production defaults are 10 GiB to start and 5 GiB to keep dispatching:
# the ieforge default of 4 ckpts/cell (~1.75 GB) is what exhausts the 2 TB Lustre
# quota. CI runners have ~6 GB free, so a hardcoded 10 GiB made every executing driver
# test die at the preflight before reaching the behaviour under test -- passing locally
# and failing in CI purely on the host's free space.
MIN_FREE_KB="${MIN_FREE_KB:-10485760}"
STOP_FREE_KB="${STOP_FREE_KB:-5242880}"
FOLD_SEED="${FOLD_SEED:-42}"
# Proposal training seed is FIXED at 0 by #218 §0.2: the proposal distribution is
# part of the data identity, not a factor to vary.
PROPOSAL_SEED="${PROPOSAL_SEED:-0}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-16}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$KNOWHYPER_HOME/runs/ner/span_proposals}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-$KNOWHYPER_HOME/artifacts/predictions/span_ner}"
mkdir -p "$OUTPUT_ROOT" "$PREDICTIONS_ROOT"
status="$OUTPUT_ROOT/STATUS.node${INDEX}.$$"
: >"$status"

echo "=== span-proposals node $INDEX/$total ==="
echo "cells: $CELLS"
echo "predictions root: $PREDICTIONS_ROOT"
echo "folds: $NUM_FOLDS  fold_seed: $FOLD_SEED  seed: $PROPOSAL_SEED  top_k: $CANDIDATE_TOP_K"

if command -v lfs >/dev/null 2>&1; then
  lfs quota -h -u "$(whoami)" /home 2>/dev/null | tail -2 || true
fi
df -h "$OUTPUT_ROOT" | tail -1
free_kb=$(df -P "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')
if [ "${free_kb:-0}" -lt "$MIN_FREE_KB" ]; then
  echo "FATAL only ${free_kb} KiB free under $OUTPUT_ROOT (need ${MIN_FREE_KB} KiB); refusing to dispatch" >&2
  exit 1
fi

KH_CONFIGS=$(uv run python -c 'from knowhyper.ieforge._configs import CONFIGS_DIR; print(CONFIGS_DIR)' 2>/dev/null)
: "${KH_CONFIGS:?could not resolve knowhyper CONFIGS_DIR}"

# Assert the pins and every cell compose BEFORE spending GPU time. Catches upstream
# default drift and, critically, a wrong-directory submission: `abci submit` snapshots
# cwd, so submitting from the main repo ships a tree without this branch's configs and
# Hydra reports the experiment as unknown -- which reads like a config bug.
if ! uv run python scripts/analysis/verify_span_knowledge_compose.py --proposals; then
  echo "FATAL compose verification failed; refusing to dispatch" >&2
  exit 1
fi

# A breaker left by a sibling node (they share one Lustre quota) or by an earlier
# attempt means the disk is already gone. Dispatching into it loses every cell slowly
# and still reports `finished`.
if compgen -G "$OUTPUT_ROOT/.circuit_breaker.*" >/dev/null 2>&1; then
  echo "FATAL circuit breaker already present under $OUTPUT_ROOT; refusing to dispatch" >&2
  ls -1 "$OUTPUT_ROOT"/.circuit_breaker.* >&2
  exit 1
fi

declare -a pids=() labels=()
breaker_tripped=0
dispatched=0
gpu=0
# Word splitting on $CELLS is intended; pathname expansion is NOT.
set -f
requested=0
for _spec in $CELLS; do requested=$(( requested + 1 )); done

for spec in $CELLS; do
  IFS=: read -r backbone dataset fold extra_field <<<"$spec"
  if [ -z "${fold:-}" ]; then
    echo "FAIL $spec (malformed; want backbone:dataset:fold)" >>"$status"
    continue
  fi
  if [ -n "${extra_field:-}" ]; then
    echo "FAIL $spec (too many fields; want backbone:dataset:fold)" >>"$status"
    continue
  fi
  case "$backbone" in
    pubmedbert|biolinkbert) : ;;
    *) echo "FAIL $spec (unknown backbone $backbone)" >>"$status"; continue ;;
  esac
  case "$dataset" in
    bc5cdr_chem|bc2gm|bc5cdr_disease|jnlpba|ncbi_disease) : ;;
    *) echo "FAIL $spec (unknown dataset $dataset)" >>"$status"; continue ;;
  esac
  if [ "$fold" != full ]; then
    case "$fold" in
      ''|*[!0-9]*) echo "FAIL $spec (fold '$fold' is neither an integer nor 'full')" >>"$status"; continue ;;
    esac
    if [ "$fold" -ge "$NUM_FOLDS" ]; then
      echo "FAIL $spec (fold $fold out of range 0..$(( NUM_FOLDS - 1 )))" >>"$status"
      continue
    fi
  fi

  model="${backbone}_span_classifier"
  if [ "$fold" = full ]; then
    leaf=full
    splits="validation,test"
  else
    leaf="oof_fold${fold}"
    splits=train
  fi
  RUN_DIR="$OUTPUT_ROOT/$backbone/$dataset/$leaf"
  OUT_DIR="$PREDICTIONS_ROOT/$backbone/$dataset/$leaf"

  # Completion is judged by the EXPORTED ARTIFACT plus its manifest, not by the
  # checkpoint: the checkpoint is an intermediate, and run checkpoints get cleaned.
  complete=1
  for split in ${splits//,/ }; do
    [ -s "$OUT_DIR/$split.jsonl" ] || complete=0
  done
  [ -s "$OUT_DIR/predictions.manifest.json" ] || complete=0
  if [ "$complete" = 1 ]; then
    echo "SKIP $spec (finished: $OUT_DIR)" >>"$status"
    continue
  fi

  mkdir -p "$RUN_DIR"
  (
    declare -a fold_overrides=()
    if [ "$fold" != full ]; then
      # role=train is the FIT side. span_ner_predict.py re-derives the holdout from
      # the same (num_folds, fold, fold_seed), so the two stages cannot disagree.
      fold_overrides+=(
        "train_dataset=blurb_ner_oof"
        "train_dataset.num_folds=$NUM_FOLDS"
        "train_dataset.fold=$fold"
        "train_dataset.fold_seed=$FOLD_SEED"
        "train_dataset.role=train"
      )
    fi

    # `trainer=benchmark_nolog logger=none` is MANDATORY and the searchpath with it.
    # ieforge's `logger: null` resolves to logger=True == CSVLogger, which aborts at
    # a validation-epoch end and leaves NO checkpoint. This is the knowledge-free
    # first pass, so it takes `benchmark_nolog`, NOT `difflr_benchmark_nolog`:
    # there is no knowledge branch to give a 10x LR.
    if ! CUDA_VISIBLE_DEVICES="$gpu" uv run ieforge train \
        experiment=span_ner \
        dataset="blurb_ner_$dataset" \
        model="$model" \
        task.max_span_width=12 \
        trainer=benchmark_nolog \
        logger=none \
        trainer.max_epochs="$EPOCHS" \
        trainer.limit_test_batches=0 \
        trainer.callbacks.0.save_top_k=1 \
        trainer.callbacks.0.save_last=false \
        dataset.batch_size="$BATCH_SIZE" \
        dataset.num_workers="$NUM_WORKERS" \
        seed="$PROPOSAL_SEED" \
        "${fold_overrides[@]}" \
        output_dir="$RUN_DIR" \
        paths.output_dir="$RUN_DIR" \
        hydra.run.dir="$RUN_DIR" \
        "hydra.searchpath=[file://$KH_CONFIGS]" >"$RUN_DIR/train.log" 2>&1; then
      echo "FAIL $spec (train)" >>"$status"
      # Circuit breaker: a quota wall fails every remaining cell the same way, so
      # signal siblings instead of letting each one rediscover it. CLAUDE.md requires
      # this on any multi-cell job script, and `df` free space is NOT a substitute --
      # a Lustre QUOTA can be exhausted while the filesystem still reports space.
      if grep -qi "Disk quota exceeded\|No space left on device" "$RUN_DIR/train.log" 2>/dev/null; then
        echo "QUOTA" >"$OUTPUT_ROOT/.circuit_breaker.node${INDEX}"
      fi
      exit 1
    fi

    ckpt=$(find "$RUN_DIR/checkpoints" -type f -name 'epoch_*.ckpt' 2>/dev/null | sort | head -1)
    if [ -z "$ckpt" ]; then
      echo "FAIL $spec (no checkpoint under $RUN_DIR/checkpoints)" >>"$status"
      exit 1
    fi
    # Inventory it now: the filename is the only durable record of best epoch and
    # best val f1, and run checkpoints get cleaned later.
    ls -1 "$RUN_DIR/checkpoints" >"$RUN_DIR/best_checkpoint_name.txt" 2>/dev/null

    declare -a oof_args=()
    if [ "$fold" != full ]; then
      oof_args+=(
        "--oof-num-folds" "$NUM_FOLDS"
        "--oof-fold" "$fold"
        "--fold-seed" "$FOLD_SEED"
      )
    fi
    if ! uv run python scripts/span_ner_predict.py \
        --ckpt "$ckpt" \
        --dataset "blurb_ner_$dataset" \
        --model "$model" \
        --experiment span_ner \
        --searchpath "$KH_CONFIGS" \
        --splits "$splits" \
        --output "$OUT_DIR" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --candidate-top-k "$CANDIDATE_TOP_K" \
        "${oof_args[@]}" >>"$RUN_DIR/predict.log" 2>&1; then
      echo "FAIL $spec (predict)" >>"$status"
      if grep -qi "Disk quota exceeded\|No space left on device" "$RUN_DIR/predict.log" 2>/dev/null; then
        echo "QUOTA" >"$OUTPUT_ROOT/.circuit_breaker.node${INDEX}"
      fi
      exit 1
    fi
    echo "OK   $spec" >>"$status"
    exit 0
  ) &
  pids+=("$!")
  labels+=("$spec (gpu $gpu)")
  dispatched=$(( dispatched + 1 ))
  gpu=$(( (gpu + 1) % 8 ))

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

count_lines() { grep -c "$1" "$status" 2>/dev/null | tr -dc '0-9' || true; }
ok_lines=$(count_lines '^OK'); ok_lines=${ok_lines:-0}
skipped_lines=$(count_lines '^SKIP'); skipped_lines=${skipped_lines:-0}
failed_lines=$(count_lines '^FAIL'); failed_lines=${failed_lines:-0}
# Account by STATUS LINES, not by `dispatched`. Every cell ends up with exactly one
# of OK / SKIP / FAIL: pre-flight rejections write FAIL and `continue`, and each
# dispatched cell's subshell writes OK or FAIL. Summing `dispatched + failed` instead
# double-counts a dispatched-then-failed cell, and that inflation can push `accounted`
# up to `requested` while cells the breaker never reached go unreported.
accounted=$(( ok_lines + skipped_lines + failed_lines ))
if [ "$accounted" -lt "$requested" ]; then
  echo "UNDISPATCHED $(( requested - accounted )) cells" >>"$status"
  rc=1
fi
# Any FAIL fails the node, including cells rejected before dispatch: they never
# enter `pids`, so `wait` cannot see them, and exit 0 would read as "nothing to do".
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
