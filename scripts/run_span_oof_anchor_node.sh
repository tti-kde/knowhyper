#!/usr/bin/env bash
# #233: train bs=16 OUT-OF-FOLD anchors for the candidate-specific text route.
#
# WHY THIS EXISTS
# ---------------
# Phase 0.5 had to fit its probe inside validation, because the only bs=16 anchor available
# was a single FULL-TRAIN model. Its train scores are in-sample: anchor-decoded train F1
# 0.9933, leaving FOUR correctable positives on the whole split (the OOF-*proposal* column
# gives 1507 on the same data, which is the control proving this is in-sample-ness and not a
# pool bug). Consuming half of validation was acceptable for a screen, but a LEARNED method
# must not reuse that split.
#
# So this produces, per backbone x fold, a bs=16 anchor trained on 4/5 of train, whose
# held-out fifth can be scored out-of-sample. Five folds tile the whole train split.
#
# WHAT MUST MATCH, AND WHY EACH ONE IS SILENT WHEN WRONG
# -----------------------------------------------------
#   experiment=span_ner_base*  -> bs=16. `experiment=span_ner` is ieforge's config and the
#                                 proposal driver overrides it to 64; inheriting 64 here
#                                 would undertrain the anchor and STILL report a plausible
#                                 F1 (biolinkbert/chem 0.7214 vs 0.9247). Verified by the
#                                 compose gate below, not assumed.
#   NUM_FOLDS=5 / FOLD_SEED=42 -> identical to `run_span_ner_proposals_node.sh`, so fold k
#                                 here holds out exactly the instances fold k held out
#                                 there. A different seed silently produces a DIFFERENT
#                                 holdout, and the anchor would then be in-sample on part of
#                                 the lattice it is meant to score out-of-sample.
#   role=train                 -> the FIT side (4/5). `role=holdout` would train on the
#                                 fifth we need to score.
#   trainer=benchmark_nolog    -> `logger=none` is mandatory; ieforge's `logger: null`
#     + logger=none               resolves to logger=True == CSVLogger, which aborts at a
#                                 validation-epoch end and leaves NO checkpoint. This arm is
#                                 knowledge-free, so `benchmark_nolog`, NOT `difflr_*`.
#   patience=100               -> matches `ner_span_text_phase3`, the tree
#                                 `resolve_baseline_checkpoint` accepts. Without it the run
#                                 is protocol-incompatible with every recorded anchor.
#
# Usage:
#   KNOWHYPER_HOME=... ./scripts/run_span_oof_anchor_node.sh
#   BACKBONES="pubmedbert" FOLDS="0 1" ./scripts/run_span_oof_anchor_node.sh   # subset
set -euo pipefail

: "${KNOWHYPER_HOME:?KNOWHYPER_HOME must be set to an absolute path}"
case "$KNOWHYPER_HOME" in
  /*) ;;
  *) echo "FATAL KNOWHYPER_HOME must be absolute, got '$KNOWHYPER_HOME'" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KH_CONFIGS="$REPO_ROOT/src/knowhyper/configs"

DATASET="${DATASET:-bc5cdr_disease}"
BACKBONES="${BACKBONES:-pubmedbert biolinkbert}"
FOLDS="${FOLDS:-0 1 2 3 4}"
# Pinned to the proposal driver's values. Changing either silently changes the holdout.
NUM_FOLDS="${NUM_FOLDS:-5}"
FOLD_SEED="${FOLD_SEED:-42}"
# The anchor seed. #220's phase3 anchors are seeds 3-9; 3 is the one Phase 0.5 used.
SEED="${SEED:-3}"
EPOCHS="${EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPUS="${GPUS:-0 1 2 3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$KNOWHYPER_HOME/runs/ner_span_oof_anchor}"

# The bs=16 anchor experiment per dataset. bc5cdr_chem is the unsuffixed root.
case "$DATASET" in
  bc5cdr_chem)    EXPERIMENT="span_ner_base" ;;
  bc2gm)          EXPERIMENT="span_ner_base_bc2gm" ;;
  bc5cdr_disease) EXPERIMENT="span_ner_base_bc5cdr_disease" ;;
  *) echo "FATAL unknown DATASET '$DATASET'" >&2; exit 2 ;;
esac

# Free-space preflight. Overridable because CI runners have ~6 GB and would otherwise fail
# 27 tests on disk alone (see CLAUDE.md).
MIN_FREE_KB="${MIN_FREE_KB:-10485760}"
free_kb="$(df -Pk "$KNOWHYPER_HOME" | awk 'NR==2 {print $4}')"
if [ "$free_kb" -lt "$MIN_FREE_KB" ]; then
  echo "FATAL only ${free_kb}KB free under $KNOWHYPER_HOME, need ${MIN_FREE_KB}KB" >&2
  exit 2
fi

status="$(mktemp)"
trap 'rm -f "$status"' EXIT

# ---- Compose gate: prove bs=16 and the fold plumbing BEFORE spending GPU hours. ----
echo "== compose gate =="
if ! uv run python - <<PY
import sys
from hydra import compose, initialize_config_dir
from ieforge import CONFIGS_DIR
with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_DIR)):
    cfg = compose(config_name="train", overrides=[
        "experiment=$EXPERIMENT",
        "model=pubmedbert_span_classifier",
        "dataset=blurb_ner_$DATASET",
        "trainer=benchmark_nolog", "logger=none",
        "train_dataset=blurb_ner_oof",
        "train_dataset.num_folds=$NUM_FOLDS",
        "train_dataset.fold=0",
        "train_dataset.fold_seed=$FOLD_SEED",
        "train_dataset.role=train",
        "hydra.searchpath=[file://$KH_CONFIGS]",
    ])
bad = []
if int(cfg.dataset.batch_size) != 16:
    bad.append(f"dataset.batch_size={cfg.dataset.batch_size}, expected 16 (bs=64 undertrains span NER)")
if int(cfg.train_dataset.batch_size) != 16:
    bad.append(f"train_dataset.batch_size={cfg.train_dataset.batch_size}, expected 16")
if int(cfg.train_dataset.num_folds) != $NUM_FOLDS:
    bad.append(f"num_folds={cfg.train_dataset.num_folds}")
if int(cfg.train_dataset.fold_seed) != $FOLD_SEED:
    bad.append(f"fold_seed={cfg.train_dataset.fold_seed}, must match the proposal driver")
if str(cfg.train_dataset.role) != "train":
    bad.append(f"role={cfg.train_dataset.role}, expected train (the FIT side)")
if int(cfg.task.max_span_width) != 12 or str(cfg.task.decoder) != "max_total_score":
    bad.append(f"task pins drifted: width={cfg.task.max_span_width} decoder={cfg.task.decoder}")
if bad:
    for line in bad:
        print("COMPOSE FAIL:", line)
    sys.exit(1)
print(f"compose OK: bs={cfg.dataset.batch_size} folds={cfg.train_dataset.num_folds} "
      f"fold_seed={cfg.train_dataset.fold_seed} role={cfg.train_dataset.role} "
      f"width={cfg.task.max_span_width} decoder={cfg.task.decoder}")
PY
then
  echo "FATAL compose gate failed; not dispatching" >&2
  exit 2
fi

gpu_list=($GPUS)
gpu_i=0
declare -a pids=()

for backbone in $BACKBONES; do
  for fold in $FOLDS; do
    RUN_DIR="$OUTPUT_ROOT/$backbone/$DATASET/oof_fold$fold/seed_$SEED"
    # Completion is judged by a CHECKPOINT existing, not by the directory existing: a
    # killed run leaves the directory behind and would otherwise be skipped forever.
    if compgen -G "$RUN_DIR/checkpoints/*.ckpt" >/dev/null; then
      echo "SKIP $backbone/fold$fold (checkpoint present)" >>"$status"
      continue
    fi
    mkdir -p "$RUN_DIR"
    gpu="${gpu_list[$gpu_i]}"
    gpu_i=$(( (gpu_i + 1) % ${#gpu_list[@]} ))

    (
      if CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 uv run ieforge train \
          experiment="$EXPERIMENT" \
          model="${backbone}_span_classifier" \
          dataset="blurb_ner_$DATASET" \
          trainer=benchmark_nolog \
          logger=none \
          trainer.max_epochs="$EPOCHS" \
          trainer.callbacks.0.save_top_k=1 \
          trainer.callbacks.0.save_last=false \
          trainer.callbacks.1.patience=100 \
          dataset.num_workers="$NUM_WORKERS" \
          train_dataset=blurb_ner_oof \
          train_dataset.num_folds="$NUM_FOLDS" \
          train_dataset.fold="$fold" \
          train_dataset.fold_seed="$FOLD_SEED" \
          train_dataset.role=train \
          seed="$SEED" \
          paths.output_dir="$RUN_DIR" \
          hydra.run.dir="$RUN_DIR" \
          "hydra.searchpath=[file://$KH_CONFIGS]" >"$RUN_DIR/driver.log" 2>&1; then
        if compgen -G "$RUN_DIR/checkpoints/*.ckpt" >/dev/null; then
          echo "OK $backbone/fold$fold" >>"$status"
        else
          # Exit 0 with no checkpoint is the CSVLogger failure mode. Do not call it OK.
          echo "FAIL $backbone/fold$fold (exit 0 but no checkpoint)" >>"$status"
        fi
      else
        echo "FAIL $backbone/fold$fold (training exited nonzero)" >>"$status"
      fi
    ) &
    pids+=($!)
    # One cell per GPU at a time: 1 GPU = 1 training run.
    if [ "${#pids[@]}" -ge "${#gpu_list[@]}" ]; then
      # `wait` on an EMPTY array succeeds under `set -u` in some shells and would silently
      # report success after losing every cell -- guard the expansion.
      if [ "${#pids[@]}" -gt 0 ]; then
        wait "${pids[@]}" || true
      fi
      pids=()
    fi
  done
done
if [ "${#pids[@]}" -gt 0 ]; then
  wait "${pids[@]}" || true
fi

echo "== summary =="
cat "$status"
count() { grep -c "$1" "$status" 2>/dev/null || true; }
ok=$(count '^OK'); failed=$(count '^FAIL'); skipped=$(count '^SKIP')
ok=${ok:-0}; failed=${failed:-0}; skipped=${skipped:-0}
echo "ok=$ok failed=$failed skipped=$skipped"
if [ "$failed" -gt 0 ]; then
  echo "FATAL $failed cell(s) failed" >&2
  exit 1
fi
