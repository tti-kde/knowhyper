#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 EXPERIMENT" >&2
  exit 2
fi

experiment="$1"
workers_per_split="${KNOWHYPER_PRECOMPUTE_WORKERS_PER_SPLIT:-8}"
read -r -a splits <<< "${KNOWHYPER_PRECOMPUTE_SPLITS:-train validation test}"
artifact_root="${PRECOMPUTE_ARTIFACT_ROOT:-artifacts/subhg_precompute/${experiment}}"

if (( workers_per_split < 1 )); then
  echo "KNOWHYPER_PRECOMPUTE_WORKERS_PER_SPLIT must be >= 1" >&2
  exit 2
fi
if (( ${#splits[@]} == 0 )); then
  echo "KNOWHYPER_PRECOMPUTE_SPLITS must contain at least one split" >&2
  exit 2
fi

export KNOWHYPER_HOME="${KNOWHYPER_HOME:-${HOME}/.knowhyper}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$artifact_root/logs"

pids=()
for split in "${splits[@]}"; do
  (
    export KNOWHYPER_PRECOMPUTE_WORKERS="$workers_per_split"
    # KNOWHYPER_PRECOMPUTE_EXTRA carries extra Hydra overrides, space separated.
    # Needed for sweeps over knobs that participate in the CACHE FINGERPRINT (top_k,
    # khop_budget, semantics_weight): each value is a different cache, so it must be
    # set at precompute time -- passing it only at training time would read the wrong
    # cache and silently fall back to live retrieval on every miss.
    read -r -a _extra <<< "${KNOWHYPER_PRECOMPUTE_EXTRA:-}"
    uv run python -m knowhyper.scripts.precompute_subhg_cache \
      experiment="$experiment" \
      ++dataset.split="$split" \
      "${_extra[@]}"
  ) >"$artifact_root/logs/${split}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

{
  printf 'experiment=%s\n' "$experiment"
  printf 'extra_overrides=%s\n' "${KNOWHYPER_PRECOMPUTE_EXTRA:-}"
  printf 'workers_per_split=%s\n' "$workers_per_split"
  printf 'splits=%s\n' "${splits[*]}"
  printf 'knowhyper_home=%s\n' "$KNOWHYPER_HOME"
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
  for split in "${splits[@]}"; do
    grep 'precompute complete:' "$artifact_root/logs/${split}.log" | tail -n 1 || true
  done
} >"$artifact_root/summary.txt"

if (( failed )); then
  echo "one or more split precompute processes failed" >&2
  exit 1
fi
