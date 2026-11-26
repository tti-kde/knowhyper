#!/usr/bin/env bash
# Build every SubHG cache the RE matrix needs, packed onto ONE rt_HF node.
#
# An rt_HF node is 192 CPU / 8 GPU owned outright. Precompute is CPU-bound
# (`augment + build_incidence` is ~20 ms/graph, single-threaded per worker), so
# the work parallelizes across (experiment, split) units that write to different
# directories -- no shared-writer hazard.
#
# Six experiments, one per cache fingerprint (verified with
# scripts/resolve_cache_fingerprint.py, which self-checks its own recipe):
#
#   khp_re_xattn             8bf72b74a1435a17  chemprot fusion, non-strict
#   re_serialized_subhg      731c244b79f7c5d2  chemprot text + serialized
#   khp_ddi_xattn            45caf65e8a07ea90  ddi fusion, non-strict
#   ddi_serialized_subhg     c3e22a6c67339292  ddi text + serialized
#   khp_biored_xattn         3a484c25253cf019  biored id-only (text/serialized too)
#   khp_biored_xattn_surface cb831d8592bc38aa  biored surface fallback
#
# The text and serialized arms do NOT share the fusion caches on ChemProt/DDI --
# the serializer participates in cache identity. Assuming they did would have
# left two arms silently reading a cache built for a different serialization.
#
# Usage (inside an abci submit command block):
#   EXPERIMENTS="khp_re_xattn re_serialized_subhg ..." bash scripts/run_re_matrix_precompute_node.sh
set -uo pipefail

# ABCI defaults. Set HERE, not passed via `abci submit --`: anything after `--`
# containing a shell operator is treated as a single filename (that is an
# Exit_status=127), and the runner's own `uv sync` runs BEFORE this script, so an
# export in the submit line would land too late anyway.
if [ -n "${PBS_JOBID:-}" ]; then
  R="$HOME"
  export KNOWHYPER_HOME="${KNOWHYPER_HOME:-$R/.knowhyper}"
  export HF_HOME="${HF_HOME:-$R/.cache/huggingface}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

: "${KNOWHYPER_HOME:?KNOWHYPER_HOME must be exported}"

# All six by default; override to resume a subset after a partial node.
EXPERIMENTS="${EXPERIMENTS:-khp_re_xattn re_serialized_subhg khp_ddi_xattn ddi_serialized_subhg khp_biored_xattn khp_biored_xattn_surface}"
read -r -a experiments <<< "$EXPERIMENTS"

# 192 CPU / (6 experiments x 3 splits) = 10 per unit, leaving headroom for the
# parent processes and page cache. Sized from the CONCURRENT total, not from a
# single-unit probe: a unit measured alone is optimistic, and when walltime kills
# land they hit the slowest arm -- the one most worth measuring.
WORKERS_PER_SPLIT="${KNOWHYPER_PRECOMPUTE_WORKERS_PER_SPLIT:-10}"

export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

ART_ROOT="${PRECOMPUTE_ARTIFACT_ROOT:-$PWD/artifacts/re_matrix_precompute}"
mkdir -p "$ART_ROOT"

echo "=== preflight ==="
echo "node CPUs: $(nproc)"
echo "KNOWHYPER_HOME=$KNOWHYPER_HOME"
echo "workers/split=$WORKERS_PER_SPLIT  experiments=${#experiments[@]}"
df -h "$KNOWHYPER_HOME" 2>/dev/null | tail -1
# Quota preflight: a fail-tolerant loop that runs out of disk reports "finished"
# after losing every unit, which is worse than dying early.
if command -v lfs >/dev/null 2>&1; then
  lfs quota -h -u "$(whoami)" /home 2>/dev/null | tail -2
fi

pids=(); names=()
for exp in "${experiments[@]}"; do
  (
    export KNOWHYPER_PRECOMPUTE_WORKERS_PER_SPLIT="$WORKERS_PER_SPLIT"
    export PRECOMPUTE_ARTIFACT_ROOT="$ART_ROOT/$exp"
    bash scripts/run_subhg_precompute_all_cpu.sh "$exp"
  ) >"$ART_ROOT/${exp}.log" 2>&1 &
  pids+=("$!"); names+=("$exp")
  echo "dispatched $exp (pid $!)"
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "OK   ${names[$i]}"
  else
    echo "FAIL ${names[$i]} -- see $ART_ROOT/${names[$i]}.log"
    failed=1
  fi
done

echo "=== per-cache result ==="
# Report what is actually on disk. "The job exited 0" and "the cache is usable"
# are different claims: a complete cache full of EMPTY graphs passes a file
# count and then aborts training in validation_step (BioRED, 80 cells, 07-27).
for exp in "${experiments[@]}"; do
  s="$ART_ROOT/$exp/summary.txt"
  [ -f "$s" ] && grep -h 'precompute complete:' "$s" | sed "s/^/  $exp: /" || echo "  $exp: no summary"
done

echo "=== cache dirs now present ==="
for d in "$KNOWHYPER_HOME"/cache/subhg/v2/*/; do
  fp=$(basename "$d")
  for ds in "$d"*/; do
    [ -d "$ds" ] || continue
    line="  $fp $(basename "$ds"):"
    for sp in "$ds"*/; do
      [ -d "$sp" ] || continue
      line="$line $(basename "$sp")=$(ls "$sp" 2>/dev/null | wc -l)"
    done
    echo "$line"
  done
done

exit "$failed"
