#!/usr/bin/env bash
set -uo pipefail

# One rt_HF node's share of the #218 §0.3 SubHG precompute: ONE backbone x dataset.
#
# This is a thin `abci sweep` adapter, NOT a new precompute implementation. The work
# is done by `scripts/run_ner_strict_precompute_node.sh`, which already carries every
# lesson this stage needs and is fully parameterised by EXPERIMENT / PREDICTIONS_ROOT:
#   - 50 shards/split x 3 splits = 150 procs, measured ~7.3 instances/s (job 2075365)
#   - CUDA_VISIBLE_DEVICES="" -- 150 shards each loading SapBERT exhausted VRAM and
#     left a PARTIAL cache from a job that had passed every preflight
#   - pin preflight (core 0.12.13), strict-linking preflight, uniform-scoring refusal
#   - offline dataset instantiation before dispatching 150 shards
#   - per-fingerprint file counting with a fail-if-zero gate
# Do not re-implement any of that here.
#
# What this adds is only the #218 mapping: `--backbone`/`--dataset` select the span
# experiment and the backbone-scoped predictions root. edge and EV share one
# retrieval fingerprint per dataset (EV augmentation happens after raw retrieval), so
# precompute runs ONCE per backbone x dataset against the edge experiment -- not once
# per condition.
#
# CELLS format: `backbone:dataset`, one cell per node line.

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
CELL=$(sed -n "${INDEX}p" "$NODE_LINES" | tr -d '[:space:]')
[ -n "$CELL" ] || { echo "FATAL empty node line at index $INDEX" >&2; exit 1; }

PH="$HOME"
export UV_OFFLINE=1
export KNOWHYPER_HOME="${KNOWHYPER_HOME:-$PH/.knowhyper}"
export HF_HOME="${HF_HOME:-$PH/.cache/huggingface}"
[ -d "$HF_HOME" ] || { echo "FATAL HF_HOME does not exist: $HF_HOME" >&2; exit 1; }

IFS=: read -r backbone dataset extra_field <<<"$CELL"
if [ -z "${dataset:-}" ] || [ -n "${extra_field:-}" ]; then
  echo "FATAL malformed cell '$CELL'; want backbone:dataset" >&2; exit 1
fi
case "$backbone" in
  pubmedbert|biolinkbert) : ;;
  *) echo "FATAL unknown backbone $backbone" >&2; exit 1 ;;
esac
STRICT_REQUIRED=1
case "$dataset" in
  bc5cdr_chem) EXPERIMENT=khp_span_ner_edge ;;
  bc2gm) EXPERIMENT=khp_span_ner_edge_bc2gm ;;
  bc5cdr_disease) EXPERIMENT=khp_span_ner_edge_bc5cdr_disease ;;
  # The available JNLPBA loader collapses its five source labels to `entity`.
  # Its preregistered profile therefore links without a fabricated PrimeHKG type.
  # The delegated precompute driver is strict by default, so make this one
  # deliberate exception explicit rather than weakening every dataset.
  jnlpba)
    EXPERIMENT=khp_span_ner_edge_jnlpba
    STRICT_REQUIRED=0
    ;;
  ncbi_disease) EXPERIMENT=khp_span_ner_edge_ncbi_disease ;;
  *) echo "FATAL unknown dataset $dataset" >&2; exit 1 ;;
esac

PREDICTIONS_ROOT="$KNOWHYPER_HOME/artifacts/predictions/span_ner/$backbone/$dataset"
for split in train validation test; do
  [ -s "$PREDICTIONS_ROOT/$split.jsonl" ] || {
    echo "FATAL missing prediction file: $PREDICTIONS_ROOT/$split.jsonl" >&2
    echo "run the §0.2 proposal sweep and finalize_span_ner_proposals.py first" >&2
    exit 1
  }
done

echo "=== span precompute node $INDEX/$total ==="
echo "cell        : $CELL"
echo "experiment  : $EXPERIMENT"
echo "predictions : $PREDICTIONS_ROOT"

# Delegate. WORKERS_PER_SPLIT stays at the proven default (50/split = 150 procs);
# raise it only with a `cpupercent` measurement, per that script's own note -- the
# work is NFS-latency bound on small JSON writes, not CPU bound.
export EXPERIMENT PREDICTIONS_ROOT STRICT_REQUIRED
# The ABCI runner does not pull source/artifacts from array jobs. Keep shard logs
# and the summary in central state beside the cache so the precompute provenance
# survives the submitting worktree and is visible to every later stage.
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$KNOWHYPER_HOME/analysis/ner/span_precompute_extension/$backbone/$dataset}"
exec uv run bash scripts/run_ner_strict_precompute_node.sh
