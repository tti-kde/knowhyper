#!/usr/bin/env bash
# Drive the four DDI LLM-baseline cells on one ABCI node (#314, Agent B).
#
# Sequential on GPU 0 of an rt_HF node: validation noKG, validation KG, test
# noKG, test KG.  Validation runs first and is a plumbing/parser check only --
# it is scored, reported, and never used to choose anything.
#
# The two splits read DIFFERENT SubHG caches, and that is deliberate and
# recorded:
#   * test      -> 45caf65e8a07ea90, the exact cache the recorded DDI fusion
#                  arms read.  This is the arm that gets cited.
#   * validation-> 2a5edac8b67e7449, the only DDI validation cache on disk
#                  (45caf65e8a07ea90 has a test split only).  Its fingerprint
#                  differs (0.13.2 contract, #272) but NOT its content: on DDI
#                  test, cache_equivalence_2026_09_17.txt measured identical
#                  hyperedge AND seed-node sets on 5716/5716 instances
#                  (1.000000).  Even so nothing is claimed from validation --
#                  it exists to check the parser and the evidence plumbing.
#
# Every cell is fail-fast: a failure stops the script so a half-written cell is
# never mistaken for a completed one.
set -euo pipefail

: "${OUTPUT_ROOT:?OUTPUT_ROOT missing}"
: "${KNOWHYPER_HOME:?KNOWHYPER_HOME missing}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

TEST_CACHE="$KNOWHYPER_HOME/cache/subhg/v2/45caf65e8a07ea90/ddi/test"
VAL_CACHE="$KNOWHYPER_HOME/cache/subhg/v2/2a5edac8b67e7449/ddi/validation"

RUNNER="scripts/llm_baseline/llm_ddi_baseline.py"

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "KNOWHYPER_HOME=$KNOWHYPER_HOME"
echo "HF_HOME=${HF_HOME:-unset}"
ls -d "$TEST_CACHE" >/dev/null || { echo "missing test cache $TEST_CACHE" >&2; exit 2; }

run_cell() {
  local arm="$1" split="$2" condition="$3" batch="$4" cache="$5"
  echo "=== cell arm=$arm split=$split condition=$condition bs=$batch ==="
  local extra=()
  if [ "$arm" = "KG" ]; then
    extra=(--cache-split-dir "$cache")
  fi
  uv run python "$RUNNER" \
    --arm "$arm" --split "$split" \
    --batch-size "$batch" \
    --output-root "$OUTPUT_ROOT" --condition "$condition" \
    "${extra[@]}"
  echo "=== done $condition ==="
}

# Validation first: parser + plumbing sanity, on the smaller split.
run_cell noKG validation qwen35_9b_noKG_val 16 ""
if [ -d "$VAL_CACHE" ]; then
  run_cell KG validation qwen35_9b_KG_val 8 "$VAL_CACHE"
else
  echo "WARNING: no DDI validation SubHG cache at $VAL_CACHE; skipping validation KG" >&2
fi

# Test: the arms that are reported.
run_cell noKG test qwen35_9b_noKG 16 ""
run_cell KG test qwen35_9b_KG 8 "$TEST_CACHE"

echo "=== all cells done ==="
find "$OUTPUT_ROOT" -name cell.json -print
