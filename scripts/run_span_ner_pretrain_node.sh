#!/usr/bin/env bash
set -uo pipefail

# One rt_HF node's share of #218 §0.4: cache-matched cross-modal MAE encoders.
#
# Six encoders total, one per backbone x dataset. The graph encoder itself is
# language-backbone independent, but the SubHG STREAM it pretrains on is not: the
# sub-hypergraphs were retrieved from that backbone's candidate lattice, so the
# pretrain root is split per backbone. Reusing one encoder across differently-linked
# caches is exactly what reproduced the earlier warm-start nulls.
#
# The encoder MUST be pretrained on the SAME prediction-scoped train cache the end
# task reads. That is why --cache-dir points at
#   <subhg>/<retrieval-fp>/<dataset.name>@pred-<train-sha>/train
# and not at a dataset-level directory: the `@pred-<sha>` leaf IS the matching.
#
# Fixed by #218 §0.4: objective cross_modal, entity view on, skip_modalities name,
# mask ratio 0.5, 10 epochs, lr 3e-4, grad clip 1.0, FEATURES 0.12.15.
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
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -d "$HF_HOME" ] || { echo "FATAL HF_HOME does not exist: $HF_HOME" >&2; exit 1; }
export TOKENIZERS_PARALLELISM=false

EPOCHS="${EPOCHS:-10}"
MASK_RATIO="${MASK_RATIO:-0.5}"
LR="${LR:-3e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SEED="${SEED:-0}"
SUBHG_ROOT="${SUBHG_ROOT:-$KNOWHYPER_HOME/cache/subhg/v2}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-$KNOWHYPER_HOME/artifacts/predictions/span_ner}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-$KNOWHYPER_HOME/artifacts/pretrain/span_ner}"
INCIDENCE_ROOT="${INCIDENCE_ROOT:-$KNOWHYPER_HOME/cache/incidence/v1/span_ner}"

IFS=: read -r backbone dataset extra_field <<<"$CELL"
if [ -z "${dataset:-}" ] || [ -n "${extra_field:-}" ]; then
  echo "FATAL malformed cell '$CELL'; want backbone:dataset" >&2; exit 1
fi
case "$backbone" in
  pubmedbert|biolinkbert) : ;;
  *) echo "FATAL unknown backbone $backbone" >&2; exit 1 ;;
esac
case "$dataset" in
  bc5cdr_chem) EXPERIMENT=khp_span_ner_edge ;;
  bc2gm) EXPERIMENT=khp_span_ner_edge_bc2gm ;;
  bc5cdr_disease) EXPERIMENT=khp_span_ner_edge_bc5cdr_disease ;;
  jnlpba) EXPERIMENT=khp_span_ner_edge_jnlpba ;;
  ncbi_disease) EXPERIMENT=khp_span_ner_edge_ncbi_disease ;;
  *) echo "FATAL unknown dataset $dataset" >&2; exit 1 ;;
esac

echo "=== span pretrain node $INDEX/$total ==="
echo "cell       : $CELL"
echo "experiment : $EXPERIMENT"

# Resolve the retrieval fingerprint from the CONFIG, never from a literal. A hash
# written in a comment goes stale the moment any profile key changes, and a wrong
# fingerprint names a directory that either does not exist or belongs to another arm.
FINGERPRINT=$(uv run python - "$EXPERIMENT" <<'PY'
import sys
sys.path.insert(0, "scripts")
from resolve_cache_fingerprint import resolve
print(resolve(sys.argv[1])[0])
PY
)
: "${FINGERPRINT:?could not resolve the retrieval fingerprint}"

# The train prediction sha is what scopes the cache leaf; recompute it the same way
# NERPredictionStore does (sha256 of the raw bytes, first 16 hex).
PRED_FILE="$PREDICTIONS_ROOT/$backbone/$dataset/train.jsonl"
[ -s "$PRED_FILE" ] || { echo "FATAL missing $PRED_FILE" >&2; exit 1; }
TRAIN_SHA=$(uv run python -c "
import hashlib,sys
print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:16])
" "$PRED_FILE")
: "${TRAIN_SHA:?could not compute the train prediction sha}"

CACHE_DIR="$SUBHG_ROOT/$FINGERPRINT/blurb_ner_${dataset}@pred-${TRAIN_SHA}/train"
echo "fingerprint: $FINGERPRINT"
echo "train sha  : $TRAIN_SHA"
echo "cache dir  : $CACHE_DIR"

# Fail closed on an absent or partial cache. Pretraining on a fraction of the stream
# produces a complete-looking encoder whose manifest cannot reveal the shortfall.
if [ ! -d "$CACHE_DIR" ]; then
  echo "FATAL cache dir does not exist: $CACHE_DIR" >&2
  echo "run the §0.3 precompute sweep first" >&2
  exit 1
fi
cached=$(find "$CACHE_DIR" -name '*.json' | wc -l)
expected=$(grep -c . "$PRED_FILE")
echo "cached     : $cached / $expected"
if [ "$cached" -lt "$expected" ]; then
  echo "FATAL cache is incomplete ($cached < $expected); refusing to pretrain on a subset" >&2
  exit 1
fi

LINK_MODE=strict
if [ "$dataset" = jnlpba ]; then LINK_MODE=nonstrict; fi
OUT="$PRETRAIN_ROOT/$backbone/$dataset/cross_modal/${LINK_MODE}_${FINGERPRINT}/encoder.pt"
if [ -s "$OUT" ] && [ -s "$OUT.manifest.json" ]; then
  echo "SKIP (encoder and manifest already present: $OUT)"
  exit 0
fi
mkdir -p "$(dirname "$OUT")"

# --incidence-cache matters: augment + build_incidence is the per-epoch CPU
# bottleneck (~20 ms/graph), so without it every epoch rebuilds from scratch. Keyed
# per fingerprint AND prediction sha, since the stream differs by both.
INCIDENCE="$INCIDENCE_ROOT/$backbone/$dataset/${FINGERPRINT}_${TRAIN_SHA}"
mkdir -p "$INCIDENCE"

set -x
CUDA_VISIBLE_DEVICES=0 uv run python -m knowhyper.pretrain.run_pretrain \
  --cache-dir "$CACHE_DIR" \
  --out "$OUT" \
  --incidence-cache "$INCIDENCE" \
  --objective cross_modal \
  --entity-view \
  --skip-modalities name \
  --mask-ratio "$MASK_RATIO" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --grad-clip "$GRAD_CLIP" \
  --seed "$SEED"
rc=$?
set +x

if [ "$rc" -ne 0 ]; then
  echo "FAIL pretrain exited $rc for $CELL" >&2
  exit "$rc"
fi
# The manifest is mandatory: it is the only record pairing this encoder with the
# FEATURES pin.
#
# NOTE: `run_pretrain.py` names `attr_vector_rows` in a comment as the field that
# distinguishes a working attribute build from the all-zero-vector one, but it never
# actually WRITES that field, so it is not printed here -- doing so would show `None`
# every time and read as a check that had passed. The available evidence that the
# attributes were live is the LOSS CURVE (a degenerate build predicts zeros from zeros
# and its loss starts at ~0 instead of descending), which
# the per-epoch loss logs record.
if [ ! -s "$OUT.manifest.json" ]; then
  echo "FATAL encoder written without a manifest: $OUT" >&2
  exit 1
fi
echo "OK   $CELL -> $OUT"
uv run python -c "
import json,sys
m=json.load(open(sys.argv[1]))
md=m.get('metadata',{})
print('  features_revision:', m.get('features_revision'))
print('  objective        :', md.get('objective'), 'mask_ratio', md.get('mask_ratio'))
print('  sha256           :', m.get('sha256','')[:16])
" "$OUT.manifest.json"
