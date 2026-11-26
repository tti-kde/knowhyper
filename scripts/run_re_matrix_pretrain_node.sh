#!/usr/bin/env bash
# Pretrain every cross-modal encoder the RE matrix's ev_warm arms need, packed
# onto ONE rt_HF node (8 GPU, one encoder per GPU).
#
# WHY ALL OF THEM ARE BEING REBUILT. On 2026-07-28 every existing encoder was
# audited from its manifest and all four read `"revision": "0.12.3"` while the
# project is pinned to core 0.12.13. The ChemProt one is the proof of the
# retracted ev_warm_strict result: its `cache_dir` is the NEW-pin cache
# deea89854216235e while its `revision` is the OLD core -- sub-HGs from one build,
# node features from another. None of them can be reused.
#
# The pin is no longer passed on the command line at all: run_pretrain.py defaults
# --revision/--fingerprint from knowhyper.pins and logs what it used, and the
# manifest records it. Gate G4 is then a grep, not a memory exercise.
#
# Directory name is `<strict|nonstrict>_<cache fingerprint>` because
# scripts/run_matrix_cell.py:_resolve_encoder matches on BOTH halves. Matching
# only the strict token would hand a run an encoder pretrained on a different
# PrimeHKG pin -- warm-start only ever helped when pretrain and end-task caches
# matched (DDI +0.0091 p=.002), so a cross-pin match manufactures the null.
#
# Usage (inside an abci submit command block):
#   bash scripts/run_re_matrix_pretrain_node.sh
set -uo pipefail

# ABCI defaults, set HERE for the same reasons as the precompute node script:
# a shell operator after `abci submit --` becomes a single filename
# (Exit_status=127), and the runner's own `uv sync` runs BEFORE this script.
if [ -n "${PBS_JOBID:-}" ]; then
  R="$HOME"
  export KNOWHYPER_HOME="${KNOWHYPER_HOME:-$R/.knowhyper}"
  export HF_HOME="${HF_HOME:-$R/.cache/huggingface}"
fi

: "${KNOWHYPER_HOME:?KNOWHYPER_HOME must be exported}"

CACHE="$KNOWHYPER_HOME/cache/subhg/v2"
INC="$KNOWHYPER_HOME/cache/incidence/v1"
OUT="$KNOWHYPER_HOME/artifacts/pretrain"

# dataset:condition:fingerprint  -- one encoder per entry, one GPU each.
# Fingerprints from scripts/resolve_cache_fingerprint.py (self-checking recipe).
SPECS="${SPECS:-\
chemprot:strict:deea89854216235e \
chemprot:nonstrict:8bf72b74a1435a17 \
ddi:strict:bb1087dfc4b9b88e \
ddi:nonstrict:45caf65e8a07ea90 \
biored:surface:cb831d8592bc38aa}"
read -r -a specs <<< "$SPECS"

EPOCHS="${EPOCHS:-10}"
MASK_RATIO="${MASK_RATIO:-0.5}"
LR="${LR:-3e-4}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "=== preflight ==="
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)  CPUs: $(nproc)"
command -v lfs >/dev/null 2>&1 && lfs quota -h -u "$(whoami)" /home 2>/dev/null | tail -2

# Refuse to start on a missing cache rather than half-finishing the node. A
# pretrain against an absent cache does not error usefully -- it streams zero
# graphs and writes a useless encoder.
missing=0
for spec in "${specs[@]}"; do
  IFS=: read -r ds cond fp <<< "$spec"
  d="$CACHE/$fp/$ds/train"
  n=$([ -d "$d" ] && ls "$d" 2>/dev/null | wc -l || echo 0)
  printf "  %-9s %-10s %s train=%s\n" "$ds" "$cond" "$fp" "$n"
  [ "$n" -lt 100 ] && { echo "    ^^ MISSING or too small -- skipping"; missing=1; }
done
if (( missing )); then
  echo "NOTE: some caches are absent; those entries will be skipped, not faked."
fi

gpu=0; pids=(); names=()
for spec in "${specs[@]}"; do
  IFS=: read -r ds cond fp <<< "$spec"
  cdir="$CACHE/$fp/$ds/train"
  [ -d "$cdir" ] && [ "$(ls "$cdir" 2>/dev/null | wc -l)" -ge 100 ] || continue

  odir="$OUT/$ds/cross_modal/${cond}_${fp}"
  mkdir -p "$odir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    # --incidence-cache matters: augment + build_incidence is the per-epoch CPU
    # bottleneck (~20 ms/graph), so without it 10 epochs rebuild from scratch
    # every time. Keyed per fingerprint so two caches never share one.
    uv run python -m knowhyper.pretrain.run_pretrain \
      --cache-dir "$cdir" \
      --out "$odir/encoder.pt" \
      --incidence-cache "$INC/$fp/$ds/train" \
      --objective cross_modal \
      --entity-view \
      --skip-modalities name \
      --mask-ratio "$MASK_RATIO" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --grad-clip 1.0
  ) >"$odir/pretrain.log" 2>&1 &
  pids+=("$!"); names+=("$ds/$cond/$fp"); echo "dispatched $ds:$cond on GPU $gpu"
  gpu=$(( (gpu + 1) % 8 ))
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "OK   ${names[$i]}"; else echo "FAIL ${names[$i]}"; failed=1; fi
done

# GATE G4, checked here rather than trusted later: the manifest must record the
# core the end task will read. This is the exact check that would have caught the
# retracted ev_warm_strict result.
echo "=== G4: encoder provenance ==="
for spec in "${specs[@]}"; do
  IFS=: read -r ds cond fp <<< "$spec"
  m="$OUT/$ds/cross_modal/${cond}_${fp}/encoder.pt.manifest.json"
  if [ -f "$m" ]; then
    rev=$(grep -o '"revision": *"[^"]*"' "$m" | head -1 | sed 's/.*: *"//;s/"//')
    cd_=$(grep -o '"cache_dir": *"[^"]*"' "$m" | head -1 | sed 's/.*: *"//;s/"//')
    ok="OK  "; [ "$rev" = "0.12.13" ] || ok="WRONG-PIN"
    case "$cd_" in *"/$fp/"*) ;; *) ok="WRONG-CACHE";; esac
    printf "  %-9s %-10s rev=%-8s %s\n" "$ds" "$cond" "$rev" "$ok"
    [ "$ok" = "OK  " ] || failed=1
  else
    printf "  %-9s %-10s NO MANIFEST\n" "$ds" "$cond"
  fi
done

exit "$failed"
