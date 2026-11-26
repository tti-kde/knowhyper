#!/usr/bin/env bash
set -euo pipefail

# NER SubHG precompute for ONE arm, packed on one rt_HF node (192 CPU / 8 GPU).
#
# Why this exists rather than run_ner_precompute_split.sh: that script hardcodes
# three things that silently defeat a strict run --
#   --experiment khp_ner            (so a strict arm would precompute non-strict)
#   knowhyper.primehkg_rev=v0.12.3  (overriding the config's pin back to old core)
#   predictions_root=.../predictions (hard decode, NOT the K=16 candidate lattice)
# All three are parameters here, and every one of them changes the cache
# fingerprint, so getting one wrong produces a valid-looking cache for the wrong
# arm. It is kept rather than edited so the reproduction path of the 2026-07-19
# runs stays intact.
#
# The strict arms share ONE cache (71b330506dd2ee18 on core 0.12.13 with the
# weighted chem profile; the non-strict pair is 849c05bd0d2c712a):
#   khp_ner_strict and khp_ner_ev_strict differ only after retrieval, so
#   precompute ONCE per strict setting, not once per arm.
#   Confirm with `scripts/resolve_cache_fingerprint.py <arm>` before trusting a
#   hash written in a comment -- adding the semantics weights re-fingerprinted
#   every NER arm, so `b6b45dcf4d54dcc4` (quoted elsewhere as the measurement
#   source) is the UNIFORM predecessor, not a target of this script.
#
# Usage (env vars, all overridable):
#   EXPERIMENT=khp_ner_strict \
#   PREDICTIONS_ROOT=$HOME/.abci-runner/shared/ner_bc5cdr_chem_seed0/candidate_predictions_k16 \
#   bash scripts/run_ner_strict_precompute_node.sh

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Precompute is CPU work (SapBERT linking + retrieval), and it runs 150 shards at
# once. Each shard that can see a GPU loads SapBERT onto it, so 150 of them exhaust
# VRAM: the BC5CDR-Disease run died with `torch.AcceleratorError: CUDA error: out of
# memory` in ~30 of 150 shards while the rest finished, i.e. a partial cache from a
# job that had already passed every preflight. ner_precompute.py's own usage block
# says to run with CUDA_VISIBLE_DEVICES="" -- honour it here rather than leaving it
# to the caller. This also frees the node's GPUs for anything co-scheduled.
export CUDA_VISIBLE_DEVICES=""

EXPERIMENT="${EXPERIMENT:?EXPERIMENT missing (e.g. khp_ner_strict)}"
# NOTE: keep this message brace-free -- a '{...}' inside a ':?' word terminates the
# expansion early and the literal message leaks into the value.
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:?PREDICTIONS_ROOT missing: dir holding per-split .jsonl}"
SPLITS="${SPLITS:-train validation test}"
# 3 splits x this many shards. Measured on the weighted strict run (job 2075365,
# 20/split = 60 procs): `resources_used.cpupercent = 2790`, i.e. ~28 of the node's
# 192 cores busy, and ~7.3 instances/s. The work is NFS-latency bound on small JSON
# writes, not CPU bound, so oversubscribing cores is the right move -- 50/split
# (150 procs) still leaves the node far from saturated. Raise further only with a
# cpupercent measurement in hand; the cache writes one file per instance, so the
# ceiling is the filesystem, not the scheduler.
WORKERS_PER_SPLIT="${WORKERS_PER_SPLIT:-50}"
KNOWHYPER_HOME="${KNOWHYPER_HOME:-${HOME}/.knowhyper}"
CACHE_ROOT="${CACHE_ROOT:-${KNOWHYPER_HOME}/cache/subhg/v2}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/ner_precompute_${EXPERIMENT}}"
LINK_CHUNK_SIZE="${LINK_CHUNK_SIZE:-0}"

for split in $SPLITS; do
  f="${PREDICTIONS_ROOT}/${split}.jsonl"
  [[ -s "$f" ]] || { echo "missing prediction file: $f" >&2; exit 1; }
done

KH_CONFIGS="$(python -c 'from knowhyper.ieforge._configs import CONFIGS_DIR; print(CONFIGS_DIR)')"
mkdir -p "$ARTIFACT_ROOT"

echo "=== NER precompute ==="
echo "experiment       : $EXPERIMENT"
echo "predictions_root : $PREDICTIONS_ROOT"
echo "splits           : $SPLITS"
echo "workers/split    : $WORKERS_PER_SPLIT"
echo "cache_root       : $CACHE_ROOT"
echo "configs          : $KH_CONFIGS"

# Fail before any compute if the pin the config resolves to is not the intended
# one. A wrong pin still produces a consistent cache -- under a fingerprint no
# training arm will ever look up.
python - "$EXPERIMENT" "$KH_CONFIGS" <<'PY'
import sys
from hydra import initialize_config_dir, compose
import ieforge, os
exp, kh = sys.argv[1], sys.argv[2]
base = os.path.join(os.path.dirname(ieforge.__file__), "configs")
with initialize_config_dir(config_dir=base, version_base=None):
    cfg = compose(config_name="train",
                  overrides=[f"experiment={exp}", f"hydra.searchpath=[file://{kh}]"],
                  return_hydra_config=True)
inj = cfg.knowledge.pipeline.injector
strict = inj.linker.strict_type_match
sw = inj.retriever.inner.get("semantics_weight")
n_weights = 0 if sw is None else len(sw)
print(f"resolved: core={cfg.knowhyper.dataset_revision} "
      f"fp={cfg.knowhyper.dataset_fingerprint[:16]} strict={strict} weights={n_weights}")
if cfg.knowhyper.dataset_revision != "0.12.13":
    raise SystemExit(f"expected core 0.12.13, got {cfg.knowhyper.dataset_revision}")
if exp.endswith("_strict") and not strict:
    raise SystemExit("arm name says strict but strict_type_match resolved False")
# The name check above is one-directional and misses the case that matters here:
# every NER arm in this campaign links strictly, but only some SAY so in their name
# (`khp_ner_strict_bc2gm` does, `ner_retrieved_text_bc2gm` does not). This script
# exists to build STRICT caches, so require it from the CONFIG regardless of naming
# -- the same lesson `_resolve_encoder` learned when a suffix rule silently asked
# for a nonstrict encoder that should not exist. Set STRICT_REQUIRED=0 to build a
# deliberately non-strict cache.
if os.environ.get("STRICT_REQUIRED", "1") == "1" and not strict:
    raise SystemExit(
        f"{exp}: strict_type_match resolved False, but this script builds strict "
        "caches. Pass STRICT_REQUIRED=0 if a non-strict cache is genuinely wanted."
    )
# Uniform scoring must be caught HERE, not at training time. Precompute is where the
# cache is created: a uniform run burns 150 processes and writes a complete, valid
# cache that spends ~63% of its top_k budget on drug-consequence edges (measured;
# see caching_ner_chem.yaml). The second pass would then correctly refuse to train
# against it, after the compute is gone.
if not n_weights:
    raise SystemExit(f"{exp}: uniform scoring -- see docs/EXPERIMENTS.md 3.2")
print(f"cache target: {cfg.knowhyper.dataset_fingerprint[:16]} (verify against "
      f"scripts/resolve_cache_fingerprint.py before trusting a hash in a comment)")

# Instantiate the dataset for real, offline, BEFORE dispatching 60 shards. The HF
# cache has three independent parts (hub/datasets--*, datasets/ Arrow, hub/models--*)
# and having primehkg cached says nothing about the task dataset: run 2075083 died
# 60x with ConnectionError("Couldn't reach 'tti-kde/blurb-ner' ... OfflineModeIsEnabled")
# because only blurb-re and biored were present on the reservation. One cheap
# instantiation here turns that into a 10-second failure with a clear cause.
from hydra.utils import instantiate
from omegaconf import OmegaConf
# Same key-stripping ner_precompute.py does: cfg.dataset carries datamodule-only
# keys the dataset constructor does not accept.
raw = OmegaConf.to_container(cfg.dataset, resolve=True)
ds_cfg = OmegaConf.create({k: v for k, v in raw.items()
                           if k not in {"batch_size", "num_workers", "task_family"}})
try:
    ds = instantiate(ds_cfg, split="test")
except Exception as e:  # noqa: BLE001 - want the offline cause verbatim
    raise SystemExit(f"dataset preflight FAILED (is the HF cache complete here?): {e!r}")
print(f"dataset preflight OK: {type(ds).__name__} n={len(ds)}")
PY

pids=()
for split in $SPLITS; do
  (
    sub=()
    for ((shard = 0; shard < WORKERS_PER_SPLIT; shard++)); do
      python scripts/ner_precompute.py \
        --experiment "$EXPERIMENT" \
        --searchpath "$KH_CONFIGS" \
        --predictions "$PREDICTIONS_ROOT" \
        --splits "$split" \
        --cache-root "$CACHE_ROOT" \
        --num-shards "$WORKERS_PER_SPLIT" \
        --shard-index "$shard" \
        --link-chunk-size "$LINK_CHUNK_SIZE" \
        --log-every 250 \
        >"${ARTIFACT_ROOT}/${split}-shard${shard}.log" 2>&1 &
      sub+=("$!")
    done
    rc=0
    for p in "${sub[@]}"; do wait "$p" || rc=1; done
    exit "$rc"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then
  echo "one or more precompute shards failed -- see ${ARTIFACT_ROOT}/*.log" >&2
  exit 1
fi

python - "$EXPERIMENT" "$PREDICTIONS_ROOT" "$CACHE_ROOT" "$ARTIFACT_ROOT/summary.json" "$SPLITS" <<'PY'
import hashlib, json, sys
from pathlib import Path

# The cache DIRECTORY is `method_fingerprint()` -- a 16-char sha256 over the WHOLE
# resolved retrieval profile (cache/keys.py:95-103) -- NOT `dataset_fingerprint`,
# which is one 64-char field inside it. An earlier version of this block used the
# latter and so looked in a directory that can never exist: every correct run
# reported zero and failed itself.
#
# Reuse `resolve_cache_fingerprint.resolve()` rather than rebuilding the recipe. It
# is fiddly on purpose -- `vector_candidate_sidecars` must be a sorted tuple of
# 3-tuples, and several keys are included only when non-null -- and CLAUDE.md is
# explicit that hand-computing a fingerprint yields "a plausible hash that matches
# nothing on disk". One implementation, self-checked in that script.
sys.path.insert(0, "scripts")
from resolve_cache_fingerprint import resolve  # noqa: E402
from knowhyper.span_cache_scope import (  # noqa: E402
    PredictionScope,
    verify_scope_completeness,
)

exp, pred_root, cache_root, out, splits = sys.argv[1:]
fingerprint, _profile, _dataset = resolve(exp)


summary = {"status": "completed", "experiment": exp, "predictions_root": pred_root,
           "cache_root": cache_root, "fingerprint": fingerprint, "splits": {}}

# Count ONLY the prediction-scoped leaf this invocation reads. Two backbones share
# the retrieval fingerprint by design but must have different `@pred-<sha>` leaves.
# A recursive count under the fingerprint lets one completed backbone make its
# sibling pass with zero files in the sibling's own scope.
fp_dir = Path(cache_root) / fingerprint
empty = []
for split in splits.split():
    p = Path(pred_root) / f"{split}.jsonl"
    raw = p.read_bytes()
    prediction_sha = hashlib.sha256(raw).hexdigest()[:16]
    rows = sum(1 for line in raw.decode("utf-8").splitlines() if line.strip())
    scope = PredictionScope(
        backbone=Path(pred_root).parent.name,
        dataset=Path(pred_root).name,
        split=split,
        prediction_sha=prediction_sha,
        rows=rows,
        path=p,
    )
    report = verify_scope_completeness(scope, fp_dir)
    n_cache = report.cached
    summary["splits"][split] = {
        "prediction_count": rows,
        "prediction_sha256": hashlib.sha256(raw).hexdigest(),
        "scoped_dataset": scope.scoped_dataset,
        "cache_json_count": n_cache,
    }
    if not report.complete:
        empty.append(split)
Path(out).write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if empty:
    raise SystemExit(
        f"precompute left an incomplete prediction scope under {fp_dir} for "
        f"split(s) {empty} -- the training arm would read a partial cache")
PY
