# KnowHyper

Code for **KnowHyper**, which injects a multimodal biomedical knowledge
hypergraph (**PrimeHKG**) into biomedical language models for relation
extraction (RE) and named entity recognition (NER).

> *Paper:* EACL 2027 (Findings). The citation will be added on publication.

This repository is a monorepo with everything needed to rebuild the resource
and rerun the experiments:

| Path | What it is |
|---|---|
| `src/knowhyper/`, `scripts/` | KnowHyper: retrieval, Entity View, AllSet encoder, fusion and NER verifier (ieforge plugins), Hydra configs, run and analysis scripts |
| `packages/primehkg/` | PrimeHKG: construction from the original providers, entity linking, sub-hypergraph retrieval |
| `packages/ieforge/` | ieforge: the training and evaluation framework the plugins run in, plus the benchmark builders |

## Releases

Each paper is a tag. Check out the tag to get the exact code used for it.

| Tag | Alias | Paper | Components |
|---|---|---|---|
| `2026.08.0` | `eacl2027` | EACL 2027 | knowhyper (submission code of 2026-08-04, plus the LLM baseline and bootstrap added for the camera-ready), primehkg 0.12.14 (builds core 0.12.13), ieforge 0.1.11 |

Third-party dependencies are pinned by `uv.lock` (Python 3.12, PyTorch 2.8.0,
PyTorch Geometric 2.7.0, Transformers 5.10.2).

## Setup

```bash
uv sync                                   # .venv with all three packages (editable)
export KNOWHYPER_HOME=$HOME/.knowhyper    # caches, encoders, runs (absolute path)
```

## Data

Neither the benchmarks nor PrimeHKG are redistributed here: their sources carry
heterogeneous terms (for example, CORUM is CC BY-NC 4.0, CTD has its own terms
and the DDI corpus is CC BY-NC). Build both locally.

### 1. Benchmarks

Build ChemProt and DDI (BLURB preprocessing), BioRED and the five BLURB NER sets
with the ieforge builders into one directory, and point the loaders at it:

```bash
cd packages/ieforge
uv run python scripts/build_blurb_re.py --dataset chemprot --out-dir $DATA/blurb-re
uv run python scripts/build_blurb_re.py --dataset ddi      --out-dir $DATA/blurb-re
uv run python scripts/build_blurb_ner.py --out-dir $DATA/blurb-ner
uv run python scripts/build_biored.py --biored-zip raw_data/BIORED.zip --out-dir $DATA/biored
export IEFORGE_DATA_DIR=$DATA
```

The raw inputs each builder needs (BLURB `data_generation`, DDICorpus-2013,
BioRED, the MTL-Bioinformatics-2016 NER data) are listed in
[`packages/ieforge/docs/getting-started/datasets.md`](packages/ieforge/docs/getting-started/datasets.md).
Split sizes (train / validation / test): ChemProt 18,035 / 11,268 / 15,745,
DDI 25,296 / 2,496 / 5,716, BioRED 22,896 / 6,659 / 7,591.

### 2. PrimeHKG

Follow [`packages/primehkg/REPRODUCE.md`](packages/primehkg/REPRODUCE.md). It
downloads each source from its provider, builds the core hypergraph (the paper's
build has 137,587 entities and 4,983,742 hyperedges) and the attribute features
(PubMedBERT text, ChemBERTa SMILES and ESM-2 sequence vectors), and writes both
to local directories in the Hub layout. Then:

```bash
fp() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["inputs_fingerprint"])' "$1/build_manifest.json"; }
export PRIMEHKG_CORE=/path/to/core            # local directory (or a Hub repo id)
export PRIMEHKG_CORE_FINGERPRINT=$(fp $PRIMEHKG_CORE)
export PRIMEHKG_FEATURES=/path/to/features
export PRIMEHKG_FEATURES_FINGERPRINT=$(fp $PRIMEHKG_FEATURES)
```

Every loader checks the build against these fingerprints, and they are part of
every cache and encoder address, so artifacts from different builds never mix.
A rebuild cannot reproduce the paper's fingerprints, because they hash absolute
input paths. Several providers also serve only their current release, so a build
made today differs from the 2026 one. `REPRODUCE.md` lists the sha256 of every
input the paper's build read.

## Running the experiments

All commands run from the repository root with the environment above. Runs are
written under `$KNOWHYPER_HOME`, one directory per dataset, arm and seed.

### Retrieval caches

Sub-hypergraph retrieval is precomputed once per retrieval profile (CPU only):

```bash
KNOWHYPER_PRECOMPUTE_WORKERS_PER_SPLIT=16 \
  bash scripts/run_subhg_precompute_all_cpu.sh khp_ddi_xattn
uv run python scripts/resolve_cache_fingerprint.py khp_ddi_xattn   # address and build status
```

Profiles used in the paper: `khp_re_xattn`, `khp_re_xattn_strict`,
`khp_ddi_xattn`, `khp_ddi_xattn_strict`, `khp_biored_xattn_surface`, and the
text routes `re_serialized_subhg`, `ddi_serialized_subhg` and
`biored_serialized_subhg_surface`. `scripts/run_re_matrix_precompute_node.sh`
builds a list of them in parallel (`EXPERIMENTS="..."`). A training run whose
cache is missing still works, but retrieves live at about three seconds per
instance.

### Cross-modal pretraining (the "+ cross-modal pretraining" rows)

```bash
FP=$(uv run python scripts/resolve_cache_fingerprint.py khp_ddi_xattn | awk '$1=="khp_ddi_xattn"{print $2}')
SPECS="ddi:nonstrict:$FP" bash scripts/run_re_matrix_pretrain_node.sh
```

This writes
`$KNOWHYPER_HOME/artifacts/pretrain/<dataset>/cross_modal/<strict|nonstrict>_<cache>/encoder.pt`
with its manifest. A warm-start run refuses an encoder pretrained on another
cache.

### RE table

One cell is one dataset × arm × seed (train, validate, test, export predictions):

```bash
uv run python scripts/run_matrix_cell.py --backbone pubmedbert --dataset ddi \
  --method ev_warm --seed 0 --output-root $KNOWHYPER_HOME/runs/matrix \
  --encoder $KNOWHYPER_HOME/artifacts/pretrain
uv run python scripts/aggregate_matrix.py --output-root $KNOWHYPER_HOME/runs/matrix
```

| Paper row | ChemProt | DDI | BioRED |
|---|---|---|---|
| Text-only backbone | `base` | `base` | `base` |
| Structure only | `edge_strict` (×0.3) | `edge_strict` (×1.5) | `edge_surface` (×1.5) |
| + multimodal attributes | `ev_cold_strict` (×1.5) | `ev_cold` | `ev_cold_surface` (×5) |
| + cross-modal pretraining | `ev_warm` (×1.5) | `ev_warm` | `ev_warm_surface` (×1.0) |

- **Seeds and backbones:** seeds are 0–9. `--backbone` is one of `pubmedbert`,
  `biolinkbert`, `biobert` or `scibert`.
- **Learning rates:** the knowledge branch trains at 10× the backbone learning
  rate by default.
- **Selected multipliers:** the multipliers in parentheses are the per-dataset
  selections marked ‡ in the paper. Pass one as
  `--extra-override trainer.callbacks.3.knowledge_lr_mult=<m> --variant lrmult<m>`.
- **Ablation and control arms:** these are listed with their configs in
  `src/knowhyper/matrix.py`:
  - text routes: `text`, `text_off`, `serialized`, `serialized_shuffled`
  - 2-section control: `binary`
  - `ev_nosmiles`, `endpoint`, `endpoint_shuffled`
  - `graft_*`, `edge_top3`, and others

Analyses:
- **Prediction export:** `scripts/analysis/export_re_predictions.py`
- **Retrieval strata:** `scripts/analysis/matrix_linking_strata.py`
- **Paired contrasts:** `scripts/analysis/c6_pair_binary.py`
- **Attention shares:** `scripts/analysis/extract_attention.py` with `aggregate_attention.py`
- **Instance-level bootstrap intervals:** `scripts/bootstrap/re_paired_bootstrap.py`
  with `scripts/bootstrap/contrasts.yaml`. Edit the run roots in the YAML to yours.

### NER

The NER pipeline is a sequence of node scripts. Each takes `--index N
--node-lines FILE`, where each line of the file lists the cells for one node and
is written by the `scripts/plan_*.py` planners. Each script packs one run per GPU.

1. **Text-only span model and candidate proposals,** with out-of-fold training
   candidates: `run_span_ner_proposals_node.sh`, `run_span_oof_anchor_node.sh`,
   `finalize_span_ner_proposals.py`.
2. **Retrieval caches for the candidates:** `run_span_ner_precompute_node.sh`.
3. **Encoder pretraining:** `run_span_ner_pretrain_node.sh`.
4. **The verifier arms** (Structure only, + attributes and + pretraining, i.e.
   `khp_span_late_{edge,ev,ev_warm}*`): run `plan_span_late_extension.py`, then
   `run_span_late_logit_node.sh`. Aggregate with `span_late_logit_confirm.py`.
5. **Retrieved-text routes:** `plan_span_ner_text.py` with
   `run_span_ner_text_index.sh`, and `plan_span_late_text_cumulative.py` with
   `run_span_late_text_cumulative_node.sh`.

Each script documents its inputs at the top.

### Instruction-tuned LLM baseline (DDI)

```bash
uv run python scripts/llm_baseline/llm_ddi_baseline.py --arm noKG --split test \
  --output-root $KNOWHYPER_HOME/runs/llm_baseline --condition qwen35_9b_noKG   # add --zero-shot for the zero-shot arm
uv run python scripts/llm_baseline/llm_ddi_baseline.py --arm KG --split test \
  --cache-split-dir $KNOWHYPER_HOME/cache/subhg/v2/<khp_ddi_xattn cache>/ddi/test \
  --output-root $KNOWHYPER_HOME/runs/llm_baseline --condition qwen35_9b_KG
```

## License

MIT (see `LICENSE`). Data obtained from the original providers remains under
their own terms; see `packages/primehkg/REPRODUCE.md`.
