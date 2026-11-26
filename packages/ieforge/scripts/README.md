# Scripts

One-off data generation and maintenance scripts. These are not part of the IEForge package and are not installed as CLI commands.

The datasets they build are not redistributed. Every `build_*.py` script
accepts `--out-dir DIR`, which writes `DIR/<config>/<split>-00000-of-00001.parquet`
locally instead of pushing to the Hugging Face Hub (no `HF_TOKEN` needed). Use
`DIR=$IEFORGE_DATA_DIR/<repo-name>` (e.g. `$IEFORGE_DATA_DIR/blurb-re`) so the
ieforge loaders find it; see
[`docs/getting-started/datasets.md`](../docs/getting-started/datasets.md). The
`--hf-repo` push mode below is unchanged.

```bash
export IEFORGE_DATA_DIR=$PWD/ieforge_data
uv run python scripts/build_blurb_ner.py --out-dir "$IEFORGE_DATA_DIR/blurb-ner"
uv run python scripts/build_blurb_re.py --dataset chemprot --out-dir "$IEFORGE_DATA_DIR/blurb-re"
uv run python scripts/build_biored.py --biored-zip raw_data/BIORED.zip --out-dir "$IEFORGE_DATA_DIR/biored"
```

## build_blurb_ner.py

Generates all 5 BLURB NER datasets from [Crichton et al. MTL-Bioinformatics-2016](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) and uploads them to HuggingFace Hub as `tti-kde/blurb-ner`.

Replicates the `preprocess_ner_data()` logic from BLURB's official `preprocessor.py`: strips entity type suffixes from IOB labels, keeping only the BIO prefix character (`B-Chemical` → `B`, `I-Disease` → `I`).

### Prerequisites

```bash
# Clone Crichton corpus
git clone https://github.com/cambridgeltl/MTL-Bioinformatics-2016 raw_data/MTL-Bioinformatics-2016

# Log in to HuggingFace (for upload only)
huggingface-cli login
```

### Usage

```bash
# Verify stats against BLURB paper (no upload)
uv run python scripts/build_blurb_ner.py \
  --hf-repo tti-kde/blurb-ner \
  --data-root raw_data/MTL-Bioinformatics-2016/data \
  --dry-run

# Generate and upload
uv run python scripts/build_blurb_ner.py \
  --hf-repo tti-kde/blurb-ner \
  --data-root raw_data/MTL-Bioinformatics-2016/data
```

The script aborts the upload if any split fails the entity mention count check.

### Output

Each task is pushed as a separate config to the HF dataset repo:

| Config | Entities | Train / Val / Test sentences |
|--------|----------|------------------------------|
| `bc5cdr-chem` | Chemical | 4,560 / 4,581 / 4,797 |
| `bc5cdr-disease` | Disease | 4,560 / 4,581 / 4,797 |
| `ncbi-disease` | Disease | 5,424 / 923 / 940 |
| `bc2gm` | Gene | 12,574 / 2,519 / 5,038 |
| `jnlpba` | DNA, RNA, cell_line, cell_type, protein | 16,807 / 1,739 / 3,856 |

Each record has two fields: `tokens` (list of strings) and `ner_tags` (single-character BIO labels, e.g. `B`, `I`, `O`). The entity type is supplied by the IEForge dataset config, not stored per token.

## build_blurb_re.py

Generates BLURB RE datasets from BLURB-compatible ChemProt, DDI, and GAD raw sources and uploads them to HuggingFace Hub as `tti-kde/blurb-re`.

Replicates the `preprocess_re_data()` logic from BLURB's official `preprocessor.py` exactly:

- `customized_sent_tokenize`: NLTK sentence tokenization followed by splitting on `\t` (and `\n`, which `bigbio` uses to join title+abstract where the raw TSV uses `\t`)
- Entity dummification with `@{TYPE}$` tokens (e.g. `@CHEMICAL$`, `@GENE-N$`, `@GENE-Y$`); overlapping spans produce `@CHEM-GENE$`
- Bidirectional relation lookup (checks both `(e1_id, e2_id)` and `(e2_id, e1_id)`)
- `clean_sent`: removes `\n`, `\t`, `\r` from dummified text

### Prerequisites

```bash
# HF_TOKEN must be set for upload
export HF_TOKEN=<your_token>

# DDI needs raw DDICorpus-2013 plus BLURB indexing files.
# GAD needs BLURB-compatible raw TSVs under raw_data/GAD/1/.
# The full official BLURB download workflow can provide the GAD layout via
# download_BLURB_data.sh fetching REdata.zip; data_generation.tar.gz alone cannot.
```

### Usage

```bash
# Verify record counts (no upload)
uv run python scripts/build_blurb_re.py \
  --hf-repo tti-kde/blurb-re \
  --dry-run

# Generate and upload
uv run python scripts/build_blurb_re.py \
  --hf-repo tti-kde/blurb-re
```

The script aborts the upload if any split fails the expected record count check.

### Output

Each task is pushed as a separate config to the HF dataset repo:

| Config | Train / Val / Test instances |
|--------|------------------------------|
| `chemprot` | 18,035 / 11,268 / 15,745 |
| `ddi` | 25,296 / 2,496 / 5,716 |
| `gad` | 4,261 / 535 / 534 |

Each record has fields: `id`, `text` (dummified sentence), `entities` (list with `start`, `end`, `text`, `label`), `relations` (list with `head`, `tail`, `label`).

## build_biored.py

Builds the BioRED pair-level dataset from the NCBI `BIORED.zip` PubTator
files and uploads it to Hugging Face Hub as `tti-kde/biored` with the
`biored` config.

The builder reads `BioRED/Train.PubTator`, `BioRED/Dev.PubTator`, and
`BioRED/Test.PubTator`. It emits one record per normalized PubTator ID pair with
marker-encoded text, relation label, novelty label, entity IDs, entity types,
and sentence-distance metadata.

```bash
uv run python scripts/build_biored.py \
  --biored-zip raw_data/BIORED.zip \
  --dry-run

HF_TOKEN="$HF_TOKEN" uv run python scripts/build_biored.py \
  --biored-zip raw_data/BIORED.zip \
  --hf-repo tti-kde/biored \
  --config-name biored
```
