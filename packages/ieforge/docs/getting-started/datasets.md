# Datasets

ieforge's benchmark loaders (BLURB NER / RE and BioRED) were developed against Hugging Face dataset repos under `tti-kde/*`.
**Those repos are private and are not redistributed**: the source corpora carry
their own licences and redistribution terms. To run the benchmarks, build the
datasets locally from the original corpora and point ieforge at them with
`IEFORGE_DATA_DIR`.

## `IEFORGE_DATA_DIR`

| `IEFORGE_DATA_DIR` | What a loader reads for repo `tti-kde/<name>`, config `<config>`, split `<split>` |
|---|---|
| unset or empty | the Hub repo, exactly as before: `datasets.load_dataset("tti-kde/<name>", <config>, split=<split>, token=$HF_TOKEN)` |
| set | `$IEFORGE_DATA_DIR/<name>/<config>/<split>-*.parquet`, read with the `parquet` builder (no network, no token) |

The local layout is the one `DatasetDict.push_to_hub(..., config_name=<config>)`
writes. A missing split raises `FileNotFoundError` naming the searched
path; there is no silent fallback to the Hub.

```text
$IEFORGE_DATA_DIR/
├── blurb-re/{chemprot,ddi,gad}/{train,validation,test}-00000-of-00001.parquet
├── blurb-ner/{bc5cdr-chem,bc5cdr-disease,ncbi-disease,bc2gm,jnlpba}/...
└── biored/biored/...
```

Rows, column order and features are those of the parquet files (the `datasets`
features are stored in the parquet schema metadata), so loading a local copy of
the Hub files gives the same records as loading the Hub repo. This was checked
for `blurb-re` (chemprot, ddi), `biored` and `blurb-ner` (all five configs), all
splits, against these Hub commits:

| Repo | Commit |
|---|---|
| `tti-kde/blurb-re` | `05a6afd5e4bf6bce0fd32731960daddfe0b94f27` |
| `tti-kde/biored` | `ad47435bef131617344102025671f46bac374be5` |
| `tti-kde/blurb-ner` | `3042dd37bfb97a3f40d5dddb5bc53f38a650bcd1` |

and `scripts/build_biored.py --out-dir` on the NCBI `BIORED.zip` (sha256
`c3032230bd89d22a0923d0df6ae943bc8ea37fba7e42dafa7a8dec21bac02d47`) reproduced
the `tti-kde/biored` parquet files row for row (same schema and `datasets`
feature metadata; 22,896 / 6,659 / 7,591 pairs).

The Hub `tti-kde/blurb-re` repo holds `chemprot` and `ddi` only; `gad` can be
built locally (see below) but was never published there.

## Building the datasets

Every build script keeps its original Hub push mode and adds `--out-dir`, which
writes the layout above for one repo directory and does not need `HF_TOKEN`.
Paths such as `raw_data/...` are relative to the directory you run the script
from unless noted; run them from `packages/ieforge`.

```bash
export IEFORGE_DATA_DIR=$PWD/ieforge_data
```

### `blurb-re` (ChemProt, DDI, GAD) — `scripts/build_blurb_re.py`

Raw inputs:

- **ChemProt**: downloaded automatically from the Hugging Face Hub dataset
  `bigbio/chemprot` (`chemprot_bigbio_kb`); needs network access.
- **DDI**: the DDI corpus 2013
  ([DDICorpus-2013.zip](https://github.com/isegura/DDICorpus)) extracted to
  `packages/ieforge/raw_data/ddi/DDICorpus/`, and BLURB's
  [`data_generation.tar.gz`](https://microsoft.github.io/BLURB/sample_code/data_generation.tar.gz)
  extracted to `packages/ieforge/raw_data/data_generation/` (the official file
  split in `indexing/DDI/{train,dev,test}_files.tsv`). These two paths are
  resolved relative to the package directory, not the working directory.
- **GAD**: BLURB-compatible TSVs at `raw_data/GAD/1/{train,test}.tsv`
  (`--gad-root`), from the full BLURB download workflow
  (`download_BLURB_data.sh`, `REdata.zip`); the sample-code archive alone does
  not contain them.

```bash
uv run python scripts/build_blurb_re.py --dataset chemprot --out-dir "$IEFORGE_DATA_DIR/blurb-re"
uv run python scripts/build_blurb_re.py --dataset ddi      --out-dir "$IEFORGE_DATA_DIR/blurb-re"
```

Expected counts (checked by the script): ChemProt 18,035 / 11,268 / 15,745,
DDI 25,296 / 2,496 / 5,716, GAD 4,261 / 535 / 534.

### `blurb-ner` — `scripts/build_blurb_ner.py`

Raw input: the Crichton et al. corpus,
`git clone https://github.com/cambridgeltl/MTL-Bioinformatics-2016 raw_data/MTL-Bioinformatics-2016`
(`--data-root raw_data/MTL-Bioinformatics-2016/data`, the default).

```bash
uv run python scripts/build_blurb_ner.py --out-dir "$IEFORGE_DATA_DIR/blurb-ner"
```

The script refuses to write when a split's statistics differ from the BLURB paper.

### `biored` — `scripts/build_biored.py`

Raw input: NCBI's `BIORED.zip`
(<https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/>), which holds
`BioRED/{Train,Dev,Test}.PubTator`.

```bash
uv run python scripts/build_biored.py --biored-zip raw_data/BIORED.zip \
  --out-dir "$IEFORGE_DATA_DIR/biored"
```

See [`scripts/README.md`](../../scripts/README.md) for each script's details.
