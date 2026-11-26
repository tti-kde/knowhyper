# IEForge

IEForge is the training and evaluation framework KnowHyper runs in: Hydra
configs and PyTorch Lightning modules for NER and relation extraction, with a
plugin interface that KnowHyper uses to add its knowledge components.

This copy is ieforge `v0.1.11`, the version the KnowHyper EACL 2027 experiments
ran on. Hydra configs ship inside the package (`ieforge.CONFIGS_DIR`).

## Benchmarks

The benchmarks are **not redistributed**: their source corpora carry their own
terms. Build them locally from the original corpora:

```bash
export IEFORGE_DATA_DIR=$PWD/ieforge_data
uv run python scripts/build_blurb_re.py --dataset chemprot --out-dir "$IEFORGE_DATA_DIR/blurb-re"
uv run python scripts/build_blurb_re.py --dataset ddi      --out-dir "$IEFORGE_DATA_DIR/blurb-re"
uv run python scripts/build_blurb_ner.py --out-dir "$IEFORGE_DATA_DIR/blurb-ner"
uv run python scripts/build_biored.py --biored-zip raw_data/BIORED.zip --out-dir "$IEFORGE_DATA_DIR/biored"
```

With `IEFORGE_DATA_DIR` set, every loader reads
`$IEFORGE_DATA_DIR/<name>/<config>/<split>-*.parquet` and never contacts the
Hugging Face Hub. The raw inputs each builder needs are listed in
[`docs/getting-started/datasets.md`](docs/getting-started/datasets.md), and each
script's options in [`scripts/README.md`](scripts/README.md).

## Training

KnowHyper experiments are launched through the repository root's scripts (see
the root README). The bare framework is driven by `ieforge train` / `ieforge
eval` with Hydra overrides, e.g.:

```bash
uv run ieforge train dataset=blurb_ner_bc2gm model=pubmedbert_token_classifier
```

## License

MIT (see `LICENSE`).
