# PrimeHKG

PrimeHKG builds a biomedical knowledge hypergraph from PrimeKG and curated
sources (CORUM, Complex Portal, Reactome, SIGNOR, CTD, DrugComb, STITCH). One
typed hyperedge holds a whole multi-participant event, and each entity's text,
SMILES and protein-sequence attributes are separate tables. It also provides
the entity linking and sub-hypergraph retrieval that KnowHyper uses.

This copy is PrimeHKG `v0.12.14`, the version the KnowHyper EACL 2027
experiments ran on (core `0.12.13`, features `0.12.15`).

## Rebuilding the hypergraph

The integrated hypergraph and its features are **not redistributed**. Rebuild
them from the original providers with **[`REPRODUCE.md`](REPRODUCE.md)**, which
lists every command, the sha256 of every input the paper's build read, and the
licence of each source.

The main command-line tools:

| Command | Role |
|---|---|
| `primehkg-primekg-download` | Download the PrimeKG files from Harvard Dataverse |
| `primehkg-curated-fetch` | Download and convert the curated sources ([`docs/curated/fetch.md`](docs/curated/fetch.md)) |
| `primehkg-build` | Build the core tables (`nodes`, `hyperedges`, `support`) and `build_manifest.json` |
| `primehkg-features-build`, `primehkg-embed-build` | Build the attribute tables and their vectors |
| `primehkg-hf-push --out-dir DIR` | Write a build to a local directory in the Hub layout (no upload) |

## Loading a local build

`load_core_dataset` accepts a local directory written by `primehkg-hf-push
--out-dir` in place of a Hub repo id. The Hub is not contacted, and the
fingerprint is checked against the directory's `build_manifest.json`:

```python
from primehkg import load_core_dataset

ds = load_core_dataset(
    hub_repo_id="./primehkg-core-hub",
    revision="0.12.13",                      # ignored for a local directory
    expected_fingerprint="<inputs_fingerprint of your build>",
)
```

KnowHyper reads the same directory through `PRIMEHKG_CORE` (see the repository
root README).

## License

The code is MIT (see `LICENSE`). Data obtained from the original providers
remains under their own terms; see `REPRODUCE.md`.
