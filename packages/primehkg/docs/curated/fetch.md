# Curated sources: automated fetch → TSV conversion

PrimeHKG can ingest curated n-ary sources (CORUM / Reactome) via `primehkg-build`:

- CORUM complexes → `protein_complex` hyperedges
- Reactome reaction events → `reaction_event` hyperedges (with roles)

This page documents the **official** automation path: download upstream files, convert to the PrimeHKG TSV contracts, and run `primehkg-build`.

## Licenses / redistribution

You are responsible for complying with upstream licensing and redistribution terms.
PrimeHKG’s default workflows are local-first; Hub publication is optional.

## 1) Fetch + convert

This CLI downloads upstream sources and writes PrimeHKG-ready TSVs.

```bash
uv run primehkg-curated-fetch \
  --out-dir /path/to/curated \
  --corum-version "corum:unknown" \
  --reactome-version "reactome:current"
```

If CORUM download fails with SSL verification errors (common on some clusters), use:

```bash
uv run primehkg-curated-fetch \
  --out-dir /path/to/curated \
  --insecure-ssl \
  --corum-version "corum:unknown" \
  --reactome-version "reactome:current"
```

Outputs under `--out-dir`:

- `corum.tsv` (tab-separated): `complex_id`, `uniprot_id`, `complex_name`
- `reactome.tsv` (tab-separated): `reaction_id`, `entity_id`, `role`
- `sources.json` (provenance): URLs + SHA256 digests + timestamps

## 2) Build a DatasetDict with curated hyperedges

```bash
uv run primehkg-build \
  --kg-csv /path/to/kg.csv \
  --out-dir /path/to/datasetdict \
  --corum-tsv /path/to/curated/corum.tsv \
  --corum-version "corum:unknown" \
  --reactome-tsv /path/to/curated/reactome.tsv \
  --reactome-version "reactome:current" \
  --emit-build-report
```

## TSV contracts (PrimeHKG inputs)

### CORUM (`corum.tsv`)

Header:

- `complex_id` (string)
- `uniprot_id` (string)
- `complex_name` (optional; may be empty)

Each row is one complex participant.

### Reactome (`reactome.tsv`)

Header:

- `reaction_id` (string; stable reaction identifier)
- `entity_id` (string; UniProt accession or `CHEBI:xxxx`)
- `role` (string; one of `input`, `output`, `catalyst`, `cofactor`, `positive_regulator`, `negative_regulator`)

Default automation uses Reactome’s `reactome_reaction_exporter.txt` (mostly protein-centric).

## Reproducibility notes

`sources.json` records:
- download URL
- SHA256 digest of the downloaded payload
- conversion timestamp

This is intended to be copied into build provenance (or stored alongside a build manifest).
