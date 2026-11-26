# Rebuilding PrimeHKG core 0.12.13 and features 0.12.15

The KnowHyper paper (EACL 2027) used:

| Artifact | Release | `inputs_fingerprint` | Size |
|----------|---------|----------------------|------|
| Core hypergraph | **0.12.13** | `00e2c7ea123617ab2ab1a8880756255a766ded479261bb37c9610919aa4c9c04` | 137,587 nodes / 4,983,742 hyperedges / 4,983,742 support rows |
| Features (source tables) | 0.12.14 tables | `71e8bd1e3a64832652c92e199ffd61c48a6a7dcee393a1f31169538c1468008f` | text 293,373 / sequence 27,205 / property 432,176 rows |
| Features (attribute embeddings) | **0.12.15** | same features fingerprint | `node_features_embedding` 500,495 rows |
| Text features (R2a text baseline) | 0.12.10 | not covered here | — |

This directory is the PrimeHKG **v0.12.14** package. Core 0.12.13 was built by the same
builder code (v0.12.14 changed only features / entity-view code), and v0.12.14 is the
runtime KnowHyper ran.

**The integrated graph, features and embeddings are not redistributed.** Each provider has
its own terms (see [Source licences](#source-licences)). You rebuild locally from the
original providers with the commands below.

## What will and will not reproduce

- **`inputs_fingerprint` will not equal `00e2c7ea…`.** The fingerprint hashes the
  manifest's `sources[]` *including each input's absolute path*, and builder parameters
  that record absolute directories (`enrichment_profile.params.enrichment_input_dir`,
  `primekg_dir`). A rebuild on another machine always gets a new fingerprint even with
  byte-identical inputs. Compare inputs with the per-input sha256 table below, and
  outputs with the expected counts.
- **Several providers serve only a "current" release** (CTD, ChEBI, HGNC, MONDO, HPO, GO,
  UniProt, Reactome, SIGNOR, Complex Portal, UniChem, ChEMBL). A rebuild today will
  download newer files than in July 2026, so the sha256s and some counts will differ.
  PrimeKG (Harvard Dataverse DOI 10.7910/DVN/IXA7BM), STITCH v5.0, STRING aliases and
  DrugComb v1.4 are versioned and should match.
- The 0.12.13 manifest records `"unknown"` for most `*_version` parameters
  (e.g. `ctd_version: ctd:unknown`, `stitch_version: stitch:unknown`), and
  `chebi_synonyms_version: chebi:2026-07-07`. The recipe below therefore passes no version
  flags except `--chebi-synonyms-version`, so the recorded builder parameters match.

## 0. Environment

```bash
cd packages/primehkg
uv sync                    # runtime CLIs
uv sync --group embed      # + torch / transformers / sentence-transformers (embeddings, label sidecar)
```

A full core build needs a large-memory host (tens of GB of RAM; the reference builds used
a server-class host). Embedding builds want a GPU (`--devices auto`), but run on CPU.

## 1. Provider downloads

```bash
mkdir -p data/primekg data/exports out/curated

# 1a. PrimeKG (Harvard Dataverse, CC0)
uv run primehkg-primekg-download --out-dir data/primekg \
  --files kg.csv,nodes.tab,disease_features.tab,drug_features.tab

# 1b. Extended export maps (MONDO/HPO/GO/pathway/anatomy/exposure maps from PrimeKG +
#     UniProt, ChEMBL, HGNC complete set JSON, MONDO OBO, HPO OBO, NCBI RefSeq).
#     30–60+ min (ChEMBL per-drug lookups, UniProt idmapping).
scripts/download-primehkg-extended-exports.sh --primekg-dir data/primekg --out-dir data/exports
#   For the features build you also need DrugBank-derived SMILES/properties, which require
#   your own DrugBank licence and full-database XML:
#   uv run primehkg-export-mappings --primekg-dir data/primekg --out-dir data/exports \
#       --drugbank-xml /path/to/drugbank_full_database.xml

# 1c. Curated sources: CORUM, Reactome (content service, CHEBI participants), SIGNOR,
#     Complex Portal, DrugComb v1.4, STITCH v5.0 + UniChem + STRING aliases, CTD,
#     ChEBI names / definitions / synonyms.
uv run primehkg-curated-fetch --out-dir out/curated \
  --reactome-mode content-service \
  --emit-stitch --emit-ctd \
  --emit-chebi-names --emit-chebi-definitions --emit-chebi-synonyms
#   Add --insecure-ssl if your cluster CA bundle breaks HTTPS (see docs/curated/fetch.md).
#   ctdbase.org often truncates the ~42 MB chem-gene download; if the fetcher reports
#   0 rows, pre-fetch with a resuming download (curl -C -) and re-run.
```

`--enrich-uniprot-tsv` (step 2) expects a **reviewed human UniProtKB (Swiss-Prot)** TSV
with at least `Entry`, `Gene Names` and `GeneID` columns (file name used for 0.12.13:
`uniprot_human_reviewed.tsv`). The exact UniProt query of the 0.12.13 file was not
recorded; a REST export such as
`https://rest.uniprot.org/uniprotkb/stream?format=tsv&query=reviewed:true+AND+organism_id:9606&fields=accession,gene_names,xref_geneid`
has the required columns. Without this flag, curated UniProt-keyed proteins are not
unified onto PrimeKG GeneID nodes and ~13K extra `primekg:protein:<UniProt>` nodes appear.

## 2. Core build (0.12.13 recipe)

```bash
uv run primehkg-build \
  --kg-csv data/primekg/kg.csv \
  --out-dir out/primehkg_core_01213 \
  --build-profile primehkg \
  --enrichment-profile extended \
  --enrichment-input-dir data/exports \
  --enrich-go-tsv data/exports/go_map.tsv \
  --enrich-pathway-tsv data/exports/pathway_map.tsv \
  --enrich-anatomy-tsv data/exports/anatomy_map.tsv \
  --enrich-uniprot-tsv data/exports/uniprot_human_reviewed.tsv \
  --enrich-hgnc-json data/exports/hgnc.json \
  --enrich-mondo-obo data/exports/mondo.obo \
  --corum-tsv out/curated/corum.tsv \
  --reactome-tsv out/curated/reactome.tsv \
  --signor-tsv out/curated/signor.tsv \
  --complexportal-tsv out/curated/complexportal.tsv \
  --drugcomb-summary-tsv out/curated/drugcomb_summary.csv \
  --drugcomb-identifiers-tsv out/curated/drugcomb_drug_identifiers.tsv \
  --stitch-tsv out/curated/stitch.tsv \
  --unichem-cid-map-tsv out/curated/unichem_cid_map.tsv \
  --string-ensp-geneid-tsv out/curated/string_ensp_geneid.tsv \
  --stitch-score-threshold 700 \
  --ctd-tsv out/curated/ctd.tsv \
  --ctd-mesh-cid-tsv out/curated/ctd_mesh_cid.tsv \
  --chebi-names-tsv out/curated/chebi_names.tsv \
  --chebi-definitions-tsv out/curated/chebi_definitions.tsv \
  --chebi-synonyms-tsv out/curated/chebi_synonyms.tsv \
  --chebi-synonyms-version "chebi:<download date>" \
  --emit-build-report \
  --emit-inverted-index-sidecar \
  --emit-he-row-index-sidecar \
  --emit-graph-csr-sidecar \
  --emit-label-embedding-sidecar          # SapBERT (cambridgeltl/SapBERT-from-PubMedBERT-fulltext); needs --group embed
```

`--enrichment-profile extended` auto-resolves `nodes.tab` / feature tabs from the
`kg.csv` directory and `hgnc.tsv`, `mondo_map.tsv`, `hpo_map.tsv`, `chembl_drugs.tsv`
from `--enrichment-input-dir`. ClinVar is not part of the default (variant-free)
`primehkg` profile and was not used. Before building, you can check that no input of the
reference build is missing from your argv:

```bash
uv run python scripts/check-build-sources.py --against <dir with the 0.12.13 build_manifest.json> -- <your primehkg-build argv>
```

### Post-build sidecars

```bash
# chemical_xref_v1 (lossless CTD MeSH->PubChem and UniChem PubChem xrefs; both TSVs are
# written by `primehkg-curated-fetch --emit-ctd --emit-stitch`)
uv run primehkg-chemical-xref-build --core-dir out/primehkg_core_01213 \
  --ctd-mesh-pubchem-v2 out/curated/ctd_mesh_pubchem_v2.tsv \
  --unichem-pubchem-xref-v2 out/curated/unichem_pubchem_xref_v2.tsv

# vector_node_v1 / vector_hyperedge_v1 (PubMedBERT display-name and hyperedge-sentence
# vectors; optional, used by vector retrieval only)
uv run primehkg-he-text-build --core-dir out/primehkg_core_01213 --out-dir out/he_text
uv run primehkg-embed-build --core-dir out/primehkg_core_01213 --he-text-dir out/he_text \
  --targets node,hyperedge --node-input-kinds text --encoder transformers \
  --text-model-id NeuML/pubmedbert-base-embeddings --devices auto \
  --cache-dir out/embed_cache --out-dir out/embed_core
uv run primehkg-vector-sidecar-build --embedding-dir out/embed_core --target node \
  --core-dir out/primehkg_core_01213 --input-kind text --source-modality display_name \
  --model-id NeuML/pubmedbert-base-embeddings --preprocess-version text_nfkc_strip_v1 \
  --out-dir out/primehkg_core_01213/sidecars/vector_node_v1
uv run primehkg-vector-sidecar-build --embedding-dir out/embed_core --target hyperedge \
  --core-dir out/primehkg_core_01213 --source-template-id natural_sentence_v1 \
  --model-id NeuML/pubmedbert-base-embeddings --preprocess-version text_nfkc_strip_v1 \
  --out-dir out/primehkg_core_01213/sidecars/vector_hyperedge_v1
```

(If a sidecar CLI does not add itself to `build_manifest.json` `sidecars{}`, add the entry
by hand, matching the layout in `docs/builds/layout.md`.)

### Expected core output (0.12.13)

| Metric | Value |
|--------|------:|
| `outputs.nodes` / `hyperedges` / `support` | 137,587 / 4,983,742 / 4,983,742 |
| `effective_segments` | complexportal, corum, ctd, disease_protein_bundle, drug_disease_shared_target_bundle, drugcomb, moa_triad, phenotype, protein_disease_bundle, protein_protein_bundle, reactome, registry_atomic, signor, stitch |
| Hyperedges per builder segment | registry_atomic 3,834,963 · moa_triad 557,198 · phenotype 215,101 · ctd 178,692 · stitch 67,945 · drugcomb 33,440 · signor 23,903 · drug_disease_shared_target_bundle 19,238 · protein_protein_bundle 16,977 · reactome 16,008 · corum 7,743 · protein_disease_bundle 7,454 · disease_protein_bundle 3,405 · complexportal 2,187 |
| Genuine `nodes.synonyms` coverage | 37,817 / 137,587 (metabolite 3,482, protein 20,899, disease 11,372, drug 2,064) |
| `graph_csr_v1` connections | 10,892,258 |
| `label_embedding_v1` | 492,490 vectors, dim 768 |
| `chemical_xref_v1` rows | 393,746 |

### Per-input sha256 of the 0.12.13 build (from its `build_manifest.json`)

| Manifest source | sha256 |
|-----------------|--------|
| `chebi_definitions_tsv` | `07bd5c0a77216dba28e47a6fc645adbf3ef83af7d12b9d37f0cb646f3c452c74` |
| `chebi_names_tsv` | `effd9d864e31386e358749c217b76507528809f14e56147c5e776bcd75c69e20` |
| `chebi_synonyms_tsv` | `ae7856545473aadd10531dfd03b5c3f5c179a4dc844edad0b520d7abdea6c38c` |
| `complexportal_tsv` | `29d3aeec7ede35e8686de3cf73620b5871c3e68d7ca069385153a9b28e80012a` |
| `corum_tsv` | `f4acbf79229a741213a6af654db8881e080bb91d893f2f5551892398f01b4fa8` |
| `ctd_mesh_cid_tsv` | `129b89b2164209215f7d07d1f784e5546e6ac84daa985e9e5c2e5ebce00736a9` |
| `ctd_tsv` | `327035edac75cad2682c9913c2373936a108f2051d8c7283cd817d5d26fbea67` |
| `drugcomb_identifiers_tsv` | `2cb9ee0e71666b6effa21ef7a2b99d425c60966b698f6a6e5d7a3cc5c1590bdc` |
| `drugcomb_summary_tsv` | `e08e3d35bfa4bea011afe1b05b7025acde5c669c5e8d265866b5e12b08037a2f` |
| `enrich_anatomy_tsv` | `2c059a650edcdb2d2c42713f048802c3f5e9ed704e6689cd423a947b78dd4cc8` |
| `enrich_chembl_tsv` | `97fac21a111ddac0cb16cce5e876556f17136e1292e096e30d91ebab0f897286` |
| `enrich_go_tsv` | `c1de0a3d35401748223a8a23c2f19308a29ea18afee62e839d162b2c3b34954b` |
| `enrich_hgnc_json` | `091dde88c8ccac4b6d8998c4c7e2c07c6c459992bdbc88d9f22f0f0cd95a13df` |
| `enrich_hgnc_tsv` | `171cad9bea2ac9dae4e25d5ed2d2b0a8e8859be0499a249e3698de21333fbd1e` |
| `enrich_hpo_tsv` | `7b4b39e66c6bd73a15031882c4076b9e15fd5dad007d66435edfad343a2951c6` |
| `enrich_mondo_obo` | `fffd4c01bd56a5b105c3b752fcbb735335c6b18a0ae82f8f50cff9f8ab375217` |
| `enrich_mondo_tsv` | `145fef64e98421d4d91aeb970b0c18d2eb5016ed0aa5f8a6e54c54bda9c0e442` |
| `enrich_pathway_tsv` | `adcdc0aa33de192a6c9641211efb7a2ff5ebc39d94c7b429629a59f1c82e1951` |
| `enrich_uniprot_tsv` | `7b3b7d2869316823f93d180a864ef9f3d179ea49422c18a41bd8cb93f9f3621d` |
| `primekg_disease_features_tab` | `bdd18b39aae9c80e37c3716c2bfabaf854f6a5a7e44c8d2e5b082f0dd27124cb` |
| `primekg_drug_features_tab` | `d9db67a3ae9e4e1bea6f50ac55c9e3152c30f9a128b45cec0d5b003fabdc30c6` |
| `primekg_kg` | `0a5a010fcd88dcec572a573d98c26c1c244a11567092b6161d3ded611cfb5c69` |
| `primekg_nodes_tab` | `18ebcf24887e8529b95c7a7c96c81089cc7c5ba22d53a5c05d56480331961af8` |
| `reactome_tsv` | `0dff35296e2355766ed92224ccd1943dd3602315e2a3e25f385a299ffc1c042a` |
| `relation_registry_yaml` | `18a64e741ab2e925e0f025ef2e7b62c9e599e86fa5fe1070b12c7d7eefc3867f` (= packaged `src/primehkg/policy/relation_registry.yaml`) |
| `signor_tsv` | `c482d005bdd9ccd6b4d0942e086799d0ab2e3d087fd80e7f960a7ce97b35a525` |
| `stitch_tsv` | `097dc2dccde623df4120e68ebfe52777b16de551f689144966e86159e92cc4c3` |
| `string_ensp_geneid_tsv` | `ab5caac4bbf380fdead3c9394a3cd5834d7732f1ff9d68cae82286dd2a3318b5` |
| `unichem_cid_map_tsv` | `94c8786a3f7a3911831d916a00d269c5bfcfe34f467bb84bc7ebcb7450968847` |

Sidecar input digests recorded in `chemical_xref_v1/meta.json`: `ctd_mesh_pubchem_v2`
`551b851f11947b939076597a44b4cc012d8c59ed928b8be315c6f80e5b6ba3e6`,
`unichem_pubchem_xref_v2` `c42bb0c7f3059d5c356fed0f0f34f323ed1765a9aa9d6af87a6b88af6d9fbeef`.

## 3. Features build (0.12.14 tables = the tables under 0.12.15)

```bash
uv run primehkg-features-build \
  --core-dir out/primehkg_core_01213 \
  --features-profile full-v1 \
  --features-input-dir data/exports \
  --mondo-obo data/exports/mondo.obo \
  --chebi-definitions-tsv out/curated/chebi_definitions.tsv \
  --unichem-cid-map-tsv out/curated/unichem_cid_map.tsv \
  --out-dir out/primehkg_features_01214 \
  --emit-manifest
```

`full-v1` resolves `uniprot.tsv`, `chembl_drugs.tsv`, `drugbank_smiles.tsv`,
`drugbank_properties.tsv`, `mondo_map.tsv`, `hpo_map.tsv`, `hgnc.tsv`,
`nucleotide_map.tsv`, `go_map.tsv`, `pathway_map.tsv`, `anatomy_map.tsv`,
`exposure_map.tsv`, `drug_features.tab`, `disease_features.tab` and `kg.csv` from
`--features-input-dir` / the core manifest; explicit flags win. With `--core-dir` it also
emits `synonym_context` text rows (37,817) from core `nodes.synonyms`. Expect text
293,373 / sequence 27,205 / property 432,176 rows; `modality_coverage.pass` is `false`
in the reference build too (metabolite and protein below the 0.95 threshold), so do not
pass `--strict-coverage`.

Per-input sha256 of the reference features build (its `build_manifest.json`):

| Manifest source | sha256 |
|-----------------|--------|
| `anatomy_tsv` | `2c059a650edcdb2d2c42713f048802c3f5e9ed704e6689cd423a947b78dd4cc8` |
| `chebi_definitions_tsv` | `07bd5c0a77216dba28e47a6fc645adbf3ef83af7d12b9d37f0cb646f3c452c74` |
| `chembl_tsv` | `97fac21a111ddac0cb16cce5e876556f17136e1292e096e30d91ebab0f897286` |
| `disease_features_tab` | `bdd18b39aae9c80e37c3716c2bfabaf854f6a5a7e44c8d2e5b082f0dd27124cb` |
| `drug_features_tab` | `d9db67a3ae9e4e1bea6f50ac55c9e3152c30f9a128b45cec0d5b003fabdc30c6` |
| `drugbank_properties_tsv` | `06f512095dfe6206379110c3ef7070d71c6274a887dab59e657c3a6c87618f42` |
| `drugbank_smiles_tsv` | `096048823f763cb445374a97718f377b16432c0b75de0f89aaf089b7a2818e0b` |
| `exposure_tsv` | `94868802cda091dd92d2fb3fec60b749e4809d1fa224a243914370de13a3e03d` |
| `go_tsv` | `c1de0a3d35401748223a8a23c2f19308a29ea18afee62e839d162b2c3b34954b` |
| `hgnc_tsv` | `171cad9bea2ac9dae4e25d5ed2d2b0a8e8859be0499a249e3698de21333fbd1e` |
| `hpo_tsv` | `7b4b39e66c6bd73a15031882c4076b9e15fd5dad007d66435edfad343a2951c6` |
| `kg_csv` | `0a5a010fcd88dcec572a573d98c26c1c244a11567092b6161d3ded611cfb5c69` |
| `mondo_obo` | `fffd4c01bd56a5b105c3b752fcbb735335c6b18a0ae82f8f50cff9f8ab375217` |
| `mondo_tsv` | `145fef64e98421d4d91aeb970b0c18d2eb5016ed0aa5f8a6e54c54bda9c0e442` |
| `nucleotide_tsv` | `b73414b24146b6b80c2e98c01cf9787e55aa187de533da3c2d98019486c7e1a1` |
| `pathway_tsv` | `adcdc0aa33de192a6c9641211efb7a2ff5ebc39d94c7b429629a59f1c82e1951` |
| `primehkg_core_build` | `a9b3426dd8c6d70fab3a4c3327130f6894baefb394c8c3b0fcdbbfc87062cbd6` |
| `text_provenance_registry` | `b063aa6be60b175e743f97ab53b2056ad24037a4047d06bea242965736deb856` (= packaged `src/primehkg/policy/text_provenance_registry.yaml`) |
| `unichem_cid_map_tsv` | `94c8786a3f7a3911831d916a00d269c5bfcfe34f467bb84bc7ebcb7450968847` |
| `uniprot_tsv` | `5285fe14de31119b2e4c3935e12bfbdb5b1995c5910141b392192aec95de3ed0` |

Note: the features `uniprot_tsv` (full human UniProt export with `Length` / `Mass` /
`Function` / `Sequence`) is a different file from the core `enrich_uniprot_tsv`
(reviewed-only, gene mapping); both are needed.

## 4. Attribute embeddings (features 0.12.15)

**0.12.15 is not a code change in the features or embedding builders.** `git diff
v0.12.14 v0.12.15` touches only the inverted-index / he-row-index sidecar writers and
release scripts (#468: write index JSON via atomic replace so hard-linked build trees are
not clobbered, and a release gate for stale index fingerprints). 0.12.15 re-published
`node_features_embedding` from the unchanged 0.12.14 features tables because the 0.12.14
Hub push had replaced the embeddings with a text-only slice (dropping ChemBERTa / ESM2
rows). There is therefore nothing to port: run the v0.12.14 embedding builder with all
four node input kinds:

```bash
uv run primehkg-embed-build \
  --core-dir out/primehkg_core_01213 \
  --features-dir out/primehkg_features_01214 \
  --targets node \
  --node-input-kinds text,protein_sequence,smiles,property_vector \
  --encoder transformers \
  --text-model-id NeuML/pubmedbert-base-embeddings \
  --protein-model-id facebook/esm2_t6_8M_UR50D \
  --smiles-model-id DeepChem/ChemBERTa-77M-MLM \
  --preprocess-version text_nfkc_strip_v1 \
  --devices auto \
  --cache-dir out/embed_cache \
  --out-dir out/primehkg_embed_01215
```

`--property-model-id` keeps its default, which reuses the text encoder
(`NeuML/pubmedbert-base-embeddings`) for `property_numeric_v1`, as in 0.12.15. The
reference 0.12.15 table was assembled from two runs of this builder (one text run, one
protein / SMILES / property run) merged into one table; a single run with all four input
kinds produces the same row set. Expected `embedding_manifest.json`:

| Encoder | `source_modality` | Rows | Dim |
|---------|-------------------|-----:|----:|
| `NeuML/pubmedbert-base-embeddings` | display_name 137,587 · description 126,929 · definition 83,826 · synonym_context 37,817 · drug text fields (atc_1–4, category, group, half_life, indication, mechanism_of_action, pathway, pharmacodynamics, protein_binding, state) 44,801 | 430,960 | 768 |
| `NeuML/pubmedbert-base-embeddings` | property_numeric_v1 | 42,366 | 768 |
| `facebook/esm2_t6_8M_UR50D` | protein | 20,219 | 320 |
| `DeepChem/ChemBERTa-77M-MLM` | smiles | 6,950 | 384 |
| **Total** | | **500,495** | |

Nucleotide sequences (36 source rows) were not embedded. GPU encoding is not bit-exact
across hardware; expect small float differences, identical row keys.

## 5. Loading a local rebuild

Either load the build directory directly:

```python
from primehkg import load_core_dataset
ds = load_core_dataset("out/primehkg_core_01213", expected_fingerprint="<your fingerprint>")
```

or write it in the Hub layout (what a consumer that passes `hub_repo_id` + `revision`
expects) and point `hub_repo_id` at that directory:

```bash
uv run primehkg-hf-push --dataset-dir out/primehkg_core_01213 --out-dir out/hub/primehkg-core
uv run primehkg-hf-push --dataset-dir out/primehkg_features_01214 --out-dir out/hub/primehkg-features
```

```python
ds = load_core_dataset(hub_repo_id="out/hub/primehkg-core", revision="0.12.13",
                       expected_fingerprint="<your fingerprint>")   # revision ignored, logged
```

A local directory is read in place (the Hub is never contacted), it must contain
`build_manifest.json`, and `expected_fingerprint` is checked against it. `datasets`
caches an Arrow copy of the Parquet shards under `HF_DATASETS_CACHE` (about 3 GB for core).

## Source licences

You are responsible for complying with each provider's terms. Summary (check the
providers for the current text):

| Source | Licence / terms |
|--------|-----------------|
| PrimeKG | CC0 1.0; its constituent databases keep their own terms |
| CTD | Free for research use; cite CTD, and notify CTD before redistributing data or derived products (ctdbase.org legal notice) |
| STITCH v5.0 | CC BY 4.0 |
| DrugComb v1.4 | CC BY 4.0 |
| SIGNOR | CC BY 4.0 |
| Reactome | CC0 1.0 |
| CORUM | CC BY-NC 4.0 (non-commercial) |
| Complex Portal | CC0 1.0 |
| Also used for enrichment / features | ChEBI (CC BY 4.0), UniProt (CC BY 4.0), HGNC, MONDO, HPO, GO, UniChem, ChEMBL, NCBI RefSeq, STRING aliases (CC BY 4.0), DrugBank (requires your own licence; only for features SMILES / properties) |

Because of the non-commercial (CORUM), notification (CTD) and licensed (DrugBank) terms,
the integrated graph, features and embeddings are **not redistributed**; only the code
and this recipe are.
