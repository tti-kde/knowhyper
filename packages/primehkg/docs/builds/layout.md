# Build artifact layout

PrimeHKG separates **git-tracked code** from **local build outputs**. Fingerprints in
`build_manifest.json` → `inputs_fingerprint` are the canonical identity for comparing builds.

## Python path helpers

Single fingerprint dir convention (use in scripts and sidecar writers):

```python
from primehkg.io.build_artifacts import (
    GRAPH_CSR_SIDECAR_NAME,
    build_snapshot_dir,
    build_snapshot_sidecar_dir,
)

fp = "…"  # from build_manifest.json
snapshot = build_snapshot_dir(fp)  # artifacts/builds/<fp>/
csr = build_snapshot_sidecar_dir(fp, GRAPH_CSR_SIDECAR_NAME)
```

## Snapshot tree (git-free, recommended)

After `primehkg-build`:

```
artifacts/builds/<inputs_fingerprint>/
  build_manifest.json       # sources, builders, counts, optional sidecars map
  build_report.json         # audit metrics (with --emit-build-report)
  snapshot_meta.json        # optional human label (not in fingerprint)
  sidecars/
    graph_csr_v1/           # optional (--emit-graph-csr-sidecar)
```

The CSR sidecar is written here by the build CLI when `--emit-graph-csr-sidecar` is set.
`build_manifest.json` → `sidecars.graph_csr_v1` records both the artifacts path and the
dataset-relative mirror path.

## Build output directory (`--out-dir`)

A single `primehkg-build` run also writes:

```
<out-dir>/
  nodes/                    # HF DatasetDict split
  hyperedges/
  support/
  build_manifest.json
  build_report.json         # optional (--emit-build-report)
  sidecars/graph_csr_v1/    # optional mirror of artifacts sidecar
  builder/                  # optional (--use-builder-cache)
    <builder_name>/<cache_key>/
      candidates.parquet
      facts.parquet
      support.parquet
      stats.json
```

## Workflow

1. Build: `uv run primehkg-build --kg-csv … --out-dir /tmp/dd --emit-build-report [--emit-graph-csr-sidecar]`
2. Compare fingerprints and `build_report.json` under `artifacts/builds/<fp>/`
