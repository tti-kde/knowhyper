#!/usr/bin/env bash
# Download/build maintainer-owned extended export TSVs for PrimeHKG enrichment/features.
#
# Writes TSVs under data/exports/ (gitignored) from data/primekg/ plus public APIs:
#   mondo_map.tsv, hpo_map.tsv, go_map.tsv, pathway_map.tsv, anatomy_map.tsv,
#   uniprot.tsv, chembl_drugs.tsv, hgnc.tsv
#
# Usage (from repo root):
#   scripts/download-primehkg-extended-exports.sh
#   scripts/download-primehkg-extended-exports.sh --skip-network
#   scripts/download-primehkg-extended-exports.sh --out-dir /tmp/primehkg_exports
#
# See: docs/features/build.md (extended TSV filenames)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRIMEKG_DIR="${REPO_ROOT}/data/primekg"
OUT_DIR="${REPO_ROOT}/data/exports"

usage() {
  cat <<'USAGE'
Usage:
  scripts/download-primehkg-extended-exports.sh [OPTIONS]

Options:
  --primekg-dir PATH   PrimeKG bundle directory (default: data/primekg)
  --out-dir PATH       Export output directory (default: data/exports)
  --skip-network       Local-only: mondo/hpo/pathway/anatomy maps (no UniProt/ChEMBL/HGNC/OBO downloads)
  --skip-hpo-definitions  Skip HPO OBO download; keep hp_label only
  --max-drugs N        Cap ChEMBL lookups (smoke/debug)
  --force              Regenerate all exports even when outputs look complete
  --timeout-s SECONDS  Per-request timeout (default: 120)
  -h, --help           Show this help

Notes:
  Full run may take 30–60+ minutes (ChEMBL per-drug lookups + UniProt idmapping).
  Requires network access and a local PrimeKG bundle (uv run primehkg-primekg-download …).
USAGE
}

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --primekg-dir)
      PRIMEKG_DIR="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --skip-hpo-definitions | --max-drugs | --timeout-s | --force)
      EXTRA_ARGS+=("$1" "${2:-}")
      shift 2
      ;;
    --skip-network)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "{\"error\":\"unknown argument: $1\"}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${PRIMEKG_DIR}/kg.csv" ]]; then
  echo "{\"error\":\"missing ${PRIMEKG_DIR}/kg.csv — run primehkg-primekg-download first\"}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
uv run primehkg-export-mappings \
  --primekg-dir "${PRIMEKG_DIR}" \
  --out-dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"

echo "{\"type\":\"extended-exports-ready\",\"out_dir\":\"${OUT_DIR}\"}"
