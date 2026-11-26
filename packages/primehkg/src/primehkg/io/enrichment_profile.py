"""Enrichment profile resolution for ``primehkg-build`` (T1 / #197)."""

from __future__ import annotations

import json
import pathlib
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from primehkg.io.build_cli import BuildCliArgs

REFERENCE_DISEASE_FEATURES = "disease_features.tab"
REFERENCE_DRUG_FEATURES = "drug_features.tab"
REFERENCE_NODES_TAB = "nodes.tab"

CORE_MANIFEST_SOURCE_TO_ARG: dict[str, str] = {
    "enrich_hgnc_tsv": "enrich_hgnc_tsv",
    "enrich_uniprot_tsv": "enrich_uniprot_tsv",
    "enrich_mondo_tsv": "enrich_mondo_tsv",
    "enrich_hpo_tsv": "enrich_hpo_tsv",
    "enrich_chembl_tsv": "enrich_chembl_tsv",
    "enrich_drugbank_tsv": "enrich_drugbank_tsv",
    "enrich_rxnorm_tsv": "enrich_rxnorm_tsv",
}

ENRICHMENT_INPUT_DIR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "enrich_hgnc_tsv": ("hgnc.tsv",),
    "enrich_hgnc_json": ("hgnc.json",),
    "enrich_uniprot_tsv": ("uniprot.tsv", "uniprot_human.tsv"),
    "enrich_mondo_tsv": ("mondo_map.tsv", "mondo.tsv"),
    "enrich_mondo_obo": ("mondo.obo",),
    "enrich_hpo_tsv": ("hpo_map.tsv", "hpo.tsv"),
    "enrich_chembl_tsv": ("chembl_drugs.tsv", "chembl.tsv"),
    "enrich_drugbank_tsv": ("drugbank_map.tsv", "drugbank.tsv"),
    "enrich_rxnorm_tsv": ("rxnorm_map.tsv", "rxnorm.tsv"),
}

EXPLICIT_ONLY_ENRICH_ARGS = frozenset(
    {
        "enrich_go_tsv",
        "enrich_pathway_tsv",
        "enrich_anatomy_tsv",
    }
)


class EnrichmentProfile(StrEnum):
    """Named node enrichment presets for ``primehkg-build``."""

    NONE = "none"
    REFERENCE = "reference"
    EXTENDED = "extended"


def parse_enrichment_profile(raw: str) -> EnrichmentProfile:
    """Parse ``--enrichment-profile`` (default ``none``)."""
    key = raw.strip().lower()
    try:
        return EnrichmentProfile(key)
    except ValueError as exc:
        allowed = ", ".join(p.value for p in EnrichmentProfile)
        msg = f"unknown enrichment profile {raw!r}; expected one of: {allowed}"
        raise ValueError(msg) from exc


def _resolve_file(base: pathlib.Path, name: str, current: Path | None) -> Path | None:
    if current is not None:
        return current
    candidate = base / name
    return candidate if candidate.is_file() else None


def _resolve_from_input_dir(input_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for arg_name, filenames in ENRICHMENT_INPUT_DIR_CANDIDATES.items():
        for filename in filenames:
            candidate = input_dir / filename
            if candidate.is_file():
                out[arg_name] = candidate
                break
    return out


def _load_manifest_enrich_sources(manifest_dir: Path) -> dict[str, Path]:
    manifest_path = manifest_dir / "build_manifest.json"
    if not manifest_path.is_file():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = data.get("sources")
    if not isinstance(sources, list):
        return {}
    out: dict[str, Path] = {}
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        path = entry.get("path")
        if (
            isinstance(name, str)
            and isinstance(path, str)
            and name in CORE_MANIFEST_SOURCE_TO_ARG
        ):
            candidate = Path(path)
            if candidate.is_file():
                out[CORE_MANIFEST_SOURCE_TO_ARG[name]] = candidate
    return out


def _apply_reference_tier(args: BuildCliArgs) -> BuildCliArgs:
    primekg_dir = (
        args.primekg_dir if args.primekg_dir is not None else args.kg_csv.parent
    )
    disease_tab = _resolve_file(
        primekg_dir,
        REFERENCE_DISEASE_FEATURES,
        args.enrich_primekg_disease_features_tab,
    )
    drug_tab = _resolve_file(
        primekg_dir, REFERENCE_DRUG_FEATURES, args.enrich_primekg_drug_features_tab
    )
    nodes_tab = _resolve_file(
        primekg_dir, REFERENCE_NODES_TAB, args.enrich_primekg_nodes_tab
    )

    ingest_names = args.ingest_primekg_names
    if nodes_tab is not None:
        ingest_names = False

    return replace(
        args,
        primekg_dir=primekg_dir,
        enrich_primekg_disease_features_tab=disease_tab,
        enrich_primekg_drug_features_tab=drug_tab,
        enrich_primekg_nodes_tab=nodes_tab,
        ingest_primekg_names=ingest_names,
    )


def apply_enrichment_profile(args: BuildCliArgs) -> BuildCliArgs:
    """Resolve Reference / Extended enrichment paths for ``primehkg-build``."""
    profile = parse_enrichment_profile(args.enrichment_profile)
    if profile is EnrichmentProfile.NONE:
        return args

    args = _apply_reference_tier(args)
    if profile is not EnrichmentProfile.EXTENDED:
        return args

    from_manifest = _load_manifest_enrich_sources(args.out_dir)
    from_input_dir: dict[str, Path] = {}
    if args.enrichment_input_dir is not None:
        from_input_dir = _resolve_from_input_dir(args.enrichment_input_dir)

    merged = {**from_input_dir, **from_manifest}
    updates: dict[str, Path] = {}
    for key in ENRICHMENT_INPUT_DIR_CANDIDATES:
        if key in EXPLICIT_ONLY_ENRICH_ARGS:
            continue
        if getattr(args, key) is not None:
            continue
        resolved = merged.get(key)
        if resolved is not None:
            updates[key] = resolved

    if not updates:
        return args
    return replace(args, **updates)


def enrichment_profile_manifest_builder(
    profile: EnrichmentProfile,
    *,
    primekg_dir: Path | None,
    enrichment_input_dir: Path | None = None,
) -> dict[str, object]:
    """One ``builders[]`` entry recording the enrichment profile."""
    params: dict[str, object] = {"profile": profile.value}
    if primekg_dir is not None:
        params["primekg_dir"] = str(primekg_dir.resolve())
    if enrichment_input_dir is not None:
        params["enrichment_input_dir"] = str(enrichment_input_dir.resolve())
    return {"name": "enrichment_profile", "version": 1, "params": params}
