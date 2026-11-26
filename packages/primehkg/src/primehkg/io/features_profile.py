"""Features profile resolution for ``primehkg-features-build`` (T2 / #203)."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

CORE_MANIFEST_SOURCE_TO_ARG: dict[str, str] = {
    "enrich_uniprot_tsv": "uniprot_tsv",
    "enrich_chembl_tsv": "chembl_tsv",
    "enrich_mondo_tsv": "mondo_tsv",
    "enrich_hpo_tsv": "hpo_tsv",
    "enrich_hgnc_tsv": "hgnc_tsv",
    "primekg_drug_features_tab": "drug_features_tab",
    "primekg_disease_features_tab": "disease_features_tab",
    "primekg_kg": "kg_csv",
}

INPUT_DIR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "uniprot_tsv": ("uniprot.tsv",),
    "chembl_tsv": ("chembl.tsv", "chembl_drugs.tsv"),
    "drugbank_smiles_tsv": ("drugbank_smiles.tsv",),
    "drugbank_properties_tsv": ("drugbank_properties.tsv",),
    "mondo_tsv": ("mondo.tsv", "mondo_map.tsv"),
    "hpo_tsv": ("hpo.tsv", "hpo_map.tsv"),
    "structure_skeleton_json": ("structure_skeleton.json",),
    "drug_features_tab": ("drug_features.tab",),
    "disease_features_tab": ("disease_features.tab",),
    "kg_csv": ("kg.csv",),
    "hgnc_tsv": ("hgnc.tsv",),
    "nucleotide_tsv": ("nucleotide_map.tsv",),
    "go_tsv": ("go_map.tsv", "go.tsv"),
    "pathway_tsv": ("pathway_map.tsv", "pathway.tsv"),
    "anatomy_tsv": ("anatomy_map.tsv", "anatomy.tsv"),
    "chebi_definitions_tsv": ("chebi_definitions.tsv",),
    "exposure_tsv": ("exposure_map.tsv", "exposure.tsv"),
}


class FeaturesProfile(StrEnum):
    """Named features-build presets aligned with publish profile v1."""

    NONE = "none"
    REFERENCE = "reference"
    EXTENDED = "extended"
    FULL_V1_P1 = "full-v1-p1"
    FULL_V1 = "full-v1"


@dataclass(frozen=True)
class FeaturesBuildArgs:
    """Resolved CLI inputs for ``primehkg-features-build``."""

    out_dir: Path
    core_dir: Path | None = None
    uniprot_tsv: Path | None = None
    chembl_tsv: Path | None = None
    drugbank_smiles_tsv: Path | None = None
    drugbank_properties_tsv: Path | None = None
    mondo_tsv: Path | None = None
    mondo_obo: Path | None = None
    hpo_tsv: Path | None = None
    drug_features_tab: Path | None = None
    disease_features_tab: Path | None = None
    kg_csv: Path | None = None
    hgnc_tsv: Path | None = None
    nucleotide_tsv: Path | None = None
    go_tsv: Path | None = None
    pathway_tsv: Path | None = None
    anatomy_tsv: Path | None = None
    chebi_definitions_tsv: Path | None = None
    unichem_cid_map_tsv: Path | None = None
    exposure_tsv: Path | None = None
    structure_skeleton_json: Path | None = None
    emit_manifest: bool = True
    features_profile: str = "none"
    features_input_dir: Path | None = None
    strict_coverage: bool = False
    text_provenance_registry: Path | None = None
    strict_text_provenance: bool = False
    emit_text_provenance_inventory: bool = True


def parse_features_profile(raw: str) -> FeaturesProfile:
    """Parse ``--features-profile`` (default ``none``)."""
    key = raw.strip().lower()
    try:
        return FeaturesProfile(key)
    except ValueError as exc:
        allowed = ", ".join(p.value for p in FeaturesProfile)
        msg = f"unknown features profile {raw!r}; expected one of: {allowed}"
        raise ValueError(msg) from exc


def _load_core_manifest_sources(core_dir: Path) -> dict[str, Path]:
    manifest_path = core_dir / "build_manifest.json"
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


def _resolve_from_input_dir(input_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for arg_name, filenames in INPUT_DIR_CANDIDATES.items():
        for filename in filenames:
            candidate = input_dir / filename
            if candidate.is_file():
                out[arg_name] = candidate
                break
    return out


def _merge_path(
    current: Path | None, resolved: dict[str, Path], key: str
) -> Path | None:
    if current is not None:
        return current
    return resolved.get(key)


def apply_features_profile(args: FeaturesBuildArgs) -> FeaturesBuildArgs:
    """Resolve TSV paths from core manifest and optional input dir."""
    profile = parse_features_profile(args.features_profile)
    if profile is FeaturesProfile.NONE:
        return args

    if args.core_dir is None:
        msg = f"features profile {profile.value!r} requires --core-dir"
        raise ValueError(msg)

    from_manifest = _load_core_manifest_sources(args.core_dir)
    from_input_dir: dict[str, Path] = {}
    if args.features_input_dir is not None and profile in (
        FeaturesProfile.EXTENDED,
        FeaturesProfile.FULL_V1_P1,
        FeaturesProfile.FULL_V1,
    ):
        from_input_dir = _resolve_from_input_dir(args.features_input_dir)

    # Explicit --features-input-dir wins over the core build manifest: a user pointing the
    # features build at an exports dir is designating it authoritative for feature inputs,
    # whereas the core manifest's enrich_* sources are build-lineage records (e.g.
    # enrich_uniprot_tsv -> uniprot_human_reviewed.tsv, a reviewed sequence export with no
    # Length/Mass columns -> would yield 0 protein property rows). See
    # docs/measurements/2026-06-13-variant-node-features-property-0.8.2-shark.md.
    merged = {**from_manifest, **from_input_dir}

    if profile is FeaturesProfile.FULL_V1:
        missing = [
            key
            for key in ("go_tsv", "pathway_tsv", "anatomy_tsv", "exposure_tsv")
            if _merge_path(getattr(args, key), merged, key) is None
        ]
        if missing:
            msg = (
                f"features profile full-v1 requires {missing}; "
                "set --features-input-dir or explicit --go-tsv / "
                "--pathway-tsv / --anatomy-tsv / --exposure-tsv"
            )
            raise ValueError(msg)

    return replace(
        args,
        uniprot_tsv=_merge_path(args.uniprot_tsv, merged, "uniprot_tsv"),
        chembl_tsv=_merge_path(args.chembl_tsv, merged, "chembl_tsv"),
        drugbank_smiles_tsv=_merge_path(
            args.drugbank_smiles_tsv, merged, "drugbank_smiles_tsv"
        ),
        drugbank_properties_tsv=_merge_path(
            args.drugbank_properties_tsv, merged, "drugbank_properties_tsv"
        ),
        mondo_tsv=_merge_path(args.mondo_tsv, merged, "mondo_tsv"),
        hpo_tsv=_merge_path(args.hpo_tsv, merged, "hpo_tsv"),
        drug_features_tab=_merge_path(
            args.drug_features_tab, merged, "drug_features_tab"
        ),
        disease_features_tab=_merge_path(
            args.disease_features_tab, merged, "disease_features_tab"
        ),
        kg_csv=_merge_path(args.kg_csv, merged, "kg_csv"),
        hgnc_tsv=_merge_path(args.hgnc_tsv, merged, "hgnc_tsv"),
        nucleotide_tsv=_merge_path(args.nucleotide_tsv, merged, "nucleotide_tsv"),
        go_tsv=_merge_path(args.go_tsv, merged, "go_tsv"),
        pathway_tsv=_merge_path(args.pathway_tsv, merged, "pathway_tsv"),
        anatomy_tsv=_merge_path(args.anatomy_tsv, merged, "anatomy_tsv"),
        chebi_definitions_tsv=_merge_path(
            args.chebi_definitions_tsv, merged, "chebi_definitions_tsv"
        ),
        exposure_tsv=_merge_path(args.exposure_tsv, merged, "exposure_tsv"),
        structure_skeleton_json=_merge_path(
            args.structure_skeleton_json, merged, "structure_skeleton_json"
        ),
    )


def features_profile_manifest_builder(
    profile: FeaturesProfile,
    *,
    core_dir: Path | None,
    features_input_dir: Path | None,
) -> dict[str, Any]:
    """One ``builders[]`` entry recording the features profile."""
    params: dict[str, Any] = {"profile": profile.value}
    if core_dir is not None:
        params["core_dir"] = str(core_dir.resolve())
    if features_input_dir is not None:
        params["features_input_dir"] = str(features_input_dir.resolve())
    return {"name": "features_profile", "version": 1, "params": params}


def has_features_input(args: FeaturesBuildArgs) -> bool:
    return any(
        getattr(args, field) is not None
        for field in (
            "uniprot_tsv",
            "chembl_tsv",
            "drugbank_smiles_tsv",
            "drugbank_properties_tsv",
            "mondo_tsv",
            "hpo_tsv",
            "drug_features_tab",
            "disease_features_tab",
            "kg_csv",
            "hgnc_tsv",
            "nucleotide_tsv",
            "go_tsv",
            "pathway_tsv",
            "anatomy_tsv",
            "chebi_definitions_tsv",
            "exposure_tsv",
            "structure_skeleton_json",
        )
    )
