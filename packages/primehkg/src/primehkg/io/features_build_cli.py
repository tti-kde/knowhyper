from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import typer
from datasets import DatasetDict, load_from_disk
from typer.main import get_command

from primehkg.features.build import (
    build_features_tables,
    finalize_text_provenance,
    write_features_dataset_to_disk,
)
from primehkg.features.chebi_drug_text import ChebiDrugProjectionStats
from primehkg.features.coverage import (
    compute_modality_coverage,
    compute_property_coverage,
)
from primehkg.features.provenance import (
    TEXT_PROVENANCE_SCHEMA,
    ProvenanceConflictError,
    StrictTextProvenanceError,
    provenance_inventory,
    registry_source_inventory,
    resolve_text_provenance_registry_path,
)
from primehkg.io.build_manifest import manifest_source_file, sha256_file
from primehkg.io.cli_console import cli_error
from primehkg.io.features_profile import (
    FeaturesBuildArgs,
    FeaturesProfile,
    apply_features_profile,
    features_profile_manifest_builder,
    has_features_input,
    parse_features_profile,
)

app = typer.Typer(
    rich_markup_mode="rich",
    help="Build optional primehkg-features tables from core nodes + enrichment TSVs.",
)

OUT_DIR_OPT = typer.Option(..., "--out-dir", help="Output DatasetDict directory.")
CORE_DIR_OPT = typer.Option(
    None,
    "--core-dir",
    exists=True,
    file_okay=False,
    dir_okay=True,
    help="Existing primehkg-core build (filters rows to its nodes split).",
)
UNIPROT_TSV_OPT = typer.Option(
    None,
    "--uniprot-tsv",
    exists=True,
    dir_okay=False,
    readable=True,
)
CHEMBL_TSV_OPT = typer.Option(
    None,
    "--chembl-tsv",
    exists=True,
    dir_okay=False,
    readable=True,
)
DRUGBANK_SMILES_TSV_OPT = typer.Option(
    None,
    "--drugbank-smiles-tsv",
    exists=True,
    dir_okay=False,
    readable=True,
    help="DrugBank SMILES gap-fill export (after --chembl-tsv primary rows).",
)
DRUGBANK_PROPERTIES_TSV_OPT = typer.Option(
    None,
    "--drugbank-properties-tsv",
    exists=True,
    dir_okay=False,
    readable=True,
    help="DrugBank physicochemical properties export (from XML).",
)
MONDO_TSV_OPT = typer.Option(
    None,
    "--mondo-tsv",
    exists=True,
    dir_okay=False,
    readable=True,
)
MONDO_OBO_OPT = typer.Option(
    None,
    "--mondo-obo",
    exists=True,
    dir_okay=False,
    readable=True,
    help="MONDO OBO file for disease definitions (requires --mondo-tsv).",
)
HPO_TSV_OPT = typer.Option(
    None,
    "--hpo-tsv",
    exists=True,
    dir_okay=False,
    readable=True,
)
DRUG_FEATURES_TAB_OPT = typer.Option(
    None,
    "--drug-features-tab",
    exists=True,
    dir_okay=False,
    readable=True,
    help="PrimeKG drug_features.tab (requires --kg-csv).",
)
DISEASE_FEATURES_TAB_OPT = typer.Option(
    None,
    "--disease-features-tab",
    exists=True,
    dir_okay=False,
    readable=True,
    help="PrimeKG disease_features.tab (requires --kg-csv).",
)
KG_CSV_OPT = typer.Option(
    None,
    "--kg-csv",
    exists=True,
    dir_okay=False,
    readable=True,
    help="PrimeKG kg.csv for drug tab node_index mapping.",
)
HGNC_TSV_OPT = typer.Option(
    None,
    "--hgnc-tsv",
    exists=True,
    dir_okay=False,
    readable=True,
)
NUCLEOTIDE_TSV_OPT = typer.Option(
    None,
    "--nucleotide-tsv",
    exists=True,
    dir_okay=False,
    readable=True,
    help="nucleotide_map.tsv (RefSeq RNA for non-coding protein nodes).",
)
GO_TSV_OPT = typer.Option(
    None,
    "--go-tsv",
    exists=True,
    dir_okay=False,
    readable=True,
)
PATHWAY_TSV_OPT = typer.Option(
    None,
    "--pathway-tsv",
    exists=True,
    dir_okay=False,
    readable=True,
)
ANATOMY_TSV_OPT = typer.Option(
    None,
    "--anatomy-tsv",
    exists=True,
    dir_okay=False,
    readable=True,
)
CHEBI_DEFINITIONS_TSV_OPT = typer.Option(
    None,
    "--chebi-definitions-tsv",
    exists=True,
    dir_okay=False,
    readable=True,
)
UNICHEM_CID_MAP_TSV_OPT = typer.Option(
    None,
    "--unichem-cid-map-tsv",
    exists=True,
    dir_okay=False,
    readable=True,
    help="UniChem drugbank↔chebi map for ChEBI definition projection onto drug nodes.",
)
EXPOSURE_TSV_OPT = typer.Option(
    None,
    "--exposure-tsv",
    exists=True,
    dir_okay=False,
    readable=True,
)
STRUCTURE_SKELETON_JSON_OPT = typer.Option(
    None,
    "--structure-skeleton-json",
    exists=True,
    dir_okay=False,
    readable=True,
    help="JSON from structure feature skeleton export.",
)
EMIT_MANIFEST_OPT = typer.Option(
    True,
    "--emit-manifest/--no-emit-manifest",
    help="Write build_manifest.json next to the features DatasetDict.",
)
FEATURES_PROFILE_OPT = typer.Option(
    "none",
    "--features-profile",
    help=(
        "Features preset: none (default), reference (resolve TSVs from core "
        "build_manifest.json), or extended (+ --features-input-dir filenames)."
    ),
)
FEATURES_INPUT_DIR_OPT = typer.Option(
    None,
    "--features-input-dir",
    exists=True,
    file_okay=False,
    dir_okay=True,
    help="Directory with standard export filenames (extended profile).",
)
STRICT_COVERAGE_OPT = typer.Option(
    False,
    "--strict-coverage/--no-strict-coverage",
    help="Exit 1 when modality coverage KPI fails (full-v1-p1 / full-v1 profiles).",
)
TEXT_PROVENANCE_REGISTRY_OPT = typer.Option(
    None,
    "--text-provenance-registry",
    exists=True,
    dir_okay=False,
    readable=True,
    help="YAML text provenance registry (default: packaged policy file).",
)
STRICT_TEXT_PROVENANCE_OPT = typer.Option(
    False,
    "--strict-text-provenance/--no-strict-text-provenance",
    help="Exit 1 on unknown, PENDING, or allow-without-release text provenance.",
)
EMIT_TEXT_PROVENANCE_INVENTORY_OPT = typer.Option(
    True,
    "--emit-text-provenance-inventory/--no-emit-text-provenance-inventory",
    help="Write text_provenance_inventory.json beside build_manifest.json.",
)

_MODALITY_COVERAGE_PROFILES = frozenset({"full-v1-p1", "full-v1"})


def _entity_definition_crosswalk_manifest(
    stats: ChebiDrugProjectionStats,
    *,
    unichem_cid_map_tsv: Path,
    chebi_definitions_tsv: Path,
) -> dict[str, Any]:
    return {
        "rule": "unambiguous_drugbank_chebi_only",
        "unichem_cid_map": {
            "path": str(unichem_cid_map_tsv.resolve()),
            "sha256": sha256_file(unichem_cid_map_tsv),
        },
        "chebi_definitions": {
            "path": str(chebi_definitions_tsv.resolve()),
            "sha256": sha256_file(chebi_definitions_tsv),
        },
        "kept": stats.kept,
        "dropped_ambiguous_drugbank": stats.dropped_ambiguous_drugbank,
        "dropped_ambiguous_chebi": stats.dropped_ambiguous_chebi,
        "dropped_missing_definition": stats.dropped_missing_definition,
        "dropped_missing_endpoint": stats.dropped_missing_endpoint,
    }


def _load_core_nodes(core_dir: Path) -> list[dict[str, Any]]:
    loaded = load_from_disk(str(core_dir))
    if not isinstance(loaded, DatasetDict):
        msg = f"expected DatasetDict at {core_dir!r}, got {type(loaded).__name__}"
        raise TypeError(msg)
    nodes_split = loaded["nodes"]
    return [dict(row) for row in nodes_split]


def _text_provenance_artifact_paths(args: FeaturesBuildArgs) -> dict[str, Path]:
    """Map registry ``source`` keys to build input paths for artifact digests."""
    paths: dict[str, Path] = {}
    if args.mondo_tsv is not None:
        paths["MONDO"] = (
            args.mondo_obo if args.mondo_obo is not None else args.mondo_tsv
        )
    if args.hpo_tsv is not None:
        paths["HPO"] = args.hpo_tsv
    if args.go_tsv is not None:
        paths["GO"] = args.go_tsv
    if args.chebi_definitions_tsv is not None:
        paths["chebi"] = args.chebi_definitions_tsv
    if args.hgnc_tsv is not None:
        paths["HGNC"] = args.hgnc_tsv
    if args.uniprot_tsv is not None:
        paths["UniProt"] = args.uniprot_tsv
    if args.pathway_tsv is not None:
        paths["pathway"] = args.pathway_tsv
    if args.anatomy_tsv is not None:
        paths["anatomy"] = args.anatomy_tsv
    if args.exposure_tsv is not None:
        paths["exposure"] = args.exposure_tsv
    # Drug and disease PrimeKG tabs share source="PrimeKG"; attach a digest only
    # when exactly one tab is present so rows never inherit the wrong file hash.
    primekg_tabs = [
        p for p in (args.drug_features_tab, args.disease_features_tab) if p is not None
    ]
    if len(primekg_tabs) == 1:
        paths["PrimeKG"] = primekg_tabs[0]
    return paths


def _run_build(args: FeaturesBuildArgs) -> None:
    profile = parse_features_profile(args.features_profile)
    try:
        resolved = apply_features_profile(args)
    except ValueError as exc:
        cli_error(str(exc))
        raise typer.Exit(2) from exc

    if not has_features_input(resolved):
        cli_error(
            "pass at least one of --uniprot-tsv, --chembl-tsv, --drugbank-smiles-tsv, "
            "--drugbank-properties-tsv, "
            "--mondo-tsv, --hpo-tsv, "
            "--hgnc-tsv, --nucleotide-tsv, --drug-features-tab, --disease-features-tab "
            "(with --kg-csv), "
            "--go-tsv, --pathway-tsv, --anatomy-tsv, --exposure-tsv, --structure-skeleton-json, "
            "or use --features-profile reference/extended/full-v1-p1/full-v1 "
            "with resolvable inputs in core build_manifest.json or "
            "--features-input-dir"
        )
        raise typer.Exit(2)

    out_path = Path(resolved.out_dir)
    core_path = Path(resolved.core_dir) if resolved.core_dir is not None else None

    build_stats: dict[str, Any] = {}
    tables = build_features_tables(
        core_dir=core_path,
        uniprot_tsv=resolved.uniprot_tsv,
        chembl_tsv=resolved.chembl_tsv,
        drugbank_smiles_tsv=resolved.drugbank_smiles_tsv,
        drugbank_properties_tsv=resolved.drugbank_properties_tsv,
        mondo_tsv=resolved.mondo_tsv,
        mondo_obo=resolved.mondo_obo,
        hpo_tsv=resolved.hpo_tsv,
        structure_skeleton_json=resolved.structure_skeleton_json,
        drug_features_tab=resolved.drug_features_tab,
        disease_features_tab=resolved.disease_features_tab,
        kg_csv=resolved.kg_csv,
        hgnc_tsv=resolved.hgnc_tsv,
        nucleotide_tsv=resolved.nucleotide_tsv,
        go_tsv=resolved.go_tsv,
        pathway_tsv=resolved.pathway_tsv,
        anatomy_tsv=resolved.anatomy_tsv,
        chebi_definitions_tsv=resolved.chebi_definitions_tsv,
        unichem_cid_map_tsv=resolved.unichem_cid_map_tsv,
        exposure_tsv=resolved.exposure_tsv,
        stats_out=build_stats,
    )

    artifact_paths = _text_provenance_artifact_paths(resolved)
    registry_path = resolve_text_provenance_registry_path(
        resolved.text_provenance_registry
    )
    try:
        tables, provenance_completeness, provenance_registry, mondo_stats = (
            finalize_text_provenance(
                tables,
                registry_path=resolved.text_provenance_registry,
                artifact_paths_by_source=artifact_paths,
                strict=resolved.strict_text_provenance,
            )
        )
    except (StrictTextProvenanceError, ProvenanceConflictError) as exc:
        cli_error(str(exc))
        raise typer.Exit(1) from exc

    modality_coverage: dict[str, Any] | None = None
    property_coverage: dict[str, Any] | None = None
    profile_key = resolved.features_profile.strip().lower()
    core_nodes: list[dict[str, Any]] | None = None
    if profile_key in _MODALITY_COVERAGE_PROFILES:
        if core_path is None:
            cli_error(
                f"features profile {profile_key!r} requires --core-dir for modality coverage"
            )
            raise typer.Exit(2)
        core_nodes = _load_core_nodes(core_path)
        modality_coverage = compute_modality_coverage(
            nodes=core_nodes,
            text_rows=tables.get("node_features_text", []),
            sequence_rows=tables.get("node_features_sequence", []),
            profile=profile_key,
        )
        if resolved.strict_coverage and not modality_coverage.get("pass"):
            cli_error(f"modality coverage failed: {modality_coverage}")
            raise typer.Exit(1)

    if resolved.emit_manifest and core_path is not None:
        if core_nodes is None:
            core_nodes = _load_core_nodes(core_path)
        property_coverage = compute_property_coverage(
            nodes=core_nodes,
            property_rows=tables.get("node_features_property", []),
        )

    sources: list[dict[str, Any]] | None = None
    manifest_builders: list[dict[str, Any]] | None = None
    if resolved.emit_manifest:
        sources = []
        if core_path is not None:
            manifest_path = core_path / "build_manifest.json"
            if manifest_path.is_file():
                sources.append(
                    manifest_source_file(name="primehkg_core_build", path=manifest_path)
                )
        if resolved.uniprot_tsv is not None:
            sources.append(
                manifest_source_file(name="uniprot_tsv", path=resolved.uniprot_tsv)
            )
        if resolved.chembl_tsv is not None:
            sources.append(
                manifest_source_file(name="chembl_tsv", path=resolved.chembl_tsv)
            )
        if resolved.drugbank_smiles_tsv is not None:
            sources.append(
                manifest_source_file(
                    name="drugbank_smiles_tsv",
                    path=resolved.drugbank_smiles_tsv,
                )
            )
        if resolved.drugbank_properties_tsv is not None:
            sources.append(
                manifest_source_file(
                    name="drugbank_properties_tsv",
                    path=resolved.drugbank_properties_tsv,
                )
            )
        if resolved.mondo_tsv is not None:
            sources.append(
                manifest_source_file(name="mondo_tsv", path=resolved.mondo_tsv)
            )
        if resolved.mondo_obo is not None:
            sources.append(
                manifest_source_file(name="mondo_obo", path=resolved.mondo_obo)
            )
        if resolved.hpo_tsv is not None:
            sources.append(manifest_source_file(name="hpo_tsv", path=resolved.hpo_tsv))
        if resolved.hgnc_tsv is not None:
            sources.append(
                manifest_source_file(name="hgnc_tsv", path=resolved.hgnc_tsv)
            )
        if resolved.nucleotide_tsv is not None:
            sources.append(
                manifest_source_file(
                    name="nucleotide_tsv", path=resolved.nucleotide_tsv
                )
            )
        if resolved.go_tsv is not None:
            sources.append(manifest_source_file(name="go_tsv", path=resolved.go_tsv))
        if resolved.pathway_tsv is not None:
            sources.append(
                manifest_source_file(name="pathway_tsv", path=resolved.pathway_tsv)
            )
        if resolved.anatomy_tsv is not None:
            sources.append(
                manifest_source_file(name="anatomy_tsv", path=resolved.anatomy_tsv)
            )
        if resolved.chebi_definitions_tsv is not None:
            sources.append(
                manifest_source_file(
                    name="chebi_definitions_tsv", path=resolved.chebi_definitions_tsv
                )
            )
        if resolved.unichem_cid_map_tsv is not None:
            sources.append(
                manifest_source_file(
                    name="unichem_cid_map_tsv", path=resolved.unichem_cid_map_tsv
                )
            )
        if resolved.exposure_tsv is not None:
            sources.append(
                manifest_source_file(name="exposure_tsv", path=resolved.exposure_tsv)
            )
        if resolved.drug_features_tab is not None:
            sources.append(
                manifest_source_file(
                    name="drug_features_tab", path=resolved.drug_features_tab
                )
            )
        if resolved.disease_features_tab is not None:
            sources.append(
                manifest_source_file(
                    name="disease_features_tab", path=resolved.disease_features_tab
                )
            )
        if resolved.kg_csv is not None:
            sources.append(manifest_source_file(name="kg_csv", path=resolved.kg_csv))
        if resolved.structure_skeleton_json is not None:
            sources.append(
                manifest_source_file(
                    name="structure_skeleton_json",
                    path=resolved.structure_skeleton_json,
                )
            )
        sources.append(
            manifest_source_file(
                name="text_provenance_registry",
                path=registry_path,
            )
        )
        if profile is not FeaturesProfile.NONE:
            manifest_builders = [
                features_profile_manifest_builder(
                    profile,
                    core_dir=core_path,
                    features_input_dir=resolved.features_input_dir,
                )
            ]

    provenance_meta: dict[str, Any] | None = None
    inventory_payload: dict[str, Any] | None = None
    if resolved.emit_manifest:
        artifact_sha_by_source = {
            source: manifest_source_file(name=source, path=path)["sha256"]
            for source, path in artifact_paths.items()
        }
        provenance_meta = {
            "text_provenance_schema": TEXT_PROVENANCE_SCHEMA,
            "provenance_registry_sha256": provenance_registry.digest,
            "text_provenance_sources": registry_source_inventory(
                provenance_registry,
                artifact_sha_by_source=artifact_sha_by_source,
            ),
            "provenance_completeness": provenance_completeness,
            "mondo_definition_ambiguity": mondo_stats,
        }
        crosswalk_stats = build_stats.get("entity_definition_crosswalk")
        if (
            isinstance(crosswalk_stats, ChebiDrugProjectionStats)
            and resolved.chebi_definitions_tsv is not None
            and resolved.unichem_cid_map_tsv is not None
        ):
            provenance_meta["entity_definition_crosswalk"] = (
                _entity_definition_crosswalk_manifest(
                    crosswalk_stats,
                    unichem_cid_map_tsv=resolved.unichem_cid_map_tsv,
                    chebi_definitions_tsv=resolved.chebi_definitions_tsv,
                )
            )
        if resolved.emit_text_provenance_inventory:
            text_rows = tables.get("node_features_text", [])
            inventory_payload = provenance_inventory(
                text_rows if isinstance(text_rows, list) else [],
                provenance_registry,
                mondo_definition_ambiguity=mondo_stats,
            )

    write_features_dataset_to_disk(
        out_path,
        tables,
        manifest_sources=sources if resolved.emit_manifest else None,
        manifest_builders=manifest_builders,
        modality_coverage=modality_coverage if resolved.emit_manifest else None,
        property_coverage=property_coverage if resolved.emit_manifest else None,
        provenance_meta=provenance_meta,
        text_provenance_inventory=inventory_payload,
    )
    raise typer.Exit(0)


@app.command()
def build(
    out_dir: Path = OUT_DIR_OPT,
    core_dir: Path | None = CORE_DIR_OPT,
    uniprot_tsv: Path | None = UNIPROT_TSV_OPT,
    chembl_tsv: Path | None = CHEMBL_TSV_OPT,
    drugbank_smiles_tsv: Path | None = DRUGBANK_SMILES_TSV_OPT,
    drugbank_properties_tsv: Path | None = DRUGBANK_PROPERTIES_TSV_OPT,
    mondo_tsv: Path | None = MONDO_TSV_OPT,
    mondo_obo: Path | None = MONDO_OBO_OPT,
    hpo_tsv: Path | None = HPO_TSV_OPT,
    drug_features_tab: Path | None = DRUG_FEATURES_TAB_OPT,
    disease_features_tab: Path | None = DISEASE_FEATURES_TAB_OPT,
    kg_csv: Path | None = KG_CSV_OPT,
    hgnc_tsv: Path | None = HGNC_TSV_OPT,
    nucleotide_tsv: Path | None = NUCLEOTIDE_TSV_OPT,
    go_tsv: Path | None = GO_TSV_OPT,
    pathway_tsv: Path | None = PATHWAY_TSV_OPT,
    anatomy_tsv: Path | None = ANATOMY_TSV_OPT,
    chebi_definitions_tsv: Path | None = CHEBI_DEFINITIONS_TSV_OPT,
    unichem_cid_map_tsv: Path | None = UNICHEM_CID_MAP_TSV_OPT,
    exposure_tsv: Path | None = EXPOSURE_TSV_OPT,
    structure_skeleton_json: Path | None = STRUCTURE_SKELETON_JSON_OPT,
    emit_manifest: bool = EMIT_MANIFEST_OPT,
    features_profile: str = FEATURES_PROFILE_OPT,
    features_input_dir: Path | None = FEATURES_INPUT_DIR_OPT,
    strict_coverage: bool = STRICT_COVERAGE_OPT,
    text_provenance_registry: Path | None = TEXT_PROVENANCE_REGISTRY_OPT,
    strict_text_provenance: bool = STRICT_TEXT_PROVENANCE_OPT,
    emit_text_provenance_inventory: bool = EMIT_TEXT_PROVENANCE_INVENTORY_OPT,
) -> None:
    _run_build(
        FeaturesBuildArgs(
            out_dir=out_dir,
            core_dir=core_dir,
            uniprot_tsv=uniprot_tsv,
            chembl_tsv=chembl_tsv,
            drugbank_smiles_tsv=drugbank_smiles_tsv,
            drugbank_properties_tsv=drugbank_properties_tsv,
            mondo_tsv=mondo_tsv,
            mondo_obo=mondo_obo,
            hpo_tsv=hpo_tsv,
            drug_features_tab=drug_features_tab,
            disease_features_tab=disease_features_tab,
            kg_csv=kg_csv,
            hgnc_tsv=hgnc_tsv,
            nucleotide_tsv=nucleotide_tsv,
            go_tsv=go_tsv,
            pathway_tsv=pathway_tsv,
            anatomy_tsv=anatomy_tsv,
            chebi_definitions_tsv=chebi_definitions_tsv,
            unichem_cid_map_tsv=unichem_cid_map_tsv,
            exposure_tsv=exposure_tsv,
            structure_skeleton_json=structure_skeleton_json,
            emit_manifest=emit_manifest,
            features_profile=features_profile,
            features_input_dir=features_input_dir,
            strict_coverage=strict_coverage,
            text_provenance_registry=text_provenance_registry,
            strict_text_provenance=strict_text_provenance,
            emit_text_provenance_inventory=emit_text_provenance_inventory,
        )
    )


def _argv_for_features_build_cli(argv: list[str] | None) -> list[str]:
    a = list(argv) if argv is not None else list(sys.argv[1:])
    if a and a[0] == "build":
        return a[1:]
    return a


def main(argv: list[str] | None = None) -> int:
    cmd = get_command(app)
    args_list = _argv_for_features_build_cli(argv)
    try:
        rc = cmd.main(
            args=args_list,
            prog_name="primehkg-features-build",
            standalone_mode=False,
        )
    except SystemExit as e:
        if isinstance(e.code, int):
            return e.code
        return 1
    return int(rc) if rc is not None else 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
