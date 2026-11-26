from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, cast

import typer
from typer.main import get_command

from primehkg.audit.anatomy_enrichment import compute_anatomy_enrichment_coverage
from primehkg.audit.go_enrichment import compute_go_enrichment_coverage
from primehkg.audit.pathway_enrichment import compute_pathway_enrichment_coverage
from primehkg.audit.relation_coverage import build_relation_coverage_audit
from primehkg.build_profiles.profile import (
    effective_segments,
    parse_build_profile,
)
from primehkg.builders import (
    build_clinvar_variant_hyperedges,
    build_complexportal_protein_complex_hyperedges,
    build_corum_protein_complex_hyperedges,
    build_ctd_regulation_hyperedges,
    build_disease_protein_bundle_hyperedges,
    build_drug_disease_shared_target_bundle_hyperedges,
    build_drugcomb_combination_hyperedges,
    build_moa_triad_hyperedges,
    build_phenotype_hyperedges,
    build_protein_disease_bundle_hyperedges,
    build_protein_protein_bundle_hyperedges,
    build_reactome_reaction_event_hyperedges,
    build_registry_atomic_hyperedges,
    build_signor_regulation_hyperedges,
    build_stitch_drug_protein_hyperedges,
)
from primehkg.builders.merge import merge_hypergraph_segments
from primehkg.builders.phenotype import PHENOTYPE_BUILDER_VERSION
from primehkg.determinism import (
    CURATED_FETCH_DETERMINISM_VERSION,
    DETERMINISM_POLICY_VERSION,
)
from primehkg.io.build_manifest import (
    DEFAULT_MANIFEST_SCHEMA_VERSION,
    manifest_source_file,
    merge_manifest_sidecars,
    primekg_kg_source,
)
from primehkg.io.builder_cache import (
    build_drug_disease_shared_target_bundle_with_cache,
    build_moa_triad_with_cache,
    build_phenotype_with_cache,
    build_registry_atomic_with_cache,
    registry_atomic_builder_entry,
)
from primehkg.io.chemical_xref_sidecar import emit_chemical_xref_sidecar_for_build
from primehkg.io.cli_console import cli_error, cli_warning
from primehkg.io.consumer_sidecars import write_consumer_sidecars
from primehkg.io.curated_node_ids import remap_protein_updates_to_geneid
from primehkg.io.curated_sources import (
    load_ctd_mesh_cid,
    load_string_ensp_geneid,
    load_unichem_cid_map,
    read_mondo_to_primekg_disease,
)
from primehkg.io.enrichment_profile import (
    EnrichmentProfile,
    apply_enrichment_profile,
    enrichment_profile_manifest_builder,
    parse_enrichment_profile,
)
from primehkg.io.graph_csr_sidecar import emit_graph_csr_sidecar_for_build
from primehkg.io.he_row_index_sidecar import emit_he_row_index_sidecar_for_build
from primehkg.io.hf_dataset import write_hf_tables_to_disk
from primehkg.io.inverted_index_sidecar import emit_inverted_index_sidecar_for_build
from primehkg.io.label_embedding_sidecar import emit_label_embedding_sidecar_for_build
from primehkg.io.nodes_from_triplets import (
    build_minimal_nodes_from_triplets,
    extend_nodes_with_node_ids,
)
from primehkg.io.primekg import PrimeKGLoadError, load_primekg_triplets
from primehkg.io.primekg_edges import (
    load_primekg_edges_triplets,
    merge_triplets_dedupe_kg_first,
)
from primehkg.io.primekg_feature_tabs import (
    build_node_index_to_node_id,
    read_primekg_disease_features_tab,
    read_primekg_drug_features_tab,
    read_primekg_nodes_tab,
)
from primehkg.io.primekg_names import collect_primekg_endpoint_name_updates
from primehkg.nodes.enrich import enrich_nodes_v1
from primehkg.nodes.feature_refs_policy import (
    feature_refs_enrichment_manifest_builder,
    validate_nodes_feature_refs_indirection,
)
from primehkg.nodes.sources import (
    bind_mondo_synonym_updates_to_nodes,
    read_anatomy_updates,
    read_chebi_definition_updates,
    read_chebi_name_updates,
    read_chebi_synonym_updates,
    read_chembl_updates,
    read_drugbank_updates,
    read_drugcomb_synonym_updates,
    read_go_updates,
    read_hgnc_synonym_updates,
    read_hgnc_updates,
    read_hpo_updates,
    read_mondo_synonym_updates,
    read_mondo_updates,
    read_pathway_updates,
    read_rxnorm_updates,
    read_umls_updates,
    read_uniprot_geneid_map,
    read_uniprot_updates,
    read_xrefs_json,
    read_xrefs_tsv,
)
from primehkg.policy.atomic_admissions import COVERAGE_REGISTRY_ADMISSIONS


@dataclass(frozen=True)
class BuildCliArgs:
    """CLI options for ``primehkg-build`` (Typer ``build`` command)."""

    kg_csv: Path
    out_dir: Path
    schema_version: str
    primekg_version: str
    corum_tsv: Path | None
    corum_version: str
    reactome_tsv: Path | None
    reactome_version: str
    signor_tsv: Path | None
    signor_version: str
    stitch_tsv: Path | None
    stitch_version: str
    stitch_score_threshold: int
    ctd_tsv: Path | None
    ctd_mesh_cid_tsv: Path | None
    ctd_version: str
    unichem_cid_map_tsv: Path | None
    string_ensp_geneid_tsv: Path | None
    complexportal_tsv: Path | None
    complexportal_version: str
    drugcomb_summary_tsv: Path | None
    drugcomb_identifiers_tsv: Path | None
    drugcomb_version: str
    clinvar_tsv: Path | None
    clinvar_version: str
    strict: bool
    non_strict: bool
    emit_build_report: bool
    emit_graph_csr_sidecar: bool
    emit_inverted_index_sidecar: bool
    emit_he_row_index_sidecar: bool
    emit_label_embedding_sidecar: bool
    emit_chemical_xref_sidecar: bool
    ctd_mesh_pubchem_v2_tsv: Path | None
    unichem_pubchem_xref_v2_tsv: Path | None
    materialize_unambiguous_mesh_xrefs: bool
    use_builder_cache: bool
    artifacts_root: Path
    enrich_nodes: Path | None
    enrich_xrefs_tsv: Path | None
    ingest_primekg_names: bool
    enrich_hgnc_tsv: Path | None
    enrich_hgnc_json: Path | None
    enrich_uniprot_tsv: Path | None
    enrich_hpo_tsv: Path | None
    enrich_go_tsv: Path | None
    enrich_pathway_tsv: Path | None
    enrich_anatomy_tsv: Path | None
    enrich_mondo_tsv: Path | None
    enrich_mondo_obo: Path | None
    enrich_drugbank_tsv: Path | None
    chebi_names_tsv: Path | None
    chebi_names_version: str
    chebi_synonyms_tsv: Path | None
    chebi_synonyms_version: str
    chebi_definitions_tsv: Path | None
    chebi_definitions_version: str
    enrich_rxnorm_tsv: Path | None
    enrich_chembl_tsv: Path | None
    enrich_umls_tsv: Path | None
    display_name_priority_json: Path | None
    enrich_primekg_disease_features_tab: Path | None
    enrich_primekg_drug_features_tab: Path | None
    enrich_primekg_nodes_tab: Path | None
    primekg_edges_csv: Path | None
    build_profile: str
    enrichment_profile: str
    enrichment_input_dir: Path | None
    primekg_dir: Path | None
    include_segments: tuple[str, ...] = ()
    exclude_segments: tuple[str, ...] = ()


# segment slug -> BuildCliArgs attribute(s) that must be provided when that segment runs.
# Drives the strict required-input check from the effective segment set.
_SEGMENT_REQUIRED_SOURCES: dict[str, tuple[str, ...]] = {
    "corum": ("corum_tsv",),
    "reactome": ("reactome_tsv",),
    "signor": ("signor_tsv",),
    "complexportal": ("complexportal_tsv",),
    "drugcomb": ("drugcomb_summary_tsv", "drugcomb_identifiers_tsv"),
    "clinvar": ("clinvar_tsv",),
    "stitch": ("stitch_tsv", "unichem_cid_map_tsv", "string_ensp_geneid_tsv"),
    "ctd": ("ctd_tsv", "ctd_mesh_cid_tsv", "unichem_cid_map_tsv"),
}


def _load_display_name_priority(path: str | Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"failed to read display-name priority json: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = "display-name priority json must be an object mapping node_type -> list[source]"
        raise TypeError(msg)
    out: dict[str, list[str]] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not k.strip():
            msg = (
                "display-name priority json keys must be non-empty strings (node_type)"
            )
            raise TypeError(msg)
        if not isinstance(v, list):
            msg = f"display-name priority for node_type {k!r} must be a list of strings"
            raise TypeError(msg)
        sources: list[str] = []
        for s in v:
            if not isinstance(s, str) or not s.strip():
                msg = f"display-name priority for node_type {k!r} must be a list of non-empty strings"
                raise TypeError(msg)
            sources.append(s.strip())
        out[k.strip()] = sources
    return out


def _apath(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    return Path(raw)


def _optional_manifest_sources(
    args: BuildCliArgs,
    *,
    corum_used: bool,
    reactome_used: bool,
    signor_used: bool,
    stitch_used: bool,
    ctd_used: bool,
    complexportal_used: bool,
    drugcomb_used: bool,
    drugcomb_synonyms_used: bool,
    chebi_synonyms_used: bool,
    hgnc_synonyms_used: bool,
    clinvar_used: bool,
) -> list[dict[str, Any]]:
    """Collect optional local inputs that affect the build, for ``build_manifest.json``."""
    out: list[dict[str, Any]] = []

    def add(name: str, path: Path | None) -> None:
        if path is not None and path.is_file():
            out.append(manifest_source_file(name=name, path=path))

    if corum_used:
        add("corum_tsv", _apath(args.corum_tsv))
    if reactome_used:
        add("reactome_tsv", _apath(args.reactome_tsv))
    if signor_used:
        add("signor_tsv", _apath(args.signor_tsv))
    if stitch_used:
        add("stitch_tsv", _apath(args.stitch_tsv))
        add("unichem_cid_map_tsv", _apath(args.unichem_cid_map_tsv))
        add("string_ensp_geneid_tsv", _apath(args.string_ensp_geneid_tsv))
    if ctd_used:
        add("ctd_tsv", _apath(args.ctd_tsv))
        add("ctd_mesh_cid_tsv", _apath(args.ctd_mesh_cid_tsv))
    if complexportal_used:
        add("complexportal_tsv", _apath(args.complexportal_tsv))
    if drugcomb_used:
        add("drugcomb_summary_tsv", _apath(args.drugcomb_summary_tsv))
        add("drugcomb_identifiers_tsv", _apath(args.drugcomb_identifiers_tsv))
    elif drugcomb_synonyms_used:
        # Synonym write-back can run without HE edges; still pin identifiers.
        add("drugcomb_identifiers_tsv", _apath(args.drugcomb_identifiers_tsv))
    if chebi_synonyms_used:
        add("chebi_synonyms_tsv", _apath(args.chebi_synonyms_tsv))
    if hgnc_synonyms_used:
        add("enrich_hgnc_json", _apath(args.enrich_hgnc_json))
    if clinvar_used:
        add("clinvar_tsv", _apath(args.clinvar_tsv))
    add("primekg_edges_csv", _apath(args.primekg_edges_csv))
    add(
        "primekg_disease_features_tab", _apath(args.enrich_primekg_disease_features_tab)
    )
    add("primekg_drug_features_tab", _apath(args.enrich_primekg_drug_features_tab))
    add("primekg_nodes_tab", _apath(args.enrich_primekg_nodes_tab))
    add("enrich_xrefs_tsv", _apath(args.enrich_xrefs_tsv))
    add("enrich_hgnc_tsv", _apath(args.enrich_hgnc_tsv))
    add("enrich_hpo_tsv", _apath(args.enrich_hpo_tsv))
    add("enrich_go_tsv", _apath(args.enrich_go_tsv))
    add("enrich_pathway_tsv", _apath(args.enrich_pathway_tsv))
    add("enrich_anatomy_tsv", _apath(args.enrich_anatomy_tsv))
    add("enrich_uniprot_tsv", _apath(args.enrich_uniprot_tsv))
    add("enrich_mondo_tsv", _apath(args.enrich_mondo_tsv))
    add("enrich_mondo_obo", _apath(args.enrich_mondo_obo))
    add("enrich_drugbank_tsv", _apath(args.enrich_drugbank_tsv))
    add("chebi_names_tsv", _apath(args.chebi_names_tsv))
    add("chebi_definitions_tsv", _apath(args.chebi_definitions_tsv))
    add("enrich_rxnorm_tsv", _apath(args.enrich_rxnorm_tsv))
    add("enrich_chembl_tsv", _apath(args.enrich_chembl_tsv))
    add("enrich_umls_tsv", _apath(args.enrich_umls_tsv))
    add("display_name_priority_json", _apath(args.display_name_priority_json))
    if args.enrich_nodes:
        enrich_fixture = Path(args.enrich_nodes) / "drug_xrefs.json"
        add("enrich_nodes_drug_xrefs_json", enrich_fixture)
    return out


def _feature_refs_enrichment_sources(args: BuildCliArgs) -> list[str]:
    mapping: list[tuple[Path | None, str]] = [
        (_apath(args.enrich_uniprot_tsv), "uniprot"),
        (_apath(args.enrich_chembl_tsv), "chembl"),
        (_apath(args.enrich_mondo_tsv), "mondo"),
        (_apath(args.enrich_mondo_obo), "mondo_obo"),
        (_apath(args.enrich_hpo_tsv), "hpo"),
        (_apath(args.enrich_go_tsv), "go"),
        (_apath(args.enrich_pathway_tsv), "pathway"),
        (_apath(args.enrich_anatomy_tsv), "anatomy"),
        (_apath(args.enrich_umls_tsv), "umls"),
    ]
    return sorted(label for path, label in mapping if path is not None)


def _relation_registry_manifest_source() -> dict[str, Any]:
    reg_path = resources.files("primehkg.policy").joinpath("relation_registry.yaml")
    return manifest_source_file(name="relation_registry_yaml", path=Path(str(reg_path)))


def run_build(args: BuildCliArgs) -> int:
    if args.strict and args.non_strict:
        cli_error("choose at most one of --strict / --non-strict")
        return 2
    if args.emit_chemical_xref_sidecar:
        missing_v2: list[str] = []
        if args.ctd_mesh_pubchem_v2_tsv is None:
            missing_v2.append("--ctd-mesh-pubchem-v2")
        elif not Path(args.ctd_mesh_pubchem_v2_tsv).is_file():
            cli_error(
                f"--ctd-mesh-pubchem-v2 is not a file: {args.ctd_mesh_pubchem_v2_tsv}"
            )
            return 2
        if args.unichem_pubchem_xref_v2_tsv is None:
            missing_v2.append("--unichem-pubchem-xref-v2")
        elif not Path(args.unichem_pubchem_xref_v2_tsv).is_file():
            cli_error(
                "--unichem-pubchem-xref-v2 is not a file: "
                f"{args.unichem_pubchem_xref_v2_tsv}"
            )
            return 2
        if missing_v2:
            cli_error(
                "--emit-chemical-xref-sidecar requires " + " and ".join(missing_v2)
            )
            return 2
    strict = bool(args.strict) or not bool(args.non_strict)

    kg_csv = Path(args.kg_csv)
    if not kg_csv.is_file():
        cli_error(f"kg.csv path is not a file: {kg_csv}")
        return 2
    out_dir = Path(args.out_dir)

    try:
        enrichment = parse_enrichment_profile(args.enrichment_profile)
    except ValueError as exc:
        cli_error(str(exc))
        return 2

    args = apply_enrichment_profile(args)
    if enrichment in (
        EnrichmentProfile.REFERENCE,
        EnrichmentProfile.EXTENDED,
    ) and not any(
        (
            args.enrich_primekg_disease_features_tab,
            args.enrich_primekg_drug_features_tab,
            args.enrich_primekg_nodes_tab,
        )
    ):
        cli_warning(
            "enrichment profile 'reference': no PrimeKG tab files found under "
            f"{args.primekg_dir} (expected disease_features.tab, drug_features.tab, "
            "and/or nodes.tab)"
        )

    try:
        display_name_priority_by_type = _load_display_name_priority(
            args.display_name_priority_json
        )
        display_name_priority_by_type.setdefault("metabolite", ["chebi"])
    except (ValueError, TypeError) as exc:
        cli_error(str(exc))
        return 2

    try:
        profile = parse_build_profile(args.build_profile)
    except ValueError as exc:
        cli_error(str(exc))
        return 2

    try:
        effective = effective_segments(
            profile,
            include=frozenset(args.include_segments),
            exclude=frozenset(args.exclude_segments),
        )
    except ValueError as exc:
        cli_error(str(exc))
        return 2

    missing = sorted(
        {
            src
            for seg in effective
            for src in _SEGMENT_REQUIRED_SOURCES.get(seg, ())
            if getattr(args, src) is None
        }
    )
    if missing:
        flags = ", ".join(
            "--" + src.removesuffix("_tsv").replace("_", "-") + "-tsv"
            for src in missing
        )
        msg = (
            f"build profile {profile.value!r} with segments {sorted(effective)} requires: "
            f"{flags}; pass them, exclude the segment(s), or use --build-profile smoke"
        )
        if strict:
            cli_error(msg)
            return 2
        cli_warning(msg)

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        triplets = load_primekg_triplets(kg_csv, strict=strict)
    except PrimeKGLoadError as exc:
        cli_error(str(exc))
        return 2
    except ValueError as exc:
        cli_error(str(exc))
        return 2

    if args.primekg_edges_csv:
        edges_path = Path(args.primekg_edges_csv)
        if not edges_path.is_file():
            if strict:
                cli_error(f"--primekg-edges-csv is not a file: {edges_path}")
                return 2
            cli_warning(f"skipping --primekg-edges-csv (not a file): {edges_path}")
        else:
            try:
                index_for_edges = build_node_index_to_node_id(kg_csv, strict=strict)
            except (OSError, ValueError, csv.Error) as exc:
                if strict:
                    cli_error(
                        f"--primekg-edges-csv requires kg.csv node_index columns: {exc}"
                    )
                    return 2
                cli_warning(
                    f"skipping --primekg-edges-csv (no node_index join map): {exc}"
                )
            else:
                try:
                    edge_triplets = load_primekg_edges_triplets(
                        edges_path,
                        index_for_edges,
                        strict=strict,
                    )
                except (OSError, ValueError, csv.Error) as exc:
                    if strict:
                        cli_error(f"failed to read --primekg-edges-csv: {exc}")
                        return 2
                    cli_warning(f"skipping --primekg-edges-csv (parse error): {exc}")
                else:
                    triplets = merge_triplets_dedupe_kg_first(triplets, edge_triplets)

    cache_sources: list[dict[str, Any]] = [
        primekg_kg_source(path=kg_csv),
        _relation_registry_manifest_source(),
    ]
    schema_ver = str(args.schema_version)
    moa_entry: dict[str, Any] = {
        "name": "moa_triad",
        "version": 1,
        "params": {"primekg_version": args.primekg_version},
    }
    emit_phenotype_composed = "phenotype_composed" in effective
    pheno_entry: dict[str, Any] = {
        "name": "phenotype",
        "version": PHENOTYPE_BUILDER_VERSION,
        "params": {"emit_composed": emit_phenotype_composed},
    }
    shared_target_entry: dict[str, Any] = {
        "name": "drug_disease_shared_target_bundle",
        "version": 2,
        "params": {
            "primekg_version": args.primekg_version,
            "max_drugs_per_bundle": 2,
            "max_eligible_drugs": 8,
            "max_bundles_per_protein_disease": 32,
        },
    }
    registry_atomic_entry = registry_atomic_builder_entry(
        primekg_version=str(args.primekg_version)
    )
    segments: list[tuple[str, dict[str, Any]]] = []

    if "moa_triad" in effective:
        if args.use_builder_cache:
            moa = build_moa_triad_with_cache(
                out_base=out_dir,
                schema_version=schema_ver,
                sources=cache_sources,
                builder_entry=moa_entry,
                triplets=triplets,
            )
        else:
            moa = build_moa_triad_hyperedges(
                triplets, source_version=str(args.primekg_version)
            )
        segments.append(("moa_triad", moa))

    if "phenotype" in effective:
        if args.use_builder_cache:
            pheno = build_phenotype_with_cache(
                out_base=out_dir,
                schema_version=schema_ver,
                sources=cache_sources,
                builder_entry=pheno_entry,
                triplets=triplets,
                emit_composed=emit_phenotype_composed,
            )
        else:
            pheno = build_phenotype_hyperedges(
                triplets, emit_composed=emit_phenotype_composed
            )
        segments.append(("phenotype", pheno))

    if "registry_atomic" in effective:
        if args.use_builder_cache:
            registry_atomic = build_registry_atomic_with_cache(
                out_base=out_dir,
                schema_version=schema_ver,
                sources=cache_sources,
                builder_entry=registry_atomic_entry,
                triplets=triplets,
            )
        else:
            registry_atomic = build_registry_atomic_hyperedges(
                triplets,
                admissions=COVERAGE_REGISTRY_ADMISSIONS,
                source_version=str(args.primekg_version),
            )
        segments.append(("registry_atomic", registry_atomic))

    if "drug_disease_shared_target_bundle" in effective:
        if args.use_builder_cache:
            bundles = build_drug_disease_shared_target_bundle_with_cache(
                out_base=out_dir,
                schema_version=schema_ver,
                sources=cache_sources,
                builder_entry=shared_target_entry,
                triplets=triplets,
            )
        else:
            bundles = build_drug_disease_shared_target_bundle_hyperedges(
                triplets,
                source_version=str(args.primekg_version),
                max_drugs_per_bundle=2,
                max_eligible_drugs=8,
                max_bundles_per_protein_disease=32,
            )
        segments.append(("drug_disease_shared_target_bundle", bundles))

    if "disease_protein_bundle" in effective:
        disease_bundles = build_disease_protein_bundle_hyperedges(
            triplets,
            source_version=str(args.primekg_version),
            max_proteins_per_disease=25,
        )
        segments.append(("disease_protein_bundle", disease_bundles))

    if "protein_disease_bundle" in effective:
        protein_bundles = build_protein_disease_bundle_hyperedges(
            triplets,
            source_version=str(args.primekg_version),
            max_diseases_per_protein=25,
        )
        segments.append(("protein_disease_bundle", protein_bundles))

    if "protein_protein_bundle" in effective:
        ppi_bundles = build_protein_protein_bundle_hyperedges(
            triplets,
            source_version=str(args.primekg_version),
            max_partners_per_anchor=8,
        )
        segments.append(("protein_protein_bundle", ppi_bundles))
    corum_used = False
    reactome_used = False
    signor_used = False
    stitch_used = False
    stitch_stats: dict[str, Any] = {}
    ctd_used = False
    ctd_stats: dict[str, Any] = {}
    complexportal_used = False
    drugcomb_used = False
    drugcomb_synonyms_used = False
    chebi_synonyms_used = False
    hgnc_synonyms_used = False
    mondo_synonyms_used = False
    clinvar_used = False
    clinvar_node_updates: dict[str, Any] = {}

    uniprot_to_geneid: dict[str, str] = {}
    if args.enrich_uniprot_tsv:
        geneid_tsv_path = Path(args.enrich_uniprot_tsv)
        if geneid_tsv_path.is_file():
            try:
                uniprot_to_geneid = read_uniprot_geneid_map(geneid_tsv_path)
            except (OSError, ValueError, csv.Error) as exc:
                cli_warning(f"skipping uniprot->geneid remap: {exc}")
                uniprot_to_geneid = {}

    nodes = build_minimal_nodes_from_triplets(triplets)
    index_to_node_id: dict[int, str] | None = None

    if args.corum_tsv and "corum" in effective:
        corum_path = Path(args.corum_tsv)
        if not corum_path.is_file():
            if strict:
                cli_error(f"--corum-tsv is not a file: {corum_path}")
                return 2
            cli_warning(f"skipping --corum-tsv (not a file): {corum_path}")
        else:
            try:
                corum_segment = build_corum_protein_complex_hyperedges(
                    corum_path,
                    corum_version=str(args.corum_version),
                    uniprot_to_geneid=uniprot_to_geneid,
                )
            except (OSError, ValueError, csv.Error) as exc:
                if strict:
                    cli_error(f"failed to read --corum-tsv: {exc}")
                    return 2
                cli_warning(f"skipping --corum-tsv (parse error): {exc}")
            else:
                segments.append(("corum", corum_segment))
                corum_used = True

    if args.reactome_tsv and "reactome" in effective:
        reactome_path = Path(args.reactome_tsv)
        if not reactome_path.is_file():
            if strict:
                cli_error(f"--reactome-tsv is not a file: {reactome_path}")
                return 2
            cli_warning(f"skipping --reactome-tsv (not a file): {reactome_path}")
        else:
            try:
                reactome_segment = build_reactome_reaction_event_hyperedges(
                    reactome_path,
                    reactome_version=str(args.reactome_version),
                    uniprot_to_geneid=uniprot_to_geneid,
                )
            except (OSError, ValueError, csv.Error) as exc:
                if strict:
                    cli_error(f"failed to read --reactome-tsv: {exc}")
                    return 2
                cli_warning(f"skipping --reactome-tsv (parse error): {exc}")
            else:
                segments.append(("reactome", reactome_segment))
                reactome_used = True

    if args.signor_tsv and "signor" in effective:
        signor_path = Path(args.signor_tsv)
        if not signor_path.is_file():
            if strict:
                cli_error(f"--signor-tsv is not a file: {signor_path}")
                return 2
            cli_warning(f"skipping --signor-tsv (not a file): {signor_path}")
        else:
            try:
                signor_segment = build_signor_regulation_hyperedges(
                    signor_path,
                    signor_version=str(args.signor_version),
                    uniprot_to_geneid=uniprot_to_geneid,
                )
            except (OSError, ValueError, csv.Error) as exc:
                if strict:
                    cli_error(f"failed to read --signor-tsv: {exc}")
                    return 2
                cli_warning(f"skipping --signor-tsv (parse error): {exc}")
            else:
                segments.append(("signor", signor_segment))
                signor_used = True

    # Existing nodes for the stitch existing-node-only constraint (#408): base
    # triplet nodes + every node_id referenced by the non-stitch segments built so far.
    stitch_existing_node_ids: set[str] = {
        str(n["node_id"]) for n in nodes if isinstance(n, dict) and n.get("node_id")
    }
    for _slug, _seg in segments:
        for _he in _seg.get("hyperedges", []):
            for _nid in _he.get("node_ids", []):
                if isinstance(_nid, str):
                    stitch_existing_node_ids.add(_nid)

    if args.stitch_tsv and "stitch" in effective:
        stitch_path = Path(args.stitch_tsv)
        cidmap_path = (
            Path(args.unichem_cid_map_tsv) if args.unichem_cid_map_tsv else None
        )
        ensp_path = (
            Path(args.string_ensp_geneid_tsv) if args.string_ensp_geneid_tsv else None
        )
        if not stitch_path.is_file() or cidmap_path is None or ensp_path is None:
            if strict:
                cli_error(
                    "--stitch-tsv requires --unichem-cid-map-tsv and "
                    "--string-ensp-geneid-tsv (all files)"
                )
                return 2
            cli_warning("skipping stitch segment (missing stitch/unichem/ensp file)")
        else:
            try:
                cid_map = load_unichem_cid_map(cidmap_path)
                ensp_to_geneid = load_string_ensp_geneid(ensp_path)
                stitch_segment = build_stitch_drug_protein_hyperedges(
                    stitch_path,
                    cid_map=cid_map,
                    ensp_to_geneid=ensp_to_geneid,
                    score_threshold=int(args.stitch_score_threshold),
                    stitch_version=str(args.stitch_version),
                    existing_node_ids=stitch_existing_node_ids,
                )
            except (OSError, ValueError, csv.Error) as exc:
                if strict:
                    cli_error(f"failed to build stitch segment: {exc}")
                    return 2
                cli_warning(f"skipping stitch segment (error): {exc}")
            else:
                stitch_stats = stitch_segment.get("stats", {})
                if stitch_stats.get("edges_emitted", 0) == 0:
                    cli_warning(
                        "stitch segment resolved 0 drug_protein edges "
                        f"(rows={stitch_stats.get('total_rows', 0)}, "
                        f"passing_threshold={stitch_stats.get('rows_passing_threshold', 0)}); "
                        "check the cid map / ensp map coverage and score threshold"
                    )
                segments.append(("stitch", stitch_segment))
                stitch_used = True

    # Existing nodes for the ctd existing-node-only constraint: base triplet
    # nodes + every node_id referenced by the segments built so far (incl. stitch).
    ctd_existing_node_ids: set[str] = {
        str(n["node_id"]) for n in nodes if isinstance(n, dict) and n.get("node_id")
    }
    for _slug, _seg in segments:
        for _he in _seg.get("hyperedges", []):
            for _nid in _he.get("node_ids", []):
                if isinstance(_nid, str):
                    ctd_existing_node_ids.add(_nid)

    if args.ctd_tsv and "ctd" in effective:
        ctd_path = Path(args.ctd_tsv)
        ctd_meshcid_path = (
            Path(args.ctd_mesh_cid_tsv) if args.ctd_mesh_cid_tsv else None
        )
        ctd_cidmap_path = (
            Path(args.unichem_cid_map_tsv) if args.unichem_cid_map_tsv else None
        )
        if (
            not ctd_path.is_file()
            or ctd_meshcid_path is None
            or ctd_cidmap_path is None
        ):
            if strict:
                cli_error(
                    "--ctd-tsv requires --ctd-mesh-cid-tsv and --unichem-cid-map-tsv (all files)"
                )
                return 2
            cli_warning("skipping ctd segment (missing ctd/mesh-cid/unichem file)")
        else:
            try:
                mesh_to_cid = load_ctd_mesh_cid(ctd_meshcid_path)
                ctd_cid_map = load_unichem_cid_map(ctd_cidmap_path)
                ctd_segment = build_ctd_regulation_hyperedges(
                    ctd_path,
                    mesh_to_cid=mesh_to_cid,
                    cid_map=ctd_cid_map,
                    existing_node_ids=ctd_existing_node_ids,
                    ctd_version=str(args.ctd_version),
                )
            except (OSError, ValueError, csv.Error) as exc:
                if strict:
                    cli_error(f"failed to build ctd segment: {exc}")
                    return 2
                cli_warning(f"skipping ctd segment (error): {exc}")
            else:
                ctd_stats = ctd_segment.get("stats", {})
                if ctd_stats.get("edges_emitted", 0) == 0:
                    cli_warning(
                        "ctd segment resolved 0 regulation edges "
                        f"(rows={ctd_stats.get('total_rows', 0)}, "
                        f"directed={ctd_stats.get('directed_rows', 0)}); "
                        "check the mesh->cid / unichem map coverage"
                    )
                segments.append(("ctd", ctd_segment))
                ctd_used = True

    if args.complexportal_tsv and "complexportal" in effective:
        complexportal_path = Path(args.complexportal_tsv)
        if not complexportal_path.is_file():
            if strict:
                cli_error(f"--complexportal-tsv is not a file: {complexportal_path}")
                return 2
            cli_warning(
                f"skipping --complexportal-tsv (not a file): {complexportal_path}"
            )
        else:
            try:
                complexportal_segment = build_complexportal_protein_complex_hyperedges(
                    complexportal_path,
                    complexportal_version=str(args.complexportal_version),
                    uniprot_to_geneid=uniprot_to_geneid,
                )
            except (OSError, ValueError, csv.Error) as exc:
                if strict:
                    cli_error(f"failed to read --complexportal-tsv: {exc}")
                    return 2
                cli_warning(f"skipping --complexportal-tsv (parse error): {exc}")
            else:
                segments.append(("complexportal", complexportal_segment))
                complexportal_used = True

    if (
        args.drugcomb_summary_tsv
        and args.drugcomb_identifiers_tsv
        and "drugcomb" in effective
    ):
        dc_summary = Path(args.drugcomb_summary_tsv)
        dc_ids = Path(args.drugcomb_identifiers_tsv)
        if not dc_summary.is_file() or not dc_ids.is_file():
            if strict:
                if not dc_summary.is_file():
                    cli_error(f"--drugcomb-summary-tsv is not a file: {dc_summary}")
                else:
                    cli_error(f"--drugcomb-identifiers-tsv is not a file: {dc_ids}")
                return 2
            cli_warning(
                "skipping --drugcomb-summary-tsv / --drugcomb-identifiers-tsv "
                f"(not files): {dc_summary} / {dc_ids}"
            )
        else:
            try:
                drugcomb_known = {
                    nid
                    for r in nodes
                    if (nid := r.get("node_id")) and isinstance(nid, str)
                }
                drugcomb_segment = build_drugcomb_combination_hyperedges(
                    dc_summary,
                    dc_ids,
                    drugcomb_version=str(args.drugcomb_version),
                    known_node_ids=drugcomb_known,
                )
            except (OSError, ValueError, csv.Error) as exc:
                if strict:
                    cli_error(f"failed to read --drugcomb-* inputs: {exc}")
                    return 2
                cli_warning(f"skipping --drugcomb-* (parse error): {exc}")
            else:
                segments.append(("drugcomb", drugcomb_segment))
                drugcomb_used = True

    if args.clinvar_tsv and "clinvar" in effective:
        clinvar_path = Path(args.clinvar_tsv)
        if not clinvar_path.is_file():
            if strict:
                cli_error(f"--clinvar-tsv is not a file: {clinvar_path}")
                return 2
            cli_warning(f"skipping --clinvar-tsv (not a file): {clinvar_path}")
        else:
            try:
                mondo_to_disease: dict[str, str] = {}
                if args.enrich_primekg_nodes_tab:
                    nt = Path(args.enrich_primekg_nodes_tab)
                    if nt.is_file():
                        mondo_to_disease = read_mondo_to_primekg_disease(nt)
                if not mondo_to_disease:
                    cli_warning(
                        "clinvar: no nodes.tab MONDO bridge; variant_disease will be empty"
                    )
                clinvar_known = {
                    nid
                    for r in nodes
                    if (nid := r.get("node_id")) and isinstance(nid, str)
                }
                clinvar_segment = build_clinvar_variant_hyperedges(
                    clinvar_path,
                    clinvar_version=str(args.clinvar_version),
                    mondo_to_disease=mondo_to_disease,
                    known_node_ids=clinvar_known,
                )
            except (OSError, ValueError, csv.Error) as exc:
                if strict:
                    cli_error(f"failed to read --clinvar-tsv: {exc}")
                    return 2
                cli_warning(f"skipping --clinvar-tsv (parse error): {exc}")
            else:
                segments.append(("clinvar", cast("dict[str, Any]", clinvar_segment)))
                clinvar_used = True
                clinvar_node_updates = clinvar_segment.get("node_updates") or {}

    merged_bundle = merge_hypergraph_segments(segments)
    dedup_stats = merged_bundle["hyperedge_dedup"]
    merged = {
        "hyperedges": merged_bundle["hyperedges"],
        "support": merged_bundle["support"],
    }

    merged_node_ids: set[str] = set()
    for he in merged["hyperedges"]:
        if not isinstance(he, dict):
            continue
        ids = he.get("node_ids")
        if isinstance(ids, list):
            merged_node_ids.update(nid for nid in ids if isinstance(nid, str))
    nodes = extend_nodes_with_node_ids(nodes, node_ids=merged_node_ids)

    if args.ingest_primekg_names and not args.enrich_primekg_nodes_tab:
        try:
            name_updates = collect_primekg_endpoint_name_updates(
                kg_csv,
                strict=strict,
                source_version=str(args.primekg_version),
            )
        except (OSError, PrimeKGLoadError, ValueError) as exc:
            if strict:
                cli_error(f"failed to ingest PrimeKG endpoint names: {exc}")
                return 2
            cli_warning(f"skipping PrimeKG endpoint names ingestion: {exc}")
            name_updates = {}
        if name_updates:
            nodes = enrich_nodes_v1(
                base_nodes=nodes,
                node_updates=name_updates,
                display_name_priority_by_type=display_name_priority_by_type,
            )

    if (
        args.enrich_primekg_disease_features_tab
        or args.enrich_primekg_drug_features_tab
    ):
        try:
            index_to_node_id = build_node_index_to_node_id(kg_csv, strict=strict)
        except (OSError, ValueError, csv.Error) as exc:
            if strict:
                cli_error(
                    f"failed to build PrimeKG node_index join map from kg.csv: {exc}"
                )
                return 2
            cli_warning(
                "skipping PrimeKG feature ingestion (failed to build node_index join map): "
                f"{exc}"
            )
            index_to_node_id = None

    if args.enrich_primekg_disease_features_tab:
        features_path = Path(args.enrich_primekg_disease_features_tab)
        if not features_path.is_file():
            if strict:
                cli_error(f"disease_features.tab path is not a file: {features_path}")
                return 2
            cli_warning(
                "skipping PrimeKG disease_features.tab ingestion (path is not a file): "
                f"{features_path}"
            )
        else:
            updates: dict[str, dict[str, object]] = {}
            if index_to_node_id is None:
                cli_warning(
                    "skipping PrimeKG disease_features.tab ingestion (no node_index join map)"
                )
            else:
                try:
                    updates = read_primekg_disease_features_tab(
                        features_path,
                        index_to_node_id=index_to_node_id,
                        strict=strict,
                    )
                except (OSError, ValueError, csv.Error) as exc:
                    if strict:
                        cli_error(
                            f"failed to ingest PrimeKG disease features tab: {exc}"
                        )
                        return 2
                    cli_warning(
                        f"skipping PrimeKG disease_features.tab ingestion: {exc}"
                    )
            if updates:
                nodes = enrich_nodes_v1(
                    base_nodes=nodes,
                    node_updates=updates,
                    display_name_priority_by_type=display_name_priority_by_type,
                )

    if args.enrich_primekg_drug_features_tab:
        features_path = Path(args.enrich_primekg_drug_features_tab)
        if not features_path.is_file():
            if strict:
                cli_error(f"drug_features.tab path is not a file: {features_path}")
                return 2
            cli_warning(
                "skipping PrimeKG drug_features.tab ingestion (path is not a file): "
                f"{features_path}"
            )
        else:
            updates: dict[str, dict[str, object]] = {}
            if index_to_node_id is None:
                cli_warning(
                    "skipping PrimeKG drug_features.tab ingestion (no node_index join map)"
                )
            else:
                try:
                    updates = read_primekg_drug_features_tab(
                        features_path,
                        index_to_node_id=index_to_node_id,
                        strict=strict,
                    )
                except (OSError, ValueError, csv.Error) as exc:
                    if strict:
                        cli_error(f"failed to ingest PrimeKG drug features tab: {exc}")
                        return 2
                    cli_warning(f"skipping PrimeKG drug_features.tab ingestion: {exc}")
            if updates:
                nodes = enrich_nodes_v1(
                    base_nodes=nodes,
                    node_updates=updates,
                    display_name_priority_by_type=display_name_priority_by_type,
                )

    if args.enrich_primekg_nodes_tab:
        nodes_tab_path = Path(args.enrich_primekg_nodes_tab)
        if not nodes_tab_path.is_file():
            if strict:
                cli_error(f"nodes.tab path is not a file: {nodes_tab_path}")
                return 2
            cli_warning(
                "skipping PrimeKG nodes.tab ingestion (path is not a file): "
                f"{nodes_tab_path}"
            )
        else:
            try:
                updates = read_primekg_nodes_tab(nodes_tab_path, strict=strict)
            except (OSError, ValueError, csv.Error) as exc:
                if strict:
                    cli_error(f"failed to ingest PrimeKG nodes.tab: {exc}")
                    return 2
                cli_warning(f"skipping PrimeKG nodes.tab ingestion: {exc}")
                updates = {}
            if updates:
                nodes = enrich_nodes_v1(
                    base_nodes=nodes,
                    node_updates=updates,
                    display_name_priority_by_type=display_name_priority_by_type,
                )

    if args.enrich_nodes:
        enrich_dir = Path(args.enrich_nodes)
        if not enrich_dir.is_dir():
            cli_error(f"--enrich-nodes is not a directory: {enrich_dir}")
            return 2
        fixture_path = enrich_dir / "drug_xrefs.json"
        if not fixture_path.is_file():
            cli_error(f"missing enrichment fixture: {fixture_path}")
            return 2
        try:
            xrefs_by_node = read_xrefs_json(fixture_path)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            cli_error(f"failed to read enrichment fixture: {exc}")
            return 2

        node_updates = {
            node_id: {
                "xrefs": xrefs,
                "source": "local_fixture",
                "source_version": "drug_xrefs.json",
                "evidence": "drug_xrefs_fixture",
            }
            for node_id, xrefs in xrefs_by_node.items()
        }
        nodes = enrich_nodes_v1(
            base_nodes=nodes,
            node_updates=node_updates,
            display_name_priority_by_type=display_name_priority_by_type,
        )
    if args.enrich_xrefs_tsv:
        tsv_path = Path(args.enrich_xrefs_tsv)
        if not tsv_path.is_file():
            cli_error(f"--enrich-xrefs-tsv is not a file: {tsv_path}")
            return 2
        try:
            xrefs_by_node = read_xrefs_tsv(tsv_path)
        except (OSError, ValueError, csv.Error) as exc:
            cli_error(f"failed to read xrefs tsv: {exc}")
            return 2

        node_updates = {
            node_id: {
                "xrefs": xrefs,
                "source": "local_tsv",
                "source_version": tsv_path.name,
                "evidence": "xrefs_tsv",
            }
            for node_id, xrefs in xrefs_by_node.items()
        }
        nodes = enrich_nodes_v1(
            base_nodes=nodes,
            node_updates=node_updates,
            display_name_priority_by_type=display_name_priority_by_type,
        )
    if args.enrich_hgnc_tsv:
        hgnc_tsv_path = Path(args.enrich_hgnc_tsv)
        if not hgnc_tsv_path.is_file():
            cli_error(f"--enrich-hgnc-tsv is not a file: {hgnc_tsv_path}")
            return 2
        try:
            hgnc_updates = read_hgnc_updates(hgnc_tsv_path)
        except (OSError, ValueError, csv.Error) as exc:
            cli_error(f"failed to read hgnc tsv: {exc}")
            return 2

        if hgnc_updates:
            nodes = enrich_nodes_v1(
                base_nodes=nodes,
                node_updates=hgnc_updates,
                display_name_priority_by_type=display_name_priority_by_type,
            )
    if args.enrich_uniprot_tsv:
        uniprot_tsv_path = Path(args.enrich_uniprot_tsv)
        if not uniprot_tsv_path.is_file():
            cli_error(f"--enrich-uniprot-tsv is not a file: {uniprot_tsv_path}")
            return 2
        try:
            uniprot_updates = read_uniprot_updates(uniprot_tsv_path)
        except (OSError, ValueError, csv.Error) as exc:
            cli_error(f"failed to read uniprot tsv: {exc}")
            return 2

        uniprot_updates = remap_protein_updates_to_geneid(
            uniprot_updates, uniprot_to_geneid
        )
        if uniprot_updates:
            nodes = enrich_nodes_v1(
                base_nodes=nodes,
                node_updates=uniprot_updates,
                display_name_priority_by_type=display_name_priority_by_type,
            )
    if args.enrich_hpo_tsv:
        hpo_tsv_path = Path(args.enrich_hpo_tsv)
        if not hpo_tsv_path.is_file():
            cli_error(f"--enrich-hpo-tsv is not a file: {hpo_tsv_path}")
            return 2
        try:
            hpo_updates = read_hpo_updates(hpo_tsv_path)
        except (OSError, ValueError, csv.Error) as exc:
            cli_error(f"failed to read hpo tsv: {exc}")
            return 2

        if hpo_updates:
            nodes = enrich_nodes_v1(
                base_nodes=nodes,
                node_updates=hpo_updates,
                display_name_priority_by_type=display_name_priority_by_type,
            )
    go_enrichment_coverage: dict[str, Any] | None = None
    if args.enrich_go_tsv:
        go_tsv_path = Path(args.enrich_go_tsv)
        if not go_tsv_path.is_file():
            cli_error(f"--enrich-go-tsv is not a file: {go_tsv_path}")
            return 2
        try:
            go_updates = read_go_updates(go_tsv_path)
        except (OSError, ValueError, csv.Error) as exc:
            cli_error(f"failed to read go tsv: {exc}")
            return 2

        if go_updates:
            nodes = enrich_nodes_v1(
                base_nodes=nodes,
                node_updates=go_updates,
                display_name_priority_by_type=display_name_priority_by_type,
            )
        go_enrichment_coverage = compute_go_enrichment_coverage(
            nodes,
            source="GO",
            source_path=str(go_tsv_path),
        )
    pathway_enrichment_coverage: dict[str, Any] | None = None
    if args.enrich_pathway_tsv:
        pathway_tsv_path = Path(args.enrich_pathway_tsv)
        if not pathway_tsv_path.is_file():
            cli_error(f"--enrich-pathway-tsv is not a file: {pathway_tsv_path}")
            return 2
        try:
            pathway_updates = read_pathway_updates(pathway_tsv_path)
        except (OSError, ValueError, csv.Error) as exc:
            cli_error(f"failed to read pathway tsv: {exc}")
            return 2

        if pathway_updates:
            nodes = enrich_nodes_v1(
                base_nodes=nodes,
                node_updates=pathway_updates,
                display_name_priority_by_type=display_name_priority_by_type,
            )
        pathway_enrichment_coverage = compute_pathway_enrichment_coverage(
            nodes,
            source="PrimeKG",
            source_path=str(pathway_tsv_path),
        )
    anatomy_enrichment_coverage: dict[str, Any] | None = None
    if args.enrich_anatomy_tsv:
        anatomy_tsv_path = Path(args.enrich_anatomy_tsv)
        if not anatomy_tsv_path.is_file():
            cli_error(f"--enrich-anatomy-tsv is not a file: {anatomy_tsv_path}")
            return 2
        try:
            anatomy_updates = read_anatomy_updates(anatomy_tsv_path)
        except (OSError, ValueError, csv.Error) as exc:
            cli_error(f"failed to read anatomy tsv: {exc}")
            return 2

        if anatomy_updates:
            nodes = enrich_nodes_v1(
                base_nodes=nodes,
                node_updates=anatomy_updates,
                display_name_priority_by_type=display_name_priority_by_type,
            )
        anatomy_enrichment_coverage = compute_anatomy_enrichment_coverage(
            nodes,
            source="PrimeKG",
            source_path=str(anatomy_tsv_path),
        )
    if args.enrich_mondo_obo and not args.enrich_mondo_tsv:
        cli_error("--enrich-mondo-obo requires --enrich-mondo-tsv")
        return 2
    if args.enrich_mondo_tsv:
        mondo_tsv_path = Path(args.enrich_mondo_tsv)
        if not mondo_tsv_path.is_file():
            cli_error(f"--enrich-mondo-tsv is not a file: {mondo_tsv_path}")
            return 2
        mondo_obo_path: Path | None = None
        if args.enrich_mondo_obo:
            mondo_obo_path = Path(args.enrich_mondo_obo)
            if not mondo_obo_path.is_file():
                cli_error(f"--enrich-mondo-obo is not a file: {mondo_obo_path}")
                return 2
        try:
            mondo_updates = read_mondo_updates(mondo_tsv_path, mondo_obo=mondo_obo_path)
        except (OSError, ValueError, csv.Error) as exc:
            cli_error(f"failed to read mondo tsv: {exc}")
            return 2

        if mondo_updates:
            nodes = enrich_nodes_v1(
                base_nodes=nodes,
                node_updates=mondo_updates,
                display_name_priority_by_type=display_name_priority_by_type,
            )
    if args.enrich_drugbank_tsv:
        drugbank_tsv_path = Path(args.enrich_drugbank_tsv)
        if not drugbank_tsv_path.is_file():
            cli_error(f"--enrich-drugbank-tsv is not a file: {drugbank_tsv_path}")
            return 2
        try:
            drugbank_updates = read_drugbank_updates(drugbank_tsv_path)
        except (OSError, ValueError, csv.Error) as exc:
            cli_error(f"failed to read drugbank tsv: {exc}")
            return 2

        if drugbank_updates:
            nodes = enrich_nodes_v1(
                base_nodes=nodes,
                node_updates=drugbank_updates,
                display_name_priority_by_type=display_name_priority_by_type,
            )
    if args.chebi_names_tsv:
        chebi_names_tsv_path = Path(args.chebi_names_tsv)
        if not chebi_names_tsv_path.is_file():
            cli_error(f"--chebi-names-tsv is not a file: {chebi_names_tsv_path}")
            return 2
        try:
            chebi_name_updates = read_chebi_name_updates(
                chebi_names_tsv_path,
                source_version=str(args.chebi_names_version),
            )
        except (OSError, ValueError, csv.Error) as exc:
            cli_error(f"failed to read chebi names tsv: {exc}")
            return 2

        if chebi_name_updates:
            matched = sum(
                1 for n in nodes if str(n.get("node_id")) in chebi_name_updates
            )
            if matched == 0:
                cli_warning(
                    f"--chebi-names-tsv parsed {len(chebi_name_updates)} entries but matched "
                    "0 graph node ids — check the ChEBI release / accession namespace"
                )
            nodes = enrich_nodes_v1(
                base_nodes=nodes,
                node_updates=chebi_name_updates,
                display_name_priority_by_type=display_name_priority_by_type,
            )
        else:
            cli_warning(
                f"--chebi-names-tsv {chebi_names_tsv_path} yielded 0 usable rows; "
                "no metabolite names resolved"
            )
    if args.chebi_definitions_tsv:
        chebi_definitions_tsv_path = Path(args.chebi_definitions_tsv)
        if not chebi_definitions_tsv_path.is_file():
            cli_error(
                f"--chebi-definitions-tsv is not a file: {chebi_definitions_tsv_path}"
            )
            return 2
        try:
            chebi_definition_updates = read_chebi_definition_updates(
                chebi_definitions_tsv_path,
                source_version=str(args.chebi_definitions_version),
            )
        except (OSError, ValueError, csv.Error) as exc:
            cli_error(f"failed to read chebi definitions tsv: {exc}")
            return 2

        if chebi_definition_updates:
            matched = sum(
                1 for n in nodes if str(n.get("node_id")) in chebi_definition_updates
            )
            if matched == 0:
                cli_warning(
                    f"--chebi-definitions-tsv parsed {len(chebi_definition_updates)} "
                    "entries but matched 0 graph node ids — check the ChEBI release / accession namespace"
                )
            nodes = enrich_nodes_v1(
                base_nodes=nodes,
                node_updates=chebi_definition_updates,
                display_name_priority_by_type=display_name_priority_by_type,
            )
        else:
            cli_warning(
                f"--chebi-definitions-tsv {chebi_definitions_tsv_path} yielded 0 usable rows; "
                "no metabolite definitions resolved"
            )
    if args.enrich_rxnorm_tsv:
        rxnorm_tsv_path = Path(args.enrich_rxnorm_tsv)
        if not rxnorm_tsv_path.is_file():
            cli_error(f"--enrich-rxnorm-tsv is not a file: {rxnorm_tsv_path}")
            return 2
        try:
            rxnorm_updates = read_rxnorm_updates(rxnorm_tsv_path)
        except (OSError, ValueError, csv.Error) as exc:
            cli_error(f"failed to read rxnorm tsv: {exc}")
            return 2

        if rxnorm_updates:
            nodes = enrich_nodes_v1(
                base_nodes=nodes,
                node_updates=rxnorm_updates,
                display_name_priority_by_type=display_name_priority_by_type,
            )
    if args.enrich_chembl_tsv:
        chembl_tsv_path = Path(args.enrich_chembl_tsv)
        if not chembl_tsv_path.is_file():
            cli_error(f"--enrich-chembl-tsv is not a file: {chembl_tsv_path}")
            return 2
        try:
            chembl_updates = read_chembl_updates(chembl_tsv_path)
        except (OSError, ValueError, csv.Error) as exc:
            cli_error(f"failed to read chembl tsv: {exc}")
            return 2

        if chembl_updates:
            nodes = enrich_nodes_v1(
                base_nodes=nodes,
                node_updates=chembl_updates,
                display_name_priority_by_type=display_name_priority_by_type,
            )
    if args.enrich_umls_tsv:
        umls_tsv_path = Path(args.enrich_umls_tsv)
        if not umls_tsv_path.is_file():
            cli_error(f"--enrich-umls-tsv is not a file: {umls_tsv_path}")
            return 2
        try:
            umls_updates = read_umls_updates(umls_tsv_path)
        except (OSError, ValueError, csv.Error) as exc:
            cli_error(f"failed to read umls tsv: {exc}")
            return 2

        if umls_updates:
            nodes = enrich_nodes_v1(
                base_nodes=nodes,
                node_updates=umls_updates,
                display_name_priority_by_type=display_name_priority_by_type,
            )

    if clinvar_node_updates:
        nodes = enrich_nodes_v1(
            base_nodes=nodes,
            node_updates=clinvar_node_updates,
            display_name_priority_by_type=display_name_priority_by_type,
        )

    if args.chebi_synonyms_tsv:
        chebi_synonyms_path = Path(args.chebi_synonyms_tsv)
        if chebi_synonyms_path.is_file():
            try:
                chebi_syn_updates = read_chebi_synonym_updates(
                    chebi_synonyms_path,
                    source_version=str(args.chebi_synonyms_version),
                )
            except (OSError, ValueError, csv.Error) as exc:
                if strict:
                    cli_error(f"failed to read chebi synonym updates: {exc}")
                    return 2
                cli_warning(f"skipping chebi synonym updates (parse error): {exc}")
            else:
                if chebi_syn_updates:
                    nodes = enrich_nodes_v1(
                        base_nodes=nodes,
                        node_updates=chebi_syn_updates,
                        display_name_priority_by_type=display_name_priority_by_type,
                    )
                    chebi_synonyms_used = True

    hgnc_json_path = _apath(args.enrich_hgnc_json)
    if hgnc_json_path and hgnc_json_path.is_file():
        try:
            hgnc_syn_updates = read_hgnc_synonym_updates(hgnc_json_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if strict:
                cli_error(f"failed to read hgnc synonym updates: {exc}")
                return 2
            cli_warning(f"skipping hgnc synonym updates (parse error): {exc}")
        else:
            if hgnc_syn_updates:
                nodes = enrich_nodes_v1(
                    base_nodes=nodes,
                    node_updates=hgnc_syn_updates,
                    display_name_priority_by_type=display_name_priority_by_type,
                )
                hgnc_synonyms_used = True

    if args.enrich_mondo_obo:
        mondo_obo_path = Path(args.enrich_mondo_obo)
        if mondo_obo_path.is_file():
            try:
                mondo_syn = read_mondo_synonym_updates(mondo_obo_path)
            except (OSError, ValueError) as exc:
                if strict:
                    cli_error(f"failed to read mondo synonym updates: {exc}")
                    return 2
                cli_warning(f"skipping mondo synonym updates (parse error): {exc}")
            else:
                bound = bind_mondo_synonym_updates_to_nodes(mondo_syn, nodes)
                if bound:
                    nodes = enrich_nodes_v1(
                        base_nodes=nodes,
                        node_updates=bound,
                        display_name_priority_by_type=display_name_priority_by_type,
                    )
                    mondo_synonyms_used = True

    if (
        args.drugcomb_summary_tsv
        and args.drugcomb_identifiers_tsv
        and "drugcomb" in effective
    ):
        dc_ids = Path(args.drugcomb_identifiers_tsv)
        dc_summary = Path(args.drugcomb_summary_tsv)
        if dc_ids.is_file() and dc_summary.is_file():
            try:
                drugcomb_syn_updates = read_drugcomb_synonym_updates(
                    dc_ids,
                    source_version=str(args.drugcomb_version),
                )
            except (OSError, ValueError, csv.Error) as exc:
                if strict:
                    cli_error(f"failed to read drugcomb synonym updates: {exc}")
                    return 2
                cli_warning(f"skipping drugcomb synonym updates (parse error): {exc}")
            else:
                if drugcomb_syn_updates:
                    nodes = enrich_nodes_v1(
                        base_nodes=nodes,
                        node_updates=drugcomb_syn_updates,
                        display_name_priority_by_type=display_name_priority_by_type,
                    )
                    drugcomb_synonyms_used = True

    sources: list[dict[str, Any]] = [
        primekg_kg_source(path=kg_csv),
        _relation_registry_manifest_source(),
    ]
    sources.extend(
        _optional_manifest_sources(
            args,
            corum_used=corum_used,
            reactome_used=reactome_used,
            signor_used=signor_used,
            stitch_used=stitch_used,
            ctd_used=ctd_used,
            complexportal_used=complexportal_used,
            drugcomb_used=drugcomb_used,
            drugcomb_synonyms_used=drugcomb_synonyms_used,
            chebi_synonyms_used=chebi_synonyms_used,
            hgnc_synonyms_used=hgnc_synonyms_used,
            clinvar_used=clinvar_used,
        )
    )
    determinism_params: dict[str, Any] = {"policy_version": DETERMINISM_POLICY_VERSION}
    if (
        corum_used
        or reactome_used
        or signor_used
        or stitch_used
        or ctd_used
        or complexportal_used
        or drugcomb_used
        or drugcomb_synonyms_used
        or chebi_synonyms_used
        or hgnc_synonyms_used
        or mondo_synonyms_used
        or clinvar_used
    ):
        determinism_params["curated_fetch_version"] = CURATED_FETCH_DETERMINISM_VERSION

    builders: list[dict[str, Any]] = [
        {
            "name": "build_profile",
            "version": 2,
            "params": {
                "profile": profile.value,
                "effective_segments": sorted(effective),
            },
        },
        {"name": "determinism_policy", "version": 1, "params": determinism_params},
    ]
    if enrichment is not EnrichmentProfile.NONE:
        builders.append(
            enrichment_profile_manifest_builder(
                enrichment,
                primekg_dir=args.primekg_dir,
                enrichment_input_dir=args.enrichment_input_dir,
            )
        )
    fr_builder = feature_refs_enrichment_manifest_builder(
        enrichment_sources=_feature_refs_enrichment_sources(args)
    )
    if fr_builder is not None:
        builders.append(fr_builder)
    if "moa_triad" in effective:
        builders.append(
            {
                "name": "moa_triad",
                "version": 1,
                "params": {"primekg_version": args.primekg_version},
            }
        )
    if "phenotype" in effective:
        builders.append(
            {
                "name": "phenotype",
                "version": PHENOTYPE_BUILDER_VERSION,
                "params": {"emit_composed": emit_phenotype_composed},
            }
        )
    if "registry_atomic" in effective:
        builders.append(registry_atomic_entry)
    if "drug_disease_shared_target_bundle" in effective:
        builders.append(
            {
                "name": "drug_disease_shared_target_bundle",
                "version": 2,
                "params": {
                    "primekg_version": args.primekg_version,
                    "max_drugs_per_bundle": 2,
                    "max_eligible_drugs": 8,
                    "max_bundles_per_protein_disease": 32,
                },
            }
        )
    if "disease_protein_bundle" in effective:
        builders.append(
            {
                "name": "disease_protein_bundle",
                "version": 1,
                "params": {
                    "primekg_version": args.primekg_version,
                    "max_proteins_per_disease": 25,
                    "min_proteins_per_bundle": 2,
                },
            }
        )
    if "protein_disease_bundle" in effective:
        builders.append(
            {
                "name": "protein_disease_bundle",
                "version": 1,
                "params": {
                    "primekg_version": args.primekg_version,
                    "max_diseases_per_protein": 25,
                    "min_diseases_per_bundle": 2,
                },
            }
        )
    if "protein_protein_bundle" in effective:
        builders.append(
            {
                "name": "protein_protein_bundle",
                "version": 2,
                "params": {
                    "primekg_version": args.primekg_version,
                    "max_partners_per_anchor": 8,
                    "min_partners_per_bundle": 2,
                },
            }
        )
    if corum_used:
        builders.append(
            {
                "name": "corum",
                "version": 1,
                "params": {"corum_version": args.corum_version},
            }
        )
    if reactome_used:
        builders.append(
            {
                "name": "reactome",
                "version": 1,
                "params": {"reactome_version": args.reactome_version},
            }
        )
    if signor_used:
        builders.append(
            {
                "name": "signor",
                "version": 2,
                "params": {"signor_version": args.signor_version},
            }
        )
    if stitch_used:
        builders.append(
            {
                "name": "stitch",
                "version": 1,
                "params": {
                    "stitch_version": args.stitch_version,
                    "score_threshold": int(args.stitch_score_threshold),
                    "coverage": stitch_stats,
                },
            }
        )
    if ctd_used:
        builders.append(
            {
                "name": "ctd",
                "version": 1,
                "params": {
                    "ctd_version": args.ctd_version,
                    "coverage": ctd_stats,
                },
            }
        )
    if complexportal_used:
        builders.append(
            {
                "name": "complexportal",
                "version": 1,
                "params": {"complexportal_version": args.complexportal_version},
            }
        )
    if drugcomb_used:
        builders.append(
            {
                "name": "drugcomb",
                "version": 1,
                "params": {"drugcomb_version": args.drugcomb_version},
            }
        )
    if drugcomb_synonyms_used:
        builders.append(
            {
                "name": "drugcomb_synonyms",
                "version": 1,
                "params": {"drugcomb_version": args.drugcomb_version},
            }
        )
    if chebi_synonyms_used:
        builders.append(
            {
                "name": "chebi_synonyms",
                "version": 1,
                "params": {"chebi_synonyms_version": args.chebi_synonyms_version},
            }
        )
    if hgnc_synonyms_used:
        hgnc_json = _apath(args.enrich_hgnc_json)
        builders.append(
            {
                "name": "hgnc_synonyms",
                "version": 1,
                "params": {
                    "source_version": hgnc_json.name if hgnc_json is not None else ""
                },
            }
        )
    if mondo_synonyms_used:
        mondo_obo = _apath(args.enrich_mondo_obo)
        builders.append(
            {
                "name": "mondo_synonyms",
                "version": 1,
                "params": {
                    "source_version": mondo_obo.name if mondo_obo is not None else ""
                },
            }
        )
    if clinvar_used:
        builders.append(
            {
                # version 2 (0.8.1): variant nodes now carry HGVS name / dbSNP xref /
                # type+coords source_meta. This changes node-table content (and thus the
                # whole-build inputs_fingerprint) while he_id / HE structure are unchanged —
                # the rule_trace.version in builders/clinvar.py stays 1 on purpose so existing
                # variant he_ids are byte-identical. See docs/releases/0.8.1.md.
                "name": "clinvar",
                "version": 2,
                "params": {"clinvar_version": args.clinvar_version},
            }
        )
    if args.chebi_names_tsv:
        builders.append(
            {
                "name": "chebi_names",
                "version": 1,
                "params": {"chebi_names_version": args.chebi_names_version},
            }
        )
    if args.chebi_definitions_tsv:
        builders.append(
            {
                "name": "chebi_definitions",
                "version": 1,
                "params": {"chebi_definitions_version": args.chebi_definitions_version},
            }
        )
    if args.emit_graph_csr_sidecar:
        builders.append(
            {
                "name": "graph_csr_sidecar",
                "version": 1,
                "params": {"file_version": "graph_csr_v1"},
            }
        )
    if args.emit_inverted_index_sidecar:
        builders.append(
            {
                "name": "inverted_index_sidecar",
                "version": 1,
                "params": {"file_version": "1"},
            }
        )
    if args.emit_he_row_index_sidecar:
        builders.append(
            {
                "name": "he_row_index_sidecar",
                "version": 1,
                "params": {"file_version": "1"},
            }
        )
    if args.emit_label_embedding_sidecar:
        builders.append(
            {
                "name": "label_embedding_sidecar",
                "version": 1,
                "params": {"file_version": "1"},
            }
        )
    if args.emit_chemical_xref_sidecar:
        builders.append(
            {
                "name": "chemical_xref_sidecar",
                "version": 1,
                "params": {"file_version": "chemical_xref_v1"},
            }
        )

    fr_violations = validate_nodes_feature_refs_indirection(nodes)
    if fr_violations:
        cli_error(f"nodes.feature_refs policy violation: {fr_violations[0]}")
        return 2

    build_report_extra: dict[str, Any] | None = None
    report_parts: dict[str, Any] = {}
    if go_enrichment_coverage is not None:
        report_parts["go_enrichment_coverage"] = go_enrichment_coverage
    if pathway_enrichment_coverage is not None:
        report_parts["pathway_enrichment_coverage"] = pathway_enrichment_coverage
    if anatomy_enrichment_coverage is not None:
        report_parts["anatomy_enrichment_coverage"] = anatomy_enrichment_coverage
    if dedup_stats is not None:
        report_parts["hyperedge_dedup"] = dedup_stats
    if report_parts:
        build_report_extra = report_parts

    relation_coverage_audit = (
        build_relation_coverage_audit(
            triplets,
            merged["hyperedges"],
        )
        if args.emit_build_report
        else None
    )

    write_hf_tables_to_disk(
        out_dir,
        nodes=nodes,
        hyperedges=merged["hyperedges"],
        support=merged["support"],
        manifest_schema_version=str(args.schema_version),
        manifest_sources=sources,
        manifest_builders=builders,
        manifest_build_profile=profile.value,
        manifest_extra_top_level={"effective_segments": sorted(effective)},
        emit_build_report=bool(args.emit_build_report),
        build_report_extra=build_report_extra,
        corum_tsv=str(args.corum_tsv) if corum_used and args.corum_tsv else None,
        uniprot_to_geneid=uniprot_to_geneid,
        builder_segments=segments,
        relation_coverage_audit=relation_coverage_audit,
    )

    sidecar_patch = write_consumer_sidecars(out_dir)
    merge_manifest_sidecars(out_dir, sidecar_patch)

    if args.emit_graph_csr_sidecar:
        manifest_path = out_dir / "build_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sidecar_paths = emit_graph_csr_sidecar_for_build(
            merged["hyperedges"],
            inputs_fingerprint=str(manifest["inputs_fingerprint"]),
            out_dir=out_dir,
            artifacts_root=Path(args.artifacts_root),
        )
        merge_manifest_sidecars(out_dir, {"graph_csr_v1": sidecar_paths["dataset_dir"]})

    if args.emit_inverted_index_sidecar:
        manifest_path = out_dir / "build_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rel = emit_inverted_index_sidecar_for_build(
            merged["hyperedges"],
            inputs_fingerprint=str(manifest["inputs_fingerprint"]),
            out_dir=out_dir,
            artifacts_root=Path(args.artifacts_root),
        )
        merge_manifest_sidecars(out_dir, {"inverted_index": rel})

    if args.emit_he_row_index_sidecar:
        manifest_path = out_dir / "build_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rel = emit_he_row_index_sidecar_for_build(
            merged["hyperedges"],
            inputs_fingerprint=str(manifest["inputs_fingerprint"]),
            out_dir=out_dir,
            artifacts_root=Path(args.artifacts_root),
        )
        merge_manifest_sidecars(out_dir, {"he_row_index": rel})

    if args.emit_label_embedding_sidecar:
        from primehkg.linking.sapbert import (
            DEFAULT_SAPBERT_MODEL_ID,
            encode_texts_sapbert,
        )

        manifest_path = out_dir / "build_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        nodes_list = [dict(row) for row in nodes]
        rel = emit_label_embedding_sidecar_for_build(
            nodes_list,
            inputs_fingerprint=str(manifest["inputs_fingerprint"]),
            out_dir=out_dir,
            encode_fn=encode_texts_sapbert,
            model_id=DEFAULT_SAPBERT_MODEL_ID,
            artifacts_root=Path(args.artifacts_root),
        )
        merge_manifest_sidecars(out_dir, {"label_embedding_v1": rel})

    if args.emit_chemical_xref_sidecar:
        manifest_path = out_dir / "build_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        live_ids = {str(row["node_id"]) for row in nodes}
        ctd_v2 = args.ctd_mesh_pubchem_v2_tsv
        unichem_v2 = args.unichem_pubchem_xref_v2_tsv
        if ctd_v2 is None or unichem_v2 is None:
            cli_error(
                "--emit-chemical-xref-sidecar requires "
                "--ctd-mesh-pubchem-v2 and --unichem-pubchem-xref-v2"
            )
            return 2
        emit_chemical_xref_sidecar_for_build(
            ctd_mesh_pubchem_v2=ctd_v2,
            unichem_pubchem_xref_v2=unichem_v2,
            live_node_ids=live_ids,
            inputs_fingerprint=str(manifest["inputs_fingerprint"]),
            out_dir=out_dir,
            artifacts_root=Path(args.artifacts_root),
            merge_manifest=True,
        )
        if args.materialize_unambiguous_mesh_xrefs:
            from primehkg.core.dataset import load_core_dataset
            from primehkg.linking.chemical_xref_materialize import (
                materialize_unambiguous_mesh_xrefs_on_disk,
            )

            ds_mat = load_core_dataset(out_dir)
            index = ds_mat.chemical_xref_index
            if index is None:
                cli_error("chemical_xref_v1 unloadable after emit; cannot materialize")
                return 2
            materialize_unambiguous_mesh_xrefs_on_disk(out_dir, index)

    return 0


app = typer.Typer(
    rich_markup_mode="rich",
    help=(
        "Build a local PrimeHKG dataset from PrimeKG kg.csv "
        "(v0 builders + optional v1 enrichment)."
    ),
)


def _argv_for_build_cli(argv: list[str] | None) -> list[str]:
    """Args for the implicit ``build`` Typer command (``get_command`` returns that command)."""
    a = list(argv) if argv is not None else list(sys.argv[1:])
    if a and a[0] == "build":
        return a[1:]
    return a


@app.command("build")
def build_command(
    kg_csv: Path = typer.Option(
        ...,
        "--kg-csv",
        help="Path to PrimeKG kg.csv (supports headers {x_id,relation,y_id} or {s,r,t}).",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        help="Output directory to write a datasets.DatasetDict via save_to_disk.",
    ),
    schema_version: str = typer.Option(
        DEFAULT_MANIFEST_SCHEMA_VERSION,
        "--schema-version",
        help=f"Schema version recorded in build_manifest.json (default: {DEFAULT_MANIFEST_SCHEMA_VERSION}).",
    ),
    primekg_version: str = typer.Option(
        "unknown",
        "--primekg-version",
        help="PrimeKG source version string used in support provenance (default: unknown).",
    ),
    build_profile: str = typer.Option(
        "primehkg",
        "--build-profile",
        help=(
            "Builder preset: primehkg (default, variant-FREE PrimeKG-complete K-layer), "
            "primehkg_clinical (primehkg + ClinVar variant), or smoke (CI/fixtures). "
            "Adjust per-build with --include-segments / --exclude-segments."
        ),
    ),
    include_segments: str = typer.Option(
        "",
        "--include-segments",
        help="Comma-separated builder segments to ADD to the profile preset (e.g. clinvar).",
    ),
    exclude_segments: str = typer.Option(
        "",
        "--exclude-segments",
        help="Comma-separated builder segments to REMOVE from the profile preset.",
    ),
    corum_tsv: Path | None = typer.Option(
        None,
        "--corum-tsv",
        help="Optional CORUM complex participants TSV for curated protein complex hyperedges.",
    ),
    corum_version: str = typer.Option(
        "unknown",
        "--corum-version",
        help="CORUM source version string used in support provenance (default: unknown).",
    ),
    reactome_tsv: Path | None = typer.Option(
        None,
        "--reactome-tsv",
        help="Optional Reactome reactions TSV for curated reaction-event hyperedges.",
    ),
    reactome_version: str = typer.Option(
        "unknown",
        "--reactome-version",
        help="Reactome source version string used in support provenance (default: unknown).",
    ),
    signor_tsv: Path | None = typer.Option(
        None,
        "--signor-tsv",
        help="Optional SIGNOR human TSV (signed/directional regulation). Required under the primehkg profile.",
    ),
    signor_version: str = typer.Option(
        "signor:unknown",
        "--signor-version",
        help="Version string recorded for the SIGNOR source.",
    ),
    stitch_tsv: Path | None = typer.Option(
        None,
        "--stitch-tsv",
        help="Optional normalized STITCH links TSV (chemical-protein). Opt-in segment 'stitch'.",
    ),
    stitch_version: str = typer.Option(
        "stitch:unknown",
        "--stitch-version",
        help="Version string recorded for the STITCH source.",
    ),
    stitch_score_threshold: int = typer.Option(
        700,
        "--stitch-score-threshold",
        help="Minimum STITCH combined_score (0-1000) to emit an edge.",
    ),
    ctd_tsv: Path | None = typer.Option(
        None,
        "--ctd-tsv",
        help="Optional normalized CTD chemical-gene TSV (regulation). Required under the primehkg profile.",
    ),
    ctd_mesh_cid_tsv: Path | None = typer.Option(
        None,
        "--ctd-mesh-cid-tsv",
        help="CTD MeSH->PubChem CID map TSV (required with --ctd-tsv).",
    ),
    ctd_version: str = typer.Option(
        "ctd:unknown",
        "--ctd-version",
        help="Version string recorded for the CTD source.",
    ),
    unichem_cid_map_tsv: Path | None = typer.Option(
        None,
        "--unichem-cid-map-tsv",
        help="UniChem CID->DrugBank/ChEBI map TSV (required with --stitch-tsv and --ctd-tsv).",
    ),
    string_ensp_geneid_tsv: Path | None = typer.Option(
        None,
        "--string-ensp-geneid-tsv",
        help="STRING ENSP->GeneID map TSV (required with --stitch-tsv).",
    ),
    complexportal_tsv: Path | None = typer.Option(
        None,
        "--complexportal-tsv",
        help="Optional Complex Portal human complexes TSV (protein_complex).",
    ),
    complexportal_version: str = typer.Option(
        "complexportal:unknown",
        "--complexportal-version",
        help="Version string recorded for the Complex Portal source.",
    ),
    drugcomb_summary_tsv: Path | None = typer.Option(
        None,
        "--drugcomb-summary-tsv",
        help="Optional DrugComb summary CSV (drug_combination).",
    ),
    drugcomb_identifiers_tsv: Path | None = typer.Option(
        None,
        "--drugcomb-identifiers-tsv",
        help="DrugComb name->drugbank_id identifiers TSV (required with --drugcomb-summary-tsv).",
    ),
    drugcomb_version: str = typer.Option(
        "drugcomb:unknown",
        "--drugcomb-version",
        help="Version string recorded for the DrugComb source.",
    ),
    clinvar_tsv: Path | None = typer.Option(
        None,
        "--clinvar-tsv",
        help="Optional ClinVar variant_summary.txt.gz (variant_gene / variant_disease).",
    ),
    clinvar_version: str = typer.Option(
        "clinvar:unknown",
        "--clinvar-version",
        help="Version string recorded for the ClinVar source.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail on the first malformed CSV row (default: strict).",
    ),
    non_strict: bool = typer.Option(
        False,
        "--non-strict",
        help="Skip malformed rows in kg.csv instead of failing.",
    ),
    emit_build_report: bool = typer.Option(
        False,
        "--emit-build-report",
        help="If set, write build_report.json audit metrics next to build_manifest.json.",
    ),
    emit_graph_csr_sidecar: bool = typer.Option(
        False,
        "--emit-graph-csr-sidecar",
        help=(
            "If set, build mmap CSR sidecar under "
            "artifacts/builds/<inputs_fingerprint>/sidecars/graph_csr_v1 "
            "and mirror to <out-dir>/sidecars/graph_csr_v1."
        ),
    ),
    emit_inverted_index_sidecar: bool = typer.Option(
        False,
        "--emit-inverted-index-sidecar",
        help=(
            "If set, write inverted-index JSON sidecar under "
            "artifacts/builds/<inputs_fingerprint>/sidecars/inverted_index_v1 "
            "and mirror to <out-dir>/sidecars/inverted_index_v1/inverted_index.json."
        ),
    ),
    emit_he_row_index_sidecar: bool = typer.Option(
        False,
        "--emit-he-row-index-sidecar",
        help=(
            "If set, write he_id→row JSON sidecar under "
            "artifacts/builds/<inputs_fingerprint>/sidecars/he_row_index_v1 "
            "and mirror to <out-dir>/sidecars/he_row_index_v1/he_row_index.json."
        ),
    ),
    emit_label_embedding_sidecar: bool = typer.Option(
        False,
        "--emit-label-embedding-sidecar",
        help=(
            "If set, encode node labels with SapBERT and write "
            "sidecars/label_embedding_v1/ (requires uv sync --group embed)."
        ),
    ),
    emit_chemical_xref_sidecar: bool = typer.Option(
        False,
        "--emit-chemical-xref-sidecar",
        help=(
            "If set, compose chemical_xref_v1 from lossless CTD/UniChem v2 TSVs "
            "(requires --ctd-mesh-pubchem-v2 and --unichem-pubchem-xref-v2)."
        ),
    ),
    ctd_mesh_pubchem_v2: Path | None = typer.Option(
        None,
        "--ctd-mesh-pubchem-v2",
        help="Lossless CTD MeSH→PubChem v2 TSV (for --emit-chemical-xref-sidecar).",
    ),
    unichem_pubchem_xref_v2: Path | None = typer.Option(
        None,
        "--unichem-pubchem-xref-v2",
        help="Lossless UniChem PubChem xref v2 TSV (for --emit-chemical-xref-sidecar).",
    ),
    materialize_unambiguous_mesh_xrefs: bool = typer.Option(
        True,
        "--materialize-unambiguous-mesh-xrefs/--no-materialize-unambiguous-mesh-xrefs",
        help=(
            "After --emit-chemical-xref-sidecar, write unambiguous MeSH into "
            "nodes.xrefs.mesh (default: on)."
        ),
        rich_help_panel="Chemical xref",
    ),
    use_builder_cache: bool = typer.Option(
        False,
        "--use-builder-cache",
        help=(
            "Cache triplet-builder stages (moa_triad, phenotype, "
            "registry_atomic, drug_disease_shared_target_bundle) "
            "under <out-dir>/builder/."
        ),
    ),
    artifacts_root: Path = typer.Option(
        Path("artifacts"),
        "--artifacts-root",
        file_okay=False,
        dir_okay=True,
        help="Root directory for git-free build snapshots (default: ./artifacts).",
    ),
    enrich_nodes: Path | None = typer.Option(
        None,
        "--enrich-nodes",
        help="Optional directory containing local node enrichment fixtures (v1). Reads: drug_xrefs.json",
    ),
    enrich_xrefs_tsv: Path | None = typer.Option(
        None,
        "--enrich-xrefs-tsv",
        help="Optional TSV (node_id, namespace, id) to enrich node xrefs (v1).",
    ),
    ingest_primekg_names: bool = typer.Option(
        False,
        "--ingest-primekg-names",
        help=(
            "Read optional endpoint label columns from kg.csv (e.g. x_name/y_name) "
            "and merge into node display names with provenance (v1)."
        ),
    ),
    enrich_hgnc_tsv: Path | None = typer.Option(
        None,
        "--enrich-hgnc-tsv",
        help="Optional HGNC TSV for gene node enrichment (v1).",
    ),
    enrich_hgnc_json: Path | None = typer.Option(
        None,
        "--enrich-hgnc-json",
        help="Optional HGNC JSON (response.docs) for protein synonym enrichment (v1).",
    ),
    enrich_uniprot_tsv: Path | None = typer.Option(
        None,
        "--enrich-uniprot-tsv",
        help="Optional UniProt TSV for protein node enrichment (v1).",
    ),
    enrich_hpo_tsv: Path | None = typer.Option(
        None,
        "--enrich-hpo-tsv",
        help="Optional TSV mapping node_id to HPO (columns: node_id, hp_id; optional hp_label) (v1).",
    ),
    enrich_go_tsv: Path | None = typer.Option(
        None,
        "--enrich-go-tsv",
        help=(
            "Optional TSV mapping GO-family node_id to GO xrefs "
            "(columns: node_id, go_id; optional go_name, definition) (v1)."
        ),
    ),
    enrich_pathway_tsv: Path | None = typer.Option(
        None,
        "--enrich-pathway-tsv",
        help=(
            "Optional TSV mapping pathway node_id to Reactome xrefs "
            "(columns: node_id, reactome_id; optional pathway_name, description) (v1)."
        ),
    ),
    enrich_anatomy_tsv: Path | None = typer.Option(
        None,
        "--enrich-anatomy-tsv",
        help=(
            "Optional TSV mapping anatomy node_id to anatomy xrefs "
            "(columns: node_id, anatomy_id; optional anatomy_name, description) (v1)."
        ),
    ),
    enrich_mondo_tsv: Path | None = typer.Option(
        None,
        "--enrich-mondo-tsv",
        help="Optional TSV mapping disease node_id to MONDO xrefs (v1).",
    ),
    enrich_mondo_obo: Path | None = typer.Option(
        None,
        "--enrich-mondo-obo",
        help=(
            "Optional MONDO OBO file. When provided together with "
            "--enrich-mondo-tsv, adds MeSH/OMIM/UMLS xrefs to disease "
            "nodes transitively through their MONDO IDs."
        ),
    ),
    enrich_drugbank_tsv: Path | None = typer.Option(
        None,
        "--enrich-drugbank-tsv",
        help="Optional TSV mapping drug node_id to DrugBank xrefs (v1).",
    ),
    chebi_names_tsv: Path | None = typer.Option(
        None,
        "--chebi-names-tsv",
        help="Optional TSV (chebi_node_id<TAB>name) resolving metabolite display names.",
    ),
    chebi_names_version: str = typer.Option(
        "chebi:unknown",
        "--chebi-names-version",
        help="Source version recorded for resolved ChEBI compound names (e.g. chebi:235).",
    ),
    chebi_synonyms_tsv: Path | None = typer.Option(
        None,
        "--chebi-synonyms-tsv",
        help="Optional TSV (chebi_node_id<TAB>surface) for metabolite synonym enrichment.",
    ),
    chebi_synonyms_version: str = typer.Option(
        "chebi:unknown",
        "--chebi-synonyms-version",
        help="Source version recorded for ChEBI synonym enrichment (e.g. chebi:235).",
    ),
    chebi_definitions_tsv: Path | None = typer.Option(
        None,
        "--chebi-definitions-tsv",
        help="ChEBI definitions TSV (chebi_node_id, definition) -> text feature_refs.",
    ),
    chebi_definitions_version: str = typer.Option(
        "chebi:unknown",
        "--chebi-definitions-version",
        help="Version tag recorded for ChEBI definitions enrichment.",
    ),
    enrich_rxnorm_tsv: Path | None = typer.Option(
        None,
        "--enrich-rxnorm-tsv",
        help="Optional TSV mapping drug node_id to RxNorm xrefs (v1).",
    ),
    enrich_chembl_tsv: Path | None = typer.Option(
        None,
        "--enrich-chembl-tsv",
        help="Optional TSV mapping drug node_id to ChEMBL xrefs (v1).",
    ),
    enrich_umls_tsv: Path | None = typer.Option(
        None,
        "--enrich-umls-tsv",
        help="Optional TSV mapping node_id to UMLS CUIs (v1).",
    ),
    display_name_priority_json: Path | None = typer.Option(
        None,
        "--display-name-priority-json",
        help="Optional JSON mapping node_type -> [source, ...] for display name priority (v1).",
    ),
    enrich_primekg_disease_features_tab: Path | None = typer.Option(
        None,
        "--enrich-primekg-disease-features-tab",
        help="Optional PrimeKG disease_features.tab (join on kg.csv node_index) (v1).",
    ),
    enrich_primekg_drug_features_tab: Path | None = typer.Option(
        None,
        "--enrich-primekg-drug-features-tab",
        help="Optional PrimeKG drug_features.tab (join on kg.csv node_index) (v1).",
    ),
    enrich_primekg_nodes_tab: Path | None = typer.Option(
        None,
        "--enrich-primekg-nodes-tab",
        help="Optional PrimeKG nodes.tab for display names (v1).",
    ),
    primekg_edges_csv: Path | None = typer.Option(
        None,
        "--primekg-edges-csv",
        help=(
            "Optional PrimeKG edges.csv appended after kg.csv triplets; deduped against kg rows."
        ),
    ),
    enrichment_profile: str = typer.Option(
        "none",
        "--enrichment-profile",
        help=(
            "Node enrichment preset: none (default), reference (auto-apply PrimeKG "
            "nodes.tab / feature tabs from --primekg-dir or kg.csv parent directory), "
            "or extended (+ --enrichment-input-dir export filenames)."
        ),
    ),
    enrichment_input_dir: Path | None = typer.Option(
        None,
        "--enrichment-input-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory with standard Extended-tier export filenames (extended profile).",
    ),
    primekg_dir: Path | None = typer.Option(
        None,
        "--primekg-dir",
        help=(
            "PrimeKG bundle directory for --enrichment-profile reference "
            "(default: parent directory of --kg-csv)."
        ),
    ),
) -> None:
    bargs = BuildCliArgs(
        kg_csv=kg_csv,
        out_dir=out_dir,
        schema_version=schema_version,
        primekg_version=primekg_version,
        corum_tsv=corum_tsv,
        corum_version=corum_version,
        reactome_tsv=reactome_tsv,
        reactome_version=reactome_version,
        signor_tsv=signor_tsv,
        signor_version=signor_version,
        stitch_tsv=stitch_tsv,
        stitch_version=stitch_version,
        stitch_score_threshold=stitch_score_threshold,
        ctd_tsv=ctd_tsv,
        ctd_mesh_cid_tsv=ctd_mesh_cid_tsv,
        ctd_version=ctd_version,
        unichem_cid_map_tsv=unichem_cid_map_tsv,
        string_ensp_geneid_tsv=string_ensp_geneid_tsv,
        complexportal_tsv=complexportal_tsv,
        complexportal_version=complexportal_version,
        drugcomb_summary_tsv=drugcomb_summary_tsv,
        drugcomb_identifiers_tsv=drugcomb_identifiers_tsv,
        drugcomb_version=drugcomb_version,
        clinvar_tsv=clinvar_tsv,
        clinvar_version=clinvar_version,
        strict=strict,
        non_strict=non_strict,
        emit_build_report=emit_build_report,
        emit_graph_csr_sidecar=emit_graph_csr_sidecar,
        emit_inverted_index_sidecar=emit_inverted_index_sidecar,
        emit_he_row_index_sidecar=emit_he_row_index_sidecar,
        emit_label_embedding_sidecar=emit_label_embedding_sidecar,
        emit_chemical_xref_sidecar=emit_chemical_xref_sidecar,
        ctd_mesh_pubchem_v2_tsv=ctd_mesh_pubchem_v2,
        unichem_pubchem_xref_v2_tsv=unichem_pubchem_xref_v2,
        materialize_unambiguous_mesh_xrefs=materialize_unambiguous_mesh_xrefs,
        use_builder_cache=use_builder_cache,
        artifacts_root=artifacts_root,
        enrich_nodes=enrich_nodes,
        enrich_xrefs_tsv=enrich_xrefs_tsv,
        ingest_primekg_names=ingest_primekg_names,
        enrich_hgnc_tsv=enrich_hgnc_tsv,
        enrich_hgnc_json=enrich_hgnc_json,
        enrich_uniprot_tsv=enrich_uniprot_tsv,
        enrich_hpo_tsv=enrich_hpo_tsv,
        enrich_go_tsv=enrich_go_tsv,
        enrich_pathway_tsv=enrich_pathway_tsv,
        enrich_anatomy_tsv=enrich_anatomy_tsv,
        enrich_mondo_tsv=enrich_mondo_tsv,
        enrich_mondo_obo=enrich_mondo_obo,
        enrich_drugbank_tsv=enrich_drugbank_tsv,
        chebi_names_tsv=chebi_names_tsv,
        chebi_names_version=chebi_names_version,
        chebi_synonyms_tsv=chebi_synonyms_tsv,
        chebi_synonyms_version=chebi_synonyms_version,
        chebi_definitions_tsv=chebi_definitions_tsv,
        chebi_definitions_version=chebi_definitions_version,
        enrich_rxnorm_tsv=enrich_rxnorm_tsv,
        enrich_chembl_tsv=enrich_chembl_tsv,
        enrich_umls_tsv=enrich_umls_tsv,
        display_name_priority_json=display_name_priority_json,
        enrich_primekg_disease_features_tab=enrich_primekg_disease_features_tab,
        enrich_primekg_drug_features_tab=enrich_primekg_drug_features_tab,
        enrich_primekg_nodes_tab=enrich_primekg_nodes_tab,
        primekg_edges_csv=primekg_edges_csv,
        build_profile=build_profile,
        include_segments=tuple(
            s.strip() for s in include_segments.split(",") if s.strip()
        ),
        exclude_segments=tuple(
            s.strip() for s in exclude_segments.split(",") if s.strip()
        ),
        enrichment_profile=enrichment_profile,
        enrichment_input_dir=enrichment_input_dir,
        primekg_dir=primekg_dir,
    )
    raise typer.Exit(run_build(bargs))


def main(argv: list[str] | None = None) -> int:
    cmd = get_command(app)
    args_list = _argv_for_build_cli(argv)
    try:
        rc = cmd.main(
            args=args_list,
            prog_name="primehkg-build",
            standalone_mode=False,
        )
    except SystemExit as e:
        if isinstance(e.code, int):
            return e.code
        return 1
    return int(rc) if rc is not None else 0


def entrypoint() -> None:
    """Console script entrypoint (returns process exit code via `SystemExit`)."""
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
