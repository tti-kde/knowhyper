from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from primehkg.distribution.hub_layout import PRIMEHKG_FEATURES_HF_CONFIGS
from primehkg.features.chebi_drug_text import project_chebi_definitions_to_drugs
from primehkg.features.extract import (
    merge_smiles_gap_fill,
    read_anatomy_text_feature_rows,
    read_chebi_text_feature_rows,
    read_chembl_property_feature_rows,
    read_chembl_smiles_feature_rows,
    read_drugbank_property_feature_rows,
    read_drugbank_smiles_feature_rows,
    read_exposure_text_feature_rows,
    read_go_text_feature_rows,
    read_hgnc_text_feature_rows,
    read_hpo_text_feature_rows,
    read_mondo_text_feature_rows,
    read_nucleotide_sequence_feature_rows,
    read_pathway_text_feature_rows,
    read_uniprot_sequence_feature_rows,
    read_uniprot_text_feature_rows,
    structure_skeleton_feature_rows,
)
from primehkg.features.extract_disease_tab import (
    read_disease_features_property_rows,
    read_disease_features_text_rows,
)
from primehkg.features.extract_drug_tab import (
    read_drug_features_property_rows,
    read_drug_features_text_rows,
)
from primehkg.features.extract_hpo_property import read_hpo_property_rows
from primehkg.features.extract_uniprot_property import read_uniprot_property_rows
from primehkg.features.extract_variant_property import read_variant_property_rows
from primehkg.features.mondo_ambiguity import resolve_mondo_definition_ambiguity
from primehkg.features.provenance import (
    TextProvenanceRegistry,
    apply_text_provenance,
    load_text_provenance_registry,
    normalize_source_key,
)
from primehkg.features.synonym_context import read_synonym_context_rows
from primehkg.io.build_manifest import (
    DEFAULT_MANIFEST_SCHEMA_VERSION,
    assemble_build_manifest,
    sha256_file,
    stable_json_dumps,
    write_build_manifest_json,
)
from primehkg.io.cli_console import cli_warning
from primehkg.io.primekg_feature_tabs import build_node_index_to_node_id


def _load_core_nodes(core_dir: str | Path) -> list[dict[str, Any]]:
    """Load all core node rows once (reused for the ID filter + variant extractor).

    Materializes the full nodes table (~1.8M rows at reference scale) in memory; acceptable
    on the build host and avoids a second load_from_disk for the variant extractor.
    """
    from datasets import load_from_disk

    loaded = load_from_disk(str(core_dir))
    if not isinstance(loaded, DatasetDict):
        msg = f"expected DatasetDict at {core_dir!r}, got {type(loaded).__name__}"
        raise TypeError(msg)
    return [dict(row) for row in loaded["nodes"]]


def _core_node_ids(core_nodes: list[dict[str, Any]]) -> frozenset[str]:
    ids: set[str] = set()
    for row in core_nodes:
        nid = row.get("node_id")
        if isinstance(nid, str) and nid:
            ids.add(nid)
    return frozenset(ids)


def _warn_variant_unknowns(unknowns: dict[str, set[str]]) -> None:
    for pid, values in unknowns.items():
        if values:
            preview = ", ".join(sorted(values)[:10])
            cli_warning(
                f"variant: {len(values)} unknown {pid} value(s) "
                f"skipped (not in registry): {preview}"
            )


def _nullable_str(value: Any) -> str:
    return "" if value is None else str(value)


def _node_features_text_sort_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row["node_id"]),
        str(row["modality"]),
        str(row["source"]),
        str(row["source_version"]),
        str(row.get("source_field", "")),
        _nullable_str(row.get("upstream_release")),
        _nullable_str(row.get("license_id")),
    )


def finalize_text_provenance(
    tables: dict[str, list[dict[str, Any]]],
    *,
    registry_path: Path | None,
    artifact_paths_by_source: dict[str, Path],
    strict: bool,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
    TextProvenanceRegistry,
    dict[str, int],
]:
    """Apply text provenance registry to ``node_features_text`` and re-sort."""
    registry = load_text_provenance_registry(registry_path)
    artifact_sha_by_source: dict[str, str] = {}
    for source, path in artifact_paths_by_source.items():
        digest = sha256_file(path)
        artifact_sha_by_source[source] = digest
        artifact_sha_by_source[normalize_source_key(source)] = digest

    out = dict(tables)
    text_rows = out.get("node_features_text", [])
    if not isinstance(text_rows, list):
        msg = "table 'node_features_text' must be a list"
        raise TypeError(msg)

    resolved_rows, mondo_stats = resolve_mondo_definition_ambiguity(list(text_rows))
    merged_rows, completeness = apply_text_provenance(
        resolved_rows,
        registry,
        artifact_sha_by_source=artifact_sha_by_source,
        strict=strict,
    )
    merged_rows.sort(key=_node_features_text_sort_key)
    out["node_features_text"] = merged_rows
    return out, completeness, registry, mondo_stats


def build_features_tables(
    *,
    core_dir: str | Path | None = None,
    allowed_node_ids: frozenset[str] | None = None,
    uniprot_tsv: str | Path | None = None,
    chembl_tsv: str | Path | None = None,
    drugbank_smiles_tsv: str | Path | None = None,
    drugbank_properties_tsv: str | Path | None = None,
    mondo_tsv: str | Path | None = None,
    mondo_obo: str | Path | None = None,
    hpo_tsv: str | Path | None = None,
    structure_skeleton_json: str | Path | None = None,
    drug_features_tab: str | Path | None = None,
    disease_features_tab: str | Path | None = None,
    kg_csv: str | Path | None = None,
    hgnc_tsv: str | Path | None = None,
    nucleotide_tsv: str | Path | None = None,
    go_tsv: str | Path | None = None,
    pathway_tsv: str | Path | None = None,
    anatomy_tsv: str | Path | None = None,
    chebi_definitions_tsv: str | Path | None = None,
    unichem_cid_map_tsv: str | Path | None = None,
    exposure_tsv: str | Path | None = None,
    stats_out: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build feature tables keyed by HF config name."""
    node_ids = allowed_node_ids
    core_nodes: list[dict[str, Any]] | None = None
    if core_dir is not None:
        core_nodes = _load_core_nodes(core_dir)
        node_ids = _core_node_ids(core_nodes)

    tables: dict[str, list[dict[str, Any]]] = {
        "node_features_text": [],
        "node_features_sequence": [],
        "node_features_structure": [],
        "node_features_property": [],
    }

    if uniprot_tsv is not None:
        tables["node_features_sequence"].extend(
            read_uniprot_sequence_feature_rows(uniprot_tsv, allowed_node_ids=node_ids)
        )
        tables["node_features_text"].extend(
            read_uniprot_text_feature_rows(uniprot_tsv, allowed_node_ids=node_ids)
        )
        tables["node_features_property"].extend(
            read_uniprot_property_rows(uniprot_tsv, allowed_node_ids=node_ids)
        )
    if nucleotide_tsv is not None:
        protein_node_ids = {
            str(r["node_id"])
            for r in tables["node_features_sequence"]
            if r.get("sequence_type") == "protein" and r.get("node_id")
        }
        nucleotide_rows = read_nucleotide_sequence_feature_rows(
            nucleotide_tsv, allowed_node_ids=node_ids
        )
        tables["node_features_sequence"].extend(
            r for r in nucleotide_rows if str(r["node_id"]) not in protein_node_ids
        )
    if chembl_tsv is not None:
        tables["node_features_sequence"].extend(
            read_chembl_smiles_feature_rows(chembl_tsv, allowed_node_ids=node_ids)
        )
        tables["node_features_property"].extend(
            read_chembl_property_feature_rows(chembl_tsv, allowed_node_ids=node_ids)
        )
    if drugbank_properties_tsv is not None:
        tables["node_features_property"].extend(
            read_drugbank_property_feature_rows(
                drugbank_properties_tsv, allowed_node_ids=node_ids
            )
        )
    if drugbank_smiles_tsv is not None:
        fallback_rows = read_drugbank_smiles_feature_rows(
            drugbank_smiles_tsv,
            allowed_node_ids=node_ids,
        )
        tables["node_features_sequence"] = merge_smiles_gap_fill(
            tables["node_features_sequence"],
            fallback_rows,
        )
    if mondo_tsv is not None:
        tables["node_features_text"].extend(
            read_mondo_text_feature_rows(
                mondo_tsv,
                mondo_obo=Path(mondo_obo) if mondo_obo is not None else None,
                allowed_node_ids=node_ids,
            )
        )
    if hpo_tsv is not None:
        tables["node_features_text"].extend(
            read_hpo_text_feature_rows(hpo_tsv, allowed_node_ids=node_ids)
        )
        tables["node_features_property"].extend(
            read_hpo_property_rows(hpo_tsv, allowed_node_ids=node_ids)
        )
    if drug_features_tab is not None:
        if kg_csv is None:
            msg = "kg_csv is required when drug_features_tab is set"
            raise ValueError(msg)
        index_to_node_id = build_node_index_to_node_id(kg_csv)
        tables["node_features_text"].extend(
            read_drug_features_text_rows(
                drug_features_tab,
                index_to_node_id=index_to_node_id,
                allowed_node_ids=node_ids,
            )
        )
        tables["node_features_property"].extend(
            read_drug_features_property_rows(
                drug_features_tab,
                index_to_node_id=index_to_node_id,
                allowed_node_ids=node_ids,
            )
        )
    if disease_features_tab is not None:
        if kg_csv is None:
            msg = "kg_csv is required when disease_features_tab is set"
            raise ValueError(msg)
        index_to_node_id = build_node_index_to_node_id(kg_csv)
        tables["node_features_text"].extend(
            read_disease_features_text_rows(
                disease_features_tab,
                index_to_node_id=index_to_node_id,
                allowed_node_ids=node_ids,
            )
        )
        tables["node_features_property"].extend(
            read_disease_features_property_rows(
                disease_features_tab,
                index_to_node_id=index_to_node_id,
                allowed_node_ids=node_ids,
            )
        )
    if hgnc_tsv is not None:
        tables["node_features_text"].extend(
            read_hgnc_text_feature_rows(hgnc_tsv, allowed_node_ids=node_ids)
        )
    if go_tsv is not None:
        tables["node_features_text"].extend(
            read_go_text_feature_rows(go_tsv, allowed_node_ids=node_ids)
        )
    if pathway_tsv is not None:
        tables["node_features_text"].extend(
            read_pathway_text_feature_rows(pathway_tsv, allowed_node_ids=node_ids)
        )
    if anatomy_tsv is not None:
        tables["node_features_text"].extend(
            read_anatomy_text_feature_rows(anatomy_tsv, allowed_node_ids=node_ids)
        )
    if chebi_definitions_tsv is not None:
        tables["node_features_text"].extend(
            read_chebi_text_feature_rows(
                chebi_definitions_tsv, allowed_node_ids=node_ids
            )
        )
        if unichem_cid_map_tsv is not None:
            drug_def_rows, crosswalk_stats = project_chebi_definitions_to_drugs(
                chebi_definitions_tsv=Path(chebi_definitions_tsv),
                unichem_cid_map_tsv=Path(unichem_cid_map_tsv),
                allowed_node_ids=node_ids,
            )
            tables["node_features_text"].extend(drug_def_rows)
            if stats_out is not None:
                stats_out["entity_definition_crosswalk"] = crosswalk_stats
    if exposure_tsv is not None:
        tables["node_features_text"].extend(
            read_exposure_text_feature_rows(exposure_tsv, allowed_node_ids=node_ids)
        )
    if structure_skeleton_json is not None:
        skel = json.loads(Path(structure_skeleton_json).read_text(encoding="utf-8"))
        tables["node_features_structure"] = structure_skeleton_feature_rows(skel)

    if core_nodes is not None:
        result = read_variant_property_rows(
            core_nodes, allowed_node_ids=node_ids, collect_unknowns=True
        )
        variant_rows, variant_unknowns = result
        tables["node_features_property"].extend(variant_rows)  # ty: ignore[invalid-argument-type]  # union return not narrowed by collect_unknowns=True
        _warn_variant_unknowns(variant_unknowns)
        fp = "unknown"
        if core_dir is not None:
            man = Path(core_dir) / "build_manifest.json"
            if man.is_file():
                data = json.loads(man.read_text(encoding="utf-8"))
                raw_fp = (
                    data.get("inputs_fingerprint") if isinstance(data, dict) else None
                )
                if isinstance(raw_fp, str) and raw_fp.strip():
                    fp = raw_fp.strip()
        tables["node_features_text"].extend(
            read_synonym_context_rows(
                core_nodes,
                core_fingerprint=fp,
                allowed_node_ids=node_ids,
            )
        )

    # Stable ordering when both UniProt and ChEMBL contributed sequence rows.
    tables["node_features_sequence"].sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["sequence_type"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    tables["node_features_text"].sort(key=_node_features_text_sort_key)
    tables["node_features_property"].sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["property_id"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return tables


def _dataset_rows_for_split(
    name: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if name != "node_features_property":
        return rows
    normalized: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied.setdefault("value_numeric", None)
        copied.setdefault("value_text", None)
        normalized.append(copied)
    return normalized


def write_features_dataset_to_disk(
    out_dir: str | Path,
    tables: dict[str, list[dict[str, Any]]],
    *,
    manifest_sources: list[dict[str, Any]] | None = None,
    manifest_builders: list[dict[str, Any]] | None = None,
    modality_coverage: dict[str, Any] | None = None,
    property_coverage: dict[str, Any] | None = None,
    provenance_meta: dict[str, Any] | None = None,
    text_provenance_inventory: dict[str, Any] | None = None,
    schema_version: str = DEFAULT_MANIFEST_SCHEMA_VERSION,
) -> None:
    """Write non-empty feature tables to a local ``DatasetDict`` directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    feature_splits: dict[str, Dataset] = {}
    counts: dict[str, int] = {}
    for name in PRIMEHKG_FEATURES_HF_CONFIGS:
        rows = tables.get(name, [])
        if not isinstance(rows, list):
            msg = f"table {name!r} must be a list"
            raise TypeError(msg)
        if not rows:
            continue
        feature_splits[name] = Dataset.from_list(_dataset_rows_for_split(name, rows))
        counts[name] = len(rows)

    if not feature_splits:
        msg = (
            "no feature rows to write; pass at least one of --uniprot-tsv, --chembl-tsv, "
            "--mondo-tsv, --hpo-tsv, --hgnc-tsv, --nucleotide-tsv, "
            "--drug-features-tab (with --kg-csv), "
            "--structure-skeleton-json"
        )
        raise ValueError(msg)

    dd_out = DatasetDict(dict(feature_splits))
    dd_out.save_to_disk(str(out))

    if manifest_sources is not None:
        builders = [
            {
                "name": "primehkg_features_build",
                "version": 1,
                "params": {"tables": sorted(counts.keys())},
            }
        ]
        if manifest_builders:
            builders.extend(manifest_builders)
        manifest = assemble_build_manifest(
            schema_version=schema_version,
            sources=manifest_sources,
            builders=builders,
            counts=counts,
            extra_top_level=provenance_meta,
        )
        if modality_coverage is not None:
            manifest["modality_coverage"] = modality_coverage
        if property_coverage is not None:
            manifest["property_coverage"] = property_coverage
        write_build_manifest_json(out, manifest)
        if text_provenance_inventory is not None:
            inv_text = stable_json_dumps(text_provenance_inventory) + "\n"
            out.joinpath("text_provenance_inventory.json").write_text(
                inv_text, encoding="utf-8"
            )
