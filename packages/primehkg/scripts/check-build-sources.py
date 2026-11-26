#!/usr/bin/env python3
"""Guard against omitted enrichment flags in a reference build command.

Compares the ``sources[]`` recorded in a previous release's ``build_manifest.json``
against the source-bearing flags present in a proposed ``primehkg-build`` argv, and
reports which prior sources have no corresponding flag in the new command. This
catches the class of bug where the shark-playbook recipe block is treated as the
complete command and a materially-important enrichment flag (e.g.
``--enrich-uniprot-tsv``) is silently dropped.

Usage:
    uv run python scripts/check-build-sources.py \
        --against ../primehkg-tmp-artifacts/primehkg_0110 \
        -- \
        primehkg-build --kg-csv ... --signor-tsv ... [full argv]

Exit code 0 = every prior source is covered by a flag in the new argv (plus any
intentional new flags); exit 1 = at least one prior source has no matching flag.

The check is intentionally conservative: it maps each manifest ``source`` name to the
CLI flag that would supply it and verifies that flag (or its value) appears in the
argv. Unknown source names are reported as advisory, not failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# manifest source `name` -> the primehkg-build CLI flag that supplies it.
# Keep in sync with build_cli's _optional_manifest_sources / source recording.
SOURCE_TO_FLAG: dict[str, str] = {
    "primekg_kg": "--kg-csv",
    "primekg_nodes_tab": "--enrich-primekg-nodes-tab",
    "primekg_drug_features_tab": "--enrich-primekg-drug-features-tab",
    "primekg_disease_features_tab": "--enrich-primekg-disease-features-tab",
    "corum_tsv": "--corum-tsv",
    "reactome_tsv": "--reactome-tsv",
    "signor_tsv": "--signor-tsv",
    "complexportal_tsv": "--complexportal-tsv",
    "drugcomb_summary_tsv": "--drugcomb-summary-tsv",
    "drugcomb_identifiers_tsv": "--drugcomb-identifiers-tsv",
    "clinvar_tsv": "--clinvar-tsv",
    "stitch_tsv": "--stitch-tsv",
    "unichem_cid_map_tsv": "--unichem-cid-map-tsv",
    "string_ensp_geneid_tsv": "--string-ensp-geneid-tsv",
    "ctd_tsv": "--ctd-tsv",
    "ctd_mesh_cid_tsv": "--ctd-mesh-cid-tsv",
    "enrich_uniprot_tsv": "--enrich-uniprot-tsv",
    "enrich_mondo_tsv": "--enrich-mondo-tsv",
    "enrich_mondo_obo": "--enrich-mondo-obo",
    "enrich_hgnc_tsv": "--enrich-hgnc-tsv",
    "enrich_hpo_tsv": "--enrich-hpo-tsv",
    "enrich_chembl_tsv": "--enrich-chembl-tsv",
    "enrich_drugbank_tsv": "--enrich-drugbank-tsv",
    "enrich_rxnorm_tsv": "--enrich-rxnorm-tsv",
    "enrich_go_tsv": "--enrich-go-tsv",
    "enrich_pathway_tsv": "--enrich-pathway-tsv",
    "enrich_anatomy_tsv": "--enrich-anatomy-tsv",
    "enrich_umls_tsv": "--enrich-umls-tsv",
    "enrich_xrefs_tsv": "--enrich-xrefs-tsv",
    "chebi_names_tsv": "--chebi-names-tsv",
    "chebi_definitions_tsv": "--chebi-definitions-tsv",
    "display_name_priority_json": "--display-name-priority-json",
    # recorded but not a build input flag (packaged with the wheel):
    "relation_registry_yaml": "",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--against",
        required=True,
        help="Prior release build dir (containing build_manifest.json) to diff against.",
    )
    parser.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="The proposed primehkg-build command (after `--`).",
    )
    args = parser.parse_args()

    manifest_path = Path(args.against) / "build_manifest.json"
    if not manifest_path.is_file():
        print(f"error: no build_manifest.json under {args.against!r}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_sources = {s["name"] for s in manifest.get("sources", []) if "name" in s}

    argv = [a for a in args.argv if a != "--"]
    argv_flags = {a for a in argv if a.startswith("--")}

    missing: list[str] = []
    advisory: list[str] = []
    for name in sorted(prior_sources):
        flag = SOURCE_TO_FLAG.get(name)
        if flag is None:
            advisory.append(name)
            continue
        if flag == "":  # packaged, not a build flag
            continue
        if flag not in argv_flags:
            missing.append(f"{name}  (expected flag {flag})")

    if advisory:
        print("advisory: prior sources with no known flag mapping (verify manually):")
        for a in advisory:
            print(f"  - {a}")

    if missing:
        print(
            f"\nFAIL: {len(missing)} source(s) in {args.against} have no matching "
            "flag in the proposed build command:"
        )
        for m in missing:
            print(f"  - {m}")
        print(
            "\nAdd the missing flag(s), or confirm the source is intentionally "
            "dropped this release."
        )
        return 1

    print(
        f"OK: all {len(prior_sources)} prior sources are covered by flags in the "
        "proposed command."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
