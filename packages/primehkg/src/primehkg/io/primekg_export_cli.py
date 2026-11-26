"""CLI: build maintainer-owned extended export TSVs from PrimeKG + public APIs."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from typer.main import get_command

from primehkg.io.primekg_export_mappings import export_extended_mappings

app = typer.Typer(
    rich_markup_mode="rich",
    invoke_without_command=True,
    help=(
        "Generate extended export TSVs (mondo_map, hpo_map, uniprot, chembl_drugs, hgnc) "
        "from a local PrimeKG bundle plus public APIs."
    ),
)

PRIMEKG_DIR_OPT = typer.Option(
    Path("data/primekg"),
    "--primekg-dir",
    file_okay=False,
    dir_okay=True,
    help="Directory containing kg.csv, nodes.tab, disease_features.tab.",
)
OUT_DIR_OPT = typer.Option(
    Path("data/exports"),
    "--out-dir",
    file_okay=False,
    dir_okay=True,
    help="Output directory for generated TSV exports.",
)
SKIP_NETWORK_OPT = typer.Option(
    False,
    "--skip-network",
    help="Only write local exports (mondo_map, hpo_map labels); skip API downloads.",
)
SKIP_HPO_DEFINITIONS_OPT = typer.Option(
    False,
    "--skip-hpo-definitions",
    help="Write hpo_map.tsv with labels only (no HPO OBO download).",
)
TIMEOUT_S_OPT = typer.Option(
    120.0,
    "--timeout-s",
    min=1.0,
    help="Per-request timeout in seconds.",
)
INSECURE_SSL_OPT = typer.Option(
    False,
    "--insecure-ssl",
    help="Disable SSL certificate verification (use only when necessary).",
)
MAX_DRUGS_OPT = typer.Option(
    None,
    "--max-drugs",
    min=1,
    help="Optional cap on ChEMBL lookups (debug/smoke).",
)
FORCE_OPT = typer.Option(
    False,
    "--force",
    help="Regenerate all exports even when existing TSVs look complete.",
)
DRUGBANK_XML_OPT = typer.Option(
    None,
    "--drugbank-xml",
    exists=True,
    dir_okay=False,
    readable=True,
    help=(
        "Optional DrugBank full-database XML for drugbank_smiles.tsv gap fill "
        "(drugs missing SMILES in chembl_drugs.tsv)."
    ),
)


@app.callback()
def main(
    primekg_dir: Path = PRIMEKG_DIR_OPT,
    out_dir: Path = OUT_DIR_OPT,
    skip_network: bool = SKIP_NETWORK_OPT,
    skip_hpo_definitions: bool = SKIP_HPO_DEFINITIONS_OPT,
    timeout_s: float = TIMEOUT_S_OPT,
    insecure_ssl: bool = INSECURE_SSL_OPT,
    max_drugs: int | None = MAX_DRUGS_OPT,
    force: bool = FORCE_OPT,
    drugbank_xml: Path | None = DRUGBANK_XML_OPT,
) -> None:
    """Generate extended export TSVs under ``--out-dir``."""
    counts = export_extended_mappings(
        primekg_dir=primekg_dir,
        out_dir=out_dir,
        skip_network=skip_network,
        include_hpo_definitions=not skip_hpo_definitions,
        timeout_s=timeout_s,
        insecure_ssl=insecure_ssl,
        max_drugs=max_drugs,
        force=force,
        drugbank_xml=drugbank_xml,
    )
    payload = {
        "type": "primehkg-extended-exports",
        "out_dir": str(out_dir.resolve()),
        "counts": counts.__dict__,
    }
    typer.echo(json.dumps(payload, sort_keys=True))


def entrypoint() -> None:
    try:
        get_command(app)(standalone_mode=False)
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    entrypoint()
