"""CLI: census PrimeKG relation strings and optional registry coverage check."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from typer.main import get_command

from primehkg.policy.census import (
    count_relations_in_kg_csv,
    relations_missing_from_registry,
)
from primehkg.policy.registry import load_relation_registry

app = typer.Typer(
    rich_markup_mode="rich",
    help="Enumerate PrimeKG relation strings in kg.csv and check relation_registry.yaml coverage.",
)

KG_CSV_OPT = typer.Option(
    ...,
    "--kg-csv",
    exists=True,
    dir_okay=False,
    readable=True,
    help="PrimeKG kg.csv path.",
)
OUT_OPT = typer.Option(
    None,
    "--out",
    dir_okay=False,
    writable=True,
    help="Write TSV (relation, count). Default: stdout.",
)
CHECK_REGISTRY_OPT = typer.Option(
    False,
    "--check-registry",
    help="Exit 1 if any observed relation is missing from packaged relation_registry.yaml.",
)


@app.command()
def main(
    kg_csv: Path = KG_CSV_OPT,
    out: Path | None = OUT_OPT,
    check_registry: bool = CHECK_REGISTRY_OPT,
) -> None:
    """Stream kg.csv and print relation counts."""
    rows = count_relations_in_kg_csv(kg_csv)
    lines = ["relation\tcount", *(f"{rel}\t{count}" for rel, count in rows)]
    text = "\n".join(lines) + "\n"
    if out is None:
        sys.stdout.write(text)
    else:
        out.write_text(text, encoding="utf-8")

    if check_registry:
        reg = load_relation_registry()
        missing = relations_missing_from_registry([rel for rel, _ in rows], reg)
        if missing:
            typer.echo(
                "Relations missing from relation_registry.yaml:\n"
                + "\n".join(f"  - {m!r}" for m in missing),
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(f"Registry covers all {len(rows)} observed relation strings.")


def entrypoint() -> None:
    get_command(app)()


if __name__ == "__main__":
    entrypoint()
