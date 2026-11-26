"""CLI: build the entity-attribute view sidecar from a built core DatasetDict."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from typer.main import get_command

from primehkg.io.cli_console import cli_error
from primehkg.views.entity_view import build_entity_view_tables

app = typer.Typer(
    rich_markup_mode="rich",
    help="Build the entity-attribute view sidecar (entity_view_* tables) from a core build.",
)


def _argv(argv: list[str] | None) -> list[str]:
    a = list(argv) if argv is not None else list(sys.argv[1:])
    if a and a[0] == "run":
        return a[1:]
    return a


@app.command()
def run_command(
    core_dir: str = typer.Option(
        ..., "--core-dir", help="Path to core DatasetDict build"
    ),
    out_dir: str = typer.Option(
        ..., "--out-dir", help="Output dir for entity_view_* tables"
    ),
    source_fingerprint: str = typer.Option(
        ...,
        "--source-fingerprint",
        help="inputs_fingerprint of the core build this view derives from",
    ),
) -> None:
    from datasets import Dataset, DatasetDict, load_from_disk

    from primehkg.io.hf_dataset import normalize_record
    from primehkg.views.entity_view.manifest import (
        build_entity_view_manifest,
        dumps_manifest,
    )
    from primehkg.views.entity_view.modalities import DENSE_MODALITIES

    try:
        core = load_from_disk(core_dir)
    except Exception as exc:
        cli_error(f"failed to load core build at {core_dir!r}: {exc}")
        raise typer.Exit(1) from exc

    if not isinstance(core, DatasetDict) or "nodes" not in core:
        cli_error(f"expected a DatasetDict with a 'nodes' split at {core_dir!r}")
        raise typer.Exit(1)

    nodes = (dict(row) for row in core["nodes"])
    tables = build_entity_view_tables(nodes, source_fingerprint=source_fingerprint)

    dd = DatasetDict(
        {
            name: Dataset.from_list([normalize_record(r) for r in rows])
            for name, rows in tables.items()
        }
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(out))
    manifest = build_entity_view_manifest(
        source_inputs_fingerprint=source_fingerprint,
        source_core_dir=core_dir,
        dense_modalities=DENSE_MODALITIES,
        tables=tables,
    )
    try:
        (out / "manifest.json").write_text(dumps_manifest(manifest), encoding="utf-8")
    except OSError as exc:
        cli_error(f"failed to write manifest at {out / 'manifest.json'}: {exc}")
        raise typer.Exit(1) from exc
    typer.echo(
        f"entity_view: {len(tables['entity_view_hyperedges'])} HE, "
        f"{len(tables['entity_view_nodes'])} attribute nodes -> {out}"
    )


def entrypoint(argv: list[str] | None = None) -> None:
    # standalone_mode=False so a direct entrypoint([...]) call returns instead of
    # always calling sys.exit (click exits even on success). But that also means a
    # typer.Exit(1) is delivered as a returned code rather than a process exit — so
    # we must re-raise it, or error paths would silently exit 0.
    code = get_command(app)(args=_argv(argv), standalone_mode=False)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    entrypoint()
