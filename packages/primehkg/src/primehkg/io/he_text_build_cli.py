from __future__ import annotations

import typer
from typer.main import get_command

from primehkg.features.he_text_build import build_he_text_rows
from primehkg.io.cli_console import cli_error

app = typer.Typer(
    rich_markup_mode="rich",
    help="Build he_features_text from core hyperedges using natural language templates.",
)


@app.command()
def main(
    core_dir: str = typer.Option(
        ..., "--core-dir", help="Path to core DatasetDict build"
    ),
    template_id: str = typer.Option("natural_sentence_v1", "--template-id"),
    out_dir: str = typer.Option(
        ..., "--out-dir", help="Output directory for he_features_text"
    ),
) -> None:
    """Textualize hyperedges into he_features_text rows."""
    from pathlib import Path

    from datasets import Dataset, DatasetDict, load_from_disk

    try:
        core = load_from_disk(core_dir)
    except Exception as exc:
        cli_error(f"failed to load core build at {core_dir!r}: {exc}")
        raise typer.Exit(1) from exc

    if not isinstance(core, DatasetDict):
        cli_error(f"expected DatasetDict at {core_dir!r}, got {type(core).__name__}")
        raise typer.Exit(1)

    hyperedges = [dict(row) for row in core["hyperedges"]]
    node_labels = {
        str(row["node_id"]): str(row.get("name") or row["node_id"])
        for row in core["nodes"]
    }

    rows = build_he_text_rows(
        hyperedges=hyperedges,
        node_labels=node_labels,
        template_id=template_id,
    )
    typer.echo(f"Generated {len(rows)} HE text rows")

    ds = Dataset.from_list(rows)
    dd = DatasetDict({"he_features_text": ds})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(out))
    typer.echo(f"Saved to {out}")


def entrypoint() -> None:
    get_command(app)()


if __name__ == "__main__":
    entrypoint()
