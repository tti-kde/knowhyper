"""CLI: PrimeHKG DatasetDict -> pyg-hyper incidence JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from typer.main import get_command

from primehkg.adapters.pyg_hyper import build_pyg_hyper_incidence_from_dataset_dir
from primehkg.adapters.structure_features import build_structure_feature_skeleton

app = typer.Typer(
    rich_markup_mode="rich",
    help="Convert a PrimeHKG DatasetDict (save_to_disk) to a minimal incidence JSON.",
)

DATASET_DIR_OPT = typer.Option(
    ...,
    "--dataset-dir",
    exists=True,
    file_okay=False,
    dir_okay=True,
    readable=True,
    help="Path to a DatasetDict directory written by primehkg-build (save_to_disk).",
)
OUT_JSON_OPT = typer.Option(
    ...,
    "--out-json",
    dir_okay=False,
    writable=True,
    help="Output JSON path for incidence representation.",
)
INCLUDE_HYPEREDGE_INDEX_OPT = typer.Option(
    False,
    "--include-hyperedge-index/--no-include-hyperedge-index",
    help="If enabled, also emit a PyG-style COO `hyperedge_index` ([2, num_connections]).",
)
STRUCTURE_SKELETON_OUT_OPT = typer.Option(
    None,
    "--structure-skeleton-out",
    dir_okay=False,
    writable=True,
    help="Optional path for derived structure-only node features JSON (D-layer).",
)
ENTITY_VIEW_DIR_OPT = typer.Option(
    None,
    "--entity-view-dir",
    exists=True,
    file_okay=False,
    dir_okay=True,
    readable=True,
    help="Optional entity-attribute view sidecar dir; its HEs are folded into the incidence.",
)


def _argv_for_cli(argv: list[str] | None) -> list[str]:
    a = list(argv) if argv is not None else list(sys.argv[1:])
    if a and a[0] == "run":
        return a[1:]
    return a


@app.command()
def run_command(
    dataset_dir: Path = DATASET_DIR_OPT,
    out_json: Path = OUT_JSON_OPT,
    include_hyperedge_index: bool = INCLUDE_HYPEREDGE_INDEX_OPT,
    structure_skeleton_out: Path | None = STRUCTURE_SKELETON_OUT_OPT,
    entity_view_dir: Path | None = ENTITY_VIEW_DIR_OPT,
) -> None:
    # Redundant normalization to satisfy ruff TC003 and ensure path-like behavior.
    dataset_dir = Path(str(dataset_dir))
    out_json = Path(str(out_json))
    entity_view_dir = (
        Path(str(entity_view_dir)) if entity_view_dir is not None else None
    )
    out = build_pyg_hyper_incidence_from_dataset_dir(
        dataset_dir,
        include_hyperedge_index=include_hyperedge_index,
        entity_view_dir=entity_view_dir,
    )
    out_json.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    if structure_skeleton_out is not None:
        skeleton = build_structure_feature_skeleton(out)
        structure_skeleton_out.write_text(
            json.dumps(skeleton, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def entrypoint(argv: list[str] | None = None) -> None:
    get_command(app)(args=_argv_for_cli(argv))
