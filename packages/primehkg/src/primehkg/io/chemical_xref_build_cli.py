"""Post-build CLI: compose ``chemical_xref_v1`` against an existing core build."""

from __future__ import annotations

from pathlib import Path

import typer
from typer.main import get_command

from primehkg.core.dataset import load_core_dataset
from primehkg.io.chemical_xref_sidecar import emit_chemical_xref_sidecar_for_build

app = typer.Typer(
    rich_markup_mode="rich",
    help="Build chemical_xref_v1 sidecar from lossless CTD/UniChem v2 TSVs + core nodes.",
)


@app.command()
def main(
    core_dir: Path = typer.Option(
        ...,
        "--core-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="PrimeHKG core DatasetDict directory (with build_manifest.json).",
    ),
    ctd_mesh_pubchem_v2: Path = typer.Option(
        ...,
        "--ctd-mesh-pubchem-v2",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Lossless CTD MeSH→PubChem v2 TSV.",
    ),
    unichem_pubchem_xref_v2: Path = typer.Option(
        ...,
        "--unichem-pubchem-xref-v2",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Lossless UniChem PubChem xref v2 TSV.",
    ),
    artifacts_root: Path = typer.Option(
        Path("artifacts"),
        "--artifacts-root",
        file_okay=False,
        dir_okay=True,
        help="Artifacts root for canonical sidecar snapshot (default: ./artifacts).",
    ),
    materialize_unambiguous_mesh_xrefs: bool = typer.Option(
        True,
        "--materialize-unambiguous-mesh-xrefs/--no-materialize-unambiguous-mesh-xrefs",
        help=(
            "Write unambiguous live MeSH→node mappings into nodes.xrefs.mesh "
            "(default: on). Ambiguous MeSH stay sidecar-only."
        ),
    ),
) -> None:
    ds = load_core_dataset(core_dir)
    live_ids = {str(nid) for nid in ds.nodes["node_id"]}
    paths = emit_chemical_xref_sidecar_for_build(
        ctd_mesh_pubchem_v2=ctd_mesh_pubchem_v2,
        unichem_pubchem_xref_v2=unichem_pubchem_xref_v2,
        live_node_ids=live_ids,
        inputs_fingerprint=ds.inputs_fingerprint,
        out_dir=core_dir,
        artifacts_root=artifacts_root,
        merge_manifest=True,
    )
    typer.echo(f"wrote chemical_xref_v1 -> {paths['dataset_dir']}")
    if materialize_unambiguous_mesh_xrefs:
        # Reload so chemical_xref_index sees the new sidecar.
        ds2 = load_core_dataset(core_dir)
        index = ds2.chemical_xref_index
        if index is None:
            typer.echo("chemical_xref_v1 unloadable; skipping mesh materialization")
            raise typer.Exit(code=2)
        from primehkg.linking.chemical_xref_materialize import (
            materialize_unambiguous_mesh_xrefs_on_disk,
        )

        stats = materialize_unambiguous_mesh_xrefs_on_disk(core_dir, index)
        typer.echo(
            "materialized unambiguous mesh xrefs: "
            f"{stats['materialized']} ids / {stats['nodes_touched']} nodes"
        )


def entrypoint() -> None:
    get_command(app)()


if __name__ == "__main__":
    entrypoint()
