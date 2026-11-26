"""CLI: audit chemical crosswalk coverage for a list of external IDs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from typer.main import get_command

from primehkg.core.dataset import load_core_dataset
from primehkg.io.cli_console import cli_error
from primehkg.io.curated_node_ids import ctd_chemical_to_node_id
from primehkg.io.curated_sources import load_ctd_mesh_cid, load_unichem_cid_map
from primehkg.linking.chemical_xref_index import ChemicalXrefIndex

app = typer.Typer(
    rich_markup_mode="rich",
    help="Audit chemical xref coverage for external IDs against a pinned core.",
)


def _read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.append(s)
    return ids


def audit_chemical_crosswalk(
    *,
    index: ChemicalXrefIndex,
    external_ids: list[str],
    namespace: str,
    allowed_node_types: set[str] | None,
    scalar_mesh_to_cid: dict[str, str] | None = None,
    scalar_cid_map: dict[str, dict[str, str | None]] | None = None,
) -> dict[str, Any]:
    """Return coverage buckets for the given external IDs."""
    n = len(external_ids)
    recovered = 0
    bridge = 0
    live = 0
    ambiguous = 0
    mapping_absent = 0
    node_absent = 0
    scalar_hits = 0
    details: list[dict[str, Any]] = []

    for eid in external_ids:
        # Any row including non-live: mapping presence.
        any_cands = index.lookup(
            eid,
            namespace=namespace,
            allowed_node_types=allowed_node_types,
            live_only=False,
        )
        live_cands = index.lookup(
            eid,
            namespace=namespace,
            allowed_node_types=allowed_node_types,
            live_only=True,
        )
        row: dict[str, Any] = {
            "external_id": eid,
            "candidate_count_any": len(any_cands),
            "candidate_count_live": len(live_cands),
            "live_node_ids": [c.node_id for c in live_cands],
        }
        if any_cands:
            recovered += 1
            bridge += 1
            if live_cands:
                live += 1
                if (
                    any(c.ambiguity_count > 1 for c in live_cands)
                    or len(live_cands) > 1
                ):
                    ambiguous += 1
                    row["bucket"] = "live_ambiguous" if len(live_cands) > 1 else "live"
                else:
                    row["bucket"] = "live"
            else:
                node_absent += 1
                row["bucket"] = "node_absent"
        else:
            mapping_absent += 1
            row["bucket"] = "mapping_absent"

        if scalar_mesh_to_cid is not None and scalar_cid_map is not None:
            # Scalar baseline only defined for MeSH chemicals today.
            hit = ctd_chemical_to_node_id(eid, scalar_mesh_to_cid, scalar_cid_map)
            row["scalar_node_id"] = hit
            if hit is not None:
                scalar_hits += 1
        details.append(row)

    report: dict[str, Any] = {
        "namespace": namespace,
        "n_input": n,
        "rates": {
            "input_recovery_rate": (recovered / n) if n else 0.0,
            "bridge_candidate_rate": (bridge / n) if n else 0.0,
            "live_core_node_rate": (live / n) if n else 0.0,
            "ambiguous_rate": (ambiguous / n) if n else 0.0,
            "mapping_absent_rate": (mapping_absent / n) if n else 0.0,
            "node_absent_rate": (node_absent / n) if n else 0.0,
        },
        "counts": {
            "recovered": recovered,
            "bridge": bridge,
            "live": live,
            "ambiguous": ambiguous,
            "mapping_absent": mapping_absent,
            "node_absent": node_absent,
        },
        "details": details,
    }
    if scalar_mesh_to_cid is not None:
        report["scalar_baseline"] = {
            "hits": scalar_hits,
            "hit_rate": (scalar_hits / n) if n else 0.0,
        }
    return report


@app.command()
def main(
    core_dir: Path = typer.Option(
        ...,
        "--core-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Core DatasetDict with chemical_xref_v1 listed in the manifest.",
    ),
    external_ids: Path = typer.Option(
        ...,
        "--external-ids",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Text file with one external id per line.",
    ),
    namespace: str = typer.Option(
        ...,
        "--namespace",
        help="External id namespace (mesh|pubchem|drugbank|chebi).",
    ),
    allowed_node_types: str = typer.Option(
        "drug,chebi",
        "--allowed-node-types",
        help="Comma-separated hard type filter (empty = no filter).",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Write audit JSON to this path.",
    ),
    compare_scalar_baseline: bool = typer.Option(
        False,
        "--compare-scalar-baseline",
        help="Also score legacy scalar CTD/UniChem maps (requires --ctd-mesh-cid and --unichem-cid-map).",
    ),
    ctd_mesh_cid: Path | None = typer.Option(
        None,
        "--ctd-mesh-cid",
        help="Scalar ctd_mesh_cid.tsv for baseline compare.",
    ),
    unichem_cid_map: Path | None = typer.Option(
        None,
        "--unichem-cid-map",
        help="Scalar unichem_cid_map.tsv for baseline compare.",
    ),
) -> None:
    from primehkg.io.chemical_xref_sidecar import ChemicalXrefFingerprintError

    ds = load_core_dataset(core_dir)
    try:
        index = ds.chemical_xref_index
    except ChemicalXrefFingerprintError as exc:
        cli_error(f"chemical_xref_v1 fingerprint error: {exc}")
        raise typer.Exit(2) from exc
    if index is None:
        cli_error("chemical_xref_v1 not present or unloadable from core")
        raise typer.Exit(2)

    allowed: set[str] | None
    raw_types = [t.strip() for t in allowed_node_types.split(",") if t.strip()]
    allowed = set(raw_types) if raw_types else None

    scalar_mesh: dict[str, str] | None = None
    scalar_cid: dict[str, dict[str, str | None]] | None = None
    if compare_scalar_baseline:
        if ctd_mesh_cid is None or unichem_cid_map is None:
            cli_error(
                "--compare-scalar-baseline requires --ctd-mesh-cid and --unichem-cid-map"
            )
            raise typer.Exit(2)
        scalar_mesh = load_ctd_mesh_cid(ctd_mesh_cid)
        scalar_cid = load_unichem_cid_map(unichem_cid_map)

    report = audit_chemical_crosswalk(
        index=index,
        external_ids=_read_ids(external_ids),
        namespace=namespace,
        allowed_node_types=allowed,
        scalar_mesh_to_cid=scalar_mesh,
        scalar_cid_map=scalar_cid,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    typer.echo(f"wrote {out}")


def entrypoint() -> None:
    get_command(app)()


if __name__ == "__main__":
    entrypoint()
