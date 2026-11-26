from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import typer
from typer.main import get_command

from primehkg.io.build_manifest import merge_manifest_sidecars
from primehkg.io.cli_console import cli_error
from primehkg.io.vector_ann import build_hnsw_ann_index

app = typer.Typer(
    rich_markup_mode="rich",
    help="Build vector_node_v1 / vector_hyperedge_v1 sidecars from embedding DatasetDict.",
)

_VECTOR_EXPORT_BATCH_SIZE = 8192
_SELECTOR_COLUMNS = (
    "input_kind",
    "source_modality",
    "source_template_id",
    "model_id",
    "preprocess_version",
)


def _selector_params(
    *,
    input_kind: str | None,
    source_modality: str | None,
    source_template_id: str | None,
    model_id: str | None,
    preprocess_version: str | None,
) -> dict[str, str]:
    selector = {
        "input_kind": input_kind,
        "source_modality": source_modality,
        "source_template_id": source_template_id,
        "model_id": model_id,
        "preprocess_version": preprocess_version,
    }
    return {key: value for key, value in selector.items() if isinstance(value, str)}


def _selector_batch_predicate(batch: Any, items: list[tuple[str, str]]) -> list[bool]:
    row_count = len(batch[items[0][0]])
    return [
        all(str(batch[column][i]) == expected for column, expected in items)
        for i in range(row_count)
    ]


def _filter_by_selector(ds: Any, *, config_name: str, selector: dict[str, str]) -> Any:
    for column in selector:
        if column not in ds.column_names:
            cli_error(f"cannot select {config_name} by missing column {column!r}")
            raise typer.Exit(2)
    if not selector:
        filtered = ds
    else:
        items = list(selector.items())
        filtered = ds.filter(
            _selector_batch_predicate,
            fn_kwargs={"items": items},
            batched=True,
            desc=None,
        )
    if filtered.num_rows == 0:
        cli_error(f"selector matched no rows in {config_name}: {selector}")
        raise typer.Exit(2)
    return filtered


def _resolved_embedding_selector(ds: Any, *, config_name: str) -> dict[str, str]:
    available = [column for column in _SELECTOR_COLUMNS if column in ds.column_names]
    if not available:
        return {}

    combinations: set[tuple[str, ...]] = set()
    for batch in ds.select_columns(available).iter(
        batch_size=_VECTOR_EXPORT_BATCH_SIZE
    ):
        columns = [batch[column] for column in available]
        for values in zip(*columns, strict=True):
            combinations.add(tuple(str(value) for value in values))
            if len(combinations) > 1:
                cli_error(
                    f"{config_name} contains multiple embedding slices; "
                    "choose one with --input-kind/--source-modality/"
                    "--source-template-id/--model-id/--preprocess-version"
                )
                raise typer.Exit(2)

    if not combinations:
        cli_error(f"cannot build vector sidecar from empty {config_name}")
        raise typer.Exit(1)
    values = next(iter(combinations))
    return dict(zip(available, values, strict=True))


@app.command()
def main(
    embedding_dir: str = typer.Option(..., "--embedding-dir"),
    target: str = typer.Option(..., "--target", help="'node' or 'hyperedge'"),
    out_dir: str = typer.Option(..., "--out-dir"),
    ann_backend: str = typer.Option("brute_force", "--ann-backend"),
    hnsw_m: int = typer.Option(16, "--hnsw-m"),
    hnsw_ef_construction: int = typer.Option(200, "--hnsw-ef-construction"),
    hnsw_ef_search: int = typer.Option(64, "--hnsw-ef-search"),
    core_dir: str | None = typer.Option(None, "--core-dir"),
    input_kind: str | None = typer.Option(None, "--input-kind"),
    source_modality: str | None = typer.Option(None, "--source-modality"),
    source_template_id: str | None = typer.Option(None, "--source-template-id"),
    model_id: str | None = typer.Option(None, "--model-id"),
    preprocess_version: str | None = typer.Option(None, "--preprocess-version"),
) -> None:
    """Export mmap-friendly vector sidecar from embedding tables."""
    from datasets import DatasetDict, load_from_disk

    if target not in ("node", "hyperedge"):
        cli_error("--target must be 'node' or 'hyperedge'")
        raise typer.Exit(1)

    if ann_backend not in ("brute_force", "hnswlib"):
        cli_error("--ann-backend must be 'brute_force' or 'hnswlib'")
        raise typer.Exit(1)

    try:
        dd = load_from_disk(embedding_dir)
    except Exception as exc:
        cli_error(f"failed to load embedding dataset at {embedding_dir!r}: {exc}")
        raise typer.Exit(1) from exc

    if not isinstance(dd, DatasetDict):
        cli_error(f"expected DatasetDict at {embedding_dir!r}")
        raise typer.Exit(1)

    config_name = (
        "he_features_embedding" if target == "hyperedge" else "node_features_embedding"
    )
    if config_name not in dd:
        cli_error(f"missing config {config_name!r} in {embedding_dir!r}")
        raise typer.Exit(1)

    ds = dd[config_name]
    selector = _selector_params(
        input_kind=input_kind,
        source_modality=source_modality,
        source_template_id=source_template_id,
        model_id=model_id,
        preprocess_version=preprocess_version,
    )
    if selector:
        ds = _filter_by_selector(ds, config_name=config_name, selector=selector)
    resolved_selector = _resolved_embedding_selector(ds, config_name=config_name)

    id_col = "he_id" if target == "hyperedge" else "node_id"
    required_columns = {id_col, "vector", "model_id"}
    missing_columns = sorted(required_columns - set(ds.column_names))
    if missing_columns:
        cli_error(f"missing columns in {config_name}: {missing_columns}")
        raise typer.Exit(1)
    if ds.num_rows == 0:
        cli_error(f"cannot build {target} vector sidecar from empty {config_name}")
        raise typer.Exit(1)

    first = ds[0]
    first_vector = np.asarray(first["vector"], dtype=np.float32)
    if first_vector.ndim != 1:
        cli_error(f"vector column must contain 1-D vectors, got {first_vector.shape}")
        raise typer.Exit(1)
    dim = int(first_vector.shape[0])
    model_id = str(first["model_id"])
    template_id = str(first.get("source_template_id", "unknown"))

    fingerprint = "unknown"
    if core_dir:
        manifest_path = Path(core_dir) / "build_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fingerprint = str(manifest.get("inputs_fingerprint", "unknown"))

    file_version = f"vector_{target}_v1"
    ann_params: dict[str, int] = {}
    if ann_backend == "hnswlib":
        ann_params = {
            "M": hnsw_m,
            "ef_construction": hnsw_ef_construction,
            "ef_search": hnsw_ef_search,
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    embeddings_path = out / "embeddings.f32.npy"
    vocab_path = out / "vocab.txt"
    embeddings = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(ds.num_rows), dim),
    )

    ds_arrow = ds.with_format("arrow")
    row_offset = 0
    with vocab_path.open("w", encoding="utf-8") as vocab:
        for batch in ds_arrow.iter(batch_size=_VECTOR_EXPORT_BATCH_SIZE):
            batch_ids = [str(v) for v in batch.column(id_col).to_pylist()]
            vec_col = batch.column("vector").combine_chunks()
            n = len(vec_col)
            flat = vec_col.values.to_numpy(zero_copy_only=False)
            if flat.size != n * dim:
                cli_error(
                    f"vector batch must flatten to (*, {dim}); "
                    f"got {flat.size} values for {n} rows"
                )
                raise typer.Exit(1)
            vectors = flat.reshape(n, dim).astype(np.float32, copy=False)
            if n != len(batch_ids):
                cli_error(
                    f"id/vector batch row count mismatch: {len(batch_ids)} vs {n}"
                )
                raise typer.Exit(1)
            end = row_offset + n
            embeddings[row_offset:end] = vectors
            vocab.write("\n".join(batch_ids))
            vocab.write("\n")
            row_offset = end

    if row_offset != ds.num_rows:
        cli_error(f"wrote {row_offset} vectors but expected {ds.num_rows}")
        raise typer.Exit(1)
    embeddings.flush()
    del embeddings

    if ann_backend == "hnswlib":
        try:
            hnsw_vectors = np.load(embeddings_path, mmap_mode="r")
            build_hnsw_ann_index(
                sidecar_dir=out,
                embeddings=hnsw_vectors,
                m=hnsw_m,
                ef_construction=hnsw_ef_construction,
                ef_search=hnsw_ef_search,
            )
        except ImportError as exc:
            cli_error(
                "ann_backend=hnswlib requires hnswlib; run `uv sync --group vector`"
            )
            raise typer.Exit(1) from exc

    meta = {
        "file_version": file_version,
        "inputs_fingerprint": fingerprint,
        "model_id": model_id,
        "source_template_id": template_id,
        "selector": resolved_selector,
        "embedding_dim": dim,
        "num_vectors": int(ds.num_rows),
        "dtype": "float32",
        "ann_backend": ann_backend,
        "ann_params": ann_params,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    typer.echo(f"Written {file_version} sidecar: {ds.num_rows} vectors, dim={dim}")

    # Register the sidecar in the build manifest so the loader can find it.
    # The loader (`PrimeHKGDataset._try_load_vector_index`) only loads sidecars
    # listed in build_manifest.json -> sidecars; without this the vector index
    # is silently unavailable even though the files exist (issue #389).
    if core_dir:
        build_root = Path(core_dir)
        manifest_path = build_root / "build_manifest.json"
        if manifest_path.is_file():
            try:
                rel = out.resolve().relative_to(build_root.resolve())
            except ValueError:
                typer.echo(
                    f"warning: --out-dir {out} is not under --core-dir {core_dir}; "
                    f"skipping manifest registration of {file_version}",
                    err=True,
                )
            else:
                merge_manifest_sidecars(build_root, {file_version: str(rel)})
                typer.echo(f"Registered {file_version} -> {rel} in build_manifest.json")


def entrypoint() -> None:
    get_command(app)()


if __name__ == "__main__":
    entrypoint()
