"""CLI for v2.1 batch retrieval (JSONL in/out)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal, cast

import typer
from typer.main import get_command

from primehkg.retrieval.batch import CandidateMode, QuerySpec, retrieve_hyperedges_batch
from primehkg.retrieval.index_io import (
    load_inverted_index_json,
    load_inverted_index_json_validated,
)

app = typer.Typer(
    rich_markup_mode="rich",
    help="Batch hyperedge retrieval: JSONL queries → JSONL results (v2.1).",
)

HYPEREDGES_JSON_OPT = typer.Option(
    ...,
    "--hyperedges-json",
    exists=True,
    dir_okay=False,
    readable=True,
    help="Path to JSON array of hyperedge objects.",
)
QUERIES_JSONL_OPT = typer.Option(
    ...,
    "--queries-jsonl",
    exists=True,
    dir_okay=False,
    readable=True,
    help="Path to JSONL file: one query object per line (QuerySpec).",
)
OUT_JSONL_OPT = typer.Option(
    ...,
    "--out-jsonl",
    dir_okay=False,
    writable=True,
    help="Output JSONL path: one batch result object per line.",
)
REPRODUCIBILITY_OPT = typer.Option(
    "strict",
    "--reproducibility",
    help="Reproducibility mode (default: strict).",
)
DEFAULT_TOP_K_OPT = typer.Option(
    20, "--default-top-k", min=1, help="Default top_k when query omits it."
)
INVERTED_INDEX_JSON_OPT = typer.Option(
    None,
    "--inverted-index-json",
    dir_okay=False,
    readable=True,
    help="Optional inverted index JSON saved by primehkg.retrieval.index_io.",
)
INPUTS_FINGERPRINT_OPT = typer.Option(
    None,
    "--inputs-fingerprint",
    help=(
        "Optional expected inputs_fingerprint for strict index validation. "
        "Requires --inverted-index-json."
    ),
)
CANDIDATE_MODE_OPT = typer.Option(
    "seed_incident",
    "--candidate-mode",
    help="Candidate generation mode: seed_incident or khop_k2.",
)


def _argv_for_retrieve_batch_cli(argv: list[str] | None) -> list[str]:
    a = list(argv) if argv is not None else list(sys.argv[1:])
    if a and a[0] == "run":
        return a[1:]
    return a


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:  # pragma: no cover
        msg = f"Failed to read {path}: {exc}"
        raise typer.BadParameter(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {path}: {exc}"
        raise typer.BadParameter(msg) from exc
    if not isinstance(data, list):
        msg = f"{path} must be a JSON array."
        raise typer.BadParameter(msg)
    out: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            msg = f"{path}[{i}] must be an object."
            raise typer.BadParameter(msg)
        out.append(cast("dict[str, Any]", item))
    return out


def _read_queries_jsonl(path: Path) -> list[QuerySpec]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:  # pragma: no cover
        msg = f"Failed to read {path}: {exc}"
        raise typer.BadParameter(msg) from exc

    out: list[QuerySpec] = []
    for ln, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"{path}:{ln} invalid JSON: {exc}"
            raise typer.BadParameter(msg) from exc
        if not isinstance(obj, dict):
            msg = f"{path}:{ln} must be a JSON object."
            raise typer.BadParameter(msg)
        if "query_id" not in obj or "seed_node_ids" not in obj:
            msg = f"{path}:{ln} requires keys: query_id, seed_node_ids"
            raise typer.BadParameter(msg)
        out.append(obj)  # TypedDict validation is structural
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        msg = f"Failed to write {path}: {exc}"
        raise typer.BadParameter(msg) from exc


@app.command("run")
def run_command(
    hyperedges_json: Path = HYPEREDGES_JSON_OPT,
    queries_jsonl: Path = QUERIES_JSONL_OPT,
    out_jsonl: Path = OUT_JSONL_OPT,
    reproducibility: Literal["strict", "fast"] = REPRODUCIBILITY_OPT,
    default_top_k: int = DEFAULT_TOP_K_OPT,
    inverted_index_json: Path | None = INVERTED_INDEX_JSON_OPT,
    inputs_fingerprint: str | None = INPUTS_FINGERPRINT_OPT,
    candidate_mode: CandidateMode = CANDIDATE_MODE_OPT,
) -> None:
    if inputs_fingerprint is not None and inverted_index_json is None:
        msg = "--inputs-fingerprint requires --inverted-index-json."
        raise typer.BadParameter(msg)

    # Normalize to `Path` to ensure runtime usage.
    hyperedges_json = Path(hyperedges_json)
    queries_jsonl = Path(queries_jsonl)
    out_jsonl = Path(out_jsonl)
    inverted_index_json = Path(inverted_index_json) if inverted_index_json else None

    hyperedges = _read_json_array(hyperedges_json)
    queries = _read_queries_jsonl(queries_jsonl)

    inverted_index = None
    if inverted_index_json is not None:
        if inputs_fingerprint is not None:
            inverted_index = load_inverted_index_json_validated(
                inverted_index_json,
                expected_fingerprint=inputs_fingerprint,
                reproducibility=reproducibility,
            )
        else:
            inverted_index, _fp = load_inverted_index_json(inverted_index_json)

    rows = retrieve_hyperedges_batch(
        queries,
        cast("list[dict[str, object]]", hyperedges),
        reproducibility=reproducibility,
        inverted_index=inverted_index,
        default_top_k=default_top_k,
        candidate_mode=candidate_mode,
    )
    _write_jsonl(out_jsonl, cast("list[dict[str, Any]]", rows))
    raise typer.Exit(0)


def main(argv: list[str] | None = None) -> int:
    cmd = get_command(app)
    args_list = _argv_for_retrieve_batch_cli(argv)
    try:
        rc = cmd.main(
            args=args_list,
            prog_name="primehkg-retrieve-batch",
            standalone_mode=False,
        )
    except SystemExit as e:
        if isinstance(e.code, int):
            return e.code
        return 1
    return int(rc) if rc is not None else 0


def entrypoint() -> None:
    """Console script entrypoint (returns process exit code via `SystemExit`)."""
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
