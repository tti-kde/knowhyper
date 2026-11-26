from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from primehkg.audit.reports import build_audit_report
from primehkg.io.build_manifest import (
    DEFAULT_MANIFEST_SCHEMA_VERSION,
    assemble_build_manifest,
    stable_json_dumps,
    write_build_manifest_json,
)


def _json_dumps_sorted(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_cell(value: Any) -> Any:
    """Normalize nested structures for Arrow/Parquet (HF Hub) compatibility.

    Hugging Face datasets commonly materialize Python `{}` values as empty *struct* types in Arrow.
    PyArrow cannot write empty structs to Parquet, which breaks `push_to_hub`.

    Strategy:
    - dict -> JSON string (including empty dicts)
    - list that contains dict/list -> JSON string (keeps nested structure stable)
    - list of scalars -> unchanged
    - other scalars -> unchanged
    """
    if isinstance(value, dict):
        return _json_dumps_sorted(value)
    if isinstance(value, list):
        if value and isinstance(value[0], (dict, list)):
            return _json_dumps_sorted(value)
        return value
    return value


def normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    json_list_fields = {"sources", "support_triplets"}
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k in json_list_fields and isinstance(v, list):
            out[k] = _json_dumps_sorted(v)
            continue
        out[k] = normalize_cell(v)
    return out


# Backward-compat aliases (the public names are `normalize_cell`/`normalize_record`).
_normalize_cell = normalize_cell
_normalize_record = normalize_record


def write_hf_tables_to_disk(
    out_dir: str | Path,
    *,
    nodes: list[dict[str, Any]],
    hyperedges: list[dict[str, Any]],
    support: list[dict[str, Any]],
    manifest_schema_version: str | None = None,
    manifest_sources: list[dict[str, Any]] | None = None,
    manifest_builders: list[dict[str, Any]] | None = None,
    manifest_build_profile: str | None = None,
    manifest_extra_top_level: dict[str, Any] | None = None,
    emit_build_report: bool = False,
    build_report_extra: dict[str, Any] | None = None,
    corum_tsv: str | None = None,
    uniprot_to_geneid: dict[str, str] | None = None,
    builder_segments: list[tuple[str, dict[str, list[dict[str, Any]]]]] | None = None,
    relation_coverage_audit: dict[str, Any] | None = None,
) -> None:
    """Write three Hugging Face dataset tables (`nodes`, `hyperedges`, `support`) to disk.

    The output is a `datasets.DatasetDict` saved via `save_to_disk`, and is readable via
    `datasets.load_from_disk(out_dir)`.

    Optional reproducibility metadata: pass ``manifest_sources`` and ``manifest_builders`` together
    to also write ``build_manifest.json`` (see ``primehkg.io.build_manifest``).

    Optional audit artifact: set ``emit_build_report=True`` to write ``build_report.json`` alongside
    the manifest using ``primehkg.audit.reports.build_audit_report``.
    """
    if not isinstance(nodes, list):
        msg = f"`nodes` must be a list[dict], got {type(nodes).__name__}"
        raise TypeError(msg)
    if not isinstance(hyperedges, list):
        msg = f"`hyperedges` must be a list[dict], got {type(hyperedges).__name__}"
        raise TypeError(msg)
    if not isinstance(support, list):
        msg = f"`support` must be a list[dict], got {type(support).__name__}"
        raise TypeError(msg)

    if (manifest_sources is None) ^ (manifest_builders is None):
        msg = (
            "`manifest_sources` and `manifest_builders` must be provided together "
            "(or both omitted)."
        )
        raise ValueError(msg)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    normalized_nodes = [normalize_record(r) for r in nodes]
    normalized_hyperedges = [normalize_record(r) for r in hyperedges]
    normalized_support = [normalize_record(r) for r in support]

    dd = DatasetDict(
        {
            "nodes": Dataset.from_list(normalized_nodes),
            "hyperedges": Dataset.from_list(normalized_hyperedges),
            "support": Dataset.from_list(normalized_support),
        }
    )
    dd.save_to_disk(str(out))

    if manifest_sources is not None and manifest_builders is not None:
        schema = manifest_schema_version or DEFAULT_MANIFEST_SCHEMA_VERSION
        counts = {
            "nodes": len(normalized_nodes),
            "hyperedges": len(normalized_hyperedges),
            "support": len(normalized_support),
        }
        manifest = assemble_build_manifest(
            schema_version=schema,
            sources=manifest_sources,
            builders=manifest_builders,
            counts=counts,
            build_profile=manifest_build_profile,
            extra_top_level=manifest_extra_top_level,
        )
        write_build_manifest_json(out, manifest)

        if emit_build_report:
            report = build_audit_report(
                hyperedges,
                supports=support,
                nodes=nodes,
                corum_tsv=corum_tsv,
                uniprot_to_geneid=uniprot_to_geneid,
                builder_segments=builder_segments,
                relation_coverage_audit=relation_coverage_audit,
            )
            if build_report_extra:
                report.update(build_report_extra)
            (out / "build_report.json").write_text(
                stable_json_dumps(report) + "\n", encoding="utf-8"
            )
