from __future__ import annotations

import hashlib
import json
from collections.abc import Callable  # noqa: TC003
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from primehkg.builders.drug_disease_shared_target_bundle import (
    build_drug_disease_shared_target_bundle_hyperedges,
)
from primehkg.builders.moa_triad import build_moa_triad_hyperedges
from primehkg.builders.phenotype import build_phenotype_hyperedges
from primehkg.builders.registry_atomic import build_registry_atomic_hyperedges
from primehkg.io.build_manifest import deep_sort_jsonish, stable_json_dumps
from primehkg.policy.atomic_admissions import (
    COVERAGE_REGISTRY_ADMISSIONS,
    AtomicAdmission,
)

BuilderOutput = dict[str, list[dict[str, Any]]]


def _write_json_wrapped_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one Parquet column ``row`` holding ``stable_json_dumps`` per record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = [{"row": stable_json_dumps(r)} for r in rows]
    if wrapped:
        pq.write_table(pa.Table.from_pylist(wrapped), path)
        return
    tbl = pa.Table.from_pylist([{"row": "{}"}])
    pq.write_table(tbl.slice(0, 0), path)


def _read_json_wrapped_parquet(path: Path) -> list[dict[str, Any]]:
    return [json.loads(cell["row"]) for cell in pq.read_table(path).to_pylist()]


def compute_builder_cache_key(
    *,
    schema_version: str,
    sources: list[dict[str, Any]],
    builder_entry: dict[str, Any],
    triplets: list[dict[str, Any]],
) -> str:
    """Stable SHA-256 hex key for a triplet-input builder cache stage."""
    payload = {
        "schema_version": schema_version,
        "sources": deep_sort_jsonish(sources),
        "builder": deep_sort_jsonish(builder_entry),
        "triplets": deep_sort_jsonish(triplets),
    }
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def compute_phenotype_cache_key(
    *,
    schema_version: str,
    sources: list[dict[str, Any]],
    builder_entry: dict[str, Any],
    triplets: list[dict[str, Any]],
) -> str:
    """Alias for ``compute_builder_cache_key`` (legacy phenotype-specific alias)."""
    return compute_builder_cache_key(
        schema_version=schema_version,
        sources=sources,
        builder_entry=builder_entry,
        triplets=triplets,
    )


def builder_stage_dir(out_base: str | Path, builder_name: str, cache_key: str) -> Path:
    """On-disk stage directory: ``<out_base>/builder/<builder_name>/<cache_key>/``."""
    return Path(out_base) / "builder" / builder_name / cache_key


def _triplet_candidate_rows(triplets: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for t in triplets:
        s, r, te = t.get("s"), t.get("r"), t.get("t")
        if isinstance(s, str) and isinstance(r, str) and isinstance(te, str):
            rows.append({"s": s, "r": r, "t": te})
    rows.sort(key=lambda x: (x["s"], x["r"], x["t"]))
    return rows


def _write_candidates_parquet(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path)
        return
    tbl = pa.Table.from_pylist([{"s": "", "r": "", "t": ""}])
    pq.write_table(tbl.slice(0, 0), path)


def _try_load_builder_cache(
    stage: Path, *, expected_key: str, builder_name: str
) -> BuilderOutput | None:
    stats_path = stage / "stats.json"
    if not stats_path.is_file():
        return None
    try:
        stats: dict[str, Any] = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if (
        stats.get("builder_cache_key") != expected_key
        or stats.get("builder") != builder_name
    ):
        return None
    cand, facts_p, sup_p = (
        stage / "candidates.parquet",
        stage / "facts.parquet",
        stage / "support.parquet",
    )
    if not (cand.is_file() and facts_p.is_file() and sup_p.is_file()):
        return None
    return {
        "hyperedges": _read_json_wrapped_parquet(facts_p),
        "support": _read_json_wrapped_parquet(sup_p),
    }


def _write_builder_cache(
    stage: Path,
    *,
    builder_name: str,
    cache_key: str,
    schema_version: str,
    triplets: list[dict[str, Any]],
    fresh: BuilderOutput,
) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    _write_candidates_parquet(
        stage / "candidates.parquet", _triplet_candidate_rows(triplets)
    )
    _write_json_wrapped_parquet(stage / "facts.parquet", fresh["hyperedges"])
    _write_json_wrapped_parquet(stage / "support.parquet", fresh["support"])
    stats = {
        "builder": builder_name,
        "schema_version": schema_version,
        "builder_cache_key": cache_key,
        "counts": {
            "candidates": len(_triplet_candidate_rows(triplets)),
            "hyperedges": len(fresh["hyperedges"]),
            "support": len(fresh["support"]),
        },
    }
    (stage / "stats.json").write_text(stable_json_dumps(stats) + "\n", encoding="utf-8")


def build_triplet_builder_with_cache(
    *,
    out_base: str | Path,
    schema_version: str,
    sources: list[dict[str, Any]],
    builder_entry: dict[str, Any],
    triplets: list[dict[str, Any]],
    build_fn: Callable[..., BuilderOutput],
    build_kwargs: dict[str, Any] | None = None,
) -> BuilderOutput:
    """Run a triplet-input builder with on-disk stage cache (shared layout)."""
    builder_name = str(builder_entry["name"])
    cache_key = compute_builder_cache_key(
        schema_version=schema_version,
        sources=sources,
        builder_entry=builder_entry,
        triplets=triplets,
    )
    stage = builder_stage_dir(out_base, builder_name, cache_key)
    cached = _try_load_builder_cache(
        stage, expected_key=cache_key, builder_name=builder_name
    )
    if cached is not None:
        return cached

    kwargs = dict(build_kwargs or {})
    fresh = build_fn(triplets, **kwargs)
    _write_builder_cache(
        stage,
        builder_name=builder_name,
        cache_key=cache_key,
        schema_version=schema_version,
        triplets=triplets,
        fresh=fresh,
    )
    return fresh


def _primekg_version_from_entry(builder_entry: dict[str, Any]) -> str:
    params = builder_entry.get("params")
    if isinstance(params, dict):
        ver = params.get("primekg_version")
        if isinstance(ver, str) and ver:
            return ver
    return "unknown"


def build_phenotype_with_cache(
    *,
    out_base: str | Path,
    schema_version: str,
    sources: list[dict[str, Any]],
    builder_entry: dict[str, Any],
    triplets: list[dict[str, Any]],
    emit_composed: bool = True,
) -> BuilderOutput:
    """Run ``build_phenotype_hyperedges`` with stage cache under ``out_base/builder/phenotype/``."""
    return build_triplet_builder_with_cache(
        out_base=out_base,
        schema_version=schema_version,
        sources=sources,
        builder_entry=builder_entry,
        triplets=triplets,
        build_fn=build_phenotype_hyperedges,
        build_kwargs={"emit_composed": emit_composed},
    )


def build_moa_triad_with_cache(
    *,
    out_base: str | Path,
    schema_version: str,
    sources: list[dict[str, Any]],
    builder_entry: dict[str, Any],
    triplets: list[dict[str, Any]],
) -> BuilderOutput:
    """Run ``build_moa_triad_hyperedges`` with stage cache."""
    return build_triplet_builder_with_cache(
        out_base=out_base,
        schema_version=schema_version,
        sources=sources,
        builder_entry=builder_entry,
        triplets=triplets,
        build_fn=build_moa_triad_hyperedges,
        build_kwargs={"source_version": _primekg_version_from_entry(builder_entry)},
    )


def build_drug_disease_shared_target_bundle_with_cache(
    *,
    out_base: str | Path,
    schema_version: str,
    sources: list[dict[str, Any]],
    builder_entry: dict[str, Any],
    triplets: list[dict[str, Any]],
) -> BuilderOutput:
    """Run ``build_drug_disease_shared_target_bundle_hyperedges`` with stage cache."""
    params = builder_entry.get("params")
    kwargs: dict[str, Any] = {
        "source_version": _primekg_version_from_entry(builder_entry)
    }
    if isinstance(params, dict):
        for key in (
            "max_drugs_per_bundle",
            "max_eligible_drugs",
            "max_bundles_per_protein_disease",
        ):
            if key in params:
                kwargs[key] = int(params[key])
    return build_triplet_builder_with_cache(
        out_base=out_base,
        schema_version=schema_version,
        sources=sources,
        builder_entry=builder_entry,
        triplets=triplets,
        build_fn=build_drug_disease_shared_target_bundle_hyperedges,
        build_kwargs=kwargs,
    )


def registry_atomic_builder_entry(
    *,
    primekg_version: str,
    admissions: tuple[AtomicAdmission, ...] = COVERAGE_REGISTRY_ADMISSIONS,
) -> dict[str, Any]:
    """Manifest/cache builder descriptor for ``registry_atomic``."""
    return {
        "name": "registry_atomic",
        "version": 2,
        "params": {
            "admissions": [
                {"admission_id": a.admission_id, "relations": sorted(a.relations)}
                for a in admissions
            ],
            "primekg_version": primekg_version,
        },
    }


def build_registry_atomic_with_cache(
    *,
    out_base: str | Path,
    schema_version: str,
    sources: list[dict[str, Any]],
    builder_entry: dict[str, Any],
    triplets: list[dict[str, Any]],
    admissions: tuple[AtomicAdmission, ...] = COVERAGE_REGISTRY_ADMISSIONS,
) -> BuilderOutput:
    """Run ``build_registry_atomic_hyperedges`` with stage cache."""
    return build_triplet_builder_with_cache(
        out_base=out_base,
        schema_version=schema_version,
        sources=sources,
        builder_entry=builder_entry,
        triplets=triplets,
        build_fn=build_registry_atomic_hyperedges,
        build_kwargs={
            "admissions": admissions,
            "source_version": _primekg_version_from_entry(builder_entry),
        },
    )
