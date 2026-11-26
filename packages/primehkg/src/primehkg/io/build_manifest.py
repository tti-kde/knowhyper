from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

DEFAULT_MANIFEST_SCHEMA_VERSION = "0.1.0"


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_json_dumps(obj: Any) -> str:
    """Deterministic JSON serialization (sorted keys, compact). Used for cache keys."""
    return _stable_json(obj)


def deep_sort_jsonish(obj: Any) -> Any:
    """Return a JSON-serializable structure with deterministic dict key order and list order."""
    if isinstance(obj, dict):
        return {k: deep_sort_jsonish(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        items = [deep_sort_jsonish(x) for x in obj]
        return sorted(items, key=_stable_json)
    return obj


def sha256_file(path: str | pathlib.Path) -> str:
    """SHA-256 hex digest of file bytes (streaming, deterministic)."""
    p = pathlib.Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def primekg_kg_source(*, path: str | pathlib.Path) -> dict[str, Any]:
    """Build one manifest `sources[]` entry for a PrimeKG `kg.csv` file."""
    p = pathlib.Path(path).resolve()
    return {
        "name": "primekg_kg",
        "path": str(p),
        "sha256": sha256_file(p),
    }


def manifest_source_file(*, name: str, path: str | pathlib.Path) -> dict[str, Any]:
    """One manifest ``sources[]`` entry for an arbitrary local input file."""
    p = pathlib.Path(path).resolve()
    return {"name": name, "path": str(p), "sha256": sha256_file(p)}


def compute_inputs_fingerprint(
    *,
    schema_version: str,
    sources: list[dict[str, Any]],
    builders: list[dict[str, Any]],
    build_profile: str | None = None,
) -> str:
    """Deterministic fingerprint over schema version + normalized sources + builders."""
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "sources": deep_sort_jsonish(sources),
        "builders": deep_sort_jsonish(builders),
    }
    if build_profile is not None:
        payload["build_profile"] = build_profile
    canonical = stable_json_dumps(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assemble_build_manifest(
    *,
    schema_version: str,
    sources: list[dict[str, Any]],
    builders: list[dict[str, Any]],
    counts: dict[str, int],
    build_profile: str | None = None,
    extra_top_level: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full `build_manifest.json` document (sorted keys at top level via writer)."""
    fp = compute_inputs_fingerprint(
        schema_version=schema_version,
        sources=sources,
        builders=builders,
        build_profile=build_profile,
    )
    out: dict[str, Any] = {
        "schema_version": schema_version,
        "inputs_fingerprint": fp,
        "sources": deep_sort_jsonish(sources),
        "builders": deep_sort_jsonish(builders),
        "outputs": dict(sorted(counts.items())),
    }
    if build_profile is not None:
        out["build_profile"] = build_profile
    if extra_top_level:
        for key, value in extra_top_level.items():
            out.setdefault(key, value)
    return out


def write_build_manifest_json(
    out_dir: str | pathlib.Path, manifest: dict[str, Any]
) -> None:
    """Write `build_manifest.json` under ``out_dir`` (compact, deterministic JSON)."""
    text = stable_json_dumps(manifest) + "\n"
    pathlib.Path(out_dir).joinpath("build_manifest.json").write_text(
        text, encoding="utf-8"
    )


def merge_manifest_sidecars(
    out_dir: str | pathlib.Path,
    patch: dict[str, Any],
) -> None:
    """Merge ``patch`` into ``build_manifest.json`` → ``sidecars`` without dropping keys."""
    from primehkg.core.manifest import read_build_manifest

    root = pathlib.Path(out_dir)
    manifest = dict(read_build_manifest(root))
    sidecars = manifest.get("sidecars")
    if not isinstance(sidecars, dict):
        sidecars = {}
    sidecars.update(patch)
    manifest["sidecars"] = sidecars
    write_build_manifest_json(root, manifest)
