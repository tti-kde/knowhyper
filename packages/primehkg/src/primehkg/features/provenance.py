from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from primehkg.features.mondo_ambiguity import MONDO_AMBIGUITY_HINT_KEY

TEXT_PROVENANCE_SCHEMA = "node_features_text_provenance_v1"
PENDING_RELEASE = "PENDING"

_REQUIRED_ENTRY_FIELDS = (
    "source",
    "source_field",
    "upstream_source",
    "upstream_release",
    "retrieved_at",
    "source_url",
    "license_id",
    "license_url",
    "usage_scope",
    "redistribution_scope",
    "reviewed_at",
)


class ProvenanceConflictError(ValueError):
    """Row-local provenance hints contradict the registry."""


class StrictTextProvenanceError(ValueError):
    """Strict text provenance mode rejected one or more rows."""


def normalize_source_key(source: str) -> str:
    s = source.strip()
    if s.lower() == "chebi":
        return "chebi"
    return s


def _is_pending(value: str | None) -> bool:
    return value is not None and str(value).strip() == PENDING_RELEASE


def _concrete_release(release: str | None) -> str | None:
    if release is None or _is_pending(release):
        return None
    stripped = str(release).strip()
    return stripped or None


def _concrete_retrieved_at(value: str | None) -> str | None:
    if value is None or _is_pending(value):
        return None
    stripped = str(value).strip()
    return stripped or None


@dataclass(frozen=True, slots=True)
class TextProvenanceEntry:
    """One registry entry for a (source, source_field) text provenance pair."""

    source: str
    source_field: str
    upstream_source: str
    upstream_release: str
    retrieved_at: str
    source_url: str
    license_id: str
    license_url: str
    usage_scope: str
    redistribution_scope: str
    reviewed_at: str
    source_artifact_sha256: str | None = None
    attribution_text: str | None = None
    provenance_chain: tuple[Any, ...] | None = None


@dataclass(frozen=True, slots=True)
class TextProvenanceRegistry:
    """Loaded text provenance registry with digest and (source, source_field) lookup."""

    digest: str
    entries: tuple[TextProvenanceEntry, ...]
    _index: dict[tuple[str, str], TextProvenanceEntry]

    def get(self, source: str, source_field: str) -> TextProvenanceEntry | None:
        """Return the registry entry for ``(source, source_field)``, or None."""
        key = (normalize_source_key(source), source_field.strip())
        return self._index.get(key)


def _parse_entry(row: dict[str, Any]) -> TextProvenanceEntry:
    missing = [f for f in _REQUIRED_ENTRY_FIELDS if f not in row]
    if missing:
        msg = f"text provenance registry: entry missing required fields: {missing}"
        raise ValueError(msg)
    chain = row.get("provenance_chain")
    parsed_chain: tuple[Any, ...] | None
    if chain is None:
        parsed_chain = None
    elif isinstance(chain, list):
        parsed_chain = tuple(chain)
    else:
        msg = "text provenance registry: provenance_chain must be a list when set"
        raise TypeError(msg)
    artifact = row.get("source_artifact_sha256")
    attribution = row.get("attribution_text")
    entry = TextProvenanceEntry(
        source=str(row["source"]).strip(),
        source_field=str(row["source_field"]).strip(),
        upstream_source=str(row["upstream_source"]).strip(),
        upstream_release=str(row["upstream_release"]).strip(),
        retrieved_at=str(row["retrieved_at"]).strip(),
        source_url=str(row["source_url"]).strip(),
        license_id=str(row["license_id"]).strip(),
        license_url=str(row["license_url"]).strip(),
        usage_scope=str(row["usage_scope"]).strip(),
        redistribution_scope=str(row["redistribution_scope"]).strip(),
        reviewed_at=str(row["reviewed_at"]).strip(),
        source_artifact_sha256=str(artifact).strip() if artifact is not None else None,
        attribution_text=str(attribution).strip() if attribution is not None else None,
        provenance_chain=parsed_chain,
    )
    if entry.redistribution_scope == "attribution_required" and not (
        entry.attribution_text and entry.attribution_text.strip()
    ):
        msg = (
            "text provenance registry: attribution_required entry "
            f"({entry.source!r}, {entry.source_field!r}) missing attribution_text"
        )
        raise ValueError(msg)
    return entry


def load_text_provenance_registry(path: Path | None = None) -> TextProvenanceRegistry:
    if path is None:
        raw = (
            resources.files("primehkg.policy")
            .joinpath("text_provenance_registry.yaml")
            .read_bytes()
        )
    else:
        raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    obj: Any = yaml.safe_load(raw)
    if not isinstance(obj, dict):
        msg = "text provenance registry: top-level must be a mapping"
        raise TypeError(msg)
    if obj.get("schema") != "text_provenance_registry_v1":
        msg = "text provenance registry: schema must be text_provenance_registry_v1"
        raise ValueError(msg)
    raw_entries = obj.get("entries")
    if not isinstance(raw_entries, list):
        msg = "text provenance registry: entries must be a list"
        raise TypeError(msg)

    entries: list[TextProvenanceEntry] = []
    index: dict[tuple[str, str], TextProvenanceEntry] = {}
    for row in raw_entries:
        if not isinstance(row, dict):
            msg = "text provenance registry: each entry must be a mapping"
            raise TypeError(msg)
        entry = _parse_entry(row)
        key = (normalize_source_key(entry.source), entry.source_field)
        if key in index:
            msg = (
                "text provenance registry: duplicate entry for "
                f"({entry.source!r}, {entry.source_field!r})"
            )
            raise ValueError(msg)
        index[key] = entry
        entries.append(entry)

    return TextProvenanceRegistry(
        digest=digest,
        entries=tuple(entries),
        _index=index,
    )


def _check_conflict(
    row: dict[str, Any],
    field: str,
    registry_value: str | None,
) -> None:
    row_value = row.get(field)
    if row_value is None:
        return
    row_str = str(row_value).strip()
    if not row_str:
        return
    reg_str = "" if registry_value is None else str(registry_value).strip()
    if _is_pending(reg_str):
        return
    if row_str != reg_str:
        msg = (
            f"provenance conflict on {field!r}: row has {row_str!r}, "
            f"registry has {reg_str!r}"
        )
        raise ProvenanceConflictError(msg)


def _provenance_status(
    *,
    entry: TextProvenanceEntry | None,
    upstream_source: str | None,
    upstream_release: str | None,
    license_id: str | None,
    retrieved_at: str | None,
) -> str:
    if entry is None:
        return "unknown"
    has_upstream = bool(upstream_source and str(upstream_source).strip())
    has_release = bool(upstream_release and str(upstream_release).strip())
    has_license = bool(license_id and str(license_id).strip())
    has_retrieved = bool(retrieved_at and str(retrieved_at).strip())
    if has_upstream and has_release and has_license and has_retrieved:
        return "complete"
    return "partial"


def merge_text_provenance(
    row: dict[str, Any],
    registry: TextProvenanceRegistry,
    *,
    artifact_sha256: str | None = None,
) -> dict[str, Any]:
    out = dict(row)
    usage_scope_override = out.pop(MONDO_AMBIGUITY_HINT_KEY, None)
    source = str(row.get("source", ""))
    source_field = str(row.get("source_field", "")).strip()
    entry = registry.get(source, source_field)

    out["provenance_registry_digest"] = registry.digest

    if entry is None:
        out["provenance_status"] = "unknown"
        out["usage_scope"] = "unknown"
        out["upstream_source"] = None
        out["upstream_release"] = None
        out["upstream_record_id"] = None
        out["retrieved_at"] = None
        out["source_url"] = None
        out["source_artifact_sha256"] = None
        out["license_id"] = None
        out["license_url"] = None
        out["redistribution_scope"] = None
        out["attribution_text"] = None
        out["provenance_chain"] = None
        out["license_tag"] = None
        return out

    _check_conflict(row, "upstream_source", entry.upstream_source)
    _check_conflict(row, "upstream_release", entry.upstream_release)
    _check_conflict(row, "license_id", entry.license_id)

    upstream_release = _concrete_release(entry.upstream_release)
    license_id = entry.license_id.strip() or None
    upstream_source = entry.upstream_source.strip() or None
    retrieved_at = _concrete_retrieved_at(entry.retrieved_at)

    out["upstream_source"] = upstream_source
    out["upstream_release"] = upstream_release
    out["retrieved_at"] = retrieved_at
    out["source_url"] = entry.source_url.strip() or None
    out["license_id"] = license_id
    out["license_url"] = entry.license_url.strip() or None
    out["usage_scope"] = entry.usage_scope.strip() or "unknown"
    out["redistribution_scope"] = entry.redistribution_scope.strip() or None
    out["attribution_text"] = entry.attribution_text
    if entry.provenance_chain is not None:
        out["provenance_chain"] = json.dumps(list(entry.provenance_chain))
    else:
        out["provenance_chain"] = None

    artifact = artifact_sha256
    if artifact is None and entry.source_artifact_sha256:
        artifact = entry.source_artifact_sha256
    out["source_artifact_sha256"] = artifact

    status = _provenance_status(
        entry=entry,
        upstream_source=upstream_source,
        upstream_release=upstream_release,
        license_id=license_id,
        retrieved_at=retrieved_at,
    )
    out["provenance_status"] = status
    out["license_tag"] = license_id if status == "complete" else None
    if usage_scope_override is not None:
        out["usage_scope"] = str(usage_scope_override).strip() or out["usage_scope"]
    return out


def _strict_reject_reason(
    row: dict[str, Any],
    merged: dict[str, Any],
    registry: TextProvenanceRegistry,
) -> str | None:
    source = str(row.get("source", ""))
    source_field = str(row.get("source_field", "")).strip()
    status = str(merged.get("provenance_status", "unknown"))
    if status == "unknown":
        return (
            f"strict text provenance rejected row "
            f"({row.get('node_id')!r}, source={source!r}, "
            f"source_field={source_field!r}): unregistered source/field"
        )
    entry = registry.get(source, source_field)
    if entry is not None and _is_pending(entry.upstream_release):
        return (
            f"strict text provenance rejected row "
            f"({row.get('node_id')!r}, source={source!r}, "
            f"source_field={source_field!r}): upstream_release is PENDING"
        )
    if entry is not None and _is_pending(entry.retrieved_at):
        return (
            f"strict text provenance rejected row "
            f"({row.get('node_id')!r}, source={source!r}, "
            f"source_field={source_field!r}): retrieved_at is PENDING"
        )
    usage_scope = str(merged.get("usage_scope", "")).strip()
    upstream_release = merged.get("upstream_release")
    retrieved_at = merged.get("retrieved_at")
    if usage_scope == "allow" and not upstream_release:
        return (
            f"strict text provenance rejected row "
            f"({row.get('node_id')!r}, source={source!r}, "
            f"source_field={source_field!r}): usage_scope=allow without "
            f"concrete upstream_release"
        )
    if usage_scope == "allow" and not retrieved_at:
        return (
            f"strict text provenance rejected row "
            f"({row.get('node_id')!r}, source={source!r}, "
            f"source_field={source_field!r}): usage_scope=allow without "
            f"concrete retrieved_at"
        )
    return None


def apply_text_provenance(
    rows: list[dict[str, Any]],
    registry: TextProvenanceRegistry,
    *,
    artifact_sha_by_source: dict[str, str] | None = None,
    strict: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sha_by_source = artifact_sha_by_source or {}
    out_rows: list[dict[str, Any]] = []
    counts = {"rows_total": len(rows), "complete": 0, "partial": 0, "unknown": 0}

    for row in rows:
        source = str(row.get("source", ""))
        artifact_sha = sha_by_source.get(source)
        if artifact_sha is None:
            artifact_sha = sha_by_source.get(normalize_source_key(source))
        merged = merge_text_provenance(
            row,
            registry,
            artifact_sha256=artifact_sha,
        )
        status = str(merged.get("provenance_status", "unknown"))
        if status not in counts:
            status = "unknown"
        counts[status] += 1
        if strict:
            reject = _strict_reject_reason(row, merged, registry)
            if reject is not None:
                raise StrictTextProvenanceError(reject)
        out_rows.append(merged)

    return out_rows, counts


def packaged_text_provenance_registry_path() -> Path:
    """Filesystem path to the packaged default registry YAML."""
    import primehkg.policy

    return (
        Path(primehkg.policy.__file__).resolve().parent
        / "text_provenance_registry.yaml"
    )


def resolve_text_provenance_registry_path(path: Path | None) -> Path:
    """Return explicit path or the packaged default registry file."""
    if path is not None:
        return path
    return packaged_text_provenance_registry_path()


def registry_source_inventory(
    registry: TextProvenanceRegistry,
    *,
    artifact_sha_by_source: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Per-registry-entry manifest inventory with optional artifact digests."""
    sha_by_source = artifact_sha_by_source or {}
    items: list[dict[str, Any]] = []
    for entry in registry.entries:
        artifact = sha_by_source.get(entry.source)
        if artifact is None:
            artifact = sha_by_source.get(normalize_source_key(entry.source))
        item: dict[str, Any] = {
            "source": entry.source,
            "source_field": entry.source_field,
            "upstream_source": entry.upstream_source,
            "upstream_release": entry.upstream_release,
            "retrieved_at": entry.retrieved_at,
            "source_url": entry.source_url,
            "license_id": entry.license_id,
            "license_url": entry.license_url,
            "usage_scope": entry.usage_scope,
            "redistribution_scope": entry.redistribution_scope,
            "reviewed_at": entry.reviewed_at,
        }
        if entry.source_artifact_sha256:
            item["source_artifact_sha256"] = entry.source_artifact_sha256
        elif artifact:
            item["source_artifact_sha256"] = artifact
        if entry.attribution_text:
            item["attribution_text"] = entry.attribution_text
        if entry.provenance_chain is not None:
            item["provenance_chain"] = list(entry.provenance_chain)
        items.append(item)
    items.sort(key=lambda x: (str(x["source"]), str(x["source_field"])))
    return items


def provenance_inventory(
    rows: list[dict[str, Any]],
    registry: TextProvenanceRegistry,
    *,
    mondo_definition_ambiguity: dict[str, int] | None = None,
) -> dict[str, Any]:
    status_counts = {"complete": 0, "partial": 0, "unknown": 0}
    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        status = str(row.get("provenance_status", "unknown"))
        if status not in status_counts:
            status = "unknown"
        status_counts[status] += 1
        source = str(row.get("source", ""))
        source_bucket = by_source.setdefault(
            source,
            {"complete": 0, "partial": 0, "unknown": 0},
        )
        source_bucket[status] += 1
        scope = str(row.get("usage_scope", "") or "").strip()
        if scope:
            scopes = source_bucket.setdefault("usage_scope", {})
            assert isinstance(scopes, dict)
            scopes[scope] = int(scopes.get(scope, 0)) + 1

    out: dict[str, Any] = {
        "registry_digest": registry.digest,
        "rows_total": len(rows),
        "provenance_status": status_counts,
        "by_source": by_source,
    }
    if mondo_definition_ambiguity is not None:
        out["mondo_definition_ambiguity"] = dict(mondo_definition_ambiguity)
    return out


def assert_inventory_matches_manifest(
    inventory: dict[str, Any],
    manifest: dict[str, Any],
    text_rows: list[dict[str, Any]],
) -> None:
    dig = manifest.get("provenance_registry_sha256")
    if inventory.get("registry_digest") != dig:
        msg = (
            f"inventory registry_digest {inventory.get('registry_digest')!r} "
            f"!= manifest provenance_registry_sha256 {dig!r}"
        )
        raise AssertionError(msg)
    if int(inventory.get("rows_total", -1)) != len(text_rows):
        msg = (
            f"inventory rows_total {inventory.get('rows_total')!r} "
            f"!= len(text_rows) {len(text_rows)}"
        )
        raise AssertionError(msg)
    expected_status = {"complete": 0, "partial": 0, "unknown": 0}
    for row in text_rows:
        status = str(row.get("provenance_status", "unknown"))
        if status not in expected_status:
            status = "unknown"
        expected_status[status] += 1
    if inventory.get("provenance_status") != expected_status:
        msg = (
            f"inventory provenance_status {inventory.get('provenance_status')!r} "
            f"!= recomputed {expected_status!r}"
        )
        raise AssertionError(msg)
    amb_inv = inventory.get("mondo_definition_ambiguity")
    amb_mf = manifest.get("mondo_definition_ambiguity")
    if amb_mf is not None and amb_inv != amb_mf:
        msg = f"inventory mondo_definition_ambiguity {amb_inv!r} != manifest {amb_mf!r}"
        raise AssertionError(msg)
    mondo_rows = [r for r in text_rows if str(r.get("source", "")) == "MONDO"]
    if mondo_rows:
        bucket = (inventory.get("by_source") or {}).get("MONDO") or {}
        scopes = bucket.get("usage_scope")
        if not isinstance(scopes, dict) or not scopes:
            msg = "inventory by_source.MONDO.usage_scope required when MONDO rows exist"
            raise AssertionError(msg)
