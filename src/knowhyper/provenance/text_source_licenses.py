"""Fail-closed source registry for raw textual evidence.

PrimeHKG 0.12.9 supplies row-local license and provenance metadata for complete
rows.  Text baselines additionally resolve each selected row through this
independent, pinned downstream registry and require both decisions to agree
before the evidence can be used in a publishable experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from knowhyper.provenance.values import is_concrete_provenance_value

DEFAULT_REGISTRY_PATH = "knowhyper.provenance:text_source_licenses.yaml"


class UsageScope(StrEnum):
    """How a text source may be used by controlled text baselines."""

    ALLOW = "allow"
    INTERNAL_ONLY = "internal_only"
    EXCLUDE = "exclude"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ResolvedLicense:
    """Registry decision attached to one selected raw evidence row."""

    source: str
    usage_scope: UsageScope
    redistribution_scope: str
    license_id: str
    license_url: str
    source_release: str
    reviewed_at: str
    payload_class: str = ""
    source_field: str = ""
    upstream_source: str = ""
    upstream_release: str = ""
    registry_digest: str = ""

    @property
    def allowed(self) -> bool:
        return self.usage_scope is UsageScope.ALLOW

    def as_provenance(self) -> dict[str, str]:
        return {
            "source": self.source,
            "usage_scope": self.usage_scope.value,
            "redistribution_scope": self.redistribution_scope,
            "license_id": self.license_id,
            "license_url": self.license_url,
            "source_release": self.source_release,
            "payload_class": self.payload_class,
            "source_field": self.source_field,
            "upstream_source": self.upstream_source,
            "upstream_release": self.upstream_release,
            "reviewed_at": self.reviewed_at,
            "registry_digest": self.registry_digest,
        }


class SourceLicenseRegistry:
    """Resolve source-level license decisions for selected text evidence.

    Matching is deliberately simple and deterministic:

    1. exact ``(source, modality)`` entry
    2. exact ``(source, source_field)`` entry
    3. source-only entry

    Missing entries fail closed.  ``allow_internal=True`` is intended only for
    restricted smoke tests; publishable runs must leave it false.
    """

    def __init__(
        self,
        entries: list[dict[str, Any]],
        *,
        registry_digest: str,
        allow_internal: bool = False,
    ) -> None:
        if not entries:
            msg = "source license registry must contain at least one entry"
            raise ValueError(msg)
        self.digest = registry_digest
        self.allow_internal = allow_internal
        self._entries = tuple(self._normalize_entry(entry) for entry in entries)

    @classmethod
    def from_path(
        cls, path: str | Path, *, allow_internal: bool = False
    ) -> SourceLicenseRegistry:
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_text(text, allow_internal=allow_internal)

    @classmethod
    def from_resource(
        cls, resource: str = DEFAULT_REGISTRY_PATH, *, allow_internal: bool = False
    ) -> SourceLicenseRegistry:
        package, _, name = resource.partition(":")
        if not package or not name:
            msg = f"registry resource must be 'package:name', got {resource!r}"
            raise ValueError(msg)
        text = resources.files(package).joinpath(name).read_text(encoding="utf-8")
        return cls.from_text(text, allow_internal=allow_internal)

    @classmethod
    def from_text(
        cls, text: str, *, allow_internal: bool = False
    ) -> SourceLicenseRegistry:
        payload = yaml.safe_load(text)
        entries = payload.get("sources") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            msg = "source license registry must be a list or contain sources: []"
            raise TypeError(msg)
        digest = hashlib.sha256(_canonical_yaml_payload(payload).encode()).hexdigest()
        return cls(entries, registry_digest=digest, allow_internal=allow_internal)

    def resolve(
        self,
        *,
        source: str,
        modality: str = "",
        license_tag: str | None = None,
    ) -> ResolvedLicense:
        source = str(source or "").strip()
        modality = str(modality or "").strip()
        if not source:
            msg = "text evidence row has empty source; fail-closed"
            raise ValueError(msg)
        entry = self._match(source=source, modality=modality)
        if entry is None:
            msg = f"unknown text evidence source {source!r}; fail-closed"
            raise ValueError(msg)
        resolved = ResolvedLicense(
            source=source,
            usage_scope=UsageScope(entry["usage_scope"]),
            redistribution_scope=str(entry.get("redistribution_scope", "")),
            license_id=str(entry.get("license_id", "")),
            license_url=str(entry.get("license_url", "")),
            source_release=str(entry.get("source_release", "")),
            reviewed_at=str(entry.get("reviewed_at", "")),
            payload_class=str(entry.get("payload_class", "")),
            source_field=str(entry.get("source_field", modality)),
            upstream_source=str(entry.get("upstream_source", "")),
            upstream_release=str(entry.get("upstream_release", "")),
            registry_digest=self.digest,
        )
        if resolved.usage_scope is UsageScope.UNKNOWN:
            msg = f"source {source!r} has unknown usage scope; fail-closed"
            raise ValueError(msg)
        if resolved.usage_scope is UsageScope.EXCLUDE:
            msg = f"source {source!r} is excluded by license registry"
            raise ValueError(msg)
        if resolved.usage_scope is UsageScope.INTERNAL_ONLY and not self.allow_internal:
            msg = f"source {source!r} is internal-only; publishable selection denied"
            raise ValueError(msg)
        if license_tag not in (None, "", resolved.license_id):
            # Complete 0.12.9 rows mirror license_id into license_tag.  Legacy
            # null tags remain tolerated, but populated tags must agree with
            # the independent downstream decision.
            msg = (
                f"license tag mismatch for source {source!r}: "
                f"row={license_tag!r}, registry={resolved.license_id!r}"
            )
            raise ValueError(msg)
        return resolved

    def decision_for(
        self,
        *,
        source: str,
        modality: str = "",
        license_tag: str | None = None,
    ) -> ResolvedLicense | None:
        try:
            return self.resolve(
                source=source,
                modality=modality,
                license_tag=license_tag,
            )
        except ValueError:
            return None

    def validate_for_audience(self, audience: str) -> None:
        """Validate that registry capabilities match the requested run audience.

        ``allow_internal`` is a capability on the registry rather than a row
        selection hint.  Requiring an exact match here prevents an internal
        registry from being accidentally injected into a publishable run (or
        an internal run from silently behaving like the publishable arm).
        """
        if audience == "publishable":
            if self.allow_internal:
                msg = "publishable audience requires allow_internal=false"
                raise ValueError(msg)
            for entry in self._entries:
                if entry["usage_scope"] != UsageScope.ALLOW.value:
                    continue
                missing = [
                    field
                    for field in (
                        "license_id",
                        "license_url",
                        "source_release",
                        "upstream_source",
                        "upstream_release",
                    )
                    if not is_concrete_provenance_value(entry.get(field, ""))
                ]
                if missing:
                    msg = (
                        f"publishable source {entry['source']!r} has no concrete "
                        f"{', '.join(missing)}"
                    )
                    raise ValueError(msg)
                if (
                    not entry["license_url"]
                    .casefold()
                    .startswith(("https://", "http://"))
                ):
                    msg = (
                        f"publishable source {entry['source']!r} has invalid "
                        "license_url"
                    )
                    raise ValueError(msg)
            return
        if audience == "internal":
            if not self.allow_internal:
                msg = "internal audience requires explicit allow_internal=true"
                raise ValueError(msg)
            return
        if audience != "legacy":
            msg = f"unsupported text-evidence audience: {audience!r}"
            raise ValueError(msg)

    def _match(self, *, source: str, modality: str) -> dict[str, str] | None:
        # Raw snapshot source spelling is not canonical (notably ``chebi`` is
        # lowercase while the reviewed registry uses the official ``ChEBI``
        # name). Licensing identity is case-insensitive; provenance output still
        # preserves the raw row's exact source spelling.
        source_key = source.casefold()
        candidates = [
            entry for entry in self._entries if entry["source"].casefold() == source_key
        ]
        for entry in candidates:
            if entry.get("source_field") == modality:
                return entry
        for entry in candidates:
            if entry.get("modality") == modality:
                return entry
        for entry in candidates:
            if not entry.get("source_field") and not entry.get("modality"):
                return entry
        return None

    @staticmethod
    def _normalize_entry(entry: dict[str, Any]) -> dict[str, str]:
        if not isinstance(entry, dict):
            msg = "each source license registry entry must be a mapping"
            raise TypeError(msg)
        required = ("source", "usage_scope", "redistribution_scope", "reviewed_at")
        missing = [name for name in required if not str(entry.get(name, "")).strip()]
        if missing:
            msg = f"source license registry entry missing required fields: {missing}"
            raise ValueError(msg)
        scope = str(entry["usage_scope"]).strip()
        try:
            UsageScope(scope)
        except ValueError as exc:
            msg = f"invalid usage_scope {scope!r}"
            raise ValueError(msg) from exc
        return {
            str(key): str(value) for key, value in entry.items() if value is not None
        }


def load_source_license_registry(
    path: str | Path | None = None,
    *,
    allow_internal: bool = False,
) -> SourceLicenseRegistry:
    """Hydra-friendly source registry factory."""
    if path in (None, "", DEFAULT_REGISTRY_PATH):
        return SourceLicenseRegistry.from_resource(
            DEFAULT_REGISTRY_PATH,
            allow_internal=allow_internal,
        )
    return SourceLicenseRegistry.from_path(path, allow_internal=allow_internal)


def _canonical_yaml_payload(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
