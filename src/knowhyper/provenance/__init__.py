"""Provenance and licensing utilities for text-evidence baselines."""

from __future__ import annotations

from knowhyper.provenance.text_source_licenses import (
    DEFAULT_REGISTRY_PATH,
    ResolvedLicense,
    SourceLicenseRegistry,
    UsageScope,
    load_source_license_registry,
)

__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "ResolvedLicense",
    "SourceLicenseRegistry",
    "UsageScope",
    "load_source_license_registry",
]
