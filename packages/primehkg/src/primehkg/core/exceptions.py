"""Exceptions for PrimeHKG core dataset loading."""

from __future__ import annotations


class CoreDatasetLoadError(Exception):
    """Base error for ``CoreDataset`` / ``load_core_dataset`` failures."""


class FingerprintMismatchError(CoreDatasetLoadError):
    """Raised when ``expected_fingerprint`` does not match the build manifest."""


class HubLoadError(CoreDatasetLoadError):
    """Raised when Hugging Face Hub dataset loading fails."""
