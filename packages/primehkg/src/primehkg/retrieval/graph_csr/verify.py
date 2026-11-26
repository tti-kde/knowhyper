from __future__ import annotations


class GraphCsrFingerprintError(ValueError):
    """Raised when a graph CSR sidecar fingerprint check fails."""


class GraphCsrFingerprintMissingError(GraphCsrFingerprintError):
    """Raised when a strict sidecar fingerprint is missing."""

    def __init__(self) -> None:
        """Create the exception with a fixed message."""
        super().__init__("strict: sidecar missing inputs_fingerprint")


class GraphCsrFingerprintMismatchError(GraphCsrFingerprintError):
    """Raised when a strict sidecar fingerprint does not match."""

    def __init__(self) -> None:
        """Create the exception with a fixed message."""
        super().__init__("strict: sidecar inputs_fingerprint mismatch")


def require_matching_fingerprint(
    *,
    loaded_fingerprint: str | None,
    expected_fingerprint: str | None,
    strict: bool,
) -> None:
    if expected_fingerprint is None:
        return
    if loaded_fingerprint is None:
        if strict:
            raise GraphCsrFingerprintMissingError
        return
    if loaded_fingerprint != expected_fingerprint:
        if strict:
            raise GraphCsrFingerprintMismatchError
        return
