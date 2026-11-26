"""Shared validation for provenance release/date values."""

from __future__ import annotations

from typing import Any

_NON_CONCRETE_TOKENS = frozenset({"unknown", "unspecified", "pending"})


def is_concrete_provenance_value(value: Any) -> bool:
    """Return whether a provenance value is concrete rather than a sentinel."""
    if value is None:
        return False
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if not normalized or normalized in {"none", "n/a", "na", "null"}:
        return False
    tokens = set(normalized.replace("/", "_").split("_"))
    return not tokens.intersection(_NON_CONCRETE_TOKENS)
