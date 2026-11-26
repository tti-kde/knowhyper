from __future__ import annotations

import hashlib
from typing import Protocol, TypedDict


class TruncationContext(TypedDict):
    """Stable context for deterministic truncation."""

    query_id: str
    level: str


class Truncator(Protocol):
    """Truncation strategy interface."""

    version: str

    def truncate(
        self, ids: list[str], *, budget: int, context: TruncationContext
    ) -> list[str]:
        """Return a deterministically truncated list of IDs."""
        ...


class HashBasedTruncator:
    """Deterministic truncator based on hashed ranking."""

    def __init__(self, *, version: str = "hash_based_v1") -> None:
        """Create a hash-based truncator with the given version tag."""
        self.version = version

    def truncate(
        self, ids: list[str], *, budget: int, context: TruncationContext
    ) -> list[str]:
        """Truncate IDs deterministically using a stable hash."""
        b = max(0, int(budget))
        if b == 0:
            return []

        unique = sorted({x for x in ids if isinstance(x, str) and x})
        if len(unique) <= b:
            return unique

        qid = str(context.get("query_id", ""))
        lvl = str(context.get("level", ""))

        def key(x: str) -> str:
            payload = f"{self.version}\n{qid}\n{lvl}\n{x}".encode()
            return hashlib.sha256(payload).hexdigest()

        ranked = sorted(unique, key=lambda x: (key(x), x))
        return ranked[:b]
