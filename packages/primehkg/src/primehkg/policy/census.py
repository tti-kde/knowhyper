from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from primehkg.policy.registry import RelationEntry


def count_relations_in_kg_csv(path: Path) -> list[tuple[str, int]]:
    """Count non-empty ``relation`` values in a PrimeKG ``kg.csv`` (streaming)."""
    counts: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "relation" not in fields:
            msg = f"kg.csv missing 'relation' column (columns: {fields!r})"
            raise ValueError(msg)
        for row in reader:
            rel = str(row.get("relation", "")).strip()
            if rel:
                counts[rel] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def relations_missing_from_registry(
    observed: Sequence[str],
    registry: dict[str, RelationEntry],
) -> list[str]:
    """Return sorted relation strings that have no registry key (raw, not alias-resolved)."""
    keys = set(registry)
    return sorted({r.strip() for r in observed if r.strip() and r.strip() not in keys})
