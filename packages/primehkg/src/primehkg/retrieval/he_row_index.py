from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from datasets import Dataset


def build_he_row_index(
    hyperedges: Iterable[Mapping[str, object]],
) -> dict[str, int]:
    """Build ``he_id -> dataset row index`` with first-wins semantics."""
    idx: dict[str, int] = {}
    for i, row in enumerate(hyperedges):
        hid = row.get("he_id")
        if isinstance(hid, str) and hid not in idx:
            idx[hid] = i
    return idx


def build_he_row_index_from_dataset(hyperedges: Dataset) -> dict[str, int]:
    """Stream-build ``he_id -> row`` from a HF ``Dataset`` without list materialize."""
    return build_he_row_index(hyperedges)
