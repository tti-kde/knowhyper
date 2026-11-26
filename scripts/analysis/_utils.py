"""Shared utilities for analysis scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def extract_example_ids(batch: dict) -> list[str]:
    """Build example IDs from a batch dict.

    Handles both the standard ``example_ids`` key (ChemProt / DDI) and
    the BioRED ``pmids + id1s + id2s`` triple.  Returns an empty list
    when neither is present.
    """
    if "example_ids" in batch:
        return batch["example_ids"]
    if "pmids" in batch and "id1s" in batch and "id2s" in batch:
        return [
            f"{pmid}#{id1}-{id2}"
            for pmid, id1, id2 in zip(batch["pmids"], batch["id1s"], batch["id2s"])
        ]
    return []


def find_newest_fingerprint_dir(cache_root: Path) -> Path | None:
    """Return the most-recently-modified fingerprint directory under *cache_root*.

    A fingerprint directory is any subdirectory that contains a ``manifest.json``.
    Returns ``None`` when no qualifying directory exists.
    """
    fp_dirs = sorted(
        (
            d
            for d in cache_root.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        ),
        key=lambda d: (d / "manifest.json").stat().st_mtime,
        reverse=True,
    )
    return fp_dirs[0] if fp_dirs else None
