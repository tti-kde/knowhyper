"""JSON serialization for ``he_id -> row`` sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from primehkg.retrieval.index_io import require_matching_fingerprint

if TYPE_CHECKING:
    from collections.abc import Mapping

HE_ROW_INDEX_FILE_VERSION = "1"


def save_he_row_index_json(
    path: str | Path,
    he_row_index: Mapping[str, int],
    *,
    inputs_fingerprint: str | None = None,
) -> None:
    """Write ``he_id -> row index`` plus optional ``inputs_fingerprint`` to JSON."""
    p = Path(path)
    payload: dict[str, Any] = {
        "file_version": HE_ROW_INDEX_FILE_VERSION,
        "inputs_fingerprint": inputs_fingerprint,
        "he_id_to_row": dict(he_row_index),
    }
    p.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def load_he_row_index_json(path: str | Path) -> tuple[dict[str, int], str | None]:
    """Load he-row index JSON; returns ``(he_id_to_row, inputs_fingerprint)``."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "he_row index JSON must be an object"
        raise TypeError(msg)
    if raw.get("file_version") != HE_ROW_INDEX_FILE_VERSION:
        msg = f"unsupported he_row index file_version: {raw.get('file_version')!r}"
        raise ValueError(msg)
    mapping = raw.get("he_id_to_row")
    if not isinstance(mapping, dict):
        msg = "he_id_to_row must be a JSON object"
        raise TypeError(msg)
    idx: dict[str, int] = {}
    for k, v in mapping.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, int) and not isinstance(v, bool):
            idx[k] = v
        elif isinstance(v, float) and v.is_integer():
            idx[k] = int(v)
    fp_obj = raw.get("inputs_fingerprint")
    fp = fp_obj if isinstance(fp_obj, str) else None
    return idx, fp


def load_he_row_index_json_validated(
    path: str | Path,
    *,
    expected_fingerprint: str | None,
    reproducibility: Literal["strict", "fast"],
) -> dict[str, int]:
    """Load he-row index JSON and enforce fingerprint rules for ``strict``."""
    idx, fp = load_he_row_index_json(path)
    require_matching_fingerprint(
        loaded_fingerprint=fp,
        expected_fingerprint=expected_fingerprint,
        reproducibility=reproducibility,
    )
    return idx
