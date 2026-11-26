"""JSON serialization for inverted-index sidecars (v2.0)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

INVERTED_INDEX_FILE_VERSION = "1"


def save_inverted_index_json(
    path: str | Path,
    inverted_index: Mapping[str, list[str]],
    *,
    inputs_fingerprint: str | None = None,
) -> None:
    """Write ``node_id -> [he_id, ...]`` plus optional ``inputs_fingerprint`` to JSON."""
    p = Path(path)
    payload: dict[str, Any] = {
        "file_version": INVERTED_INDEX_FILE_VERSION,
        "inputs_fingerprint": inputs_fingerprint,
        "node_to_he_ids": dict(inverted_index),
    }
    p.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_inverted_index_json(
    path: str | Path,
) -> tuple[dict[str, list[str]], str | None]:
    """Load inverted index JSON; returns ``(node_to_he_ids, inputs_fingerprint)``."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "inverted index JSON must be an object"
        raise TypeError(msg)
    if raw.get("file_version") != INVERTED_INDEX_FILE_VERSION:
        msg = f"unsupported inverted index file_version: {raw.get('file_version')!r}"
        raise ValueError(msg)
    n2h = raw.get("node_to_he_ids")
    if not isinstance(n2h, dict):
        msg = "node_to_he_ids must be a JSON object"
        raise TypeError(msg)
    inv: dict[str, list[str]] = {}
    for k, v in n2h.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, list):
            continue
        he_ids = [x for x in v if isinstance(x, str)]
        inv[k] = he_ids
    fp_obj = raw.get("inputs_fingerprint")
    fp = fp_obj if isinstance(fp_obj, str) else None
    return inv, fp


def require_matching_fingerprint(
    *,
    loaded_fingerprint: str | None,
    expected_fingerprint: str | None,
    reproducibility: Literal["strict", "fast"],
) -> None:
    """Fail in ``strict`` mode when an expected fingerprint does not match the index."""
    if expected_fingerprint is None:
        return
    if loaded_fingerprint is None:
        if reproducibility == "strict":
            msg = (
                "strict batch retrieval: inverted index JSON is missing "
                "inputs_fingerprint but expected_fingerprint was set"
            )
            raise ValueError(msg)
        return
    if loaded_fingerprint != expected_fingerprint:
        if reproducibility == "strict":
            msg = (
                "strict batch retrieval: inverted index inputs_fingerprint "
                f"{loaded_fingerprint!r} != expected {expected_fingerprint!r}"
            )
            raise ValueError(msg)
        return
    return


def load_inverted_index_json_validated(
    path: str | Path,
    *,
    expected_fingerprint: str | None,
    reproducibility: Literal["strict", "fast"],
) -> dict[str, list[str]]:
    """Load inverted index JSON and enforce fingerprint rules for ``strict``."""
    inv, fp = load_inverted_index_json(path)
    require_matching_fingerprint(
        loaded_fingerprint=fp,
        expected_fingerprint=expected_fingerprint,
        reproducibility=reproducibility,
    )
    return inv
