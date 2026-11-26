from __future__ import annotations

import json
import re
from typing import Any

from primehkg.features.extract import _text_row
from primehkg.nodes.synonyms import is_genuine_synonym

SYNONYM_CONTEXT_K = 16
SYNONYM_CONTEXT_SEP = " | "
_CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")


def is_catalog_like_synonym(surface: str) -> bool:
    s = surface.strip()
    if not s:
        return True
    if _CAS_RE.fullmatch(s):
        return True
    alnum = [c for c in s if c.isalnum()]
    if not alnum:
        return True
    return (sum(c.isdigit() for c in alnum) / len(alnum)) >= 0.5


def rank_synonym_surfaces(
    surfaces: list[str],
    *,
    name: object = None,
    node_id: str = "",
) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in surfaces:
        if not isinstance(raw, str) or not raw.strip():
            continue
        if node_id and not is_genuine_synonym(raw, name=name, node_id=node_id):
            continue
        key = raw.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(raw.strip())
    cleaned.sort(
        key=lambda s: (
            is_catalog_like_synonym(s),
            len(s),
            s.casefold(),
        )
    )
    return cleaned


def format_synonym_context_text(
    surfaces: list[str],
    *,
    name: object = None,
    node_id: str = "",
    k: int = SYNONYM_CONTEXT_K,
) -> str | None:
    ranked = rank_synonym_surfaces(surfaces, name=name, node_id=node_id)
    if not ranked:
        return None
    return SYNONYM_CONTEXT_SEP.join(ranked[:k])


def _coerce_synonyms(value: Any) -> list[str]:
    if isinstance(value, list):
        return [s for s in value if isinstance(s, str)]
    if isinstance(value, str) and value.strip():
        try:
            obj = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(obj, list):
            return [s for s in obj if isinstance(s, str)]
    return []


def read_synonym_context_rows(
    core_nodes: list[dict[str, Any]],
    *,
    core_fingerprint: str,
    allowed_node_ids: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Emit synonym_context text rows from core ``nodes.synonyms``."""
    rows: list[dict[str, Any]] = []
    for node in core_nodes:
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        if allowed_node_ids is not None and node_id not in allowed_node_ids:
            continue
        text = format_synonym_context_text(
            _coerce_synonyms(node.get("synonyms")),
            name=node.get("name"),
            node_id=node_id,
        )
        if text is None:
            continue
        rows.append(
            _text_row(
                node_id=node_id,
                modality="synonym_context",
                text=text,
                source="primehkg-core",
                source_version=core_fingerprint,
                source_field="synonyms",
                payload_class="short_label",
            )
        )
    return rows
