"""Policy helpers for ``nodes.feature_refs`` indirection (#204 / #126)."""

from __future__ import annotations

import json
from typing import Any, cast

PRIMEHKG_FEATURES_STORAGE = "primehkg-features"
DEFAULT_JOIN_KEY = "node_id"

REQUIRED_POINTER_KEYS = frozenset(
    {"modality", "source", "source_version", "storage", "join"}
)
FORBIDDEN_POINTER_PAYLOAD_KEYS = frozenset(
    {
        "text",
        "sequence",
        "smiles",
        "payload",
        "content",
        "content_hash",
        "vector",
        "embedding",
    }
)
EMBEDDING_REQUIRED_POINTER_KEYS = frozenset(
    {"modality", "model_id", "source", "source_version", "storage", "join"}
)

FEATURE_REFS_ENRICHMENT_SOURCES: tuple[str, ...] = (
    "uniprot",
    "chembl",
    "mondo",
    "hpo",
    "umls",
)


def _parse_feature_refs(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return cast("dict[str, Any]", parsed)
    return {}


def is_indirection_pointer(ref: object) -> bool:
    """Return True when ``ref`` is a PrimeHKG features indirection pointer."""
    if not isinstance(ref, dict):
        return False
    ref_dict = cast("dict[str, Any]", ref)
    keys = {str(k) for k in ref_dict if isinstance(k, str)}
    if not keys >= REQUIRED_POINTER_KEYS:
        return False
    if keys & FORBIDDEN_POINTER_PAYLOAD_KEYS:
        return False
    storage = ref_dict.get("storage")
    join = ref_dict.get("join")
    return storage == PRIMEHKG_FEATURES_STORAGE and join == DEFAULT_JOIN_KEY


def is_embedding_indirection_pointer(ref: object) -> bool:
    """Return True when ``ref`` is an embedding indirection pointer."""
    if not isinstance(ref, dict):
        return False
    ref_dict = cast("dict[str, Any]", ref)
    keys = {str(k) for k in ref_dict if isinstance(k, str)}
    if not keys >= EMBEDDING_REQUIRED_POINTER_KEYS:
        return False
    if keys & FORBIDDEN_POINTER_PAYLOAD_KEYS:
        return False
    storage = ref_dict.get("storage")
    join = ref_dict.get("join")
    return storage == PRIMEHKG_FEATURES_STORAGE and join == DEFAULT_JOIN_KEY


def validate_nodes_feature_refs_indirection(
    nodes: list[dict[str, Any]],
) -> list[str]:
    """Return human-readable violations (empty list == pass)."""
    violations: list[str] = []
    for row in nodes:
        if not isinstance(row, dict):
            continue
        node_id = row.get("node_id")
        nid = node_id if isinstance(node_id, str) else "<unknown>"
        feature_refs = _parse_feature_refs(row.get("feature_refs"))
        for modality, items in feature_refs.items():
            if not isinstance(modality, str) or not modality:
                violations.append(f"{nid}: invalid feature_refs modality key")
                continue
            if not isinstance(items, list):
                violations.append(f"{nid}: feature_refs[{modality!r}] must be a list")
                continue
            for idx, item in enumerate(items):
                if modality == "embedding":
                    valid = is_embedding_indirection_pointer(item)
                    pointer_kind = "embedding indirection pointer"
                else:
                    valid = is_indirection_pointer(item)
                    pointer_kind = "indirection pointer"
                if not valid:
                    violations.append(
                        f"{nid}: feature_refs[{modality!r}][{idx}] is not an "
                        f"{pointer_kind} (storage={PRIMEHKG_FEATURES_STORAGE!r}, "
                        f"join={DEFAULT_JOIN_KEY!r})"
                    )
    return violations


def feature_refs_enrichment_manifest_builder(
    *,
    enrichment_sources: list[str],
) -> dict[str, object] | None:
    """Record extended enrichment inputs that may populate ``feature_refs``."""
    sources = sorted({s for s in enrichment_sources if s})
    if not sources:
        return None
    return {
        "name": "feature_refs_enrichment",
        "version": 1,
        "params": {"sources": sources, "policy": "indirection_only"},
    }
