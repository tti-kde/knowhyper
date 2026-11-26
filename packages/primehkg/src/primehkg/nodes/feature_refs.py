from __future__ import annotations

import json
from typing import Any


def _ref_sort_key(ref: dict[str, Any]) -> str:
    return json.dumps(ref, sort_keys=True, separators=(",", ":"))


def merge_feature_refs(
    base: dict[str, Any] | None,
    incoming: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge ``feature_refs`` maps deterministically (per-modality ref lists)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for src in (base, incoming):
        if not isinstance(src, dict):
            continue
        for modality in sorted(src):
            if not isinstance(modality, str) or not modality:
                continue
            items = src[modality]
            if not isinstance(items, list):
                continue
            bucket = out.setdefault(modality, [])
            seen = {_ref_sort_key(r) for r in bucket if isinstance(r, dict)}
            for item in items:
                if not isinstance(item, dict):
                    continue
                ref = {
                    str(k): v
                    for k, v in item.items()
                    if isinstance(k, str) and v is not None
                }
                key = _ref_sort_key(ref)
                if key in seen:
                    continue
                bucket.append(ref)
                seen.add(key)
            bucket.sort(key=_ref_sort_key)
    return dict(out)


def modality_pointer(
    *,
    modality: str,
    source: str,
    source_version: str,
    sequence_type: str | None = None,
    storage: str = "primehkg-features",
    join: str = "node_id",
) -> dict[str, str]:
    """Indirection pointer (payload lives in ``primehkg-features`` or external store)."""
    ref: dict[str, str] = {
        "modality": modality,
        "source": source,
        "source_version": source_version,
        "storage": storage,
        "join": join,
    }
    if sequence_type:
        ref["sequence_type"] = sequence_type
    return ref


def sequence_pointer(
    *, source: str, source_version: str, sequence_type: str
) -> dict[str, str]:
    return modality_pointer(
        modality="sequence",
        source=source,
        source_version=source_version,
        sequence_type=sequence_type,
    )


def smiles_pointer(*, source: str, source_version: str) -> dict[str, str]:
    return modality_pointer(
        modality="smiles",
        source=source,
        source_version=source_version,
        sequence_type="smiles",
    )


def text_pointer(*, modality: str, source: str, source_version: str) -> dict[str, str]:
    """Pointer into ``node_features_text`` (inner ``modality`` = definition/description/…)."""
    return modality_pointer(
        modality=modality,
        source=source,
        source_version=source_version,
    )


def property_pointer(
    *, property_id: str, source: str, source_version: str
) -> dict[str, str]:
    """Pointer into ``node_features_property`` (inner ``modality`` = ``property_id``)."""
    return modality_pointer(
        modality=property_id,
        source=source,
        source_version=source_version,
    )


def ontology_pointer(
    *,
    modality: str,
    source: str,
    source_version: str,
    namespace: str,
    storage: str = "primehkg-features",
    join: str = "node_id",
) -> dict[str, str]:
    """Pointer into ontology-linked features metadata (e.g. GO term namespace)."""
    return {
        "modality": modality,
        "source": source,
        "source_version": source_version,
        "namespace": namespace,
        "storage": storage,
        "join": join,
    }


def embedding_pointer(
    *,
    modality: str,
    model_id: str,
    source: str,
    source_version: str,
    storage: str = "primehkg-features",
    join: str = "node_id",
    config: str = "node_features_embedding",
) -> dict[str, str]:
    """Indirection pointer into ``node_features_embedding`` (maps to ``source_modality``)."""
    return {
        "modality": modality,
        "model_id": model_id,
        "source": source,
        "source_version": source_version,
        "storage": storage,
        "join": join,
        "config": config,
    }
