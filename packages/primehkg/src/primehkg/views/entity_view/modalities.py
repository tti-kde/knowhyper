from __future__ import annotations

import json
from typing import Any

# Modalities sourced from nodes.feature_refs (name and synonyms are sourced from inline columns).
_FEATURE_REF_MODALITIES: tuple[str, ...] = ("sequence", "smiles", "text")

# v1 dense-embeddable modalities: inline ``name``/``synonyms`` plus feature_refs modalities.
# property/embedding/others are excluded. Derived from _FEATURE_REF_MODALITIES so the
# two cannot drift.
DENSE_MODALITIES: tuple[str, ...] = (
    "name",
    "synonyms",
    *_FEATURE_REF_MODALITIES,
)


def embedding_source_modality(modality: str, ref: dict[str, Any]) -> str:
    """Return the ``source_modality`` a feature ref maps to in ``node_features_embedding``.

    Mirrors how ``features/embedding_inputs.py`` assigns ``source_modality``:
    - a ref carrying ``sequence_type`` (sequence -> protein/nucleotide, smiles -> smiles)
      uses that ``sequence_type`` (embedding_inputs.py:326);
    - the inline ``name`` modality is embedded as ``display_name`` (embedding_inputs.py:74);
    - text refs use the ref's own inner ``modality`` (definition/description/…)
      (embedding_inputs.py:279).
    """
    seq_type = ref.get("sequence_type")
    if isinstance(seq_type, str) and seq_type:
        return seq_type
    if modality == "name":
        return "display_name"
    if modality == "synonyms":
        return "synonym_context"
    inner = ref.get("modality")
    if isinstance(inner, str) and inner:
        return inner
    return modality


def _non_empty(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _coerce_synonym_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [s for s in value if isinstance(s, str) and s.strip()]
    if isinstance(value, str) and value.strip():
        try:
            obj = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(obj, list):
            return [s for s in obj if isinstance(s, str) and s.strip()]
    return []


def extract_entity_modalities(node: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return {modality: [feature_ref...]} for the dense-embeddable modalities present.

    ``name`` is derived from the inline ``nodes.name`` column; ``synonyms`` from
    ``nodes.synonyms``; ``sequence``/``smiles``/``text`` come from ``nodes.feature_refs``.
    Modalities with no payload are omitted.
    """
    out: dict[str, list[dict[str, Any]]] = {}

    name = _non_empty(node.get("name"))
    if name is not None:
        out["name"] = [{"modality": "name", "source": "nodes.name"}]

    if _coerce_synonym_list(node.get("synonyms")):
        out["synonyms"] = [{"modality": "synonyms", "source": "nodes.synonyms"}]

    refs = node.get("feature_refs")
    if isinstance(refs, str):
        # Persisted cores (disk/Hub) store feature_refs as a JSON string
        # (hf_dataset.normalize_record JSON-encodes nested dicts for Arrow). Decode it so
        # sequence/text/smiles modalities are not silently dropped to name-only.
        try:
            refs = json.loads(refs)
        except json.JSONDecodeError:
            refs = None
    if isinstance(refs, dict):
        for modality in _FEATURE_REF_MODALITIES:
            items = refs.get(modality)
            if not isinstance(items, list):
                continue
            ref_list = [r for r in items if isinstance(r, dict)]
            if ref_list:
                out[modality] = ref_list

    return out
