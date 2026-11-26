"""Composite feature store routing anchor and attribute node IDs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from knowhyper.features.modality_index import ModalityIndex

PAD_DIM = 768

# CORE entity_view modality name -> FEATURES `source_modality` key.
#
# These two vocabularies are NOT the same and the mismatch is silent: an absent key
# makes `ModalityIndex.lookup` return None and `features()` leaves the row as
# `torch.zeros(768)`, so the attr node is still IN the graph, still attended over,
# and contributes nothing. Measured 2026-07-29 by attention attribution on the NER
# ev_cold checkpoints: `attr:synonyms` received EXACTLY 0.00% of attention mass in
# 801/801 attributed instances while `attr:text` was nonzero in every one.
#
# Verified against all 6 `node_features_embedding` parquet files at features 0.12.14:
#   PRESENT: display_name 137587, description 76707, definition 56274,
#            synonym_context 37817, ...
#   ABSENT : "synonyms", "smiles", "protein", "sequence"
#
# So `synonyms` was a fixable NAME mismatch -- the vectors exist under
# `synonym_context`.
#
# `smiles`/`sequence` are a different problem, tracked upstream as primehkg #470: the
# RAW data is present in 0.12.14 (`node_features_sequence`: smiles 6,950 / protein
# 20,219, identical to the older snapshot) but the EMBEDDING pass was not applied to
# them, so `node_features_embedding` -- the only table this store reads -- has zero
# rows for either. Neither build_manifest lists that table under `outputs`, so it
# comes from a separate pass which covered the text modalities (adding
# synonym_context) and skipped these two.
#
# They are left mapped to their (currently absent) keys rather than deleted, so the
# zero stays VISIBLE to attention attribution instead of being hidden by dropping the
# node. Once #470 lands, these two start carrying signal with no code change here.
ATTR_MODALITY_TO_SOURCE: dict[str, str] = {
    "name": "display_name",
    # The whole point of the NER synonym arm. Was defaulting to "synonyms", which
    # matches nothing.
    "synonyms": "synonym_context",
    "smiles": "smiles",
    "sequence": "protein",
}

# "text" is 1:N in primehkg (one umbrella → multiple source_modalities).
# Try each in order until one is found for the entity.
_TEXT_FALLBACK_CHAIN: tuple[str, ...] = (
    "description",
    "definition",
    "mechanism_of_action",
    "indication",
    "pharmacodynamics",
)


def _parse_attr_node_id(attr_id: str) -> tuple[str, str] | None:
    """Parse ``attr:{entity_node_id}:{modality}`` → (entity_node_id, modality)."""
    if not attr_id.startswith("attr:"):
        return None
    last_colon = attr_id.rfind(":")
    if last_colon <= 5:
        return None
    return attr_id[5:last_colon], attr_id[last_colon + 1 :]


class AttributeNodeFeatureStore:
    """Resolve per-attribute-node embeddings from the modality index."""

    dim = PAD_DIM

    def __init__(self, *, modality_index: ModalityIndex) -> None:
        self._index = modality_index

    def _resolve(self, entity_nid: str, modality: str) -> list[float] | None:
        """Resolve a modality to its vector, using fallback chain for text."""
        if modality == "text":
            for source in _TEXT_FALLBACK_CHAIN:
                vec = self._index.lookup(entity_nid, source)
                if vec is not None:
                    return vec
            return None
        source = ATTR_MODALITY_TO_SOURCE.get(modality, modality)
        return self._index.lookup(entity_nid, source)

    def features(self, node_ids: Sequence[str]) -> dict[str, Any]:
        out = torch.zeros(len(node_ids), PAD_DIM)
        for i, nid in enumerate(node_ids):
            parsed = _parse_attr_node_id(nid)
            if parsed is None:
                continue
            entity_nid, modality = parsed
            vec = self._resolve(entity_nid, modality)
            if vec is not None:
                real_dim = len(vec)
                out[i, :real_dim] = torch.as_tensor(vec, dtype=torch.float32)
        return {"node_embeds": out}

    def missing(self, node_ids: Sequence[str]) -> torch.Tensor:
        flags: list[bool] = []
        for nid in node_ids:
            parsed = _parse_attr_node_id(nid)
            if parsed is None:
                flags.append(True)
                continue
            entity_nid, modality = parsed
            flags.append(self._resolve(entity_nid, modality) is None)
        return torch.tensor(flags, dtype=torch.bool)


class CompositeFeatureStore:
    """Route ``primekg:*`` to anchor store, ``attr:*`` to attribute store."""

    dim = PAD_DIM

    def __init__(
        self,
        *,
        anchor_store: Any,
        attr_store: AttributeNodeFeatureStore,
    ) -> None:
        self._anchor = anchor_store
        self._attr = attr_store
        self._anchor_dim = int(anchor_store.dim)

    def features(self, node_ids: Sequence[str]) -> dict[str, Any]:
        anchor_ids: list[tuple[int, str]] = []
        attr_ids: list[tuple[int, str]] = []
        for i, nid in enumerate(node_ids):
            if nid.startswith("attr:"):
                attr_ids.append((i, nid))
            else:
                anchor_ids.append((i, nid))

        out = torch.zeros(len(node_ids), PAD_DIM)

        if anchor_ids:
            a_nids = [nid for _, nid in anchor_ids]
            a_result = self._anchor.features(a_nids)
            a_embeds = a_result["node_embeds"]
            for j, (i, _) in enumerate(anchor_ids):
                out[i, : self._anchor_dim] = a_embeds[j]

        if attr_ids:
            t_nids = [nid for _, nid in attr_ids]
            t_result = self._attr.features(t_nids)
            t_embeds = t_result["node_embeds"]
            for j, (i, _) in enumerate(attr_ids):
                out[i] = t_embeds[j]

        return {"node_embeds": out}

    def missing(self, node_ids: Sequence[str]) -> torch.Tensor:
        flags = torch.zeros(len(node_ids), dtype=torch.bool)
        anchor_ids: list[tuple[int, str]] = []
        attr_ids: list[tuple[int, str]] = []
        for i, nid in enumerate(node_ids):
            if nid.startswith("attr:"):
                attr_ids.append((i, nid))
            else:
                anchor_ids.append((i, nid))
        if anchor_ids:
            a_miss = self._anchor.missing([nid for _, nid in anchor_ids])
            for j, (i, _) in enumerate(anchor_ids):
                flags[i] = a_miss[j]
        if attr_ids:
            t_miss = self._attr.missing([nid for _, nid in attr_ids])
            for j, (i, _) in enumerate(attr_ids):
                flags[i] = t_miss[j]
        return flags
