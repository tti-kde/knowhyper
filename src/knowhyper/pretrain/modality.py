"""Modality dims and role↔modality mapping for cross-modal pre-training."""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowhyper.encoders.vocab import ROLE_VOCAB, Vocab

if TYPE_CHECKING:
    from knowhyper.encoders.allset import Incidence

MODALITY_DIMS: dict[str, int] = {
    "name": 16,
    "sequence": 320,
    "smiles": 384,
    "text": 768,
}

ROLE_TO_MODALITY: dict[str, str] = {f"attribute:{m}": m for m in MODALITY_DIMS}


def attribute_role_ids() -> dict[int, str]:
    """Map each ``attribute:*`` role's vocab index → modality name."""
    v = Vocab(list(ROLE_VOCAB))
    return {v.index(role): modality for role, modality in ROLE_TO_MODALITY.items()}


def node_attribute_modalities(inc: Incidence) -> dict[int, str]:
    """Map node index → modality for nodes that are attribute nodes.

    ``inc.role_id`` is per-membership (length M, parallel to ``inc.node_idx``), so a
    node's attribute role is found by scanning memberships, not by indexing role_id
    with node position. A node is an attribute node iff one of its memberships carries
    an ``attribute:*`` role; its modality is that role's modality. Attribute nodes are
    private to one entity_attribute hyperedge, so the mapping is unambiguous.
    """
    role_to_mod = attribute_role_ids()
    node_ids = inc.node_idx.tolist()
    role_ids = inc.role_id.tolist()
    out: dict[int, str] = {}
    for n, r in zip(node_ids, role_ids, strict=True):
        mod = role_to_mod.get(int(r))
        if mod is not None:
            out[int(n)] = mod
    return out
