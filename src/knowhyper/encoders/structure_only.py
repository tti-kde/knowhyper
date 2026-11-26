"""Lightweight structure-only encoder for the vertical slice and ablations.

Pulls node features from a feature store, projects to the LM hidden size, and
mean-pools to a single graph embedding. No hypergraph message passing — that is
the HNN encoder's job (deferred).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from knowhyper.pipeline.types import KnowledgeRepr, SubHypergraph


class StructureOnlyEncoder(nn.Module):
    """Encode a sub-hypergraph by projecting and mean-pooling node features."""

    def __init__(self, feature_store: Any, hidden_size: int = 768) -> None:
        """Store the feature store and build the projection to hidden_size."""
        super().__init__()
        self.feature_store = feature_store
        self.hidden_size = hidden_size
        self.proj = nn.Linear(feature_store.dim, hidden_size)

    def encode(self, sub: SubHypergraph) -> KnowledgeRepr:
        """Project node features and mean-pool to a single graph embedding."""
        device = self.proj.weight.device
        dtype = self.proj.weight.dtype
        if not sub.node_ids:
            return KnowledgeRepr(
                node_embeds=torch.zeros(
                    0, self.hidden_size, device=device, dtype=dtype
                ),
                node_mask=torch.zeros(0, device=device),
                graph_embed=torch.zeros(self.hidden_size, device=device, dtype=dtype),
            )
        feats = self.feature_store.features(sub.node_ids)["node_embeds"].to(device)
        node_embeds = self.proj(feats)
        graph_embed = node_embeds.mean(dim=0)
        node_mask = torch.ones(node_embeds.shape[0], device=device)
        return KnowledgeRepr(
            node_embeds=node_embeds, node_mask=node_mask, graph_embed=graph_embed
        )
