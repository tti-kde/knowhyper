"""Integrators: decide WHAT of a KnowledgeRepr to hand to the ieforge model.

The actual fusion op (prepend vs cross-attention) lives in the ieforge model
wrapper, not here. SoftPromptIntegrator emits a prompt tensor; FeatureIntegrator
passes raw tensors through for a cross-attention model; TextIntegrator is a
deferred stub for text-serialization baselines.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from knowhyper.pipeline.types import IntegrationPayload, KnowledgeRepr


class SoftPromptIntegrator(nn.Module):
    """Map a pooled graph embedding to K soft-prompt tokens at the LM hidden size.

    Note: this is an ``nn.Module`` whose projection produces grad-requiring
    tensors. When used inside the ieforge ``KnowledgeCollator``, the prompt
    tensors are produced during collation, so the DataLoader must run with
    ``num_workers=0`` (grad-bearing, non-leaf tensors cannot be pickled to
    worker processes). A future design may move the projection into the model
    ``forward`` so the collator emits only leaf/detached features.
    """

    def __init__(self, prompt_length: int = 8, hidden_size: int = 768) -> None:
        """Build the prompt generator for ``prompt_length`` tokens of ``hidden_size``."""
        super().__init__()
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, prompt_length * hidden_size)

    def to_payload(self, repr_: KnowledgeRepr) -> IntegrationPayload:
        """Project a 1-D pooled graph_embed into prompt_length soft-prompt tokens."""
        if repr_.graph_embed is None:
            msg = "SoftPromptIntegrator requires KnowledgeRepr.graph_embed"
            raise ValueError(msg)
        if repr_.graph_embed.dim() != 1:
            msg = (
                "SoftPromptIntegrator expects a 1-D pooled graph_embed of shape "
                f"(hidden_size,), got shape {tuple(repr_.graph_embed.shape)}"
            )
            raise ValueError(msg)
        flat = self.proj(repr_.graph_embed)
        prompt = flat.reshape(self.prompt_length, self.hidden_size)
        mask = torch.ones(self.prompt_length, dtype=torch.long, device=prompt.device)
        return IntegrationPayload(prompt_embeds=prompt, prompt_attention_mask=mask)


class FeatureIntegrator:
    """Pass present tensors through under their source keys for a feature model."""

    _FIELDS = (
        "node_embeds",
        "node_mask",
        "hyperedge_embeds",
        "hyperedge_mask",
        "graph_embed",
    )

    def to_payload(self, repr_: KnowledgeRepr) -> IntegrationPayload:
        """Collect the present tensor fields under their source keys."""
        features = {
            field_: getattr(repr_, field_)
            for field_ in self._FIELDS
            if getattr(repr_, field_) is not None
        }
        return IntegrationPayload(graph_features=features)


class TextIntegrator:
    """Deferred: text-serialization integration for baselines."""

    def to_payload(self, repr_: KnowledgeRepr) -> IntegrationPayload:
        """Raise: text-serialization integration is not implemented yet."""
        msg = "TextIntegrator is a deferred stub; not implemented in the first slice"
        raise NotImplementedError(msg)


class IncidenceIntegrator:
    """Pass raw Incidence leaf tensors through to the collator under ``incidence_*`` keys.

    This integrator requires ``KnowledgeRepr.incidence`` to be set (produced by
    Task 1's ``build_incidence``). All output tensors are leaves with no grad, so
    they are safe to pickle across DataLoader worker processes (num_workers > 0).

    The class attribute ``needs_incidence = True`` is read by the Task 4 injector
    to decide whether to call ``build_incidence`` before ``to_payload``.
    """

    needs_incidence: bool = True

    def to_payload(self, repr_: KnowledgeRepr) -> IntegrationPayload:
        """Build an IntegrationPayload whose graph_features hold incidence leaf tensors."""
        if repr_.incidence is None:
            msg = "IncidenceIntegrator requires KnowledgeRepr.incidence"
            raise ValueError(msg)
        inc = repr_.incidence
        graph_features: dict[str, Any] = {
            "incidence_node_feats": inc.node_feats,
            "incidence_node_missing": inc.node_missing,
            "incidence_node_idx": inc.node_idx,
            "incidence_edge_idx": inc.edge_idx,
            "incidence_role_id": inc.role_id,
            "incidence_sem_id": inc.sem_id,
            "incidence_pol_id": inc.pol_id,
            "incidence_num_nodes": torch.tensor([inc.num_nodes]),
            "incidence_num_edges": torch.tensor([inc.num_edges]),
        }
        if inc.node_modality_mask is not None and inc.node_slot_modality is not None:
            m = int(inc.node_modality_mask.shape[1])
            graph_features["incidence_node_modality_mask"] = inc.node_modality_mask
            graph_features["incidence_node_slot_modality"] = inc.node_slot_modality
            graph_features["incidence_modality_m"] = m
            graph_features["incidence_pad_dim"] = int(inc.node_feats.shape[1]) // m
            if inc.node_semantics_id is not None:
                graph_features["incidence_node_semantics"] = inc.node_semantics_id
            if inc.node_is_seed is not None:
                graph_features["incidence_node_is_seed"] = inc.node_is_seed
        return IntegrationPayload(graph_features=graph_features)
