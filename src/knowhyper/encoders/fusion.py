"""Collapse a node's per-modality embeddings into one fused vector.

Modalities have DIFFERENT real dims (text 768 / smiles 384 / protein 320), so each
modality gets its own projection ``W_m: real_dim_m -> out_dim``. The transport tensor
is zero-padded to ``PAD_DIM`` for rectangular batching; ``W_m`` reads only the real
``dim_m`` columns, so the pad is inert. ``slot_modality`` (long; -1 for unused slots)
selects which ``W_m`` applies per slot.

``ModalityFusion`` is the swap point. ``PerModalityAttentionFusion`` does
per-modality projection then attention pooling; with ``num_relations > 0`` it
applies NATIVE-style ReDaF — a relation-dependent temperature
``sigmoid(rel_gate(relation_hint))`` on the attention logits. All implementations
zero all-missing rows so the encoder's node-missing mask still applies and no NaN
reaches the graph.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import init


def _modality_seed(name: str) -> int:
    """Return a stable integer seed derived from the modality name.

    Uses SHA-256 so the result is identical across processes and Python runs
    (unlike ``hash()`` which is salted per-process with PYTHONHASHSEED).
    """
    h = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**63)


def _reinit_linear(lin: nn.Linear, g: torch.Generator) -> None:
    """Re-initialize a Linear layer from a seeded generator.

    Replicates PyTorch's default init (kaiming_uniform_ weight, uniform_ bias)
    but draws exclusively from ``g``, leaving the global RNG untouched.

    For a 2-D weight matrix of shape (out_features, in_features) the fan_in is
    ``weight.size(1)`` — identical to what ``nn.init._calculate_fan_in_and_fan_out``
    returns for a plain Linear, but without accessing a private symbol.
    """
    init.kaiming_uniform_(lin.weight, a=math.sqrt(5), generator=g)
    if lin.bias is not None:
        fan_in = lin.weight.size(1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(lin.bias, -bound, bound, generator=g)


class ModalityFusion(Protocol):
    """Fuse ``(N, M, PAD_DIM)`` embeds + ``(N, M)`` mask + ``(N, M)`` slot ids to ``(N, out_dim)``."""

    def __call__(
        self,
        modality_embeds: Tensor,
        modality_mask: Tensor,
        slot_modality: Tensor,
        relation_hint: Tensor | None = None,
    ) -> Tensor:
        """Fuse modality embeddings using slot modality indicators.

        Args:
            modality_embeds: Shape (N, M, PAD_DIM).
            modality_mask: Shape (N, M).
            slot_modality: Shape (N, M) with long ids.
            relation_hint: Optional relation context.

        Returns:
            Shape (N, out_dim).
        """
        ...


class _PerModalityBase(nn.Module):
    """Shared per-modality projection: ``W_m: real_dim_m -> out_dim`` for each modality."""

    def __init__(
        self, modality_dims: dict[str, int], out_dim: int, proj_depth: int = 1
    ) -> None:
        """Build a per-modality projection (1-layer Linear or 2-layer MLP).

        Each modality's parameters are initialized from a deterministic per-modality
        generator seeded by a stable hash of the modality NAME (not slot index), so
        adding a new modality to the set cannot shift the initialization of existing
        modalities.  The global torch RNG is saved and restored around the entire
        projection-construction block so subclass parameters drawn after ``super().__init__``
        (e.g. ``self.query``) are also unaffected by modality count.

        Also builds a stable slot-id -> modality vocab.
        """
        super().__init__()
        if proj_depth not in (1, 2):
            msg = f"proj_depth must be 1 or 2, got {proj_depth}"
            raise ValueError(msg)
        self.vocab = sorted(modality_dims)  # slot-id == index into this list
        self.out_dim = out_dim

        def _make(real_dim: int) -> nn.Module:
            if proj_depth == 1:
                return nn.Linear(real_dim, out_dim)
            return nn.Sequential(
                nn.Linear(real_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
            )

        # Save the global RNG state; projection construction must not consume any
        # global draws so that parameters built by subclasses after super().__init__()
        # land at the same position in the global RNG regardless of modality count.
        rng_state = torch.get_rng_state()
        try:
            self.proj = nn.ModuleDict({m: _make(modality_dims[m]) for m in self.vocab})
            # Re-initialize each modality's parameters from a per-name seeded generator.
            for m, module in self.proj.items():
                g = torch.Generator().manual_seed(_modality_seed(m))
                if isinstance(module, nn.Linear):
                    _reinit_linear(module, g)
                else:
                    # proj_depth == 2: Sequential(Linear, ReLU, Linear)
                    assert isinstance(module, nn.Sequential)
                    first_lin = module[0]
                    second_lin = module[2]
                    assert isinstance(first_lin, nn.Linear)
                    assert isinstance(second_lin, nn.Linear)
                    _reinit_linear(first_lin, g)
                    _reinit_linear(second_lin, g)
        finally:
            # Restore global RNG — from the caller's perspective, building self.proj
            # consumed zero draws from the global generator.
            torch.set_rng_state(rng_state)

    def _project(self, modality_embeds: Tensor, slot_modality: Tensor) -> Tensor:
        """Project each slot with its modality's ``W_m``; returns ``(N, M, out_dim)``."""
        n, m, _ = modality_embeds.shape
        h = modality_embeds.new_zeros(n, m, self.out_dim)
        for sid, modality in enumerate(self.vocab):
            sel = slot_modality == sid  # (N, M) bool
            if not bool(sel.any()):
                continue
            layer = self.proj[modality]
            if isinstance(layer, nn.Linear):
                first: nn.Linear = layer
            else:
                assert isinstance(layer, nn.Sequential)
                first = layer[0]
            dim_m = first.in_features
            # apply W_m to the real dims of the selected slots. Cast to h.dtype so
            # the index-put matches: under autocast (bf16-mixed) the Linear emits
            # bf16 while h follows modality_embeds (float32 from the cache), so the
            # assignment would otherwise raise a dtype mismatch.
            proj_out = self.proj[modality](modality_embeds[sel][:, :dim_m])
            h[sel] = proj_out.to(h.dtype)
        return h


class PerModalityAttentionFusion(_PerModalityBase):
    """Per-modality projection + masked attention pooling over slots.

    With ``num_relations > 0`` the attention logits are divided by a relation-
    dependent temperature ``sigmoid(rel_gate(relation_hint))`` (NATIVE-style ReDaF),
    so the modality-attention sharpness is conditioned on the (proxy) relation.
    ``num_relations == 0`` (default) has no gate and is byte-identical to static.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        out_dim: int,
        num_relations: int = 0,
        proj_depth: int = 1,
    ) -> None:
        """Add a learnable query; optionally a relation-gate Embedding for ReDaF."""
        super().__init__(modality_dims, out_dim, proj_depth=proj_depth)
        self.query = nn.Parameter(torch.randn(out_dim) * 0.02)
        self.rel_gate = nn.Embedding(num_relations, 1) if num_relations > 0 else None

    def forward(
        self,
        modality_embeds: Tensor,
        modality_mask: Tensor,
        slot_modality: Tensor,
        relation_hint: Tensor | None = None,
    ) -> Tensor:
        """Return ``(N, out_dim)``; all-missing rows are zero (no NaN)."""
        h = self._project(modality_embeds, slot_modality)  # (N, M, out_dim)
        scores = torch.einsum("nmd,d->nm", h, self.query)  # (N, M)
        if self.rel_gate is not None and relation_hint is not None:
            # ReDaF: divide logits by sigmoid(rg) — a per-relation learned temperature.
            temp = torch.sigmoid(self.rel_gate(relation_hint))  # (N, 1)
            scores = scores / temp
        scores = scores.masked_fill(modality_mask == 0, torch.finfo(scores.dtype).min)
        any_valid = modality_mask.any(dim=1, keepdim=True)  # (N, 1)
        scores = torch.where(any_valid, scores, torch.zeros_like(scores))
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # (N, M, 1)
        return (alpha * h).sum(dim=1) * any_valid.to(h.dtype)

    def project_only(
        self,
        modality_embeds: Tensor,
        modality_mask: Tensor,
        slot_modality: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return present-slot projections WITHOUT collapsing over slots.

        Unlike ``forward`` (which attention-pools the M slots into one vector per
        node), this returns one vector per PRESENT modality slot, so each seed
        attribute becomes its own cross-attention token (Phase A). Returns
        ``(P, out_dim)`` projected vectors and ``(P,)`` long node indices, where
        ``P`` is the number of present slots. All-missing input returns empty.
        """
        h = self._project(modality_embeds, slot_modality)  # (N, M, out_dim)
        present = modality_mask.bool()  # (N, M)
        node_rows = (
            torch.arange(h.shape[0], device=h.device)
            .unsqueeze(1)
            .expand(-1, h.shape[1])
        )  # (N, M)
        vecs = h[present]  # (P, out_dim)
        node_idx = node_rows[present]  # (P,)
        return vecs, node_idx


class PerModalityMeanFusion(_PerModalityBase):
    """Per-modality projection + masked mean over slots (ablation)."""

    def forward(
        self,
        modality_embeds: Tensor,
        modality_mask: Tensor,
        slot_modality: Tensor,
        relation_hint: Tensor | None = None,
    ) -> Tensor:
        """Return ``(N, out_dim)``; all-missing rows are zero."""
        del relation_hint
        h = self._project(modality_embeds, slot_modality)
        m = modality_mask.unsqueeze(-1)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
