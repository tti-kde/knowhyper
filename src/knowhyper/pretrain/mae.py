"""Masked-node reconstruction (hypergraph-MAE) objective for the AllSet encoder.

Masks a fraction of node feature rows with a learned mask token BEFORE the encoder
runs, then reconstructs the original projected features at the masked positions from
the encoder's node outputs. Self-supervised: no labels, no negative sampling. The
encoder is used unchanged (masking happens on its ``Incidence`` inputs), so the
fine-tune path is untouched.
"""

from __future__ import annotations

import torch
from torch import nn

from knowhyper.encoders.allset import AllSetHyperEncoder, Incidence


class MaskedNodeMAE(nn.Module):
    """Wrap an AllSet encoder with a masked-node reconstruction objective."""

    def __init__(self, encoder: AllSetHyperEncoder, mask_ratio: float = 0.15) -> None:
        """Store the encoder, mask token, and reconstruction head.

        ``mask_ratio`` is the fraction of nodes masked per graph; each graph keeps at
        least one unmasked node (graphs with <2 nodes get no masked positions).
        """
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            msg = "mask_ratio must be in (0, 1)"
            raise ValueError(msg)
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        in_dim = encoder.proj.in_features
        self.mask_token = nn.Parameter(torch.zeros(in_dim))
        self.recon_head = nn.Linear(encoder.hidden_size, encoder.hidden_size)

    def _pick_masked(
        self, num_nodes: int, generator: torch.Generator | None
    ) -> torch.Tensor:
        """Return a bool (num_nodes,) mask; empty (all-False) if num_nodes < 2."""
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        if num_nodes < 2:
            return mask
        k = max(1, round(num_nodes * self.mask_ratio))
        k = min(k, num_nodes - 1)  # keep >=1 unmasked
        perm = torch.randperm(num_nodes, generator=generator)
        mask[perm[:k]] = True
        return mask

    def forward(
        self,
        incidences: list[Incidence],
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Mask, encode, reconstruct; return {'loss', 'n_masked'}."""
        dev = self.encoder.proj.weight.device
        masked_incs: list[Incidence] = []
        masks: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for inc in incidences:
            mask = self._pick_masked(inc.num_nodes, generator).to(dev)
            feats = inc.node_feats.to(dev).clone()
            # target = projected ORIGINAL features (stop-grad), computed before masking
            with torch.no_grad():
                tgt = self.encoder.proj(inc.node_feats.to(dev))
            feats[mask] = self.mask_token
            # encode_batch ignores multimodal fields, so dropping them here is inert;
            # future multimodal-sub-HG pretraining must forward them.
            masked_incs.append(
                Incidence(
                    node_feats=feats,
                    node_missing=inc.node_missing.to(dev),
                    num_nodes=inc.num_nodes,
                    num_edges=inc.num_edges,
                    node_idx=inc.node_idx.to(dev),
                    edge_idx=inc.edge_idx.to(dev),
                    role_id=inc.role_id.to(dev),
                    sem_id=inc.sem_id.to(dev),
                    pol_id=inc.pol_id.to(dev),
                )
            )
            masks.append(mask)
            targets.append(tgt)
        out = self.encoder.encode_batch(masked_incs)  # (B, n_max, H)
        node_h = out["node_embeds"]
        total_loss = node_h.new_zeros(())
        n_masked = 0
        for g, mask in enumerate(masks):
            if not bool(mask.any()):
                continue
            n = mask.shape[0]
            pred = self.recon_head(node_h[g, :n][mask])
            tgt = targets[g][mask]
            cos = torch.nn.functional.cosine_similarity(pred, tgt, dim=-1)
            total_loss = total_loss + (1.0 - cos).sum()
            n_masked += int(mask.sum())
        loss = total_loss / max(n_masked, 1)
        return {"loss": loss, "n_masked": torch.tensor(n_masked)}
