"""SS-HT self-supervised objective for the AllSet encoder.

Ports the objective math from iMoonLab/SS-HT (TPAMI 2025, Apache-2.0): masked-node
reconstruction with re-mask + replace-noise + squared-cosine (SCE) loss, plus a
hyperedge-level contrastive term. Reconstruction target is the RAW node features
(not a co-trained projection) — the design that defeats the trivial identity solution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as f
from torch import nn

from knowhyper.encoders.allset import AllSetHyperEncoder, Incidence


def sce_loss(
    pred: torch.Tensor, target: torch.Tensor, alpha: float = 3.0
) -> torch.Tensor:
    """Squared-cosine embedding loss: mean (1 - cos)^alpha. Ported from SS-HT losses.SCELoss."""
    p = f.normalize(pred, p=2, dim=-1)
    t = f.normalize(target, p=2, dim=-1)
    return (1 - (p * t).sum(dim=-1)).pow(alpha).mean()


def ssht_mask(
    node_feats: torch.Tensor,
    mask_token: torch.Tensor,
    mask_rate: float,
    replace_rate: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (masked_feats, mask_idx).

    SS-HT masking: mask_token on most masked rows, replace_rate fraction get
    another random node's feats. Keeps >=1 node unmasked; <2-node inputs get
    no masked positions. Ported from SS-HT modules.encoding_mask_noise.
    """
    n = node_feats.shape[0]
    out = node_feats.clone()
    if n < 2:
        return out, node_feats.new_zeros(0, dtype=torch.long)
    k = max(1, round(mask_rate * n))
    k = min(k, n - 1)
    perm = torch.randperm(n, generator=generator).to(node_feats.device)
    mask_idx = perm[:k]
    n_noise = int(replace_rate * k)
    if n_noise > 0:
        pm = torch.randperm(k, generator=generator).to(node_feats.device)
        token_rows = mask_idx[pm[: k - n_noise]]
        noise_rows = mask_idx[pm[k - n_noise :]]
        src = torch.randperm(n, generator=generator).to(node_feats.device)[:n_noise]
        out[token_rows] = mask_token
        out[noise_rows] = node_feats[src]
    else:
        out[mask_idx] = mask_token
    return out, mask_idx


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.5) -> torch.Tensor:
    """Symmetric InfoNCE over matched rows of two views.

    Ported from iMoonLab/SS-HT losses.semi_loss (use_sim=False) + contrastive_loss:
    row i of z1/z2 is a positive pair, all other rows (in both views) are negatives.
    """

    def _dir(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = f.normalize(a, dim=-1)
        b = f.normalize(b, dim=-1)
        refl = torch.exp((a @ a.t()) / temp)
        betw = torch.exp((a @ b.t()) / temp)
        return -torch.log(betw.diag() / (refl.sum(1) + betw.sum(1) - refl.diag()))

    return ((_dir(z1, z2) + _dir(z2, z1)) * 0.5).mean()


class SSHTObjective(nn.Module):
    """SS-HT masked-reconstruction objective over AllSet sub-hypergraphs (attr term).

    encoder -> enc_to_dec -> re-mask (zero masked reps) -> decoder -> SCE(recon, RAW feats)
    at masked positions. Reconstructs RAW node features (anti-identity-leak); ported from
    SS-HT modules.mask_attr_prediction with a second AllSet instance as the decoder.
    """

    def __init__(
        self,
        encoder: AllSetHyperEncoder,
        *,
        mask_rate: float = 0.7,
        replace_rate: float = 0.05,
        sce_alpha: float = 3.0,
        cl_weight: float = 0.0,
        attr_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        h = encoder.hidden_size
        in_dim = encoder.proj.in_features
        self.decoder = AllSetHyperEncoder(
            feature_store=None,
            feature_dim=h,
            hidden_size=h,
            num_layers=len(encoder.layers),
            num_heads=encoder.num_heads,
        )
        self.enc_to_dec = nn.Linear(h, h, bias=False)
        self.recon_out = nn.Linear(h, in_dim)
        self.mask_token = nn.Parameter(torch.zeros(in_dim))
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.sce_alpha = sce_alpha
        self.cl_weight = cl_weight
        self.attr_weight = attr_weight
        self.edge_proj = nn.Sequential(nn.Linear(h, h), nn.ELU(), nn.Linear(h, h))
        # Testability seam: forced True in production. Lets tests disable the re-mask
        # to demonstrate it is load-bearing for the anti-identity-leak property.
        self._remask = True

    def forward(
        self,
        incidences: list[Incidence],
        *,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        dev = self.encoder.proj.weight.device
        masked_incs, masks, targets = [], [], []
        for inc in incidences:
            feats = inc.node_feats.to(dev)
            mfeats, midx = ssht_mask(
                feats, self.mask_token, self.mask_rate, self.replace_rate, generator
            )
            # encode_batch ignores multimodal fields, so dropping them here is inert;
            # future multimodal-sub-HG pretraining must forward them.
            masked_incs.append(
                Incidence(
                    node_feats=mfeats,
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
            masks.append(midx.to(dev))
            targets.append(feats)
        enc_out = self.encoder.encode_batch(masked_incs)["node_embeds"]  # (B, n, H)

        # build decoder incidences from re-masked (zeroed) hidden reps
        dec_incs = []
        for g, inc in enumerate(masked_incs):
            n = inc.num_nodes
            rep = self.enc_to_dec(enc_out[g, :n])
            if self._remask and masks[g].numel() > 0:
                rep = rep.clone()
                rep[masks[g]] = 0.0  # re-mask: remove masked-node info before decoding
            dec_incs.append(
                Incidence(
                    node_feats=rep,
                    node_missing=inc.node_missing,
                    num_nodes=inc.num_nodes,
                    num_edges=inc.num_edges,
                    node_idx=inc.node_idx,
                    edge_idx=inc.edge_idx,
                    role_id=inc.role_id,
                    sem_id=inc.sem_id,
                    pol_id=inc.pol_id,
                )
            )
        dec_out = self.decoder.encode_batch(dec_incs)["node_embeds"]  # (B, n, H)

        total = enc_out.new_zeros(())
        n_masked = 0
        for g in range(len(incidences)):
            midx = masks[g]
            if midx.numel() == 0:
                continue
            n = masked_incs[g].num_nodes
            recon = self.recon_out(dec_out[g, :n][midx])
            total = (
                total + sce_loss(recon, targets[g][midx], self.sce_alpha) * midx.numel()
            )
            n_masked += int(midx.numel())
        recon = total / max(n_masked, 1)
        if self.cl_weight > 0:
            cl = self.contrastive(incidences, generator=generator)
            total_loss = self.attr_weight * recon + self.cl_weight * cl
            return {
                "loss": total_loss,
                "loss_attr": recon,
                "loss_cl": cl,
                "n_masked": torch.tensor(n_masked),
            }
        total_loss = self.attr_weight * recon
        return {
            "loss": total_loss,
            "loss_attr": recon,
            "n_masked": torch.tensor(n_masked),
        }

    def _drop_view(
        self, inc: Incidence, p: float, generator: torch.Generator
    ) -> Incidence:
        """Feature-dropout augmentation: zero a shared subset of feature columns.

        SS-HT drop_features (iMoonLab/SS-HT modules.drop_features): sample a
        (feature_dim,) mask once per view and zero the SAME columns for all nodes,
        matching upstream's ``drop_mask = torch.empty((x.size(1),)).uniform_(0,1) < p;
        x[:, drop_mask] = 0``. Applied to raw node feats before encoding, mirrors
        upstream's ``x1 = drop_features(x, 0.4)`` in hyperedge_structure_loss. All
        tensors are moved to the encoder's device consistently to avoid silent
        multi-device Incidence.
        """
        dev = self.encoder.proj.weight.device
        feats = inc.node_feats.to(dev)
        # SS-HT drop_features: zero a SHARED subset of feature COLUMNS across all nodes.
        col_keep = (
            (torch.rand(feats.shape[1], generator=generator) >= p).float().to(dev)
        )
        # encode_batch ignores multimodal fields, so dropping them here is inert;
        # future multimodal-sub-HG pretraining must forward them.
        return Incidence(
            node_feats=feats * col_keep,
            node_missing=inc.node_missing.to(dev),
            num_nodes=inc.num_nodes,
            num_edges=inc.num_edges,
            node_idx=inc.node_idx.to(dev),
            edge_idx=inc.edge_idx.to(dev),
            role_id=inc.role_id.to(dev),
            sem_id=inc.sem_id.to(dev),
            pol_id=inc.pol_id.to(dev),
        )

    def contrastive(
        self,
        incidences: list[Incidence],
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Hyperedge-level contrastive term (attr term for the hyperedge structure).

        Two independent feature-dropout views -> encode each -> take the encoder's
        ``hyperedge_embeds`` (AllSet's node->hyperedge aggregation stands in for
        upstream's ``hg.v2e``, which KnowHyper's encoder has no equivalent of) ->
        project with ``edge_proj`` -> symmetric InfoNCE over the real (unpadded)
        edges. Ported from SS-HT modules.hyperedge_structure_loss.
        """
        v1 = [self._drop_view(i, 0.4, generator) for i in incidences]
        v2 = [self._drop_view(i, 0.4, generator) for i in incidences]
        o1 = self.encoder.encode_batch(v1)
        o2 = self.encoder.encode_batch(v2)
        e1, m1 = o1["hyperedge_embeds"], o1["hyperedge_mask"]
        e2 = o2["hyperedge_embeds"]
        # flatten real (unpadded) edges across the batch
        sel = m1.bool().reshape(-1)
        z1 = self.edge_proj(e1.reshape(-1, e1.shape[-1])[sel])
        z2 = self.edge_proj(e2.reshape(-1, e2.shape[-1])[sel])
        if z1.shape[0] < 2:
            return e1.new_zeros(())
        return info_nce(z1, z2, temp=0.5)
