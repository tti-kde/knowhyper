"""Cross-modal masked-node reconstruction objective for Lever B enriched graphs.

Masks WHOLE attribute nodes (keeping their HE membership and role) and reconstructs
their RAW modality vector (the leading modality-dim slice of the 768d node_feats row)
from the encoder's context. Because the masked node's target modality is absent from
the encoder input, this cannot be identity-solved — the failure mode that made prior
plain-MAE / SS-HT pre-training trivial on shared-embedding sub-HGs.
"""

from __future__ import annotations

import torch
from torch import nn

from knowhyper.encoders.allset import AllSetHyperEncoder, Incidence
from knowhyper.pretrain.modality import MODALITY_DIMS, node_attribute_modalities


class CrossModalMAE(nn.Module):
    """Mask whole attribute nodes; reconstruct their raw modality vectors."""

    def __init__(self, encoder: AllSetHyperEncoder, *, mask_ratio: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < mask_ratio <= 1.0:
            msg = "mask_ratio must be in (0, 1]"
            raise ValueError(msg)
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        in_dim = encoder.proj.in_features
        self.mask_token = nn.Parameter(torch.zeros(in_dim))
        self.recon_heads = nn.ModuleDict(
            {m: nn.Linear(encoder.hidden_size, d) for m, d in MODALITY_DIMS.items()}
        )

    def _maskable(self, inc: Incidence) -> list[tuple[int, str]]:
        """Return [(node_index, modality)] for present attribute nodes."""
        missing = inc.node_missing.tolist()
        return [
            (n, mod)
            for n, mod in node_attribute_modalities(inc).items()
            if not missing[n]
        ]

    def _pick(
        self, maskable: list[tuple[int, str]], generator: torch.Generator | None
    ) -> list[tuple[int, str]]:
        """Pick a fraction to mask; always keep ≥1 attribute node unmasked."""
        n = len(maskable)
        if n < 2:
            return []
        k = max(1, round(n * self.mask_ratio))
        k = min(k, n - 1)  # keep ≥1 unmasked cue
        perm = torch.randperm(n, generator=generator).tolist()
        return [maskable[j] for j in perm[:k]]

    def forward(
        self,
        incidences: list[Incidence],
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor | dict[str, float]]:
        dev = self.encoder.proj.weight.device
        masked_incs: list[Incidence] = []
        picks: list[list[tuple[int, str]]] = []
        targets: list[dict[int, torch.Tensor]] = []
        for inc in incidences:
            chosen = self._pick(self._maskable(inc), generator)
            feats = inc.node_feats.to(dev).clone()
            tgt: dict[int, torch.Tensor] = {}
            for i, mod in chosen:
                d = MODALITY_DIMS[mod]
                tgt[i] = feats[i, :d].detach().clone()
                feats[i] = self.mask_token
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
                    node_is_seed=(
                        inc.node_is_seed.to(dev)
                        if inc.node_is_seed is not None
                        else None
                    ),
                )
            )
            picks.append(chosen)
            targets.append(tgt)

        out = self.encoder.encode_batch(masked_incs)
        node_h = out["node_embeds"]  # (B, n_max, H)
        total = node_h.new_zeros(())
        n_masked = 0
        per_mod_sum: dict[str, float] = dict.fromkeys(MODALITY_DIMS, 0.0)
        per_mod_cnt: dict[str, int] = dict.fromkeys(MODALITY_DIMS, 0)
        for g, chosen in enumerate(picks):
            for i, mod in chosen:
                pred = self.recon_heads[mod](node_h[g, i])
                target = targets[g][i].to(dev)
                cos = torch.nn.functional.cosine_similarity(
                    pred.unsqueeze(0), target.unsqueeze(0), dim=-1
                )[0]
                loss_i = 1.0 - cos
                total = total + loss_i
                n_masked += 1
                per_mod_sum[mod] += float(loss_i.item())
                per_mod_cnt[mod] += 1
        loss = total / max(n_masked, 1)
        per_modality = {
            m: per_mod_sum[m] / cnt for m, cnt in per_mod_cnt.items() if cnt
        }
        return {
            "loss": loss,
            "n_masked": torch.tensor(n_masked),
            "per_modality": per_modality,
        }
