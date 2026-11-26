"""HyperGT sub-hypergraph encoder (ICASSP 2024, zeroxleo/HyperGT).

All-pair node attention with hyperedge positional encoding, ported as a
drop-in replacement for AllSetHyperEncoder.  Produces the same
``encode_batch`` dict format so the fusion model is unaware of which
encoder produced the embeddings.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn.functional as f
from torch import nn
from torch_scatter import scatter

from knowhyper.encoders.allset import AllSetHyperEncoder, Incidence
from knowhyper.encoders.vocab import POLARITY_VOCAB, ROLE_VOCAB, SEMANTICS_VOCAB, Vocab
from knowhyper.pipeline.types import KnowledgeRepr


def _create_projection_matrix(m: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    nb_full = m // d
    blocks: list[torch.Tensor] = []
    for _ in range(nb_full):
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        blocks.append(q.T)
    rem = m - nb_full * d
    if rem > 0:
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        blocks.append(q.T[:rem])
    final = torch.vstack(blocks)
    multiplier = torch.norm(torch.randn(m, d, generator=g), dim=1)
    return torch.diag(multiplier) @ final


def _softmax_kernel(
    data: torch.Tensor,
    is_query: bool,
    proj: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    normalizer = 1.0 / (data.shape[-1] ** 0.25)
    data = data * normalizer
    ratio = 1.0 / math.sqrt(proj.shape[0])
    data_dash = torch.einsum("bnhd,md->bnhm", data, proj)
    diag = data.square().sum(-1, keepdim=True) / 2.0
    # Mask padding before max reductions so they don't affect normalization
    if mask is not None:
        neg_inf = torch.finfo(data_dash.dtype).min
        data_dash = data_dash.masked_fill(~mask[:, :, None, None].bool(), neg_inf)
    if is_query:
        mx = data_dash.max(dim=-1, keepdim=True).values
        result = ratio * (torch.exp(data_dash - diag - mx) + eps)
    else:
        mx1 = data_dash.max(dim=-1, keepdim=True).values
        mx2 = mx1.max(dim=-3, keepdim=True).values
        result = ratio * (torch.exp(data_dash - diag - mx2) + eps)
    # Zero out padding positions so they contribute nothing to attention
    if mask is not None:
        result = result * mask[:, :, None, None]
    return result


def _performer_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    proj: torch.Tensor,
    mask: torch.Tensor | None = None,
    tau: float = 0.25,
) -> torch.Tensor:
    """Compute Performer attention with optional padding mask."""
    q = q / math.sqrt(tau)
    k = k / math.sqrt(tau)
    qp = _softmax_kernel(q, True, proj, mask=mask)  # (B, N, H, M)
    kp = _softmax_kernel(k, False, proj, mask=mask)  # (B, N, H, M)
    if mask is not None:
        v = v * mask[:, :, None, None]
    qp = qp.permute(1, 0, 2, 3)  # (N, B, H, M)
    kp = kp.permute(1, 0, 2, 3)
    vp = v.permute(1, 0, 2, 3)  # (N, B, H, D)
    kvs = torch.einsum("nbhm,nbhd->bhmd", kp, vp)
    num = torch.einsum("nbhm,bhmd->nbhd", qp, kvs)
    ones = torch.ones(kp.shape[0], device=kp.device)
    ks_sum = torch.einsum("nbhm,n->bhm", kp, ones)
    den = torch.einsum("nbhm,bhm->nbh", qp, ks_sum).unsqueeze(-1).clamp(min=1e-6)
    return (num / den).permute(1, 0, 2, 3)  # (B, N, H, D)


def _gumbel_performer_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    proj: torch.Tensor,
    mask: torch.Tensor | None = None,
    k_samples: int = 10,
    tau: float = 0.25,
) -> torch.Tensor:
    q = q / math.sqrt(tau)
    k = k / math.sqrt(tau)
    qp = _softmax_kernel(q, True, proj, mask=mask)
    kp = _softmax_kernel(k, False, proj, mask=mask)
    if mask is not None:
        v = v * mask[:, :, None, None]
    qp = qp.permute(1, 0, 2, 3)
    kp = kp.permute(1, 0, 2, 3)
    vp = v.permute(1, 0, 2, 3)
    gumbels = (
        -torch.empty((*kp.shape[:-1], k_samples), device=kp.device).exponential_().log()
        / tau
    )
    kt_gumbel = kp.unsqueeze(3) * gumbels.exp().unsqueeze(4)  # (N,B,H,K,M)
    kvs = torch.einsum("nbhkm,nbhd->bhkmd", kt_gumbel, vp)
    num = torch.einsum("nbhm,bhkmd->nbhkd", qp, kvs)
    ones = torch.ones(kt_gumbel.shape[0], device=kp.device)
    ks_sum = torch.einsum("nbhkm,n->bhkm", kt_gumbel, ones)
    den = torch.einsum("nbhm,bhkm->nbhk", qp, ks_sum).unsqueeze(-1).clamp(min=1e-6)
    return (num / den).mean(dim=3).permute(1, 0, 2, 3)  # (B,N,H,D)


class HyperGTConv(nn.Module):
    """Single HyperGT layer: all-pair Performer attention + relational bias."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        nb_random_features: int = 30,
        use_gumbel: bool = True,
        nb_gumbel_sample: int = 10,
        rb_order: int = 0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.nb_random_features = nb_random_features
        self.use_gumbel = use_gumbel
        self.nb_gumbel_sample = nb_gumbel_sample
        self.rb_order = rb_order

        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)
        self.Wo = nn.Linear(hidden_size, hidden_size)
        if rb_order >= 1:
            self.b = nn.Parameter(torch.full((rb_order, num_heads), 0.1))
        self.register_buffer(
            "_proj",
            _create_projection_matrix(nb_random_features, self.head_dim, seed=0),
        )

    def forward(
        self,
        z: torch.Tensor,
        adj_edge_index: torch.Tensor | None,
        tau: float = 1.0,
        mask: torch.Tensor | None = None,
        flat_v: torch.Tensor | None = None,
        flat_n_per: list[int] | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            z: (B, N_max, H) padded node features.
            adj_edge_index: (2, E_adj_total) offset-batched clique-expanded
                adjacency, or None if rb_order == 0.
            tau: Gumbel temperature.
            mask: (B, N_max) bool — True for real nodes, False for padding.
            flat_v: (total_N, H_head) flat v for relational bias scatter.
                Only needed when rb_order >= 1.
            flat_n_per: per-graph node counts. Only needed when rb_order >= 1.
        """
        bsz, n_max, _ = z.shape
        q = self.Wq(z).reshape(bsz, n_max, self.num_heads, self.head_dim)
        k = self.Wk(z).reshape(bsz, n_max, self.num_heads, self.head_dim)
        v = self.Wv(z).reshape(bsz, n_max, self.num_heads, self.head_dim)
        # register_buffer makes ty infer Tensor | Module; it is always a Tensor.
        proj = cast("torch.Tensor", self._proj)
        if self.use_gumbel and self.training:
            z_next = _gumbel_performer_attn(
                q,
                k,
                v,
                proj,
                mask=mask,
                k_samples=self.nb_gumbel_sample,
                tau=tau,
            )
        else:
            z_next = _performer_attn(q, k, v, proj, mask=mask, tau=tau)

        if (
            self.rb_order >= 1
            and adj_edge_index is not None
            and flat_v is not None
            and flat_n_per is not None
        ):
            total_n = flat_v.shape[0]
            row, col = adj_edge_index
            from torch_geometric.utils import degree

            d_in = degree(col, total_n).float().clamp(min=1)
            d_out = degree(row, total_n).float().clamp(min=1)
            d_norm_in = (1.0 / d_in[col]).sqrt()
            d_norm_out = (1.0 / d_out[row]).sqrt()
            edge_weight = d_norm_in * d_norm_out

            for i in range(self.rb_order):
                b_i = torch.sigmoid(self.b[i])  # (num_heads,)
                for h in range(self.num_heads):
                    src_vals = flat_v[row, h] * (b_i[h] * edge_weight).unsqueeze(-1)
                    conv_flat = scatter(
                        src_vals, col, dim=0, dim_size=total_n, reduce="sum"
                    )
                    # Scatter flat results back into padded z_next
                    offset = 0
                    for g in range(bsz):
                        nn_ = flat_n_per[g]
                        z_next[g, :nn_, h] += conv_flat[offset : offset + nn_]
                        offset += nn_

        return self.Wo(z_next.flatten(-2, -1))


class HyperGTEncoder(nn.Module):
    """HyperGT encoder — drop-in replacement for AllSetHyperEncoder."""

    def __init__(
        self,
        feature_store: Any = None,
        *,
        feature_dim: int | None = None,
        hidden_size: int = 768,
        num_layers: int = 2,
        num_heads: int = 4,
        role_vocab: tuple[str, ...] = ROLE_VOCAB,
        semantics_vocab: tuple[str, ...] = SEMANTICS_VOCAB,
        polarity_vocab: tuple[str, ...] = POLARITY_VOCAB,
        nb_random_features: int = 30,
        use_gumbel: bool = True,
        nb_gumbel_sample: int = 10,
        rb_order: int = 0,
        tau: float = 1.0,
        dropout: float = 0.0,
        hde_enabled: bool = False,
    ) -> None:
        super().__init__()
        if feature_store is None and feature_dim is None:
            msg = "HyperGTEncoder needs feature_store or feature_dim"
            raise ValueError(msg)
        self.feature_store = feature_store
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tau = tau
        self.dropout = dropout
        self.hde_enabled = hde_enabled
        if feature_store is not None:
            in_dim = int(feature_store.dim)
        else:
            assert feature_dim is not None
            in_dim = int(feature_dim)
        self.proj = nn.Linear(in_dim, hidden_size)
        self._role_vocab = role_vocab
        self._sem_vocab = semantics_vocab
        self._roles = Vocab(list(role_vocab))
        self._sems = Vocab(list(semantics_vocab))
        self._pols = Vocab(list(polarity_vocab))
        self.role_embed = nn.Embedding(len(self._roles), hidden_size)
        self.sem_embed = nn.Embedding(len(self._sems), hidden_size)
        self.pol_embed = nn.Embedding(len(self._pols), hidden_size)
        self.hepe_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.htepe_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.convs = nn.ModuleList(
            HyperGTConv(
                hidden_size,
                num_heads,
                nb_random_features=nb_random_features,
                use_gumbel=use_gumbel,
                nb_gumbel_sample=nb_gumbel_sample,
                rb_order=rb_order,
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_size) for _ in range(num_layers + 1)
        )
        self.pool_query = nn.Parameter(torch.randn(hidden_size))
        self._incidence_builder: AllSetHyperEncoder | None = None
        if feature_store is not None:
            self._incidence_builder = AllSetHyperEncoder(
                feature_store=feature_store,
                hidden_size=hidden_size,
                num_heads=num_heads,
                role_vocab=role_vocab,
                semantics_vocab=semantics_vocab,
                polarity_vocab=polarity_vocab,
            )

    def _build_adj_edge_index_batched(
        self,
        incidences: list[Incidence],
        n_per: list[int],
        device: torch.device,
    ) -> torch.Tensor | None:
        """Build offset-batched clique-expanded adjacency for all graphs."""
        rows_list: list[torch.Tensor] = []
        cols_list: list[torch.Tensor] = []
        node_off = 0
        for g, inc in enumerate(incidences):
            if inc.node_idx.numel() == 0:
                node_off += n_per[g]
                continue
            node_idx = inc.node_idx.to(device)
            edge_idx = inc.edge_idx.to(device)
            for e in range(inc.num_edges):
                members = node_idx[edge_idx == e] + node_off
                if members.numel() < 2:
                    continue
                src = members.repeat_interleave(members.numel())
                dst = members.repeat(members.numel())
                mask = src != dst
                rows_list.append(src[mask])
                cols_list.append(dst[mask])
            node_off += n_per[g]
        if not rows_list:
            return None
        return torch.stack([torch.cat(rows_list), torch.cat(cols_list)])

    def encode_batch(self, incidences: list[Incidence]) -> dict[str, torch.Tensor]:
        """Encode B sub-hypergraphs in one batched pass."""
        dev = self.proj.weight.device
        dt = self.proj.weight.dtype
        b = len(incidences)
        n_per = [inc.num_nodes for inc in incidences]
        e_per = [inc.num_edges for inc in incidences]
        n_max = max(n_per) if n_per else 0
        e_max = max(e_per) if e_per else 0
        if b == 0 or n_max == 0:
            return {
                "node_embeds": torch.zeros(
                    b, max(n_max, 1), self.hidden_size, device=dev, dtype=dt
                ),
                "node_mask": torch.zeros(b, max(n_max, 1), device=dev, dtype=dt),
                "hyperedge_embeds": torch.zeros(
                    b, max(e_max, 1), self.hidden_size, device=dev, dtype=dt
                ),
                "hyperedge_mask": torch.zeros(b, max(e_max, 1), device=dev, dtype=dt),
                "graph_embed": torch.zeros(b, self.hidden_size, device=dev, dtype=dt),
            }

        # --- Phase 1: offset-concat pre-attention features ---
        node_off, edge_off = 0, 0
        node_feats_list, node_missing_list = [], []
        n_idx, e_idx, role_ids, sem_ids, pol_ids = [], [], [], [], []
        graph_of_node: list[torch.Tensor] = []
        for g, inc in enumerate(incidences):
            node_feats_list.append(inc.node_feats.to(dev))
            node_missing_list.append(inc.node_missing.to(dev))
            n_idx.append(inc.node_idx.to(dev) + node_off)
            e_idx.append(inc.edge_idx.to(dev) + edge_off)
            role_ids.append(inc.role_id.to(dev))
            sem_ids.append(inc.sem_id.to(dev))
            pol_ids.append(inc.pol_id.to(dev))
            graph_of_node.append(
                torch.full((inc.num_nodes,), g, dtype=torch.long, device=dev)
            )
            node_off += inc.num_nodes
            edge_off += inc.num_edges
        total_n, total_e = node_off, edge_off

        cat_feats = torch.cat(node_feats_list, dim=0)
        cat_missing = torch.cat(node_missing_list, dim=0)
        cat_n_idx = torch.cat(n_idx, dim=0)
        cat_e_idx = torch.cat(e_idx, dim=0)
        cat_role = torch.cat(role_ids, dim=0)
        cat_sem = torch.cat(sem_ids, dim=0)
        cat_pol = torch.cat(pol_ids, dim=0)
        cat_gid = torch.cat(graph_of_node, dim=0)

        # Project node features
        node_h = self.proj(cat_feats)  # (total_N, H)

        # Role injection
        if cat_n_idx.numel() > 0:
            role_vecs = self.role_embed(cat_role)
            role_per_node = scatter(
                role_vecs, cat_n_idx, dim=0, dim_size=total_n, reduce="mean"
            )
            node_h = node_h + role_per_node

        # Semantics injection
        if cat_n_idx.numel() > 0 and cat_sem.numel() > 0:
            sem_per_member = self.sem_embed(cat_sem[cat_e_idx])
            sem_per_node = scatter(
                sem_per_member, cat_n_idx, dim=0, dim_size=total_n, reduce="mean"
            )
            node_h = node_h + sem_per_node

        # Polarity injection
        if cat_n_idx.numel() > 0 and cat_pol.numel() > 0:
            pol_per_member = self.pol_embed(cat_pol[cat_e_idx])
            pol_per_node = scatter(
                pol_per_member, cat_n_idx, dim=0, dim_size=total_n, reduce="mean"
            )
            node_h = node_h + pol_per_node

        # HEPE
        if cat_n_idx.numel() > 0:
            role_vecs_he = self.role_embed(cat_role)
            hepe_member = self.hepe_proj(role_vecs_he)
            hepe = scatter(
                hepe_member, cat_n_idx, dim=0, dim_size=total_n, reduce="mean"
            )
            node_h = node_h + hepe

        # HtEPE
        if cat_n_idx.numel() > 0 and total_e > 0:
            htepe_member = self.htepe_proj(node_h[cat_n_idx])
            htepe_edge = scatter(
                htepe_member, cat_e_idx, dim=0, dim_size=total_e, reduce="mean"
            )
            htepe_back = htepe_edge[cat_e_idx]
            htepe_node = scatter(
                htepe_back, cat_n_idx, dim=0, dim_size=total_n, reduce="mean"
            )
            node_h = node_h + htepe_node

        node_h = self.norms[0](node_h)
        node_h = f.elu(node_h)
        node_h = f.dropout(node_h, p=self.dropout, training=self.training)

        # --- Phase 2: pad to (B, N_max, H) for Performer attention ---
        z = torch.zeros(b, n_max, self.hidden_size, device=dev, dtype=node_h.dtype)
        attn_mask = torch.zeros(b, n_max, device=dev, dtype=node_h.dtype)
        offset = 0
        for g in range(b):
            nn_ = n_per[g]
            z[g, :nn_] = node_h[offset : offset + nn_]
            attn_mask[g, :nn_] = 1.0
            offset += nn_

        # Build batched adjacency for relational bias
        adj = self._build_adj_edge_index_batched(incidences, n_per, dev)

        head_dim = self.hidden_size // self.num_heads
        for i, conv in enumerate(self.convs):
            conv = cast("HyperGTConv", conv)
            z_prev = z
            # Compute flat v for relational bias (from padded z)
            flat_v = None
            if conv.rb_order >= 1 and adj is not None:
                v_padded = conv.Wv(z).reshape(b, n_max, self.num_heads, head_dim)
                flat_v = torch.cat([v_padded[g, : n_per[g]] for g in range(b)], dim=0)

            z = conv(z, adj, self.tau, mask=attn_mask, flat_v=flat_v, flat_n_per=n_per)
            z = z * attn_mask.unsqueeze(-1)
            z = z + z_prev
            z = self.norms[i + 1](z)
            z = f.dropout(z, p=self.dropout, training=self.training)
            z = z * attn_mask.unsqueeze(-1)

        # --- Phase 3: extract flat node_h, compute outputs ---
        # Extract flat node embeddings from padded tensor
        flat_node_h = torch.cat([z[g, : n_per[g]] for g in range(b)], dim=0)

        node_mask_flat = (~cat_missing).to(dtype=dt)

        # Graph-level pool via segment softmax
        from torch_geometric.utils import softmax as seg_softmax

        scores = flat_node_h @ self.pool_query
        weights = seg_softmax(scores, cat_gid, num_nodes=b).unsqueeze(-1)
        graph_embed = scatter(
            weights * flat_node_h, cat_gid, dim=0, dim_size=b, reduce="sum"
        )

        # Hyperedge embeds via scatter-mean of node embeddings
        edge_h = (
            scatter(
                flat_node_h[cat_n_idx],
                cat_e_idx,
                dim=0,
                dim_size=total_e,
                reduce="mean",
            )
            if total_e > 0 and cat_n_idx.numel() > 0
            else torch.zeros(total_e, self.hidden_size, device=dev, dtype=dt)
        )

        # Split + pad to [B, max, H]
        out_n = torch.zeros(b, n_max, self.hidden_size, device=dev, dtype=dt)
        out_nm = torch.zeros(b, n_max, device=dev, dtype=dt)
        out_e = torch.zeros(b, e_max, self.hidden_size, device=dev, dtype=dt)
        out_em = torch.zeros(b, e_max, device=dev, dtype=dt)
        no, eo = 0, 0
        for g in range(b):
            nn_ = n_per[g]
            ee_ = e_per[g]
            out_n[g, :nn_] = flat_node_h[no : no + nn_]
            out_nm[g, :nn_] = node_mask_flat[no : no + nn_]
            out_e[g, :ee_] = edge_h[eo : eo + ee_]
            out_em[g, :ee_] = 1.0
            no += nn_
            eo += ee_

        return {
            "node_embeds": out_n,
            "node_mask": out_nm,
            "hyperedge_embeds": out_e,
            "hyperedge_mask": out_em,
            "graph_embed": graph_embed,
        }

    def _empty(self) -> KnowledgeRepr:
        dev = self.proj.weight.device
        dt = self.proj.weight.dtype
        return KnowledgeRepr(
            node_embeds=torch.zeros(0, self.hidden_size, device=dev, dtype=dt),
            node_mask=torch.zeros(0, device=dev),
            hyperedge_embeds=torch.zeros(0, self.hidden_size, device=dev, dtype=dt),
            hyperedge_mask=torch.zeros(0, device=dev),
            graph_embed=torch.zeros(self.hidden_size, device=dev, dtype=dt),
        )

    def build_incidence(self, sub: Any) -> Incidence:
        """Delegate to AllSetHyperEncoder.build_incidence (same Incidence format)."""
        if self._incidence_builder is None:
            msg = "build_incidence requires a feature_store (set at init)"
            raise RuntimeError(msg)
        return self._incidence_builder.build_incidence(sub)

    def encode(self, sub: Any) -> KnowledgeRepr:
        if not sub.node_ids:
            return self._empty()
        inc = self.build_incidence(sub)
        out = self.encode_batch([inc])
        n, e = inc.num_nodes, inc.num_edges
        return KnowledgeRepr(
            node_embeds=out["node_embeds"][0, :n],
            node_mask=out["node_mask"][0, :n],
            hyperedge_embeds=out["hyperedge_embeds"][0, :e],
            hyperedge_mask=out["hyperedge_mask"][0, :e],
            graph_embed=out["graph_embed"][0],
            meta={"missing_node_count": int(inc.node_missing.sum().item())},
        )
