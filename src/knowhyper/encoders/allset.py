"""AllSetTransformer sub-hypergraph encoder (role/semantics-aware).

Two-stage permutation-invariant message passing (AllSet, Chien et al. ICLR 2022):
node->hyperedge ``f_{V->E}`` then hyperedge->node ``f_{E->V}``, both multi-head
set-attention over variable-arity edges, with HYTREL-style role/semantics
injection. Emits node + hyperedge + graph embeddings into ``KnowledgeRepr``.

This file builds incrementally: the skeleton here uses mean aggregation; the
set-attention layer and role/semantics injection land in later tasks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch_scatter import scatter

from knowhyper.encoders.vocab import (
    POLARITY_VOCAB,
    ROLE_VOCAB,
    SEMANTICS_VOCAB,
    Vocab,
    redaf_semantics_id,
)
from knowhyper.pipeline.types import KnowledgeRepr, SubHypergraph

IncidenceProjection = Literal["hypergraph", "binary_clique"]


@dataclass
class Incidence:
    """Flattened (node, hyperedge) membership tensors for one sub-hypergraph."""

    node_feats: torch.Tensor  # (N, feature_dim) raw, pre-projection
    node_missing: torch.Tensor  # (N,) bool
    num_nodes: int
    num_edges: int
    node_idx: torch.Tensor  # (M,) long — node row per membership
    edge_idx: torch.Tensor  # (M,) long — edge row per membership
    role_id: torch.Tensor  # (M,) long — role vocab id per membership
    sem_id: torch.Tensor  # (E,) long — semantics vocab id per edge
    pol_id: torch.Tensor  # (E,) long — polarity vocab id per edge
    # S_mod = modality-slot count (distinct from M = membership count above).
    node_modality_mask: torch.Tensor | None = None  # (N, S_mod) float — present slots
    node_slot_modality: torch.Tensor | None = (
        None  # (N, S_mod) long; slot ids, -1=unused
    )
    node_semantics_id: torch.Tensor | None = (
        None  # (N,) long — instance dominant semantics (ReDaF)
    )
    # node_is_seed: like node_semantics_id, set only on the multimodal path; not in the co-presence guard.
    node_is_seed: torch.Tensor | None = (
        None  # (N,) bool — True iff node id in seed_node_ids
    )

    def __post_init__(self) -> None:
        """Enforce co-presence invariant: both mask fields must be set or both None."""
        if (self.node_modality_mask is None) != (self.node_slot_modality is None):
            msg = (
                "Incidence: node_modality_mask and node_slot_modality must both be "
                "set or both be None; got one set and the other None"
            )
            raise ValueError(msg)


class AllSetHyperEncoder(nn.Module):
    """Role/semantics-aware AllSetTransformer encoder over a SubHypergraph."""

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
        hde_enabled: bool = False,
        incidence_projection: IncidenceProjection = "hypergraph",
    ) -> None:
        """Build projections, role/semantics tables, layers, and the pool head.

        ``feature_store`` is the node-feature lookup used by :meth:`build_incidence`.
        Pass ``feature_store=None`` with an explicit ``feature_dim`` for a store-free
        encoder that only runs :meth:`encode_batch` on incidences whose ``node_feats``
        are already resolved (the GPU/model path); ``build_incidence`` is unavailable
        on such an instance.
        """
        super().__init__()
        if feature_store is None and feature_dim is None:
            msg = "AllSetHyperEncoder needs feature_store or feature_dim"
            raise ValueError(msg)
        self.feature_store = feature_store
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if incidence_projection not in ("hypergraph", "binary_clique"):
            msg = (
                "incidence_projection must be 'hypergraph' or 'binary_clique', "
                f"got {incidence_projection!r}"
            )
            raise ValueError(msg)
        self.incidence_projection = incidence_projection
        if feature_store is not None:
            in_dim = int(feature_store.dim)
        else:
            assert feature_dim is not None  # guaranteed by the guard above
            in_dim = int(feature_dim)
        self.proj = nn.Linear(in_dim, hidden_size)
        # Vocab is idempotent about a leading <unk>, so pass the curated tuples as-is.
        self._roles = Vocab(list(role_vocab))
        self._sems = Vocab(list(semantics_vocab))
        self._pols = Vocab(list(polarity_vocab))
        self.role_embed = nn.Embedding(len(self._roles), hidden_size)
        self.sem_embed = nn.Embedding(len(self._sems), hidden_size)
        self.pol_embed = nn.Embedding(len(self._pols), hidden_size)
        self.hde_enabled = hde_enabled
        self.layers = nn.ModuleList(
            AllSetTransformerLayer(hidden_size, num_heads, hde_enabled=hde_enabled)
            for _ in range(num_layers)
        )
        self.pool_query = nn.Parameter(torch.randn(hidden_size))

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

    def build_incidence(self, sub: SubHypergraph) -> Incidence:
        """Extract structure and raw (pre-projection) node features from a sub-HG.

        Deterministic feature lookup + membership-tensor construction, no trainable
        ops.  For an empty sub-hypergraph, returns a zero-length ``Incidence``
        suitable for the incidence-path injector.
        """
        if self.feature_store is None:
            msg = (
                "build_incidence requires a feature_store (this encoder is store-free)"
            )
            raise RuntimeError(msg)
        original_semantics = [edge.semantics for edge in sub.hyperedges]
        dominant_semantics = (
            Counter(original_semantics).most_common(1)[0][0]
            if original_semantics
            else "<unk>"
        )
        if self.incidence_projection == "binary_clique":
            from knowhyper.pipeline.projections import binary_clique_projection

            sub = binary_clique_projection(sub)
        dev = self.proj.weight.device

        if not sub.node_ids:
            return Incidence(
                node_feats=torch.zeros(0, self.feature_store.dim, device=dev),
                node_missing=torch.zeros(0, dtype=torch.bool, device=dev),
                num_nodes=0,
                num_edges=0,
                node_idx=torch.zeros(0, dtype=torch.long, device=dev),
                edge_idx=torch.zeros(0, dtype=torch.long, device=dev),
                role_id=torch.zeros(0, dtype=torch.long, device=dev),
                sem_id=torch.zeros(0, dtype=torch.long, device=dev),
                pol_id=torch.zeros(0, dtype=torch.long, device=dev),
                node_modality_mask=None,
                node_slot_modality=None,
                node_semantics_id=None,
                node_is_seed=None,
            )

        node_order = list(sub.node_ids)
        node_row = {nid: i for i, nid in enumerate(node_order)}
        raw = self.feature_store.features(node_order)
        node_modality_mask: torch.Tensor | None = None
        node_slot_modality: torch.Tensor | None = None
        node_semantics_id: torch.Tensor | None = None
        node_is_seed: torch.Tensor | None = None
        if "modality_embeds" in raw:
            me = raw["modality_embeds"].to(dev)  # (n, M, PAD_DIM)
            n_, m_, pad_ = me.shape
            feats = me.reshape(n_, m_ * pad_)  # (n, M*PAD_DIM)
            node_modality_mask = raw["modality_mask"].to(dev)  # (n, M)
            node_slot_modality = raw["slot_modality"].to(dev)  # (n, M) long
            # ReDaF relation signal: the instance's most-common hyperedge semantics,
            # stamped on every node (instance-level, mirroring NATIVE head/tail).
            dom_sem_id = redaf_semantics_id(dominant_semantics)
            node_semantics_id = torch.full(
                (n_,), dom_sem_id, dtype=torch.long, device=dev
            )
            seed_ids = set(sub.seed_node_ids)
            node_is_seed = torch.tensor(
                [nid in seed_ids for nid in node_order],
                dtype=torch.bool,
                device=dev,
            )
        else:
            feats = raw["node_embeds"].to(dev)
        node_missing = self.feature_store.missing(node_order).to(dev)
        n_idx: list[int] = []
        e_idx: list[int] = []
        roles: list[int] = []
        sems: list[int] = []
        pols: list[int] = []
        for e, he in enumerate(sub.hyperedges):
            sems.append(self._sems.index(he.semantics))
            pols.append(self._pols.index(he.polarity))
            for nid in he.node_ids:
                if nid not in node_row:  # retriever unions participants; be safe
                    continue
                n_idx.append(node_row[nid])
                e_idx.append(e)
                roles.append(self._roles.index(he.roles.get(nid)))

        def _long(xs: list[int]) -> torch.Tensor:
            return torch.tensor(xs, dtype=torch.long, device=dev)

        return Incidence(
            node_feats=feats,
            node_missing=node_missing,
            num_nodes=len(node_order),
            num_edges=len(sub.hyperedges),
            node_idx=_long(n_idx),
            edge_idx=_long(e_idx),
            role_id=_long(roles),
            sem_id=_long(sems),
            pol_id=_long(pols),
            node_modality_mask=node_modality_mask,
            node_slot_modality=node_slot_modality,
            node_semantics_id=node_semantics_id,
            node_is_seed=node_is_seed,
        )

    def _build(self, sub: SubHypergraph) -> Incidence:
        """Thin wrapper kept for compatibility; delegates to build_incidence."""
        return self.build_incidence(sub)

    def encode_batch(self, incidences: list[Incidence]) -> dict[str, torch.Tensor]:
        """Encode B sub-hypergraphs in one batched pass; return padded tensors.

        Concatenates the per-graph incidence into a single offset-indexed graph so the
        AllSet layers (segment scatter/softmax over node_idx/edge_idx) run once over
        the whole batch, then splits + pads results back to [B, max, H].
        """
        dev = self.proj.weight.device
        dt = self.proj.weight.dtype
        b = len(incidences)
        # per-graph sizes
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
        # offset-concat membership indices
        node_off, edge_off = 0, 0
        node_feats_list, node_missing_list = [], []
        n_idx, e_idx, role_id, sem_id, pol_id = [], [], [], [], []
        graph_of_node: list[torch.Tensor] = []  # (sum N,) graph id per node
        for g, inc in enumerate(incidences):
            node_feats_list.append(inc.node_feats.to(dev))
            node_missing_list.append(inc.node_missing.to(dev))
            n_idx.append(inc.node_idx.to(dev) + node_off)
            e_idx.append(inc.edge_idx.to(dev) + edge_off)
            role_id.append(inc.role_id.to(dev))
            sem_id.append(inc.sem_id.to(dev))
            pol_id.append(inc.pol_id.to(dev))
            graph_of_node.append(
                torch.full((inc.num_nodes,), g, dtype=torch.long, device=dev)
            )
            node_off += inc.num_nodes
            edge_off += inc.num_edges
        total_n, total_e = node_off, edge_off

        def _cat(xs: list[torch.Tensor]) -> torch.Tensor:
            return torch.cat(xs, dim=0)

        # Propagate node_is_seed if any incidence has it (HDE uses seed flag).
        seed_parts = [inc.node_is_seed for inc in incidences]
        node_is_seed: torch.Tensor | None = None
        if any(s is not None for s in seed_parts):
            filled = []
            for inc, sp in zip(incidences, seed_parts, strict=True):
                if sp is not None:
                    filled.append(sp.to(dev))
                else:
                    filled.append(
                        torch.zeros(inc.num_nodes, dtype=torch.bool, device=dev)
                    )
            node_is_seed = _cat(filled)

        big = Incidence(
            node_feats=_cat(node_feats_list),
            node_missing=_cat(node_missing_list),
            num_nodes=total_n,
            num_edges=total_e,
            node_idx=_cat(n_idx),
            edge_idx=_cat(e_idx),
            role_id=_cat(role_id),
            sem_id=_cat(sem_id),
            pol_id=_cat(pol_id),
            node_is_seed=node_is_seed,
        )
        node_h = self.proj(big.node_feats)
        edge_h = torch.zeros(total_e, self.hidden_size, device=dev, dtype=node_h.dtype)
        for layer in self.layers:
            node_h, edge_h = layer(node_h, edge_h, big, self)
        node_mask_flat = (~big.node_missing).to(dtype=node_h.dtype)
        # per-graph masked-softmax pool over node_h @ pool_query
        gids = _cat(graph_of_node)
        scores = node_h @ self.pool_query
        from torch_geometric.utils import softmax as seg_softmax

        weights = seg_softmax(scores, gids, num_nodes=b).unsqueeze(-1)
        graph_embed = scatter(weights * node_h, gids, dim=0, dim_size=b, reduce="sum")
        # split + pad back to [B, max, H]
        out_n = torch.zeros(b, n_max, self.hidden_size, device=dev, dtype=dt)
        out_nm = torch.zeros(b, n_max, device=dev, dtype=dt)
        out_e = torch.zeros(b, e_max, self.hidden_size, device=dev, dtype=dt)
        out_em = torch.zeros(b, e_max, device=dev, dtype=dt)
        no, eo = 0, 0
        for g, inc in enumerate(incidences):
            nn_ = inc.num_nodes
            ee_ = inc.num_edges
            out_n[g, :nn_] = node_h[no : no + nn_]
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

    def encode(self, sub: SubHypergraph) -> KnowledgeRepr:
        """Encode the sub-hypergraph into node/hyperedge/graph embeddings.

        Delegates to ``encode_batch`` for a single graph and unwraps row 0.
        """
        if not sub.node_ids:
            return self._empty()
        inc = self.build_incidence(sub)
        out = self.encode_batch([inc])
        n = inc.num_nodes
        e = inc.num_edges
        return KnowledgeRepr(
            node_embeds=out["node_embeds"][0, :n],
            node_mask=out["node_mask"][0, :n],
            hyperedge_embeds=out["hyperedge_embeds"][0, :e],
            hyperedge_mask=out["hyperedge_mask"][0, :e],
            graph_embed=out["graph_embed"][0],
            meta={"missing_node_count": int(inc.node_missing.sum().item())},
        )


class AllSetTransformerLayer(nn.Module):
    """One AllSet layer with multi-head set-attention for both stages.

    f_{V->E}: each hyperedge attends over its member nodes (softmax within edge).
    f_{E->V}: each node attends over its incident hyperedges (softmax within node).
    A learned query per stage scores members; values are linear projections.

    When ``hde_enabled=True``, the V->E stage uses Hyperedge-Dependent Entity (HDE)
    conditioning: each membership vector is computed by an FFN over the concatenation
    of [node_h, role_embed, sem_embed, seed_flag] instead of a simple additive
    ``node_h + role_embed``.  This gives the same node genuinely different
    representations per hyperedge context (arXiv:2212.14077).
    """

    def __init__(
        self, hidden_size: int, num_heads: int, *, hde_enabled: bool = False
    ) -> None:
        """Build per-stage Q/K/V projections, output mixes, and norms."""
        super().__init__()
        if hidden_size % num_heads != 0:
            msg = f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
            raise ValueError(msg)
        self.h = num_heads
        self.dh = hidden_size // num_heads
        self.hde_enabled = hde_enabled
        # HDE conditioning FFN: [node_h, role_embed, sem_embed, seed_flag] -> hidden
        # 2*hidden for node_h+role_embed, +hidden for sem_embed, +1 for seed flag
        if hde_enabled:
            self.hde_ffn = nn.Sequential(
                nn.Linear(2 * hidden_size + hidden_size + 1, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        # V->E stage
        self.q_e = nn.Parameter(torch.randn(num_heads, self.dh))
        self.k_ve = nn.Linear(hidden_size, hidden_size)
        self.v_ve = nn.Linear(hidden_size, hidden_size)
        self.o_e = nn.Linear(hidden_size, hidden_size)
        self.norm_e = nn.LayerNorm(hidden_size)
        # E->V stage
        self.q_v = nn.Parameter(torch.randn(num_heads, self.dh))
        self.k_ev = nn.Linear(hidden_size, hidden_size)
        self.v_ev = nn.Linear(hidden_size, hidden_size)
        self.o_v = nn.Linear(hidden_size, hidden_size)
        self.norm_v = nn.LayerNorm(hidden_size)

    def _attn(
        self,
        member: torch.Tensor,  # (M, hidden) per-membership input
        group_idx: torch.Tensor,  # (M,) group id per membership
        num_groups: int,
        q: torch.Tensor,  # (h, dh) learned query
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        o_proj: nn.Linear,
    ) -> torch.Tensor:
        from torch_geometric.utils import softmax as seg_softmax

        m = member.shape[0]
        k = k_proj(member).view(m, self.h, self.dh)
        v = v_proj(member).view(m, self.h, self.dh)
        scores = (k * q).sum(-1) / (self.dh**0.5)  # (M, h)
        alpha = seg_softmax(scores, group_idx, num_nodes=num_groups)  # within group
        weighted = (alpha.unsqueeze(-1) * v).reshape(m, self.h * self.dh)
        out = scatter(weighted, group_idx, dim=0, dim_size=num_groups, reduce="sum")
        return o_proj(out)

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        inc: Incidence,
        enc: AllSetHyperEncoder,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """f_{V->E} (with role injection) then f_{E->V} (after semantics inject)."""
        if self.hde_enabled:
            # HDE: per-membership FFN([node_h, role_embed, sem_embed, seed_flag])
            node_part = node_h[inc.node_idx]  # (M, H)
            role_part = enc.role_embed(inc.role_id)  # (M, H)
            sem_per_member = enc.sem_embed(inc.sem_id[inc.edge_idx])  # (M, H)
            if inc.node_is_seed is not None:
                seed_flag = inc.node_is_seed[inc.node_idx].float().unsqueeze(-1)
            else:
                seed_flag = node_part.new_zeros(node_part.shape[0], 1)
            member_v = self.hde_ffn(
                torch.cat([node_part, role_part, sem_per_member, seed_flag], dim=-1)
            )
        else:
            # Original additive: node_h + role_embed
            member_v = node_h[inc.node_idx] + enc.role_embed(inc.role_id)
        edge_agg = self._attn(
            member_v,
            inc.edge_idx,
            inc.num_edges,
            self.q_e,
            self.k_ve,
            self.v_ve,
            self.o_e,
        )
        edge_new = self.norm_e(edge_agg + edge_h)
        if not self.hde_enabled:
            edge_new = edge_new + enc.sem_embed(inc.sem_id)
        # polarity always additive post-norm (not routed through HDE FFN)
        edge_new = edge_new + enc.pol_embed(inc.pol_id)
        # f_{E->V}: each edge-as-seen-by-node.
        member_e = edge_new[inc.edge_idx]
        node_agg = self._attn(
            member_e,
            inc.node_idx,
            inc.num_nodes,
            self.q_v,
            self.k_ev,
            self.v_ev,
            self.o_v,
        )
        node_new = self.norm_v(node_agg + node_h)
        return node_new, edge_new
