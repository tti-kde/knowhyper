"""KnowHyper ieforge model wrappers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from knowhyper.encoders.fusion import PerModalityAttentionFusion

import torch
from hydra.utils import instantiate
from ieforge.models.base import IEModel
from ieforge.registry import register
from omegaconf import DictConfig, OmegaConf
from torch import nn

logger = logging.getLogger(__name__)


@register("model", name="knowhyper_cross_attention")
class CrossAttentionKnowledgeModel(IEModel):
    """Fuse knowledge node/hyperedge embeds into the [CLS] state via cross-attention.

    The inner BERT classifier runs to ``last_hidden_state``; its ``[CLS]`` row
    queries the concatenated (node + hyperedge) knowledge tokens through one
    ``MultiheadAttention`` layer, the result is added to ``[CLS]`` through a gate
    initialized to 0 (so the model starts identical to the no-knowledge baseline),
    and the inner pooler + classifier produce the logits. When the knowledge keys
    are absent the inner model runs unchanged.

    With ``gate_mlp_enabled=True`` a learned per-instance relevance gate
    ``g=sigmoid(MLP([cls, pooled_knowledge, cls*pooled_knowledge]))`` further scales
    the fusion (``cls_fused = cls + g * gate * attn_out``), attenuating knowledge on
    instances whose retrieved sub-HG is irrelevant to the text.
    """

    # Pre-encoded embed path (legacy / unit fixtures): embeds already in the batch.
    NODE_KEY = "knowledge_node_embeds"
    NODE_MASK_KEY = "knowledge_node_mask"
    HE_KEY = "knowledge_hyperedge_embeds"
    HE_MASK_KEY = "knowledge_hyperedge_mask"
    # Incidence path (the GPU-batched production path): raw sub-HG structure in the
    # batch (from IncidenceChannel); the model's own encoder runs encode_batch.
    INC_FEATS_KEY = "knowledge_incidence_node_feats"
    INC_NODE_MISSING_KEY = "knowledge_incidence_node_missing"
    INC_NODE_IDX_KEY = "knowledge_incidence_node_idx"
    INC_EDGE_IDX_KEY = "knowledge_incidence_edge_idx"
    INC_ROLE_KEY = "knowledge_incidence_role_id"
    INC_SEM_KEY = "knowledge_incidence_sem_id"
    INC_POL_KEY = "knowledge_incidence_pol_id"
    INC_NUM_NODES_KEY = "knowledge_incidence_num_nodes"
    INC_NUM_EDGES_KEY = "knowledge_incidence_num_edges"
    # Multimodal incidence path: per-modality keys emitted by Task-5 IncidenceChannel.
    INC_MODALITY_MASK_KEY = "knowledge_incidence_node_modality_mask"
    INC_SLOT_MODALITY_KEY = "knowledge_incidence_node_slot_modality"
    INC_MODALITY_M_KEY = "knowledge_incidence_modality_m"
    INC_PAD_DIM_KEY = "knowledge_incidence_pad_dim"
    # ReDaF relation hint: per-node semantics id emitted by Task 4 IncidenceChannel.
    INC_SEMANTICS_KEY = "knowledge_incidence_node_semantics"
    # Phase A seed attribute tokens: per-node bool flag emitted by Task 3 IncidenceChannel.
    INC_SEED_KEY = "knowledge_incidence_node_is_seed"

    def __init__(
        self,
        inner: IEModel | dict[str, Any] | DictConfig,
        label_list: list[str],
        hidden_size: int = 768,
        num_heads: int = 8,
        extra_token_count: int = 0,
        name: str = "knowhyper_cross_attention",
        require_knowledge: bool = False,
        feature_dim: int | None = None,
        encoder_type: str = "allset",
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        hde_enabled: bool = False,
        fusion: Any = None,
        hypergt_nb_random_features: int = 30,
        hypergt_use_gumbel: bool = True,
        hypergt_nb_gumbel_sample: int = 10,
        hypergt_rb_order: int = 0,
        hypergt_tau: float = 1.0,
        hypergt_dropout: float = 0.0,
        gate_init: float = 0.0,
        gate_shape: str = "scalar",
        gate_mlp_enabled: bool = False,
        gate_mlp_init: str = "zeros",
        seed_attr_tokens: bool = False,
        split_gate: bool = False,
        fusion_norm: bool = False,
        pretrained_encoder_path: str | None = None,
        **kwargs: Any,  # tolerate config-level keys (e.g. tokenizer_name) read by CLI
    ) -> None:
        """Build the inner classifier, cross-attention, gate, and (optional) encoder.

        When ``feature_dim`` is given, also build a store-free AllSet encoder that
        runs the sub-HG encode on this model's device from the batched incidence
        tensors (the GPU-batched path).
        """
        super().__init__()
        del kwargs
        self.name = name
        self.label_list = label_list
        self.hidden_size = hidden_size
        self.require_knowledge = require_knowledge
        self.seed_attr_tokens = seed_attr_tokens
        self._warned_no_knowledge = False
        self.inner = self._build_inner(inner, label_list, extra_token_count)
        self._check_inner_shape()
        self._hf = self._bert()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        # Optional LayerNorm on the cross-attention output before it is gated into
        # CLS. Without it, a knowledge branch trained at a high LR grows attn_out's
        # magnitude unboundedly and blows up CLS (observed collapse at
        # knowledge_lr_mult>=10). Normalizing attn_out keeps its contribution
        # scale-bounded so the branch can train without destabilizing the trunk.
        # Default off => byte-identical to the un-normed fusion.
        self.attn_norm = nn.LayerNorm(hidden_size) if fusion_norm else None
        # Fusion gate on `cls_fused = cls + gate * attn_out`. gate_init=0.0 (default)
        # keeps the degradation guarantee (gate=0 => exact baseline); the
        # force-knowledge experiment sets gate_init=1.0 so the model starts fully
        # fusing knowledge and must actively learn to discard it. gate_shape="vector"
        # makes the gate per-hidden-dim (broadcasts over attn_out) so it cannot
        # collapse to a single scalar 0.
        if gate_shape == "scalar":
            gate_n = 1
        elif gate_shape == "vector":
            gate_n = hidden_size
        else:
            msg = f"gate_shape must be 'scalar' or 'vector', got {gate_shape!r}"
            raise ValueError(msg)
        self.gate = nn.Parameter(torch.full((gate_n,), float(gate_init)))
        # Per-instance relevance gate g=sigmoid(MLP([cls, pooled_knowledge,
        # cls*pooled_knowledge])) that attenuates fusion when the retrieved
        # knowledge is irrelevant: cls_fused = cls + g * gate * attn_out. Default
        # disabled (no gate computed; equivalent to g=1.0) so existing behavior is
        # byte-identical. The final layer bias is +4.0 so g=sigmoid(4)=0.982 at
        # startup (a measurable delta over the gate1/vector baseline).
        # gate_mlp_init controls the final-layer WEIGHTS:
        #   "zeros" -> g is exactly sigmoid(4) for all inputs at startup, but the
        #     MLP gets ~zero gradient early and barely learns in a few epochs
        #     (observed in the 3-epoch ChemProt/DDI RE runs: g stayed a near-constant
        #     ~0.987 with no per-instance signal — run-specific, not guaranteed);
        #   "randn" -> small random weights so gradients flow from step 1 and the
        #     gate can actually learn input-dependent relevance (startup g is still
        #     ~0.98 because the +4 bias dominates the small-weight contribution).
        self.gate_mlp_enabled = gate_mlp_enabled
        self.gate_mlp: nn.Sequential | None = None
        self.split_gate = split_gate
        self.gate_node: nn.Parameter | None = None
        self.gate_edge: nn.Parameter | None = None
        if split_gate:
            if gate_mlp_enabled:
                msg = "split_gate=True is mutually exclusive with gate_mlp_enabled=True"
                raise ValueError(msg)
            self.gate_node = nn.Parameter(torch.full((gate_n,), float(gate_init)))
            self.gate_edge = nn.Parameter(torch.full((gate_n,), float(gate_init)))
        if gate_mlp_enabled:
            final = nn.Linear(hidden_size, 1)
            if gate_mlp_init == "zeros":
                nn.init.zeros_(final.weight)
            elif gate_mlp_init == "randn":
                nn.init.normal_(final.weight, std=0.02)
            else:
                msg = f"gate_mlp_init must be 'zeros' or 'randn', got {gate_mlp_init!r}"
                raise ValueError(msg)
            nn.init.constant_(final.bias, 4.0)
            self.gate_mlp = nn.Sequential(
                nn.Linear(3 * hidden_size, hidden_size),
                nn.ReLU(),
                final,
            )
        # Store-free encoder for the incidence path. None when feature_dim is unset
        # (unit fixtures using the pre-encoded embed path).
        self.encoder = None
        self.encoder_type = encoder_type
        if feature_dim is not None:
            if encoder_type == "hypergt":
                from knowhyper.encoders.hypergt import HyperGTEncoder

                self.encoder = HyperGTEncoder(
                    feature_store=None,
                    feature_dim=int(feature_dim),
                    hidden_size=hidden_size,
                    num_layers=encoder_layers,
                    num_heads=encoder_heads,
                    nb_random_features=hypergt_nb_random_features,
                    use_gumbel=hypergt_use_gumbel,
                    nb_gumbel_sample=hypergt_nb_gumbel_sample,
                    rb_order=hypergt_rb_order,
                    tau=hypergt_tau,
                    dropout=hypergt_dropout,
                    hde_enabled=hde_enabled,
                )
            else:
                from knowhyper.encoders.allset import AllSetHyperEncoder

                self.encoder = AllSetHyperEncoder(
                    feature_store=None,
                    feature_dim=int(feature_dim),
                    hidden_size=hidden_size,
                    num_layers=encoder_layers,
                    num_heads=encoder_heads,
                    hde_enabled=hde_enabled,
                )
        self.hde_enabled = hde_enabled
        if pretrained_encoder_path is not None and self.encoder is not None:
            ckpt = torch.load(pretrained_encoder_path, map_location="cpu")
            sd = ckpt.get("encoder", ckpt)
            result = self.encoder.load_state_dict(sd, strict=False)
            if result.missing_keys:
                logger.info(
                    "warm-start: %d missing keys (randomly initialised): %s",
                    len(result.missing_keys),
                    ", ".join(result.missing_keys[:10]),
                )
            if result.unexpected_keys:
                logger.warning(
                    "warm-start: %d unexpected keys ignored: %s",
                    len(result.unexpected_keys),
                    ", ".join(result.unexpected_keys[:10]),
                )
            logger.info(
                "warm-started %s encoder from %s (%d tensors loaded)",
                encoder_type,
                pretrained_encoder_path,
                len(sd) - len(result.unexpected_keys),
            )
        elif pretrained_encoder_path is not None:
            # No encoder was built (feature_dim unset) — warn rather than silently
            # ignore the requested warm-start, which would waste an experiment.
            logger.warning(
                "pretrained_encoder_path=%s set but no encoder built (feature_dim "
                "unset); warm-start ignored",
                pretrained_encoder_path,
            )
        # Optional per-modality fusion (Task 6). When set, `_encode_incidence` will
        # reshape the stacked multimodal node feats and run fusion before handing
        # the (sum_n, feature_dim) tensor to the AllSet encoder.
        self.modality_fusion: nn.Module | None = None
        if fusion is not None:
            self.modality_fusion = (
                fusion if isinstance(fusion, nn.Module) else instantiate(fusion)
            )

    @staticmethod
    def _build_inner(
        inner: IEModel | dict[str, Any] | DictConfig,
        label_list: list[str],
        extra_token_count: int,
    ) -> IEModel:
        if isinstance(inner, IEModel):
            return inner
        if not isinstance(inner, (dict, DictConfig)):
            msg = "inner must be an IEModel or a Hydra config/dict with a _target_"
            raise TypeError(msg)
        inner_cfg = OmegaConf.create(inner)
        if OmegaConf.select(inner_cfg, "_target_", default=None) is None:
            msg = "inner config must include a _target_"
            raise ValueError(msg)
        built = instantiate(
            inner_cfg,
            label_list=label_list,
            extra_token_count=extra_token_count,
        )
        if not isinstance(built, IEModel):
            msg = f"inner config instantiated {type(built).__name__}, expected IEModel"
            raise TypeError(msg)
        return built

    def _bert(self) -> Any:
        hf = getattr(self.inner, "model", None)
        bert = getattr(hf, "bert", None)
        pooler = getattr(bert, "pooler", None)
        if bert is None or pooler is None or not hasattr(hf, "classifier"):
            msg = (
                "CrossAttentionKnowledgeModel requires a BERT-family inner model "
                "with .model.bert(.pooler) and .model.classifier"
            )
            raise TypeError(msg)
        return hf

    def _check_inner_shape(self) -> None:
        hf = self._bert()
        inner_h = int(hf.config.hidden_size)
        if inner_h != self.hidden_size:
            msg = f"hidden_size {self.hidden_size} != inner model hidden size {inner_h}"
            raise ValueError(msg)

    def _has_knowledge(self, batch: dict[str, Any]) -> bool:
        # The incidence integrator ALWAYS writes the incidence_* keys, even for an
        # empty sub-HG (a zero-node placeholder). So key-presence alone is not
        # "knowledge present" — an all-cache-miss batch has the keys but every graph
        # has num_nodes=0. Treat such a batch as knowledge-absent so require_knowledge
        # fails loud instead of silently running the baseline.
        if self.INC_FEATS_KEY in batch:
            return int(batch[self.INC_NUM_NODES_KEY].sum()) > 0
        return self.NODE_KEY in batch or self.HE_KEY in batch

    def forward(self, batch: dict[str, Any]) -> Any:
        """Run inner BERT, gate-fuse cross-attended knowledge into CLS, classify."""
        hf = self._hf
        if not self._has_knowledge(batch):
            if self.require_knowledge:
                msg = "knowledge keys absent from batch but require_knowledge=True"
                raise ValueError(msg)
            if not self._warned_no_knowledge:
                logger.warning(
                    "CrossAttentionKnowledgeModel: no knowledge keys in batch; "
                    "running inner (baseline) forward"
                )
                self._warned_no_knowledge = True
            return self.inner.forward(batch)

        encoded = hf.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        hidden = encoded.last_hidden_state  # [B, L, H]
        cls = hidden[:, 0]  # [B, H]

        if self.split_gate:
            assert self.gate_node is not None
            assert self.gate_edge is not None
            node_kv, node_m, he_kv, he_m = self._knowledge_kv_split(batch, cls)
            attn_node = self._attend(cls, node_kv, node_m)
            attn_edge = self._attend(cls, he_kv, he_m)
            cls_fused = cls + self.gate_node * attn_node + self.gate_edge * attn_edge
        elif self.gate_mlp is not None:
            kv, kmask = (
                self._encode_incidence(batch, cls)
                if self.INC_FEATS_KEY in batch
                else self._knowledge_tokens(batch, cls)
            )
            attn_out = self._attend(cls, kv, kmask)  # [B, H]
            g = self._relevance_gate(cls, kv, kmask)  # [B, 1]
            cls_fused = cls + g * self.gate * attn_out
        else:
            kv, kmask = (
                self._encode_incidence(batch, cls)
                if self.INC_FEATS_KEY in batch
                else self._knowledge_tokens(batch, cls)
            )
            attn_out = self._attend(cls, kv, kmask)  # [B, H]
            cls_fused = cls + self.gate * attn_out

        pooled = hf.bert.pooler(cls_fused.unsqueeze(1))  # BertPooler reads [:, 0]
        pooled = hf.dropout(pooled) if hasattr(hf, "dropout") else pooled
        logits = hf.classifier(pooled)

        loss = None
        labels = batch.get("labels")
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return _Output(logits=logits, loss=loss)

    def _encode_incidence_raw(
        self, batch: dict[str, Any], cls: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]
    ]:
        """Shared incidence encode: returns (node_e, node_m, he_e, he_m, attr_per_graph).

        Performs the modality-fusion reshape, seed-attr token extraction, the
        un-offset Incidence loop, and the AllSet encode — everything up to but
        NOT including Phase A attr-token placement.  The two callers then decide
        WHERE the attr tokens land:

        * ``_encode_incidence``       — appends attr at the combined-kv tail → [node|he|attr]
        * ``_encode_incidence_split`` — folds attr onto the node side → node_e_with_attr
        """
        if self.encoder is None:
            msg = (
                "incidence keys present but this model was built without "
                "feature_dim (no encoder); set model.feature_dim"
            )
            raise ValueError(msg)
        from knowhyper.encoders.allset import Incidence

        dev = self.encoder.proj.weight.device
        feats = batch[self.INC_FEATS_KEY].to(dev)
        if self.modality_fusion is not None and self.INC_MODALITY_M_KEY not in batch:
            msg = (
                f"modality_fusion is configured but batch lacks "
                f"{self.INC_MODALITY_M_KEY!r}; the feature store must be "
                f"MultiModalFeatureStore and the cache built with v0.12.0 multimodal keys"
            )
            raise ValueError(msg)
        # Seed attribute token state; may be populated inside the multimodal block below.
        seed_attr_vecs: torch.Tensor | None = None  # (sum_a, out_dim) or None
        seed_attr_node: torch.Tensor | None = None  # (sum_a,) long in sum_n space
        if self.modality_fusion is not None and self.INC_MODALITY_M_KEY in batch:
            fusion = cast("PerModalityAttentionFusion", self.modality_fusion)
            m = int(batch[self.INC_MODALITY_M_KEY])
            pad = int(batch[self.INC_PAD_DIM_KEY])
            stack = feats.reshape(feats.shape[0], m, pad)  # (sum_n, M, PAD_DIM)
            mod_mask = batch[self.INC_MODALITY_MASK_KEY].to(dev)  # (sum_n, M)
            slot = batch[self.INC_SLOT_MODALITY_KEY].to(dev)  # (sum_n, M) long
            rel_hint = batch.get(self.INC_SEMANTICS_KEY)
            if rel_hint is None and getattr(fusion, "rel_gate", None) is not None:
                msg = (
                    f"modality_fusion has a relation gate (ReDaF, num_relations>0) but "
                    f"batch lacks {self.INC_SEMANTICS_KEY!r}; the cache must be built with "
                    f"this version's build_incidence (which stamps node_semantics_id). "
                    f"Running without it would silently degrade ReDaF to static fusion."
                )
                raise ValueError(msg)
            if rel_hint is not None:
                rel_hint = rel_hint.to(dev)
            feats = fusion(
                stack, mod_mask, slot, relation_hint=rel_hint
            )  # (sum_n, OUT_DIM)
            # Phase A: compute seed attribute tokens (bypass AllSet; project_only per
            # seed node, one token per present modality slot).
            if self.seed_attr_tokens:
                is_seed = batch.get(self.INC_SEED_KEY)
                if is_seed is None:
                    msg = (
                        f"seed_attr_tokens=True but batch lacks {self.INC_SEED_KEY!r}; "
                        f"the cache must be built with this version's build_incidence "
                        f"(which stamps node_is_seed). Refusing to silently skip "
                        f"seed attribute tokens."
                    )
                    raise ValueError(msg)
                is_seed = is_seed.to(dev).bool()
                seed_rows = is_seed.nonzero(as_tuple=True)[0]  # (S,) indices in sum_n
                if seed_rows.numel() > 0:
                    sub_stack = stack[seed_rows]  # (S, M, PAD)
                    sub_mask = mod_mask[seed_rows]  # (S, M)
                    sub_slot = slot[seed_rows]  # (S, M)
                    vecs, local_node = fusion.project_only(
                        sub_stack, sub_mask, sub_slot
                    )  # vecs:(sum_a, out_dim), local_node:(sum_a,) in 0..S-1
                    seed_attr_vecs = vecs
                    # map local seed-row index back to the flat sum_n node index
                    seed_attr_node = seed_rows[local_node]  # (sum_a,) in sum_n space
        # HDE needs node_is_seed even when seed_attr_tokens is off.
        node_is_seed: torch.Tensor | None = None
        if self.hde_enabled or self.seed_attr_tokens:
            raw_seed = batch.get(self.INC_SEED_KEY)
            if raw_seed is not None:
                node_is_seed = raw_seed.to(dev).bool()

        node_missing = batch[self.INC_NODE_MISSING_KEY].to(dev)
        node_idx = batch[self.INC_NODE_IDX_KEY].to(dev)
        edge_idx = batch[self.INC_EDGE_IDX_KEY].to(dev)
        role_id = batch[self.INC_ROLE_KEY].to(dev)
        sem_id = batch[self.INC_SEM_KEY].to(dev)
        pol_id = batch[self.INC_POL_KEY].to(dev)
        num_nodes = [int(x) for x in batch[self.INC_NUM_NODES_KEY].tolist()]
        num_edges = [int(x) for x in batch[self.INC_NUM_EDGES_KEY].tolist()]

        incidences: list[Incidence] = []
        attr_per_graph: list[torch.Tensor] = []
        node_off = edge_off = m_off = 0
        for g in range(len(num_nodes)):
            nn_, ee_ = num_nodes[g], num_edges[g]
            hi = node_off + nn_
            m_end = m_off
            while m_end < node_idx.shape[0] and int(node_idx[m_end]) < hi:
                m_end += 1
            sl = slice(m_off, m_end)
            g_seed = (
                node_is_seed[node_off : node_off + nn_]
                if node_is_seed is not None
                else None
            )
            incidences.append(
                Incidence(
                    node_feats=feats[node_off : node_off + nn_],
                    node_missing=node_missing[node_off : node_off + nn_],
                    num_nodes=nn_,
                    num_edges=ee_,
                    node_idx=node_idx[sl] - node_off,
                    edge_idx=edge_idx[sl] - edge_off,
                    role_id=role_id[sl],
                    sem_id=sem_id[edge_off : edge_off + ee_],
                    pol_id=pol_id[edge_off : edge_off + ee_],
                    node_is_seed=g_seed,
                )
            )
            # Accumulate this graph's seed attribute vectors (Phase A).
            if seed_attr_vecs is not None and seed_attr_node is not None:
                in_g = (seed_attr_node >= node_off) & (seed_attr_node < hi)
                attr_per_graph.append(seed_attr_vecs[in_g])
            else:
                attr_per_graph.append(feats.new_zeros(0, feats.shape[-1]))
            node_off = hi
            edge_off += ee_
            m_off = m_end

        out = self.encoder.encode_batch(incidences)
        node_e = out["node_embeds"].to(device=cls.device, dtype=cls.dtype)
        node_m = out["node_mask"].to(device=cls.device, dtype=cls.dtype)
        he_e = out["hyperedge_embeds"].to(device=cls.device, dtype=cls.dtype)
        he_m = out["hyperedge_mask"].to(device=cls.device, dtype=cls.dtype)
        return node_e, node_m, he_e, he_m, attr_per_graph

    @staticmethod
    def _pad_attr_tokens(
        attr_per_graph: list[torch.Tensor], ref: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad per-graph seed attr vectors into a dense (B, a_max, H) tensor + mask.

        ``ref`` is any existing (B, T, H) tensor on the target device/dtype
        (typically ``node_e`` or ``kv``) — used only for ``new_zeros`` allocation.
        Returns ``(attr_e, attr_m)`` where ``attr_m[g, i]=1.0`` iff slot i is valid.
        """
        a_max = max(a.shape[0] for a in attr_per_graph)
        b = len(attr_per_graph)
        attr_e = ref.new_zeros(b, a_max, ref.shape[-1])
        attr_m = ref.new_zeros(b, a_max)
        for g, a in enumerate(attr_per_graph):
            if a.shape[0]:
                attr_e[g, : a.shape[0]] = a.to(device=ref.device, dtype=ref.dtype)
                attr_m[g, : a.shape[0]] = 1.0
        return attr_e, attr_m

    def _encode_incidence(
        self, batch: dict[str, Any], cls: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the batched incidence on-device into cross-attention kv + mask.

        Rebuilds per-graph Incidence from the offset-batched channel tensors, runs
        the store-free AllSet encoder, and returns concatenated node+hyperedge
        embeds (kv) and masks.  The single-gate path preserves the original token
        order [node_e | he_e | attr_e] (Phase A attrs appended at the end).
        """
        node_e, node_m, he_e, he_m, attr_per_graph = self._encode_incidence_raw(
            batch, cls
        )
        kv = torch.cat([node_e, he_e], dim=1)
        kmask = torch.cat([node_m, he_m], dim=1)
        # Phase A: append seed attribute tokens when the flag is on and at least one
        # graph produced seed attribute vectors (bypass AllSet; raw project_only output).
        if self.seed_attr_tokens and any(a.shape[0] > 0 for a in attr_per_graph):
            attr_e, attr_m = self._pad_attr_tokens(attr_per_graph, kv)
            kv = torch.cat([kv, attr_e], dim=1)
            kmask = torch.cat([kmask, attr_m], dim=1)
        return kv, kmask

    def _knowledge_tokens(
        self, batch: dict[str, Any], cls: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_kv, node_m, he_kv, he_m = self._knowledge_tokens_split(batch, cls)
        return torch.cat([node_kv, he_kv], dim=1), torch.cat([node_m, he_m], dim=1)

    def _knowledge_tokens_split(
        self, batch: dict[str, Any], cls: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (node_kv, node_mask, he_kv, he_mask) un-concatenated.

        Mirrors ``_knowledge_tokens`` but returns the node and hyperedge pieces
        separately for the split-gate path.  Validation (mask presence, length
        consistency) is identical to the original.
        """
        node_parts: list[torch.Tensor] = []
        node_masks: list[torch.Tensor] = []
        he_parts: list[torch.Tensor] = []
        he_masks: list[torch.Tensor] = []
        for ekey, mkey, parts, masks_list in (
            (self.NODE_KEY, self.NODE_MASK_KEY, node_parts, node_masks),
            (self.HE_KEY, self.HE_MASK_KEY, he_parts, he_masks),
        ):
            emb = batch.get(ekey)
            if emb is None:
                continue
            mask = batch.get(mkey)
            if mask is None:
                msg = (
                    f"{ekey!r} present but mask {mkey!r} missing; "
                    "cannot tell valid tokens from padding"
                )
                raise ValueError(msg)
            emb = emb.to(device=cls.device, dtype=cls.dtype)
            mask = mask.to(device=cls.device)
            if emb.shape[1] != mask.shape[1]:
                msg = (
                    f"knowledge embed {ekey!r} token count {emb.shape[1]} != "
                    f"mask {mkey!r} token count {mask.shape[1]}"
                )
                raise ValueError(msg)
            parts.append(emb)
            masks_list.append(mask)
        # Build a zero-token placeholder for whichever side is absent so the
        # split path always returns tensors with consistent batch dim.
        b_sz = cls.shape[0]
        h = cls.shape[-1]
        if node_parts:
            node_kv = torch.cat(node_parts, dim=1)
            node_m = torch.cat(node_masks, dim=1)
        else:
            node_kv = cls.new_zeros(b_sz, 0, h)
            node_m = cls.new_zeros(b_sz, 0)
        if he_parts:
            he_kv = torch.cat(he_parts, dim=1)
            he_m = torch.cat(he_masks, dim=1)
        else:
            he_kv = cls.new_zeros(b_sz, 0, h)
            he_m = cls.new_zeros(b_sz, 0)
        return node_kv, node_m, he_kv, he_m

    def _knowledge_kv_split(
        self, batch: dict[str, Any], cls: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (node_kv, node_mask, he_kv, he_mask) un-concatenated.

        Used by the split-gate path.  Phase A seed-attribute tokens fold onto
        the node side (they are entity attributes).  Mirrors
        ``_encode_incidence`` / ``_knowledge_tokens`` but returns the pieces
        instead of cat-ing them.
        """
        if self.INC_FEATS_KEY in batch:
            node_kv, node_m, he_kv, he_m = self._encode_incidence_split(batch, cls)
        else:
            node_kv, node_m, he_kv, he_m = self._knowledge_tokens_split(batch, cls)
        return node_kv, node_m, he_kv, he_m

    def _encode_incidence_split(
        self, batch: dict[str, Any], cls: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the batched incidence and return the 4 un-concatenated pieces.

        Delegates shared work to ``_encode_incidence_raw``, then folds Phase A
        seed-attribute tokens onto the NODE side (they are entity attributes;
        gate_node governs them).  Called only from ``_knowledge_kv_split``
        (split-gate forward path).
        """
        node_e, node_m, he_e, he_m, attr_per_graph = self._encode_incidence_raw(
            batch, cls
        )
        # Phase A: fold seed attribute tokens onto the NODE side.
        if self.seed_attr_tokens and any(a.shape[0] > 0 for a in attr_per_graph):
            attr_e, attr_m = self._pad_attr_tokens(attr_per_graph, node_e)
            node_e = torch.cat([node_e, attr_e], dim=1)
            node_m = torch.cat([node_m, attr_m], dim=1)
        return node_e, node_m, he_e, he_m

    def _relevance_gate(
        self, cls: torch.Tensor, kv: torch.Tensor, kmask: torch.Tensor
    ) -> torch.Tensor:
        """Per-instance fusion-relevance gate g in (0, 1), shape [B, 1].

        Pools the knowledge tokens by their mask (empty rows -> zero vector, no
        NaN) and feeds [cls, pooled, cls*pooled] through the gate MLP. The
        elementwise product is the text-knowledge relevance proxy; an irrelevant
        but PRESENT sub-HG drives g down, attenuating fusion. (Fully-empty rows
        are already zeroed by ``_attend`` via ``* any_valid``, so g cannot change
        their output — the gate only matters where knowledge is present.)
        """
        if self.gate_mlp is None:  # only reached when gate_mlp_enabled
            msg = "_relevance_gate called without gate_mlp"
            raise RuntimeError(msg)
        mask = kmask.to(kv.dtype).unsqueeze(-1)  # [B, T, 1]
        denom = mask.sum(dim=1).clamp(min=1.0)  # [B, 1], avoid div-by-zero
        pooled = (kv * mask).sum(dim=1) / denom  # [B, H]
        feats = torch.cat([cls, pooled, cls * pooled], dim=-1)  # [B, 3H]
        return torch.sigmoid(self.gate_mlp(feats))  # [B, 1]

    def _attend(
        self, cls: torch.Tensor, kv: torch.Tensor, kmask: torch.Tensor
    ) -> torch.Tensor:
        # key_padding_mask: True = ignore. Rows with no valid token would make the
        # whole row masked (NaN); detect and zero those rows' attention output.
        valid = kmask.bool()  # [B, Nk]
        any_valid = valid.any(dim=1)  # [B]
        key_padding = ~valid
        # An all-masked row would make every key_padding bit True -> softmax over
        # nothing -> NaN. For those rows (any_valid False) we AND the mask with
        # any_valid so NO key is masked (attention is defined), then zero the whole
        # row's output below via `* any_valid`. Non-empty rows keep their real mask.
        key_padding = key_padding & any_valid.unsqueeze(1)
        attn_out, _ = self.cross_attn(
            query=cls.unsqueeze(1),
            key=kv,
            value=kv,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        attn_out = attn_out.squeeze(1)  # [B, H]
        if self.attn_norm is not None:
            attn_out = self.attn_norm(attn_out)
        # Zero empty-sub-HG rows AFTER norm so the empty->no-fusion and gate=0
        # guarantees survive (LayerNorm's beta would otherwise make a zero row
        # nonzero).
        return attn_out * any_valid.unsqueeze(1).to(attn_out.dtype)


class _Output:
    """Minimal output object exposing .logits/.loss like a HF classifier output."""

    def __init__(self, logits: torch.Tensor, loss: torch.Tensor | None) -> None:
        self.logits = logits
        self.loss = loss


@register("model", name="knowhyper_token_cross_attention")
class TokenCrossAttentionKnowledgeModel(IEModel):
    """Fuse hypergraph knowledge into token states for NER.

    ``fusion_scope="mention"`` restricts the residual to tokens overlapping predicted
    first-pass mentions (plus an optional token window), so unrelated ``O`` tokens and
    no-mention rows are exact zero-fusion paths. ``conditional_gate=True`` further
    scales each token residual from its text/knowledge compatibility. The legacy
    full-sequence path remains available as ``fusion_scope="global"`` for ablations.
    """

    NODE_KEY = "knowledge_node_embeds"
    NODE_MASK_KEY = "knowledge_node_mask"
    HE_KEY = "knowledge_hyperedge_embeds"
    HE_MASK_KEY = "knowledge_hyperedge_mask"
    INC_FEATS_KEY = "knowledge_incidence_node_feats"
    INC_NODE_MISSING_KEY = "knowledge_incidence_node_missing"
    INC_NODE_IDX_KEY = "knowledge_incidence_node_idx"
    INC_EDGE_IDX_KEY = "knowledge_incidence_edge_idx"
    INC_ROLE_KEY = "knowledge_incidence_role_id"
    INC_SEM_KEY = "knowledge_incidence_sem_id"
    INC_POL_KEY = "knowledge_incidence_pol_id"
    INC_NUM_NODES_KEY = "knowledge_incidence_num_nodes"
    INC_NUM_EDGES_KEY = "knowledge_incidence_num_edges"
    NER_SPANS_KEY = "knowledge_ner_mention_spans"
    NER_SPAN_MASK_KEY = "knowledge_ner_mention_mask"
    INC_NODE_MENTION_KEY = "knowledge_incidence_node_mention_mask"
    INC_LOCAL_NODE_MENTION_KEY = "knowledge_incidence_local_node_mention_mask"
    INC_HE_MENTION_KEY = "knowledge_incidence_hyperedge_mention_mask"
    NER_LINK_SCORES_KEY = "knowledge_ner_mention_link_scores"
    NER_LINK_MASK_KEY = "knowledge_ner_mention_link_mask"
    NER_CONFIDENCES_KEY = "knowledge_ner_mention_confidences"
    NER_ENTROPIES_KEY = "knowledge_ner_mention_entropies"
    NER_DECODED_KEY = "knowledge_ner_mention_decoded"
    NER_LABELS_KEY = "knowledge_ner_mention_labels"

    def __init__(
        self,
        inner: IEModel | dict[str, Any] | DictConfig,
        label_list: list[str],
        hidden_size: int = 768,
        num_heads: int = 8,
        name: str = "knowhyper_token_cross_attention",
        require_knowledge: bool = False,
        feature_dim: int | None = None,
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        gate_init: float = 0.0,
        gate_shape: str = "scalar",
        fusion_norm: bool = False,
        pretrained_encoder_path: str | None = None,
        fusion_scope: str = "global",
        mention_context_window: int = 0,
        conditional_gate: bool = False,
        conditional_gate_bias: float = -2.0,
        knowledge_key_scope: str = "all",
        link_score_gate: bool = False,
        knowledge_control: str = "real",
        entity_loss_weight: float = 1.0,
        begin_loss_weight: float = 1.0,
        candidate_reranker: bool = False,
        candidate_logit_scale: float = 1.0,
        require_knowledge_per_instance: bool = False,
        tokenizer_name: str | None = None,
        extra_token_count: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            msg = f"Unexpected TokenCrossAttentionKnowledgeModel arguments: {unknown}"
            raise TypeError(msg)
        if extra_token_count != 0:
            msg = (
                "TokenCrossAttentionKnowledgeModel does not support prompt extra tokens; "
                f"got extra_token_count={extra_token_count}"
            )
            raise ValueError(msg)
        self.name = name
        self.tokenizer_name = tokenizer_name
        self.extra_token_count = extra_token_count
        self.label_list = label_list
        self.hidden_size = hidden_size
        self.require_knowledge = require_knowledge
        self.require_knowledge_per_instance = require_knowledge_per_instance
        if fusion_scope not in ("global", "mention"):
            msg = f"fusion_scope must be 'global' or 'mention', got {fusion_scope!r}"
            raise ValueError(msg)
        if mention_context_window < 0:
            msg = "mention_context_window must be non-negative"
            raise ValueError(msg)
        mention_scopes = ("mention_seed", "mention_local")
        if knowledge_key_scope not in ("all", *mention_scopes):
            msg = (
                "knowledge_key_scope must be all, mention_seed, or "
                f"mention_local, got {knowledge_key_scope!r}"
            )
            raise ValueError(msg)
        if knowledge_key_scope in mention_scopes and fusion_scope != "mention":
            msg = (
                f"knowledge_key_scope={knowledge_key_scope!r} requires "
                "fusion_scope=mention"
            )
            raise ValueError(msg)
        if link_score_gate and not conditional_gate:
            msg = "link_score_gate requires conditional_gate=True"
            raise ValueError(msg)
        if link_score_gate and knowledge_key_scope not in mention_scopes:
            msg = "link_score_gate requires a mention-specific knowledge_key_scope"
            raise ValueError(msg)
        if knowledge_control not in ("real", "shuffled", "zero"):
            msg = (
                "knowledge_control must be 'real', 'shuffled', or 'zero', "
                f"got {knowledge_control!r}"
            )
            raise ValueError(msg)
        if entity_loss_weight <= 0 or begin_loss_weight <= 0:
            msg = "NER loss weights must be positive"
            raise ValueError(msg)
        self.fusion_scope = fusion_scope
        self.mention_context_window = mention_context_window
        self.knowledge_key_scope = knowledge_key_scope
        self.link_score_gate = link_score_gate
        self.knowledge_control = knowledge_control
        if candidate_logit_scale <= 0:
            msg = "candidate_logit_scale must be positive"
            raise ValueError(msg)
        self.entity_loss_weight = float(entity_loss_weight)
        self.begin_loss_weight = float(begin_loss_weight)
        self.candidate_reranker = candidate_reranker
        self.candidate_logit_scale = float(candidate_logit_scale)
        self._warned_no_knowledge = False
        self.inner = self._build_inner(inner, label_list)
        self._hf = self._token_classifier()
        self._check_inner_shape()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size) if fusion_norm else None
        self.fusion_norm_active = self.attn_norm is not None
        gate_n = hidden_size if gate_shape == "vector" else 1
        if gate_shape not in ("scalar", "vector"):
            msg = f"gate_shape must be 'scalar' or 'vector', got {gate_shape!r}"
            raise ValueError(msg)
        self.gate = nn.Parameter(torch.full((gate_n,), float(gate_init)))
        self.conditional_gate = conditional_gate
        self.token_gate: nn.Sequential | None = None
        if conditional_gate:
            final = nn.Linear(hidden_size, 1)
            nn.init.normal_(final.weight, mean=0.0, std=0.02)
            nn.init.constant_(final.bias, float(conditional_gate_bias))
            gate_input_size = 3 * hidden_size + (2 if link_score_gate else 0)
            self.token_gate = nn.Sequential(
                nn.Linear(gate_input_size, hidden_size),
                nn.GELU(),
                final,
            )
        self.candidate_head: nn.Sequential | None = None
        if candidate_reranker:
            candidate_feature_size = 4 * hidden_size + 5
            candidate_output = nn.Linear(hidden_size, 2)
            nn.init.zeros_(candidate_output.weight)
            nn.init.zeros_(candidate_output.bias)
            self.candidate_head = nn.Sequential(
                nn.Linear(candidate_feature_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                candidate_output,
            )
        self.encoder = None
        if feature_dim is not None:
            from knowhyper.encoders.allset import AllSetHyperEncoder

            self.encoder = AllSetHyperEncoder(
                feature_store=None,
                feature_dim=int(feature_dim),
                hidden_size=hidden_size,
                num_layers=encoder_layers,
                num_heads=encoder_heads,
            )
        self.pretrained_encoder_path = pretrained_encoder_path
        self.pretrained_encoder_loaded_tensors = 0
        if pretrained_encoder_path is not None:
            if self.encoder is None:
                msg = "pretrained_encoder_path requires feature_dim and an encoder"
                raise ValueError(msg)
            ckpt = torch.load(pretrained_encoder_path, map_location="cpu")
            state_dict = ckpt.get("encoder", ckpt)
            if not isinstance(state_dict, dict):
                msg = "pretrained encoder checkpoint must contain a state dict"
                raise TypeError(msg)
            self.encoder.load_state_dict(state_dict, strict=True)
            self.pretrained_encoder_loaded_tensors = len(state_dict)
            logger.info(
                "strictly warm-started NER encoder from %s (%d tensors)",
                pretrained_encoder_path,
                len(state_dict),
            )

    @staticmethod
    def _build_inner(
        inner: IEModel | dict[str, Any] | DictConfig,
        label_list: list[str],
    ) -> IEModel:
        if isinstance(inner, IEModel):
            return inner
        if not isinstance(inner, (dict, DictConfig)):
            msg = "inner must be an IEModel or a Hydra config/dict with a _target_"
            raise TypeError(msg)
        inner_cfg = OmegaConf.create(inner)
        if OmegaConf.select(inner_cfg, "_target_", default=None) is None:
            msg = "inner config must include a _target_"
            raise ValueError(msg)
        built = instantiate(inner_cfg, label_list=label_list)
        if not isinstance(built, IEModel):
            msg = f"inner config instantiated {type(built).__name__}, expected IEModel"
            raise TypeError(msg)
        return built

    def _token_classifier(self) -> Any:
        hf = getattr(self.inner, "model", None)
        bert = getattr(hf, "bert", None)
        if bert is None or not hasattr(hf, "classifier"):
            msg = (
                "TokenCrossAttentionKnowledgeModel requires a BERT-family inner model "
                "with .model.bert and .model.classifier"
            )
            raise TypeError(msg)
        return hf

    def _check_inner_shape(self) -> None:
        inner_h = int(self._hf.config.hidden_size)
        if inner_h != self.hidden_size:
            msg = f"hidden_size {self.hidden_size} != inner model hidden size {inner_h}"
            raise ValueError(msg)

    def _has_knowledge(self, batch: dict[str, Any]) -> bool:
        if self.INC_FEATS_KEY in batch:
            return True
        return self.NODE_KEY in batch or self.HE_KEY in batch

    def _knowledge_row_mask(self, batch: dict[str, Any]) -> torch.Tensor:
        if self.INC_FEATS_KEY in batch:
            return batch[self.INC_NUM_NODES_KEY].bool()
        row_mask: torch.Tensor | None = None
        for key in (self.NODE_MASK_KEY, self.HE_MASK_KEY):
            value = batch.get(key)
            if value is None:
                continue
            present = value.bool().any(dim=1)
            row_mask = present if row_mask is None else row_mask | present
        if row_mask is None:
            return torch.zeros(batch["input_ids"].shape[0], dtype=torch.bool)
        return row_mask

    def forward(self, batch: dict[str, Any]) -> Any:
        hf = self._hf
        if not self._has_knowledge(batch):
            if self.require_knowledge:
                msg = "knowledge keys absent from batch but require_knowledge=True"
                raise ValueError(msg)
            if not self._warned_no_knowledge:
                logger.warning(
                    "TokenCrossAttentionKnowledgeModel: no knowledge keys in batch; "
                    "running inner (baseline) forward"
                )
                self._warned_no_knowledge = True
            return self.inner.forward(batch)

        knowledge_rows = self._knowledge_row_mask(batch)
        if self.require_knowledge_per_instance and not bool(knowledge_rows.all()):
            empty = (~knowledge_rows).nonzero(as_tuple=False).flatten().tolist()
            msg = f"knowledge-empty rows with require_knowledge_per_instance=True: {empty}"
            raise ValueError(msg)

        encoded = hf.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        hidden = encoded.last_hidden_state  # [B, L, H]

        if self.INC_FEATS_KEY in batch:
            kv, kmask, node_mention = self._encode_incidence(batch)
        else:
            kv, kmask = self._knowledge_tokens(batch)
            node_mention = None
        if self.knowledge_control == "shuffled" and kv.shape[0] > 1:
            kv = kv.roll(shifts=1, dims=0)
            kmask = kmask.roll(shifts=1, dims=0)
            if node_mention is not None:
                node_mention = node_mention.roll(shifts=1, dims=0)

        token_mentions: torch.Tensor | None = None
        if self.knowledge_key_scope in ("mention_seed", "mention_local"):
            if node_mention is None:
                msg = "mention_seed scope requires incidence mention ownership"
                raise ValueError(msg)
            token_mentions = self._token_mention_mask(
                batch, hidden.shape[1], hidden.device
            )
            attn_out = self._attend_mention_seed(
                hidden, kv, kmask, node_mention, token_mentions
            )
        else:
            attn_out = self._attend(hidden, kv, kmask)
        if self.knowledge_control == "zero":
            attn_out = torch.zeros_like(attn_out)

        fusion_mask = self._fusion_token_mask(batch, hidden.shape[1], hidden.device)
        attn_out = attn_out * fusion_mask.unsqueeze(2).to(attn_out.dtype)
        relevance: torch.Tensor | float = 1.0
        if self.token_gate is not None:
            gate_parts = [hidden, attn_out, hidden * attn_out]
            if self.link_score_gate:
                if token_mentions is None:
                    msg = "link_score_gate requires mention-token assignments"
                    raise ValueError(msg)
                score, linked = self._token_link_features(batch, token_mentions)
                gate_parts.extend([score.unsqueeze(2), linked.unsqueeze(2)])
            gate_input = torch.cat(gate_parts, dim=-1)
            relevance = torch.sigmoid(self.token_gate(gate_input))
        hidden_fused = hidden + relevance * self.gate * attn_out

        hidden_fused = hf.dropout(hidden_fused)
        logits = hf.classifier(hidden_fused)  # [B, L, num_labels]
        if self.candidate_head is not None:
            if token_mentions is None:
                token_mentions = self._token_mention_mask(
                    batch, hidden.shape[1], hidden.device
                )
            logits = self._apply_candidate_reranker(
                logits, hidden, attn_out, token_mentions, batch
            )

        loss = None
        labels = batch.get("labels")
        if labels is not None:
            loss = self._ner_loss(logits, labels)
        return _Output(logits=logits, loss=loss)

    def _fusion_token_mask(
        self, batch: dict[str, Any], seq_len: int, device: torch.device
    ) -> torch.Tensor:
        attention_mask = batch["attention_mask"].to(device=device).bool()
        if self.fusion_scope == "global":
            return attention_mask
        return self._token_mention_mask(batch, seq_len, device).any(dim=2)

    def _token_mention_mask(
        self, batch: dict[str, Any], seq_len: int, device: torch.device
    ) -> torch.Tensor:
        if self.NER_SPANS_KEY not in batch or self.NER_SPAN_MASK_KEY not in batch:
            msg = (
                "mention fusion requires NERMentionSpanChannel "
                f"({self.NER_SPANS_KEY!r} and {self.NER_SPAN_MASK_KEY!r})"
            )
            raise ValueError(msg)
        attention_mask = batch["attention_mask"].to(device=device).bool()
        offsets = torch.as_tensor(batch["offset_mapping"], device=device)
        if offsets.shape[:2] != (attention_mask.shape[0], seq_len):
            msg = f"offset_mapping shape {tuple(offsets.shape)} does not match tokens"
            raise ValueError(msg)
        spans = batch[self.NER_SPANS_KEY].to(device=device)
        span_mask = batch[self.NER_SPAN_MASK_KEY].to(device=device).bool()
        token_start = offsets[:, None, :, 0]
        token_end = offsets[:, None, :, 1]
        span_start = spans[:, :, None, 0]
        span_end = spans[:, :, None, 1]
        active = (token_start < span_end) & (token_end > span_start)
        active &= span_mask.unsqueeze(2)
        valid_token = (offsets[:, :, 0] < offsets[:, :, 1]) & attention_mask
        active &= valid_token.unsqueeze(1)
        if self.mention_context_window:
            batch_size, mention_count, _ = active.shape
            kernel = 2 * self.mention_context_window + 1
            active = (
                nn.functional.max_pool1d(
                    active.to(torch.float32).reshape(
                        batch_size * mention_count, 1, seq_len
                    ),
                    kernel_size=kernel,
                    stride=1,
                    padding=self.mention_context_window,
                )
                .reshape(batch_size, mention_count, seq_len)
                .bool()
            )
            active &= valid_token.unsqueeze(1)
        return active.transpose(1, 2)

    def _token_link_features(
        self, batch: dict[str, Any], token_mentions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for key in (self.NER_LINK_SCORES_KEY, self.NER_LINK_MASK_KEY):
            if key not in batch:
                msg = f"link_score_gate requires {key!r}"
                raise ValueError(msg)
        scores = batch[self.NER_LINK_SCORES_KEY].to(token_mentions.device).float()
        links = batch[self.NER_LINK_MASK_KEY].to(token_mentions.device).bool()
        expected = (token_mentions.shape[0], token_mentions.shape[2])
        if scores.shape != links.shape or tuple(scores.shape) != expected:
            msg = (
                f"mention link feature shapes must be {expected}, got "
                f"scores={tuple(scores.shape)}, links={tuple(links.shape)}"
            )
            raise ValueError(msg)
        active_links = token_mentions & links.unsqueeze(1)
        token_scores = (token_mentions.to(scores.dtype) * scores.unsqueeze(1)).amax(
            dim=2
        )
        return token_scores, active_links.any(dim=2).to(scores.dtype)

    def _apply_candidate_reranker(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        knowledge: torch.Tensor,
        token_mentions: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Project joint span/KG evidence back into deployable BIO token logits."""
        if self.candidate_head is None:
            return logits
        required = (
            self.NER_CONFIDENCES_KEY,
            self.NER_ENTROPIES_KEY,
            self.NER_DECODED_KEY,
            self.NER_LABELS_KEY,
            # Read a few lines below but previously absent from this tuple, so its absence
            # surfaced as a bare `KeyError` instead of the named diagnostic every sibling
            # key gets. Same #171 principle: a missing input must say what is missing.
            self.NER_SPAN_MASK_KEY,
        )
        missing = [key for key in required if key not in batch]
        if missing:
            msg = f"candidate_reranker requires candidate channel keys: {missing}"
            raise ValueError(msg)

        active = token_mentions.bool()
        candidate_mask = batch[self.NER_SPAN_MASK_KEY].to(hidden.device).bool()
        counts = active.sum(dim=1).clamp_min(1).to(hidden.dtype)
        weights = active.to(hidden.dtype)
        mean_hidden = torch.einsum("blm,blh->bmh", weights, hidden) / counts.unsqueeze(
            2
        )
        mean_knowledge = torch.einsum(
            "blm,blh->bmh", weights, knowledge
        ) / counts.unsqueeze(2)
        first = active & active.cumsum(dim=1).eq(1)
        last = active & active.flip(1).cumsum(dim=1).eq(1).flip(1)
        start_hidden = torch.einsum("blm,blh->bmh", first.to(hidden.dtype), hidden)
        end_hidden = torch.einsum("blm,blh->bmh", last.to(hidden.dtype), hidden)

        confidence = batch[self.NER_CONFIDENCES_KEY].to(hidden.device).float()
        entropy = batch[self.NER_ENTROPIES_KEY].to(hidden.device).float()
        decoded = batch[self.NER_DECODED_KEY].to(hidden.device).float()
        # #171: MISSING plumbing must not be zero-filled. A zero link score is numerically
        # identical to "linked nothing", and on the strict arms a high no-link rate is the
        # EXPECTED observation -- so the previous zero-fill let a broken pipe be reported as
        # the strict effect. The three states are now distinct:
        #
        #   MISSING plumbing -> raise (below)
        #   explicit NIL     -> keys present, right shape, all-False mask; no error
        #   linked evidence  -> keys present, mask True
        #
        # The shape check further down is what separates NIL from a malformed tensor.
        absent = [
            key
            for key in (self.NER_LINK_SCORES_KEY, self.NER_LINK_MASK_KEY)
            if batch.get(key) is None
        ]
        # `candidate_reranker=True` requires these keys UNCONDITIONALLY. There is no
        # tolerated-absence case, because an empty sub-HG is not an absent one:
        # `_mention_seed_features` allocates `scores`/`linked_mask` from `len(spans)` before
        # it reads any linked mention, so a sub-HG with zero nodes and zero hyperedges still
        # emits all five keys at the correct candidate width, explicitly all-False. Verified
        # by test: an empty sub-HG yields scores [0.0, 0.0] and mask [False, False] for two
        # candidates. That is the explicit-NIL state, and it flows through normally.
        #
        # An EARLIER VERSION of this guard gated the raise on
        # `require_knowledge or require_knowledge_per_instance`, which was wrong twice over:
        #   * it re-created the #171 ambiguity for `require_knowledge_per_instance=false` --
        #     all-empty sub-HGs, a knowledge channel that never ran, and broken plumbing all
        #     collapsed back onto the same zero tensor;
        #   * `require_knowledge_per_instance` means "every instance must have NON-EMPTY
        #     knowledge". It is not a statement about whether the provenance producer ran.
        #     BioRED sets it false to tolerate 46% empty sub-HGs, and borrowing that flag to
        #     also permit missing keys conflates emptiness with absence.
        #
        # If a future arm needs to run this head without mention provenance, that deserves
        # its own flag (`require_mention_provenance`), kept separate from the knowledge
        # non-emptiness requirement.
        if absent:
            msg = (
                f"candidate_reranker requires NER mention link features, missing: {absent}. "
                "An EMPTY sub-hypergraph is not an absent one -- the injector emits "
                "all-False scores and mask at the correct candidate width even for zero "
                "nodes, and that explicit NIL is handled normally. So absence here means "
                "mention provenance never ran, which zero-filling would make "
                "indistinguishable from a legitimate no-link -- exactly what strict linking "
                "produces on purpose. See issue #171. Note that "
                "require_knowledge_per_instance does NOT relax this: it requires non-empty "
                "knowledge, not that the provenance producer ran."
            )
            raise ValueError(msg)
        link_score = batch[self.NER_LINK_SCORES_KEY].to(hidden.device).float()
        link_mask = batch[self.NER_LINK_MASK_KEY].to(hidden.device).float()
        # The zero/shuffled controls act AFTER validation, never instead of it: a control
        # must not be able to hide missing plumbing. `zero` is a claim about the evidence
        # being neutralised, which presupposes the evidence arrived.
        if self.knowledge_control == "zero":
            mean_knowledge = torch.zeros_like(mean_knowledge)
            link_score = torch.zeros_like(link_score)
            link_mask = torch.zeros_like(link_mask)
        elif self.knowledge_control == "shuffled" and hidden.shape[0] > 1:
            link_score = link_score.roll(shifts=1, dims=0)
            link_mask = link_mask.roll(shifts=1, dims=0)

        expected = candidate_mask.shape
        for name, value in (
            ("confidence", confidence),
            ("entropy", entropy),
            ("decoded", decoded),
            ("link_score", link_score),
            ("link_mask", link_mask),
        ):
            if tuple(value.shape) != tuple(expected):
                msg = (
                    f"candidate {name} shape must be {tuple(expected)}, got "
                    f"{tuple(value.shape)}"
                )
                raise ValueError(msg)

        features = torch.cat(
            [
                start_hidden,
                end_hidden,
                mean_hidden,
                mean_knowledge,
                confidence.unsqueeze(2),
                entropy.unsqueeze(2),
                decoded.unsqueeze(2),
                link_score.unsqueeze(2),
                link_mask.unsqueeze(2),
            ],
            dim=2,
        )
        candidate_scores = self.candidate_head(features) * self.candidate_logit_scale
        candidate_scores = candidate_scores * candidate_mask.unsqueeze(2)

        labels = batch[self.NER_LABELS_KEY]
        if len(labels) != logits.shape[0]:
            msg = "candidate label rows do not match batch size"
            raise ValueError(msg)
        begin_ids = torch.zeros_like(candidate_mask, dtype=torch.long)
        inside_ids = torch.zeros_like(candidate_mask, dtype=torch.long)
        for row, row_labels in enumerate(labels):
            if len(row_labels) != candidate_mask.shape[1]:
                msg = "candidate label width does not match candidate tensors"
                raise ValueError(msg)
            for mention, entity_type in enumerate(row_labels):
                if not bool(candidate_mask[row, mention]):
                    continue
                begin_label = f"B-{entity_type}"
                inside_label = f"I-{entity_type}"
                if begin_label not in self.label_list:
                    msg = f"candidate type has no BIO label: {entity_type!r}"
                    raise ValueError(msg)
                begin_ids[row, mention] = self.label_list.index(begin_label)
                inside_ids[row, mention] = (
                    self.label_list.index(inside_label)
                    if inside_label in self.label_list
                    else begin_ids[row, mention]
                )

        # Aggregate overlapping candidates without in-place writes so gradients
        # reach the boundary/interior scores. Negative evidence is retained.
        negative_infinity = torch.tensor(
            float("-inf"), dtype=logits.dtype, device=logits.device
        )
        per_label_bias: list[torch.Tensor] = []
        first_scores = candidate_scores[:, None, :, 0]
        inside_scores = candidate_scores[:, None, :, 1]
        continuation = active & ~first
        for label_id in range(len(self.label_list)):
            begin_active = (
                first & candidate_mask[:, None, :] & begin_ids[:, None, :].eq(label_id)
            )
            inside_active = (
                continuation
                & candidate_mask[:, None, :]
                & inside_ids[:, None, :].eq(label_id)
            )
            begin_bias = torch.where(
                begin_active, first_scores, negative_infinity
            ).amax(dim=2)
            inside_bias = torch.where(
                inside_active, inside_scores, negative_infinity
            ).amax(dim=2)
            label_bias = torch.maximum(begin_bias, inside_bias)
            per_label_bias.append(
                torch.where(torch.isfinite(label_bias), label_bias, 0.0)
            )
        bias = torch.stack(per_label_bias, dim=2)
        return logits + bias

    def _ner_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.entity_loss_weight == 1.0 and self.begin_loss_weight == 1.0:
            return nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
            )
        flat_logits = logits.view(-1, logits.shape[-1])
        flat_labels = labels.view(-1)
        per_token = nn.functional.cross_entropy(
            flat_logits, flat_labels, ignore_index=-100, reduction="none"
        )
        valid = flat_labels != -100
        weights = torch.ones_like(per_token)
        o_index = self.label_list.index("O")
        entity = valid & (flat_labels != o_index)
        weights[entity] *= self.entity_loss_weight
        begin_ids = {
            index
            for index, label in enumerate(self.label_list)
            if label.startswith("B-")
        }
        if begin_ids:
            begin = torch.zeros_like(valid)
            for label_id in begin_ids:
                begin |= flat_labels == label_id
            weights[begin] *= self.begin_loss_weight
        return (per_token[valid] * weights[valid]).sum() / weights[valid].sum()

    def _encode_incidence(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.encoder is None:
            msg = (
                "incidence keys present but this model was built without "
                "feature_dim (no encoder); set model.feature_dim"
            )
            raise ValueError(msg)
        from knowhyper.encoders.allset import Incidence

        dev = self.encoder.proj.weight.device
        feats = batch[self.INC_FEATS_KEY].to(dev)
        node_missing = batch[self.INC_NODE_MISSING_KEY].to(dev)
        node_idx = batch[self.INC_NODE_IDX_KEY].to(dev)
        edge_idx = batch[self.INC_EDGE_IDX_KEY].to(dev)
        role_id = batch[self.INC_ROLE_KEY].to(dev)
        sem_id = batch[self.INC_SEM_KEY].to(dev)
        pol_id = batch[self.INC_POL_KEY].to(dev)
        num_nodes = [int(x) for x in batch[self.INC_NUM_NODES_KEY].tolist()]
        num_edges = [int(x) for x in batch[self.INC_NUM_EDGES_KEY].tolist()]

        incidences: list[Incidence] = []
        node_off = edge_off = m_off = 0
        for g in range(len(num_nodes)):
            nn_, ee_ = num_nodes[g], num_edges[g]
            hi = node_off + nn_
            m_end = m_off
            while m_end < node_idx.shape[0] and int(node_idx[m_end]) < hi:
                m_end += 1
            sl = slice(m_off, m_end)
            incidences.append(
                Incidence(
                    node_feats=feats[node_off : node_off + nn_],
                    node_missing=node_missing[node_off : node_off + nn_],
                    num_nodes=nn_,
                    num_edges=ee_,
                    node_idx=node_idx[sl] - node_off,
                    edge_idx=edge_idx[sl] - edge_off,
                    role_id=role_id[sl],
                    sem_id=sem_id[edge_off : edge_off + ee_],
                    pol_id=pol_id[edge_off : edge_off + ee_],
                )
            )
            node_off = hi
            edge_off += ee_
            m_off = m_end

        out = self.encoder.encode_batch(incidences)
        ref_device = batch["input_ids"].device
        ref_dtype = feats.dtype
        node_e = out["node_embeds"].to(device=ref_device, dtype=ref_dtype)
        node_m = out["node_mask"].to(device=ref_device, dtype=ref_dtype)
        he_e = out["hyperedge_embeds"].to(device=ref_device, dtype=ref_dtype)
        he_m = out["hyperedge_mask"].to(device=ref_device, dtype=ref_dtype)
        if self.knowledge_key_scope == "all":
            kv = torch.cat([node_e, he_e], dim=1)
            kmask = torch.cat([node_m, he_m], dim=1)
            return kv, kmask, None

        def pad_ownership(
            key: str, counts: list[int], padded_width: int
        ) -> torch.Tensor:
            if key not in batch:
                msg = f"{self.knowledge_key_scope} scope requires {key!r}"
                raise ValueError(msg)
            flat = batch[key].to(ref_device).bool()
            if flat.shape[0] != sum(counts):
                msg = (
                    f"mention ownership rows for {key!r} do not match incidence: "
                    f"{flat.shape[0]} != {sum(counts)}"
                )
                raise ValueError(msg)
            padded = torch.zeros(
                len(counts),
                padded_width,
                flat.shape[1],
                dtype=torch.bool,
                device=ref_device,
            )
            offset = 0
            for graph, count in enumerate(counts):
                if count:
                    padded[graph, :count] = flat[offset : offset + count]
                offset += count
            return padded

        if self.knowledge_key_scope == "mention_seed":
            ownership = pad_ownership(
                self.INC_NODE_MENTION_KEY, num_nodes, node_e.shape[1]
            )
            return node_e, node_m, ownership
        node_ownership = pad_ownership(
            self.INC_LOCAL_NODE_MENTION_KEY, num_nodes, node_e.shape[1]
        )
        hyperedge_ownership = pad_ownership(
            self.INC_HE_MENTION_KEY, num_edges, he_e.shape[1]
        )
        return (
            torch.cat([node_e, he_e], dim=1),
            torch.cat([node_m, he_m], dim=1),
            torch.cat([node_ownership, hyperedge_ownership], dim=1),
        )

    def _knowledge_tokens(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parts: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        ref_device = batch["input_ids"].device
        for ekey, mkey in (
            (self.NODE_KEY, self.NODE_MASK_KEY),
            (self.HE_KEY, self.HE_MASK_KEY),
        ):
            emb = batch.get(ekey)
            if emb is None:
                continue
            mask = batch.get(mkey)
            if mask is None:
                msg = f"{ekey!r} present but mask {mkey!r} missing"
                raise ValueError(msg)
            emb = emb.to(device=ref_device)
            mask = mask.to(device=ref_device)
            if emb.shape[1] != mask.shape[1]:
                msg = (
                    f"knowledge embed {ekey!r} token count {emb.shape[1]} != "
                    f"mask {mkey!r} token count {mask.shape[1]}"
                )
                raise ValueError(msg)
            parts.append(emb)
            masks.append(mask)
        return torch.cat(parts, dim=1), torch.cat(masks, dim=1)

    def _attend_mention_seed(
        self,
        hidden: torch.Tensor,
        kv: torch.Tensor,
        kmask: torch.Tensor,
        node_mention: torch.Tensor,
        token_mentions: torch.Tensor,
    ) -> torch.Tensor:
        if node_mention.shape[:2] != kv.shape[:2]:
            msg = (
                "node mention ownership must align with knowledge tokens: "
                f"{tuple(node_mention.shape)} vs {tuple(kv.shape)}"
            )
            raise ValueError(msg)
        if node_mention.shape[2] != token_mentions.shape[2]:
            msg = "node and token mention dimensions do not match"
            raise ValueError(msg)
        allowed = torch.einsum(
            "blm,bnm->bln",
            token_mentions.to(kv.dtype),
            node_mention.to(kv.dtype),
        ).bool()
        allowed &= kmask.bool().unsqueeze(1)
        has_real_key = allowed.any(dim=2)
        dummy = torch.zeros(
            kv.shape[0], 1, kv.shape[2], dtype=kv.dtype, device=kv.device
        )
        kv_with_dummy = torch.cat([kv, dummy], dim=1)
        allowed = torch.cat([allowed, (~has_real_key).unsqueeze(2)], dim=2)
        attn_mask = ~allowed.unsqueeze(1).expand(
            -1, self.cross_attn.num_heads, -1, -1
        ).reshape(
            kv.shape[0] * self.cross_attn.num_heads,
            hidden.shape[1],
            kv_with_dummy.shape[1],
        )
        attn_out, _ = self.cross_attn(
            query=hidden,
            key=kv_with_dummy,
            value=kv_with_dummy,
            attn_mask=attn_mask,
            need_weights=False,
        )
        if self.attn_norm is not None:
            attn_out = self.attn_norm(attn_out)
        return attn_out * has_real_key.unsqueeze(2).to(attn_out.dtype)

    def _attend(
        self, hidden: torch.Tensor, kv: torch.Tensor, kmask: torch.Tensor
    ) -> torch.Tensor:
        valid = kmask.bool()  # [B, Nk]
        any_valid = valid.any(dim=1)  # [B]
        key_padding = ~valid
        key_padding = key_padding & any_valid.unsqueeze(1)
        attn_out, _ = self.cross_attn(
            query=hidden,  # [B, L, H] — full sequence, not just CLS
            key=kv,
            value=kv,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        if self.attn_norm is not None:
            attn_out = self.attn_norm(attn_out)
        # Zero out rows where no knowledge token was valid
        return attn_out * any_valid.unsqueeze(1).unsqueeze(2).to(attn_out.dtype)


@register("model", name="knowhyper_parallel_text_gated")
class ParallelTextGatedKnowledgeModel(IEModel):
    """GRAFT-style baseline: parallel text encoding of retrieved evidence + learned gate.

    Answers the #189 question -- given the SAME retrieved PrimeHKG evidence, is
    parallel text encoding with a learned gate enough, or is learned
    node/hyperedge encoding needed? The comparison arm is ``edge_top3_matched``,
    which runs :class:`CrossAttentionKnowledgeModel` over the same 3 hyperedges.

    Deliberately **not** a variant of :class:`CrossAttentionKnowledgeModel`. That
    model fuses graph tokens into ``[CLS]`` and produces one logit vector; GRAFT
    mixes *two* logit vectors -- source-only and retrieval-conditioned -- through a
    content-sensitive gate. Bolting a second mode onto the fusion model would have
    made the two arms share a code path whose behaviour differs by config, which is
    the failure mode this project has repeatedly paid for.

    Shape:

    #. ``inner`` (an :class:`HFSequenceClassifier`) encodes the source and its
       ``[CLS]`` row goes through the inner pooler + classifier -> ``source_logits``.
    #. ``evidence_encoder`` -- an INDEPENDENT full encoder from the same initial
       checkpoint -- encodes the evidence sequence. Independent, not shared,
       because official GRAFT carries two full encoders; sharing would change the
       capacity being compared. The resource cost is measured and disclosed rather
       than hidden (Phase 5).
    #. The source ``[CLS]`` cross-attends over the evidence token states, the result
       is added to ``[CLS]``, and the SAME pooler + classifier produce
       ``conditioned_logits``. Sharing the head is load-bearing: a second classifier
       would give the conditioned path extra capacity and ``real - masked`` would
       partly measure that.
    #. ``g = sigmoid(MLP([cls, pooled_evidence, cls * pooled_evidence]))`` and
       ``logits = (1 - g) * source_logits + g * conditioned_logits``.

    The gate reads pooled EVIDENCE, not just the source: a source-only gate could
    not distinguish real evidence from ``[MASK]`` evidence, so it would learn a
    per-instance prior and ``real - masked`` would be structurally zero -- the
    experiment would return a null it manufactured.

    Rows with ``knowledge_evidence_present=False`` get ``g=0`` exactly, so their
    logits equal ``source_logits`` bit-for-bit. Combined with the collator leaving
    the source untouched, that makes the no-evidence path exactly the ``base`` arm.

    **Use ``trainer=nolog``, never ``difflr_nolog``.**
    :data:`~knowhyper.ieforge.lr_groups.BACKBONE_PREFIX` is ``"model.inner."``, so
    :class:`~knowhyper.ieforge.lr_groups.KnowledgeLRCallback` would classify this
    model's entire second BERT as "knowledge branch" and train 110M pretrained
    parameters at 10x the backbone LR. That does not error -- it just destroys the
    evidence encoder while reporting a plausible score.
    """

    EVIDENCE_IDS_KEY = "knowledge_evidence_input_ids"
    EVIDENCE_MASK_KEY = "knowledge_evidence_attention_mask"
    EVIDENCE_PRESENT_KEY = "knowledge_evidence_present"

    def __init__(
        self,
        inner: IEModel | dict[str, Any] | DictConfig,
        label_list: list[str],
        evidence_model_name: str | None = None,
        hidden_size: int = 768,
        num_heads: int = 8,
        extra_token_count: int = 0,
        name: str = "knowhyper_parallel_text_gated",
        gate_bias_init: float = 0.0,
        fusion_norm: bool = True,
        **kwargs: Any,  # tolerate config-level keys (e.g. tokenizer_name) read by CLI
    ) -> None:
        """Build the source classifier, the independent evidence encoder, and the gate.

        ``evidence_model_name`` defaults to the inner model's own checkpoint, which
        is what keeps the two encoders identically initialised. Passing a different
        checkpoint is allowed but changes what the arm measures, so it must be
        explicit rather than inferred.

        ``gate_bias_init=0.0`` starts the gate at ``sigmoid(0)=0.5``, i.e. an even
        mix of the source-only and conditioned logits. Unlike the fusion model's
        ``gate_init``, 0.0 here is NOT a knowledge-off setting -- the conditioned
        path still contributes half. There is deliberately no value of this
        parameter that silently disables the evidence branch; ``present=False`` is
        the only path to ``g=0``.
        """
        super().__init__()
        del kwargs
        self.name = name
        self.label_list = label_list
        self.hidden_size = hidden_size
        # Reused rather than re-implemented: this is the third model needing the
        # exact same "IEModel or Hydra config with a _target_" handling, and the two
        # existing copies (cross-attention, token-cross-attention) already drifted in
        # their error messages. A fourth copy would drift again.
        self.inner = CrossAttentionKnowledgeModel._build_inner(  # noqa: SLF001
            inner, label_list, extra_token_count
        )
        self._hf = self._source_bert()
        inner_hidden = int(self._hf.config.hidden_size)
        if inner_hidden != hidden_size:
            msg = f"hidden_size {hidden_size} != inner model hidden size {inner_hidden}"
            raise ValueError(msg)
        self.evidence_model_name = evidence_model_name or self._inner_checkpoint()
        self.evidence_encoder = self._build_evidence_encoder(
            self.evidence_model_name, extra_token_count
        )
        evidence_hidden = int(self.evidence_encoder.config.hidden_size)
        if evidence_hidden != hidden_size:
            msg = (
                f"evidence encoder hidden size {evidence_hidden} != {hidden_size}; "
                "the cross-attention requires one shared width"
            )
            raise ValueError(msg)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        # Same rationale as the fusion model's fusion_norm: an un-normed
        # cross-attention output grows unboundedly under a high LR and swamps CLS.
        # Default ON here because this arm has no gate_init=0 escape hatch.
        self.attn_norm = nn.LayerNorm(hidden_size) if fusion_norm else None
        gate_final = nn.Linear(hidden_size, 1)
        # Small random weights, not zeros: with zero weights the gate receives
        # ~no gradient early and stays a constant, which is exactly the
        # "gate collapses to a prior" outcome that would make real == masked by
        # construction. Observed on the fusion model's gate_mlp_init="zeros".
        nn.init.normal_(gate_final.weight, std=0.02)
        nn.init.constant_(gate_final.bias, float(gate_bias_init))
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            gate_final,
        )

    def _source_bert(self) -> Any:
        hf = getattr(self.inner, "model", None)
        bert = getattr(hf, "bert", None)
        pooler = getattr(bert, "pooler", None)
        if bert is None or pooler is None or not hasattr(hf, "classifier"):
            msg = (
                "ParallelTextGatedKnowledgeModel requires a BERT-family inner model "
                "with .model.bert(.pooler) and .model.classifier"
            )
            raise TypeError(msg)
        return hf

    def _inner_checkpoint(self) -> str:
        """Read the inner encoder's own checkpoint id so both encoders match.

        Taken from the loaded config rather than from a config key so a backbone
        swap (PubMedBERT -> BioLinkBERT) cannot leave the evidence encoder on the
        old checkpoint -- a mismatch that trains fine and reports a plausible score.
        """
        name = getattr(self._hf.config, "_name_or_path", "")
        if not name:
            msg = (
                "cannot infer the evidence encoder checkpoint from the inner model; "
                "pass evidence_model_name explicitly"
            )
            raise ValueError(msg)
        return str(name)

    @staticmethod
    def _build_evidence_encoder(model_name: str, extra_token_count: int) -> Any:
        """Load a second full encoder, head-free, from the same checkpoint."""
        from transformers import AutoModel, BertConfig, BertModel

        try:
            encoder = AutoModel.from_pretrained(model_name)
        except ValueError:
            # Mirrors HFSequenceClassifier's own fallback: some checkpoints ship no
            # `model_type` in config.json (prajjwal1/bert-tiny, which every unit test
            # here uses), and AutoModel cannot dispatch without it. Without this the
            # arm would be untestable offline while working fine on PubMedBERT --
            # i.e. the tests would be skipped rather than the model verified.
            config = BertConfig.from_pretrained(model_name)
            encoder = BertModel.from_pretrained(model_name, config=config)
        if extra_token_count < 0:
            msg = f"extra_token_count must be non-negative, got {extra_token_count}"
            raise ValueError(msg)
        if extra_token_count > 0:
            # The evidence sequence carries the same added marker tokens as the
            # source, so its embedding table must be resized identically or those
            # ids index out of range.
            embeddings = encoder.get_input_embeddings()
            encoder.resize_token_embeddings(
                int(embeddings.num_embeddings) + extra_token_count
            )
        return encoder

    def forward(self, batch: dict[str, Any]) -> Any:
        """Mix source-only and evidence-conditioned logits through the learned gate."""
        if self.EVIDENCE_IDS_KEY not in batch:
            # No evidence branch in the batch at all: this is a misconfiguration
            # (the arm's own collator always emits the key, even for empty
            # evidence), so fail loud rather than silently running the baseline
            # and reporting it as a knowledge score.
            msg = (
                f"{self.EVIDENCE_IDS_KEY} absent from batch; "
                "ParallelTextGatedKnowledgeModel requires the parallel evidence "
                "collator (a missing key would silently train the baseline)"
            )
            raise ValueError(msg)

        hf = self._hf
        encoded = hf.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        cls = encoded.last_hidden_state[:, 0]  # [B, H]
        source_logits = self._classify(cls)

        evidence_mask = batch[self.EVIDENCE_MASK_KEY]
        evidence = self.evidence_encoder(
            input_ids=batch[self.EVIDENCE_IDS_KEY],
            attention_mask=evidence_mask,
        )
        evidence_states = evidence.last_hidden_state  # [B, Le, H]

        attn_out = self._attend(cls, evidence_states, evidence_mask)
        conditioned_logits = self._classify(cls + attn_out)

        gate = self._gate(cls, evidence_states, evidence_mask)
        present = self._present(batch, evidence_mask)
        # Hard zero on absent evidence, so those rows' logits are source_logits
        # bit-for-bit. Multiplying rather than branching keeps the graph shape
        # constant, and `attn_out` is already zeroed for the same rows.
        gate = gate * present.unsqueeze(1).to(gate.dtype)
        logits = (1.0 - gate) * source_logits + gate * conditioned_logits

        loss = None
        labels = batch.get("labels")
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return _GatedOutput(
            logits=logits,
            loss=loss,
            source_logits=source_logits,
            conditioned_logits=conditioned_logits,
            knowledge_gate=gate,
        )

    def _classify(self, cls: torch.Tensor) -> torch.Tensor:
        """Pool and classify a [CLS] row through the INNER model's own head.

        Shared between the source-only and conditioned paths on purpose: a second
        head would hand the conditioned path extra capacity, so part of any
        `real - masked` delta would be head size rather than evidence content.
        """
        hf = self._hf
        pooled = hf.bert.pooler(cls.unsqueeze(1))  # BertPooler reads [:, 0]
        pooled = hf.dropout(pooled) if hasattr(hf, "dropout") else pooled
        return cast("torch.Tensor", hf.classifier(pooled))

    def _present(self, batch: dict[str, Any], evidence_mask: torch.Tensor) -> Any:
        """Per-row evidence presence, defaulting to the mask when the flag is absent.

        The flag is authoritative because a `[CLS] [SEP]` row has a valid mask but
        carries no evidence. Falling back to the mask keeps hand-built test batches
        usable without letting a real run depend on the fallback.
        """
        present = batch.get(self.EVIDENCE_PRESENT_KEY)
        if present is None:
            return evidence_mask.sum(dim=1) > 2
        return present.bool()

    def _gate(
        self, cls: torch.Tensor, states: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Content-sensitive gate g in (0, 1), shape [B, 1].

        Reads pooled evidence and its interaction with the source. A gate computed
        from `cls` alone would be blind to whether the evidence is real or masked,
        which would make `real - masked` zero by construction.
        """
        weights = mask.to(states.dtype).unsqueeze(-1)  # [B, Le, 1]
        denom = weights.sum(dim=1).clamp(min=1.0)  # [B, 1]
        pooled = (states * weights).sum(dim=1) / denom  # [B, H]
        feats = torch.cat([cls, pooled, cls * pooled], dim=-1)  # [B, 3H]
        return torch.sigmoid(self.gate_mlp(feats))  # [B, 1]

    def _attend(
        self, cls: torch.Tensor, states: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Cross-attend the source [CLS] over evidence token states -> [B, H]."""
        valid = mask.bool()
        any_valid = valid.any(dim=1)
        # An all-masked row would softmax over nothing and produce NaN. Unmask
        # those rows so attention is defined, then zero their output below.
        key_padding = (~valid) & any_valid.unsqueeze(1)
        attn_out, _ = self.cross_attn(
            query=cls.unsqueeze(1),
            key=states,
            value=states,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        attn_out = attn_out.squeeze(1)  # [B, H]
        if self.attn_norm is not None:
            attn_out = self.attn_norm(attn_out)
        # Zero AFTER the norm: LayerNorm's beta would otherwise make a zero row
        # nonzero and break the absent-evidence guarantee.
        return attn_out * any_valid.unsqueeze(1).to(attn_out.dtype)


class _GatedOutput(_Output):
    """``_Output`` plus the GRAFT-style diagnostics Phase 5 exports.

    ``source_logits`` / ``conditioned_logits`` / ``knowledge_gate`` are carried on
    the output rather than logged, so ieforge's training loop (which reads only
    ``.logits`` / ``.loss``) is unaffected while the export script can recover what
    the gate actually did per instance. A gate that collapsed to a constant is the
    main way this arm can produce a null for an uninteresting reason.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        loss: torch.Tensor | None,
        *,
        source_logits: torch.Tensor,
        conditioned_logits: torch.Tensor,
        knowledge_gate: torch.Tensor,
    ) -> None:
        super().__init__(logits=logits, loss=loss)
        self.source_logits = source_logits
        self.conditioned_logits = conditioned_logits
        self.knowledge_gate = knowledge_gate
