"""Span-level KnowHyper fusion for ieforge's span NER model."""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
from hydra.utils import instantiate
from ieforge.models.base import IEModel
from ieforge.models.hf_span_classifier import SpanClassifierOutput
from ieforge.registry import register
from omegaconf import DictConfig, OmegaConf
from torch import nn

from knowhyper.ieforge.models import TokenCrossAttentionKnowledgeModel

logger = logging.getLogger(__name__)


class SpanKnowledgeOutput(SpanClassifierOutput):
    """``SpanClassifierOutput`` plus the #218 Phase 2 diagnostics.

    Declared as a subclass rather than set as an ad-hoc attribute so the field is
    typed, mirroring `_GatedOutput` in models.py. ieforge's training loop reads only
    ``.logits``/``.loss``, so the extra field is inert there while an export callback
    can recover what the knowledge branch did per candidate.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        *,
        knowledge_diagnostics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logits=logits, **kwargs)
        self.knowledge_diagnostics = knowledge_diagnostics


@register("model", name="knowhyper_span_cross_attention")
class SpanCrossAttentionKnowledgeModel(TokenCrossAttentionKnowledgeModel):
    """Verify span candidates against mention-specific hypergraph evidence.

    ieforge 0.1.11 computes the candidate geometry and base span representation.
    This wrapper consumes its public span-feature contract, lets each span attend
    only to the graph tokens owned by the matching first-pass candidate, and adds
    the resulting evidence as a gated residual before re-scoring it through the
    public span scorer.

    The initial implementation deliberately supports one entity type per
    dataset.  All three KnowHyper NER datasets satisfy that constraint.  Failing
    closed avoids accidentally sharing one candidate's evidence across types if
    a multi-type dataset is added later.
    """

    def __init__(
        self,
        inner: IEModel | dict[str, Any] | DictConfig,
        label_list: list[str],
        hidden_size: int = 768,
        num_heads: int = 8,
        name: str = "knowhyper_span_cross_attention",
        require_knowledge: bool = True,
        require_knowledge_per_instance: bool = False,
        feature_dim: int | None = 768,
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        gate_init: float = 1.0,
        gate_shape: str = "vector",
        fusion_norm: bool = True,
        conditional_gate: bool = True,
        conditional_gate_bias: float = -2.0,
        knowledge_control: str = "real",
        pretrained_encoder_path: str | None = None,
        tokenizer_name: str | None = None,
        extra_token_count: int = 0,
    ) -> None:
        # TokenCrossAttentionKnowledgeModel.__init__ assumes a BIO token
        # classifier, so initialise IEModel directly and reuse only its
        # incidence/ownership helpers.
        IEModel.__init__(self)
        if extra_token_count != 0:
            msg = (
                "SpanCrossAttentionKnowledgeModel does not support prompt extra "
                f"tokens; got extra_token_count={extra_token_count}"
            )
            raise ValueError(msg)
        if knowledge_control not in ("real", "shuffled", "zero"):
            msg = (
                "knowledge_control must be 'real', 'shuffled', or 'zero', "
                f"got {knowledge_control!r}"
            )
            raise ValueError(msg)
        if gate_shape not in ("scalar", "vector"):
            msg = f"gate_shape must be 'scalar' or 'vector', got {gate_shape!r}"
            raise ValueError(msg)

        self.name = name
        self.tokenizer_name = tokenizer_name
        self.extra_token_count = extra_token_count
        self.label_list = list(label_list)
        if len(self.label_list) != 1:
            msg = (
                "SpanCrossAttentionKnowledgeModel currently requires exactly one "
                f"entity type, got {self.label_list!r}"
            )
            raise ValueError(msg)
        self.hidden_size = hidden_size
        self.require_knowledge = require_knowledge
        self.require_knowledge_per_instance = require_knowledge_per_instance
        self.knowledge_control = knowledge_control
        self.knowledge_key_scope = "mention_local"
        self._warned_no_knowledge = False

        self.inner = self._build_span_inner(inner, self.label_list)
        self._check_span_contract()
        span_hidden_dim = int(cast("Any", self.inner).span_hidden_dim)
        if span_hidden_dim != hidden_size:
            msg = (
                f"hidden_size {hidden_size} != inner span hidden size {span_hidden_dim}"
            )
            raise ValueError(msg)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size) if fusion_norm else None
        self.fusion_norm_active = self.attn_norm is not None
        gate_n = hidden_size if gate_shape == "vector" else 1
        self.gate = nn.Parameter(torch.full((gate_n,), float(gate_init)))

        self.conditional_gate = conditional_gate
        self.span_gate: nn.Sequential | None = None
        if conditional_gate:
            final = nn.Linear(hidden_size, 1)
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, float(conditional_gate_bias))
            self.span_gate = nn.Sequential(
                nn.Linear(3 * hidden_size + 2, hidden_size),
                nn.GELU(),
                final,
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
                "strictly warm-started span NER encoder from %s (%d tensors)",
                pretrained_encoder_path,
                len(state_dict),
            )

    @staticmethod
    def _build_span_inner(
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

    def _check_span_contract(self) -> None:
        required = (
            "span_hidden_dim",
            "score_span_representations",
        )
        missing = [name for name in required if not hasattr(self.inner, name)]
        if missing:
            msg = (
                "ieforge span model does not expose the v0.1.11 public contract: "
                f"missing {missing}"
            )
            raise TypeError(msg)

    def _base_forward_with_span_features(
        self, batch: dict[str, Any]
    ) -> tuple[SpanClassifierOutput, torch.Tensor, torch.Tensor, torch.Tensor]:
        base_output = cast("Any", self.inner).forward(batch, return_span_features=True)
        if not isinstance(base_output, SpanClassifierOutput):
            msg = "inner span model must return SpanClassifierOutput"
            raise TypeError(msg)
        span_rep = base_output.span_representations
        span_indices = base_output.span_indices
        span_mask = base_output.span_mask
        if span_rep is None or span_indices is None or span_mask is None:
            msg = (
                "ieforge span model did not populate its v0.1.11 public features "
                "with return_span_features=True"
            )
            raise TypeError(msg)
        if span_rep.ndim != 4:
            msg = f"span representations must be [B, W, K, D], got {span_rep.shape}"
            raise ValueError(msg)
        expected_indices = (*span_rep.shape[:3], 2)
        if tuple(span_indices.shape) != expected_indices:
            msg = (
                "span indices do not align with representations: "
                f"indices={tuple(span_indices.shape)}, rep={tuple(span_rep.shape)}"
            )
            raise ValueError(msg)
        if tuple(span_mask.shape) != tuple(span_rep.shape[:3]):
            msg = (
                "span mask does not align with representations: "
                f"mask={tuple(span_mask.shape)}, rep={tuple(span_rep.shape)}"
            )
            raise ValueError(msg)
        expected_logits = (*span_rep.shape[:3], len(self.label_list))
        if tuple(base_output.logits.shape) != expected_logits:
            msg = (
                "span logits do not align with representations/labels: "
                f"logits={tuple(base_output.logits.shape)}, expected={expected_logits}"
            )
            raise ValueError(msg)
        return base_output, span_rep, span_indices, span_mask

    def forward(self, batch: dict[str, Any]) -> SpanClassifierOutput:
        base_output, span_rep_4d, span_indices, span_mask = (
            self._base_forward_with_span_features(batch)
        )
        if not self._has_knowledge(batch):
            if self.require_knowledge:
                msg = "knowledge keys absent from batch but require_knowledge=True"
                raise ValueError(msg)
            if not self._warned_no_knowledge:
                logger.warning(
                    "SpanCrossAttentionKnowledgeModel: no knowledge keys in batch; "
                    "returning the inner span baseline"
                )
                self._warned_no_knowledge = True
            return base_output

        knowledge_rows = self._knowledge_row_mask(batch)
        if self.require_knowledge_per_instance and not bool(knowledge_rows.all()):
            empty = (~knowledge_rows).nonzero(as_tuple=False).flatten().tolist()
            msg = f"knowledge-empty rows with require_knowledge_per_instance=True: {empty}"
            raise ValueError(msg)

        kv, kmask, node_mention = self._encode_incidence(batch)
        if node_mention is None:
            msg = "span knowledge verification requires mention-local ownership"
            raise ValueError(msg)
        if self.knowledge_control == "shuffled" and kv.shape[0] > 1:
            kv = kv.roll(shifts=1, dims=0)
            kmask = kmask.roll(shifts=1, dims=0)
            node_mention = node_mention.roll(shifts=1, dims=0)

        batch_size, max_words, max_span_width, span_dim = span_rep_4d.shape
        expected_spans = max_words * max_span_width
        if span_dim != self.hidden_size:
            msg = (
                "span representation does not align with the knowledge model: "
                f"rep={tuple(span_rep_4d.shape)}, hidden_size={self.hidden_size}"
            )
            raise ValueError(msg)
        span_rep = span_rep_4d.reshape(batch_size, expected_spans, span_dim)

        span_to_mention = self._candidate_to_mention(
            batch,
            span_indices=span_indices,
            span_mask=span_mask,
            mention_width=node_mention.shape[2],
            device=span_rep.device,
        )
        attn_out, has_evidence = self._attend_candidate_evidence(
            span_rep,
            kv,
            kmask,
            node_mention,
            span_to_mention,
        )

        link_score, link_mask = self._candidate_link_features(
            batch, span_to_mention, span_rep.dtype
        )
        if self.knowledge_control == "zero":
            # The zero arm IS the span baseline (it returns before the gate), but it
            # still has to be readable in the Phase 2 record -- a control with no
            # diagnostics cannot be compared against the real arm, and a missing
            # entry would read as "not measured" rather than "measured, exactly zero".
            return SpanKnowledgeOutput(
                logits=base_output.logits,
                knowledge_diagnostics=self._zero_diagnostics(
                    base_output.logits, has_evidence, link_mask
                ),
            )

        relevance: torch.Tensor | float = 1.0
        if self.span_gate is not None:
            gate_input = torch.cat(
                [
                    span_rep,
                    attn_out,
                    span_rep * attn_out,
                    link_score.unsqueeze(2),
                    link_mask.unsqueeze(2),
                ],
                dim=2,
            )
            relevance = torch.sigmoid(self.span_gate(gate_input))
        fused = span_rep + relevance * self.gate * attn_out
        fused = fused.reshape(batch_size, max_words, max_span_width, span_dim)
        logits = cast("Any", self.inner).score_span_representations(fused)
        if tuple(logits.shape) != tuple(base_output.logits.shape):
            msg = (
                "public span scorer returned logits with an incompatible shape: "
                f"fused={tuple(logits.shape)}, base={tuple(base_output.logits.shape)}"
            )
            raise ValueError(msg)
        return SpanKnowledgeOutput(
            logits=logits,
            knowledge_diagnostics=self._diagnostics(
                base_logits=base_output.logits,
                fused_logits=logits,
                span_rep=span_rep,
                fused=fused.reshape(batch_size, expected_spans, span_dim),
                attn_out=attn_out,
                relevance=relevance,
                has_evidence=has_evidence,
                link_mask=link_mask,
            ),
        )

    # ------------------------------------------------------------------------
    # Phase 2 diagnostics (#218). Carried ON THE OUTPUT rather than logged, the
    # same way `_GatedOutput` does it for the GRAFT arms: ieforge's training loop
    # reads only `.logits`/`.loss`, so extra attributes are inert there while an
    # export script can recover what the branch actually did per candidate.
    #
    # Every tensor is DETACHED. Keeping them live would retain the graph for the
    # whole batch and change the memory profile of a run whose VRAM headroom is
    # itself one of the things Phase 2 measures.
    # ------------------------------------------------------------------------

    def _diagnostics(
        self,
        *,
        base_logits: torch.Tensor,
        fused_logits: torch.Tensor,
        span_rep: torch.Tensor,
        fused: torch.Tensor,
        attn_out: torch.Tensor,
        relevance: torch.Tensor | float,
        has_evidence: torch.Tensor,
        link_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Per-candidate record of what the knowledge branch did."""
        gate_relevance = None
        if isinstance(relevance, torch.Tensor):
            # [B, S, 1] -> [B, S]: one sigmoid per candidate span.
            gate_relevance = relevance.detach().squeeze(-1)
        return {
            "control": self.knowledge_control,
            "base_logits": base_logits.detach(),
            "fused_logits": fused_logits.detach(),
            # `||attn_out||` and `||fused - base||`: a residual that never moves and
            # an attention output that is always zero are different failures, and
            # aggregate F1 cannot tell them apart.
            "attn_norm": attn_out.detach().norm(dim=-1),
            "residual_norm": (fused - span_rep).detach().norm(dim=-1),
            "logit_delta": (fused_logits - base_logits).detach(),
            "gate_relevance": gate_relevance,
            # Two different questions: did the owning mention LINK, and did the
            # candidate actually receive attention keys.
            "has_evidence": has_evidence.detach(),
            "link_mask": link_mask.detach().bool(),
        }

    def _zero_diagnostics(
        self,
        base_logits: torch.Tensor,
        has_evidence: torch.Tensor,
        link_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Diagnostics for the zero arm: measured, and exactly zero."""
        zeros = torch.zeros_like(has_evidence, dtype=base_logits.dtype)
        return {
            "control": self.knowledge_control,
            "base_logits": base_logits.detach(),
            "fused_logits": base_logits.detach(),
            "attn_norm": zeros,
            "residual_norm": zeros,
            "logit_delta": torch.zeros_like(base_logits),
            "gate_relevance": None,
            "has_evidence": has_evidence.detach(),
            "link_mask": link_mask.detach().bool(),
        }

    def _candidate_to_mention(
        self,
        batch: dict[str, Any],
        *,
        span_indices: torch.Tensor,
        span_mask: torch.Tensor,
        mention_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        for key in (
            self.NER_SPANS_KEY,
            self.NER_SPAN_MASK_KEY,
            self.NER_LABELS_KEY,
        ):
            if key not in batch:
                msg = f"span knowledge verification requires {key!r}"
                raise ValueError(msg)
        # Lightning has already moved tensors to the accelerator by this point.
        # Build the sparse character-offset lookup once on CPU; calling bool()/int()
        # on each CUDA scalar would introduce one device synchronisation per
        # candidate.
        mention_spans = batch[self.NER_SPANS_KEY].detach().cpu()
        mention_mask = batch[self.NER_SPAN_MASK_KEY].detach().cpu()
        mention_labels = batch[self.NER_LABELS_KEY]
        candidate_indices = span_indices.detach().cpu()
        candidate_mask = span_mask.detach().cpu().bool()
        if candidate_indices.ndim != 4 or candidate_indices.shape[-1] != 2:
            msg = "span_indices must be [B, W, K, 2]"
            raise ValueError(msg)
        if tuple(candidate_mask.shape) != tuple(candidate_indices.shape[:3]):
            msg = "span_mask must align with span_indices"
            raise ValueError(msg)
        batch_size, max_words, max_span_width, _ = candidate_indices.shape
        if mention_spans.shape[:2] != (len(mention_labels), mention_width):
            msg = "mention span tensors do not align with mention labels"
            raise ValueError(msg)
        if batch_size != len(mention_labels):
            msg = "span candidates do not align with mention labels"
            raise ValueError(msg)

        result = torch.zeros(
            batch_size,
            max_words * max_span_width,
            mention_width,
            dtype=torch.bool,
        )
        word_char_spans = batch["word_char_spans"]
        target_label = self.label_list[0]
        for row, words in enumerate(word_char_spans):
            lookup: dict[tuple[int, int], int] = {}
            for mention in range(mention_width):
                if not bool(mention_mask[row, mention]):
                    continue
                label = str(mention_labels[row][mention])
                if label != target_label:
                    continue
                span = (
                    int(mention_spans[row, mention, 0]),
                    int(mention_spans[row, mention, 1]),
                )
                if span in lookup:
                    msg = f"duplicate first-pass candidate span is ambiguous: {span}"
                    raise ValueError(msg)
                lookup[span] = mention
            for word_slot in range(max_words):
                for width_slot in range(max_span_width):
                    if not bool(candidate_mask[row, word_slot, width_slot]):
                        continue
                    start_word = int(candidate_indices[row, word_slot, width_slot, 0])
                    end_word = int(candidate_indices[row, word_slot, width_slot, 1])
                    if not (0 <= start_word <= end_word < len(words)):
                        msg = (
                            "public span geometry is outside word_char_spans: "
                            f"row={row}, span=({start_word}, {end_word}), "
                            f"num_words={len(words)}"
                        )
                        raise ValueError(msg)
                    char_span = (
                        int(words[start_word][0]),
                        int(words[end_word][1]),
                    )
                    mention = lookup.get(char_span)
                    if mention is not None:
                        flat = word_slot * max_span_width + width_slot
                        result[row, flat, mention] = True
        return result.to(device=device)

    def _attend_candidate_evidence(
        self,
        span_rep: torch.Tensor,
        kv: torch.Tensor,
        kmask: torch.Tensor,
        node_mention: torch.Tensor,
        span_to_mention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if node_mention.shape[:2] != kv.shape[:2]:
            msg = (
                "mention ownership must align with knowledge tokens: "
                f"{tuple(node_mention.shape)} vs {tuple(kv.shape)}"
            )
            raise ValueError(msg)
        if node_mention.shape[2] != span_to_mention.shape[2]:
            msg = "knowledge and span mention dimensions do not match"
            raise ValueError(msg)

        allowed = torch.einsum(
            "bsm,bnm->bsn",
            span_to_mention.to(kv.dtype),
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
            span_rep.shape[1],
            kv_with_dummy.shape[1],
        )
        attn_out, _ = self.cross_attn(
            query=span_rep,
            key=kv_with_dummy,
            value=kv_with_dummy,
            attn_mask=attn_mask,
            need_weights=False,
        )
        if self.attn_norm is not None:
            attn_out = self.attn_norm(attn_out)
        attn_out *= has_real_key.unsqueeze(2).to(attn_out.dtype)
        return attn_out, has_real_key

    def _candidate_link_features(
        self,
        batch: dict[str, Any],
        span_to_mention: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for key in (self.NER_LINK_SCORES_KEY, self.NER_LINK_MASK_KEY):
            if key not in batch:
                msg = f"span knowledge verification requires {key!r}"
                raise ValueError(msg)
        scores = batch[self.NER_LINK_SCORES_KEY].to(span_to_mention.device).float()
        links = batch[self.NER_LINK_MASK_KEY].to(span_to_mention.device).bool()
        expected = (span_to_mention.shape[0], span_to_mention.shape[2])
        if tuple(scores.shape) != expected or tuple(links.shape) != expected:
            msg = (
                f"mention link features must be {expected}, got "
                f"scores={tuple(scores.shape)}, links={tuple(links.shape)}"
            )
            raise ValueError(msg)
        if self.knowledge_control == "shuffled" and scores.shape[0] > 1:
            scores = scores.roll(shifts=1, dims=0)
            links = links.roll(shifts=1, dims=0)
        active_links = span_to_mention & links.unsqueeze(1)
        candidate_scores = (
            span_to_mention.to(scores.dtype) * scores.unsqueeze(1)
        ).amax(dim=2)
        return candidate_scores.to(dtype), active_links.any(dim=2).to(dtype)
