"""Baseline-preserving late-logit span verifier -- #231 Variant 1.

`SpanCrossAttentionKnowledgeModel` rewrites the span representation with a knowledge
residual and re-scores it. #218 measured the cost at n=10: on BC5CDR-Chem nine contrasts
cleared Holm, all negative, all `wins 0/10`, and the mechanism was recall loss in the
first-pass confidence band 0.25-0.75 with precision unharmed. The diagnosis was not broken
plumbing -- evidenced candidates all received a non-zero residual, and the real residual was
~6.4x the shuffled one -- but an injection site that displaces boundary/existence evidence
the text encoder already holds.

This model changes only the injection site:

    base_logits  = inner(span_rep)                      # frozen, untouched
    delta        = late_head(span_rep, evidence, link)  # per candidate, non-negative
    final_logits = base_logits + evidence_mask * delta

Guarantees, each pinned by a bitwise test:

* the span representation, candidate lattice and span geometry are passed through unchanged
* a knowledge-empty / NIL / unlinked candidate gets `final_logits == base_logits` exactly
* `knowledge_control="zero"` returns exact base logits for every candidate
* the delta is non-negative (rescue-only), because Chem's failure is recall, not precision
* at initialisation the shift is under `INITIAL_LOGIT_TOLERANCE`

Bitwise rather than approximate matters here: 97.8% of candidates own no evidence (#218
measured 2.16% coverage), so a small spurious delta leaking into the unevidenced majority
would be invisible to an `allclose` check and would dominate the aggregate.

The inner baseline is FROZEN by default. #231 forbids an unfreeze sweep, and freezing is
what makes the contrast interpretable: without it, "the knowledge arm learned" and "the text
backbone became a different model" are confounded.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterator  # noqa: TC003 - runtime return annotation
from pathlib import Path
from typing import Any, cast

import torch
from ieforge.models.hf_span_classifier import (
    SpanClassifierOutput,
)

from knowhyper.ieforge.span_knowledge_model import (
    SpanCrossAttentionKnowledgeModel,
    SpanKnowledgeOutput,
)
from knowhyper.ieforge.span_late_head import RescueOnlyLateHead
from knowhyper.ieforge.span_signed_head import SignedBoundedLateHead

logger = logging.getLogger(__name__)


#: Re-exported so the compose gate and tests can name the production sources without
#: constructing a model (which would download a BERT).
PRODUCTION_EVIDENCE_SOURCES = ("graph", "text", "lexical")


class SpanLateLogitKnowledgeModel(SpanCrossAttentionKnowledgeModel):
    """Add a per-candidate non-negative logit delta to a frozen span baseline."""

    # The initial forward must sit within this of the frozen baseline. One order of
    # magnitude under the smallest contrast #218 resolved at n=10 (0.0040 F1), so a
    # starting perturbation cannot be mistaken for an effect.
    INITIAL_LOGIT_TOLERANCE = 1e-3
    INITIAL_DELTA_TOLERANCE = RescueOnlyLateHead.INITIAL_DELTA_TOLERANCE

    #: Where the residual head's evidence comes from. `graph` is the ONLY value every recorded
    #: #231 cell ran under (400 cells), so it stays the default: flipping it would silently
    #: re-interpret completed results as if they had used a different channel.
    #: `lexical` is a SEPARATE source, not `text` with the content blanked: blanking would
    #: still push M rows through the second encoder, so the arm's cost and parameter count
    #: would differ from what its name implies.
    EVIDENCE_SOURCES = ("graph", "text", "lexical", "hash_test")

    #: The sources an EXPERIMENT may select. `hash_test` is deliberately excluded: it embeds
    #: text as a deterministic hash with NO learnable parameters, which is fine for testing the
    #: plumbing and worthless as a treatment. Phase A0 came within one command of running
    #: `text_real` on it, which would have compared a fixed hash against a trained projection
    #: and -- under the pre-registered protocol -- CLOSED the text route on that basis.
    PRODUCTION_EVIDENCE_SOURCES = ("graph", "text", "lexical")

    #: Sources whose evidence is owned by a CANDIDATE rather than by a graph mention. These
    #: bypass `_attend_candidate_evidence` entirely and use the signed, bounded head; `graph`
    #: keeps the #231 path so its 400 recorded cells stay bit-identical.
    CANDIDATE_OWNED_SOURCES = ("text", "lexical", "hash_test")

    #: Batch keys the `text` source consumes: PACKED token ids, not raw strings. Tokenising
    #: inside the forward would re-tokenise the same text every epoch, make the arm's cost
    #: depend on Python string work, and leave the tokenisation outside any fingerprint.
    EVIDENCE_IDS_KEY = "candidate_evidence_input_ids"
    EVIDENCE_ATTENTION_KEY = "candidate_evidence_attention_mask"
    EVIDENCE_OWNER_KEY = "evidence_owner_index"
    EVIDENCE_MASK_KEY = "candidate_evidence_mask"

    #: `hash_test` only. Raw strings, one list per instance.
    TEXT_KEY = "knowledge_candidate_texts"
    TEXT_MENTION_COUNT_KEY = "knowledge_candidate_mention_count"
    LEXICAL_KEY = "knowledge_candidate_lexical_features"
    LEXICAL_MASK_KEY = "knowledge_candidate_lexical_mask"

    @classmethod
    def validate_evidence_source(
        cls,
        evidence_source: str,
        *,
        evidence_encoder_name: str | None = None,
        allow_test_sources: bool = True,
    ) -> None:
        """Refuse a source that cannot deliver what its name promises.

        Split out as a classmethod so the experiment compose gate can check an arm without
        building a BERT, and so the "text without an encoder" rule is stated in ONE place.
        """
        allowed = (
            cls.EVIDENCE_SOURCES
            if allow_test_sources
            else cls.PRODUCTION_EVIDENCE_SOURCES
        )
        if evidence_source not in allowed:
            msg = (
                f"evidence_source must be one of {allowed}, got {evidence_source!r}. "
                "`hash_test` is test-only and cannot be selected by an experiment."
            )
            raise ValueError(msg)
        if evidence_source == "text" and not evidence_encoder_name:
            msg = (
                "evidence_source='text' requires evidence_encoder_name. There is NO implicit "
                "fallback: the previous `getattr(self, 'text_encoder', None)` turned a missing "
                "encoder into a deterministic hash with no learnable parameters, so the arm "
                "trained and reported a plausible score while measuring nothing. Use "
                "evidence_source='hash_test' if a hash is what you actually want."
            )
            raise ValueError(msg)

    @staticmethod
    def scatter_evidence(
        *,
        pooled: torch.Tensor,
        owner_index: torch.Tensor,
        batch_size: int,
        max_candidates: int,
    ) -> torch.Tensor:
        """Scatter `[N_valid, H]` pooled evidence to `[B, M, H]` by owner index.

        Matching is by the owner index carried alongside each packed row, never by position,
        so permuting the packed rows cannot pair one candidate's text with another's span.
        Out-of-range owners RAISE: dropping them would lose a candidate's evidence silently,
        and `index_put_` would happily wrap or error deep inside autograd instead.
        """
        if owner_index.dim() != 2 or owner_index.shape[-1] != 2:
            msg = f"owner_index must be [N, 2], got {tuple(owner_index.shape)}"
            raise ValueError(msg)
        if owner_index.shape[0] != pooled.shape[0]:
            msg = (
                f"owner_index has {owner_index.shape[0]} rows but pooled has "
                f"{pooled.shape[0]}; every packed row must name its owner"
            )
            raise ValueError(msg)
        out = torch.zeros(
            (batch_size, max_candidates, pooled.shape[-1]),
            dtype=pooled.dtype,
            device=pooled.device,
        )
        if owner_index.numel() == 0:
            return out
        rows = owner_index[:, 0]
        slots = owner_index[:, 1]
        if int(rows.min()) < 0 or int(rows.max()) >= batch_size:
            msg = (
                f"owner batch index out of range: [{int(rows.min())}, {int(rows.max())}] "
                f"for batch_size={batch_size}"
            )
            raise ValueError(msg)
        if int(slots.min()) < 0 or int(slots.max()) >= max_candidates:
            msg = (
                f"owner candidate index out of range: [{int(slots.min())}, "
                f"{int(slots.max())}] for max_candidates={max_candidates}"
            )
            raise ValueError(msg)
        out[rows, slots] = pooled.to(out.dtype)
        return out

    def __init__(
        self,
        *args: Any,
        name: str = "knowhyper_span_late_logit",
        freeze_inner: bool = True,
        pretrained_baseline_path: str | None = None,
        evidence_source: str = "graph",
        evidence_encoder_name: str | None = None,
        lexical_feature_dim: int | None = None,
        residual_bound: float = 0.5,
        eligibility_band: tuple[float, float] = (0.0, 1.0),
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("name", name)
        super().__init__(*args, **kwargs)

        # Validated at CONSTRUCTION, not at the first forward: a typo in an experiment config
        # would otherwise survive dispatch and only surface after the job had started. This
        # also rejects `text` with no encoder, which used to degrade to a hash.
        self.validate_evidence_source(
            evidence_source, evidence_encoder_name=evidence_encoder_name
        )
        self.evidence_source = evidence_source
        self.evidence_encoder_name = evidence_encoder_name
        self.text_encoder: Any = None
        self.evidence_manifest: dict[str, Any] = {}
        if evidence_source == "text":
            from knowhyper.ieforge.evidence_encoder import (
                build_evidence_encoder,
                encoder_manifest,
            )

            # An INDEPENDENT encoder from the anchor's own checkpoint -- the #189 recipe. Per
            # backbone: a PubMedBERT anchor gets a PubMedBERT evidence encoder and a
            # BioLinkBERT anchor a BioLinkBERT one, so no arm becomes a cross-backbone hybrid.
            # Hidden size is checked here rather than at the scatter, where a mismatch would
            # surface as a shape error inside autograd on some later batch.
            self.text_encoder = build_evidence_encoder(
                evidence_encoder_name or "",
                expected_hidden_size=self.hidden_size,
            )
            self.evidence_manifest = encoder_manifest(
                self.text_encoder,
                checkpoint=str(evidence_encoder_name),
                trainable=True,
            )
        self._lexical_projection: torch.nn.Linear | None = None
        if evidence_source == "lexical":
            from knowhyper.ieforge.span_text_evidence import LEXICAL_FEATURE_NAMES

            expected_dim = len(LEXICAL_FEATURE_NAMES)
            configured_dim = (
                expected_dim
                if lexical_feature_dim is None
                else int(lexical_feature_dim)
            )
            if configured_dim != expected_dim:
                msg = (
                    f"lexical_feature_dim={configured_dim}, but the channel emits "
                    f"{expected_dim} features ({LEXICAL_FEATURE_NAMES}). The projection "
                    "must be constructed before the optimizer; adapting it lazily would "
                    "leave its parameters outside every optimizer group."
                )
                raise ValueError(msg)
            # Lightning creates the optimizer before the first training forward. The former
            # lazy construction produced a checkpoint tensor that no optimizer group updated.
            self._lexical_projection = torch.nn.Linear(configured_dim, self.hidden_size)

        # The candidate-owned branch's registered residual: SIGNED and BOUNDED, with a separate
        # eligibility band. Built only for text/lexical -- `graph` must keep the rescue-only
        # head, since its 400 recorded cells ran under a delta that could never suppress.
        self.eligibility_band = (float(eligibility_band[0]), float(eligibility_band[1]))
        if self.eligibility_band[0] > self.eligibility_band[1]:
            msg = (
                f"eligibility_band must be (low, high) with low <= high, got "
                f"{self.eligibility_band}. An inverted band admits nothing, so the arm would "
                "train as the frozen anchor and report a plausible null."
            )
            raise ValueError(msg)
        self.residual_bound = float(residual_bound)
        self.signed_head: SignedBoundedLateHead | None = None
        if evidence_source in self.CANDIDATE_OWNED_SOURCES:
            self.signed_head = SignedBoundedLateHead(
                span_dim=self.hidden_size,
                evidence_dim=self.hidden_size,
                num_labels=len(self.label_list),
                bound=self.residual_bound,
            )

        self.late_head = RescueOnlyLateHead(
            span_dim=self.hidden_size,
            evidence_dim=self.hidden_size,
            num_labels=len(self.label_list),
        )
        # The cross-attention fusion gates are inherited but UNUSED by this forward:
        # `_attend_candidate_evidence` uses `cross_attn` and `attn_norm` to build the
        # evidence summary, while `gate` and `span_gate` belonged to the residual-fusion
        # step this variant replaces. Measured in the Phase 0 audit: 5 dead parameters were
        # trainable. That is not merely wasteful -- `difflr` puts them in the KNOWLEDGE LR
        # group, so a param-group audit would report a branch bigger than the one that
        # exists, and weight decay would move weights that affect nothing.
        self._disable_unused_fusion_gates()
        self.pretrained_baseline_path = pretrained_baseline_path
        # Load BEFORE freezing: `requires_grad=False` does not block `load_state_dict`,
        # but doing it in this order keeps the invariant "frozen means frozen at the
        # weights it was given" easy to state.
        if pretrained_baseline_path is not None:
            self._load_frozen_baseline(pretrained_baseline_path)
        # Default True: #231 explicitly rules out an unfreeze sweep, and the safe value is
        # the one you must opt OUT of.
        self.freeze_inner = bool(freeze_inner)
        if self.freeze_inner:
            self._freeze_inner()

    def _load_frozen_baseline(self, path: str) -> None:
        """Load a #220 span baseline into `inner`, strictly.

        The checkpoint is an `IELightningModule` one, so the IE model lives under a
        `model.` prefix, which is stripped here.

        `strict=True` on purpose: a partial load leaves some layers at random init and the
        run still trains and reports a plausible number. And because a compatible-but-wrong
        checkpoint loads cleanly, provenance matching is NOT proof the right weights
        arrived -- only equal logits are, which is why #231 requires the one-batch
        reproduction check and `test_the_baseline_is_loaded_and_reproduces_its_logits`
        performs it against a donor built with different weights.
        """
        self._load_baseline_into(self.inner, path)

    @staticmethod
    def _load_baseline_into(inner: torch.nn.Module, path: str) -> None:
        """Strictly load one Lightning span checkpoint into ``inner``."""
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            msg = f"frozen baseline checkpoint not found: {checkpoint_path}"
            raise FileNotFoundError(msg)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            msg = (
                f"frozen baseline has no state_dict mapping: {checkpoint_path}. A #220 "
                "span baseline is a Lightning checkpoint."
            )
            raise TypeError(msg)
        prefix = "model."
        inner_state = {
            str(key)[len(prefix) :]: value
            for key, value in state.items()
            if str(key).startswith(prefix)
        }
        if not inner_state:
            msg = f"frozen baseline state_dict has no model.* parameters: {checkpoint_path}"
            raise ValueError(msg)
        inner.load_state_dict(inner_state, strict=True)
        logger.info(
            "loaded frozen span baseline from %s (%d tensors)",
            checkpoint_path,
            len(inner_state),
        )

    def _disable_unused_fusion_gates(self) -> None:
        """Mark the inherited-but-unused fusion gates as not requiring grad."""
        for attribute in ("gate", "span_gate"):
            module_or_param = getattr(self, attribute, None)
            if module_or_param is None:
                continue
            if isinstance(module_or_param, torch.nn.Parameter):
                module_or_param.requires_grad_(False)
            else:
                for param in module_or_param.parameters():
                    param.requires_grad_(False)

    def _freeze_inner(self) -> None:
        """Mark every inner parameter as not requiring grad.

        `requires_grad=False` rather than merely detaching the output: the optimizer audit
        (#231) enumerates `requires_grad` to prove the frozen base is not in a param group,
        and a detach-only implementation would still hand the inner weights to the
        optimizer, where weight decay would move them.
        """
        for param in self.inner.parameters():
            param.requires_grad_(False)

    def trainable_named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """Parameters an optimizer may see, for the #231 param-group audit."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param

    def forward(self, batch: dict[str, Any]) -> SpanClassifierOutput:
        base_output, span_rep_4d, span_indices, span_mask = (
            self._base_forward_with_span_features(batch)
        )
        base_logits = base_output.logits

        if not self._has_knowledge(batch):
            if self.require_knowledge:
                msg = "knowledge keys absent from batch but require_knowledge=True"
                raise ValueError(msg)
            if not self._warned_no_knowledge:
                logger.warning(
                    "SpanLateLogitKnowledgeModel: no knowledge keys in batch; "
                    "returning the frozen span baseline"
                )
                self._warned_no_knowledge = True
            return base_output

        # PER-INSTANCE guard, mirroring the parent. `forward` is a full override here, so
        # without this the flag would be declared in every model config
        # (`require_knowledge_per_instance: false`) yet never read -- turning it on would
        # silently pass knowledge-empty ROWS through as baseline logits instead of failing
        # fast. On a rescue-only head that is the worst-shaped failure available: returning the
        # frozen baseline is exactly what a correct no-evidence candidate looks like, so
        # nothing downstream could tell the two apart.
        # Guarded on the flag FIRST, not after computing the mask: `_knowledge_row_mask` falls
        # back to `batch["input_ids"]` when no node/hyperedge mask is present, which raises
        # KeyError on the reduced batches some tests build. Since the flag is false in every
        # #231 config, computing the mask unconditionally would make a disabled guard able to
        # break a working forward pass.
        if self.require_knowledge_per_instance:
            knowledge_rows = self._knowledge_row_mask(batch)
            if not bool(knowledge_rows.all()):
                empty = (~knowledge_rows).nonzero(as_tuple=False).flatten().tolist()
                msg = (
                    "knowledge-empty rows with require_knowledge_per_instance=True: "
                    f"{empty}"
                )
                raise ValueError(msg)

        batch_size, max_words, max_span_width, span_dim = span_rep_4d.shape
        expected_spans = max_words * max_span_width
        if span_dim != self.hidden_size:
            msg = (
                "span representation does not align with the knowledge model: "
                f"rep={tuple(span_rep_4d.shape)}, hidden_size={self.hidden_size}"
            )
            raise ValueError(msg)
        span_rep = span_rep_4d.reshape(batch_size, expected_spans, span_dim)

        if self.knowledge_control == "zero":
            # Exact base logits, and a diagnostics record whose deltas are exactly zero.
            # Returned BEFORE the head runs, so no rounding can creep in.
            return self._late_output(
                base_output=base_output,
                final_logits=base_logits,
                delta=torch.zeros(
                    batch_size,
                    expected_spans,
                    len(self.label_list),
                    dtype=base_logits.dtype,
                    device=base_logits.device,
                ),
                reliability=torch.zeros(
                    batch_size,
                    expected_spans,
                    dtype=base_logits.dtype,
                    device=base_logits.device,
                ),
                has_evidence=torch.zeros(
                    batch_size,
                    expected_spans,
                    dtype=torch.bool,
                    device=base_logits.device,
                ),
                link_mask=torch.zeros(
                    batch_size,
                    expected_spans,
                    dtype=torch.bool,
                    device=base_logits.device,
                ),
                attn_norm=torch.zeros(
                    batch_size,
                    expected_spans,
                    dtype=base_logits.dtype,
                    device=base_logits.device,
                ),
            )

        if self.evidence_source in self.CANDIDATE_OWNED_SOURCES:
            # The candidate-owned branch does NOT traverse the graph's mention dimension.
            # `_attend_candidate_evidence` computes reachability as
            # `einsum("bsm,bnm->bsn", span_to_mention, node_mention)`, so evidence can only
            # reach a span through a graph mention -- which is why candidate-slot ownership
            # matched nothing and every delta was zero (0/199 encoder tensors moved after 150
            # steps). Pooled candidate evidence is ALREADY per-candidate [B, M, H], so it goes
            # straight to the head.
            return self._forward_candidate_owned(
                batch=batch,
                base_output=base_output,
                base_logits=base_logits,
                span_rep=span_rep,
                span_indices=span_indices,
                span_mask=span_mask,
            )

        kv, kmask, node_mention = self._encode_evidence(batch)
        if node_mention is None:
            msg = "span knowledge verification requires mention-local ownership"
            raise ValueError(msg)
        if self.knowledge_control == "shuffled" and kv.shape[0] > 1:
            # Every evidence tensor moves under the SAME permutation. Rolling keys but not
            # ownership would produce a partially-shuffled hybrid, which is not a control.
            kv = kv.roll(shifts=1, dims=0)
            kmask = kmask.roll(shifts=1, dims=0)
            node_mention = node_mention.roll(shifts=1, dims=0)

        span_to_mention = self._candidate_to_mention(
            batch,
            span_indices=span_indices,
            span_mask=span_mask,
            mention_width=node_mention.shape[2],
            device=span_rep.device,
        )
        evidence, has_evidence = self._attend_candidate_evidence(
            span_rep, kv, kmask, node_mention, span_to_mention
        )
        link_score, link_mask = self._candidate_link_features(
            batch, span_to_mention, span_rep.dtype
        )
        if self.knowledge_control == "shuffled" and link_score.shape[0] > 1:
            link_score = link_score.roll(shifts=1, dims=0)
            link_mask = link_mask.roll(shifts=1, dims=0)

        # `_candidate_link_features` returns the mask as a FLOAT tensor (it is used as a
        # multiplier there), so it is booled explicitly -- `has_evidence & link_mask` on a
        # float raises `bitwise_and_cpu not implemented for 'Float'`.
        link_bool = link_mask.bool()
        evidence_bool = has_evidence.bool()
        # Reliability is zero wherever the candidate owns no evidence or no link, so the
        # head's output is multiplied out rather than merely masked afterwards.
        reliability = (
            link_score
            * evidence_bool.to(link_score.dtype)
            * link_bool.to(link_score.dtype)
        )
        delta = self.late_head(
            span_rep=span_rep, evidence=evidence, link_score=reliability
        )
        # Mask again, explicitly and in the logit space. `reliability` already zeroes the
        # unevidenced candidates, but this makes the base-preservation guarantee independent
        # of the head's internals -- a future head that added a bias would otherwise break
        # the bitwise equality without any test here changing.
        keep = (evidence_bool & link_bool).unsqueeze(-1).to(delta.dtype)
        delta = delta * keep

        final_logits = base_logits + delta.reshape(base_logits.shape)
        return self._late_output(
            base_output=base_output,
            final_logits=final_logits,
            delta=delta,
            reliability=reliability,
            has_evidence=evidence_bool,
            link_mask=link_bool,
            attn_norm=evidence.detach().norm(dim=-1),
        )

    def _encode_evidence(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(kv, kmask, ownership)` from the configured evidence source.

        ONE seam. The graph branch delegates to `_encode_incidence` unchanged, so the 400
        recorded #231 cells keep the exact behaviour they were run under; the text branch
        supplies the same three tensors from candidate-owned text.

        The text branch is FAIL-CLOSED for the #171 reason: a zero-filled evidence tensor is
        numerically identical to "this candidate owns no text", so a missing key must raise
        rather than be absorbed. An explicitly EMPTY text list is a different state and is
        accepted -- present-and-empty is a legitimate observation, exactly as for the link
        features.
        """
        if self.evidence_source == "graph":
            return self._encode_incidence(batch)

        if self.evidence_source == "lexical":
            return self._encode_lexical(batch)

        if self.evidence_source == "hash_test":
            missing = [
                key
                for key in (self.TEXT_KEY, self.TEXT_MENTION_COUNT_KEY)
                if key not in batch
            ]
            if missing:
                msg = (
                    f"hash_test evidence requires {missing} in the batch. Zero-filling would "
                    "be indistinguishable from a candidate that owns no text -- the #171 "
                    "ambiguity. To express 'no text', pass an explicitly empty list."
                )
                raise ValueError(msg)

            from knowhyper.ieforge.span_text_evidence import encode_candidate_text

            # No encoder ON PURPOSE here: this source exists to exercise the plumbing, and it
            # is unreachable from an experiment config (`PRODUCTION_EVIDENCE_SOURCES`).
            return encode_candidate_text(
                texts=list(batch[self.TEXT_KEY]),
                mention_count=int(batch[self.TEXT_MENTION_COUNT_KEY]),
                hidden_size=self.hidden_size,
                device=next(self.parameters()).device,
                encoder=None,
            )

        return self._encode_packed_text(batch)

    def _forward_candidate_owned(
        self,
        *,
        batch: dict[str, Any],
        base_output: Any,
        base_logits: torch.Tensor,
        span_rep: torch.Tensor,
        span_indices: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> SpanClassifierOutput:
        """Return `z_anchor + valid * eligible * b * tanh(raw)` for text/lexical.

        `valid` is `candidate_evidence_mask` -- a VALIDITY mask, not a confidence. The graph's
        `link_score` / `link_mask` are NOT multiplied in again: a candidate that owns text was
        already linked upstream, so re-gating on graph linkability is double-conditioning. They
        are recorded as diagnostics so the relationship can still be measured.
        """
        kv, valid, _ = self._encode_evidence(batch)
        if self.knowledge_control == "shuffled" and kv.shape[0] > 1:
            # Keys and validity move under the SAME permutation, or the control is a hybrid.
            kv = kv.roll(shifts=1, dims=0)
            valid = valid.roll(shifts=1, dims=0)

        # Candidate slot -> enumerated span by PUBLIC GEOMETRY. The model scores every
        # word-start x width span (often 504-1464 slots), while the proposal lattice carries
        # at most 16 candidates. Padding 16 vectors onto the front of the enumerated axis pairs
        # unrelated spans and produced an all-zero treatment on real data. Reuse the existing
        # char-span/type bridge, but NOT its graph-link gate: this is index alignment only.
        mention_width = int(batch[self.NER_SPANS_KEY].shape[1])
        if kv.shape[1] < mention_width or valid.shape[1] < mention_width:
            msg = (
                "candidate evidence width is smaller than the NER lattice width: "
                f"kv={kv.shape[1]}, valid={valid.shape[1]}, mentions={mention_width}"
            )
            raise ValueError(msg)
        if kv.shape[1] > mention_width:
            # CandidateTextChannel pads to its configured maximum. Extra slots are allowed
            # only when explicitly NIL; silently slicing live evidence would drop a candidate.
            if bool(valid[:, mention_width:].any()):
                msg = (
                    "candidate evidence beyond the NER lattice width is live; truncating it "
                    "would attach or drop evidence silently"
                )
                raise ValueError(msg)
            kv = kv[:, :mention_width]
            valid = valid[:, :mention_width]

        self._validate_candidate_ids(batch, mention_width=mention_width)
        span_to_candidate = self._candidate_to_mention(
            batch,
            span_indices=span_indices,
            span_mask=span_mask,
            mention_width=mention_width,
            device=span_rep.device,
        )
        # Each enumerated span maps to at most one (span,label) lattice candidate. Keep this
        # explicit: summing two candidate vectors would manufacture evidence not owned by
        # either candidate alone.
        if bool((span_to_candidate.sum(dim=2) > 1).any()):
            msg = (
                "an enumerated span maps to more than one candidate-owned evidence slot"
            )
            raise ValueError(msg)
        evidence = torch.einsum("bsm,bmh->bsh", span_to_candidate.to(kv.dtype), kv)
        valid_spans = (span_to_candidate & valid.bool().unsqueeze(1)).any(dim=2)
        # `span_rep` is flattened to [B, S] while `span_mask` is [B, W, K]. Flattening keeps
        # every gate on one axis.
        flat_mask = span_mask.reshape(span_mask.shape[0], -1).bool()
        # Only real spans can be intervened on; padding must stay bitwise untouched.
        valid_spans = valid_spans & flat_mask

        eligible = self._candidate_eligibility(base_logits, flat_mask)
        delta = self.signed_head(
            span_rep=span_rep,
            evidence=evidence,
            valid=valid_spans,
            eligible=eligible,
        )
        intervened = valid_spans & eligible.bool()
        # Masked a second time in logit space, so base-preservation does not depend on the
        # head's internals: a future head that added a bias would otherwise break the bitwise
        # equality with no test here changing.
        delta = delta * intervened.unsqueeze(-1).to(delta.dtype)
        final_logits = base_logits + delta.reshape(base_logits.shape)

        # DIAGNOSTICS, recorded separately and never multiplied into the delta. The graph link
        # features are kept so "does text help where the graph linked?" stays answerable, but
        # they do not gate this branch.
        link_score, link_mask = self._candidate_link_diagnostics(
            batch, span_indices, span_mask, span_rep, flat_mask
        )
        return self._late_output(
            base_output=base_output,
            final_logits=final_logits,
            delta=delta,
            has_evidence=valid_spans,
            link_mask=link_mask,
            link_score=link_score,
            eligible=eligible,
            intervened=intervened,
        )

    def _has_knowledge(self, batch: dict[str, Any]) -> bool:
        """Recognise CANDIDATE-OWNED evidence as knowledge, not just the graph keys.

        The inherited check looks for `incidence_*` / node / hyperedge keys. For a text or
        lexical arm those may be absent entirely -- the candidate evidence keys ARE the
        knowledge -- so without this override `require_knowledge=True` raises on a perfectly
        well-formed candidate-owned batch.

        Presence is not enough on its own for the graph path (an all-cache-miss batch has the
        keys with zero nodes), and the same care applies here: the keys must be present AND the
        source must be one that reads them. A `graph` arm handed candidate keys still defers to
        the inherited definition, so #231's behaviour is untouched.
        """
        if self.evidence_source in self.CANDIDATE_OWNED_SOURCES:
            candidate_keys = (
                (self.EVIDENCE_MASK_KEY, self.EVIDENCE_IDS_KEY)
                if self.evidence_source == "text"
                else (self.LEXICAL_MASK_KEY, self.TEXT_KEY)
            )
            if any(key in batch for key in candidate_keys):
                return True
        return super()._has_knowledge(batch)

    def _knowledge_row_mask(self, batch: dict[str, Any]) -> torch.Tensor:
        """Per-row knowledge presence for the candidate-owned branch.

        A row whose candidates own no evidence at all is knowledge-EMPTY, which is legitimate
        and common (97.8% of candidates own nothing). It is reported honestly so
        `require_knowledge_per_instance` means what it says, rather than being satisfied by the
        mere presence of a tensor full of False.
        """
        if self.evidence_source in self.CANDIDATE_OWNED_SOURCES:
            for key in (self.EVIDENCE_MASK_KEY, self.LEXICAL_MASK_KEY):
                value = batch.get(key)
                if value is not None:
                    return value.bool().any(dim=1)
        return super()._knowledge_row_mask(batch)

    def _validate_candidate_ids(
        self, batch: dict[str, Any], *, mention_width: int
    ) -> None:
        """Prove the packed candidate slots match NERMentionSpanChannel's slots.

        Both are built from the same query, but that fact is load-bearing rather than a reason
        to trust position. A reorder leaves every tensor shape valid and attaches each text to
        another span. CandidateTextChannel therefore carries stable ``start:end:label`` ids,
        which are checked against the actual span-channel batch here.
        """
        ids = batch.get("candidate_ids")
        if ids is None:
            msg = (
                f"candidate-owned {self.evidence_source} evidence requires stable "
                "candidate_ids for alignment"
            )
            raise ValueError(msg)
        spans = batch[self.NER_SPANS_KEY].detach().cpu()
        mask = batch[self.NER_SPAN_MASK_KEY].detach().cpu().bool()
        labels = batch[self.NER_LABELS_KEY]
        if len(ids) != spans.shape[0]:
            msg = f"candidate_ids batch {len(ids)} != NER spans batch {spans.shape[0]}"
            raise ValueError(msg)
        for row, row_ids in enumerate(ids):
            if len(row_ids) < mention_width:
                msg = f"candidate_ids row {row} has {len(row_ids)} slots, need {mention_width}"
                raise ValueError(msg)
            for slot in range(mention_width):
                expected = ""
                if bool(mask[row, slot]):
                    expected = (
                        f"{int(spans[row, slot, 0])}:{int(spans[row, slot, 1])}:"
                        f"{labels[row][slot]}"
                    )
                if str(row_ids[slot]) != expected:
                    msg = (
                        f"candidate id mismatch at row={row}, slot={slot}: "
                        f"packed={row_ids[slot]!r}, span_channel={expected!r}"
                    )
                    raise ValueError(msg)

    def _candidate_eligibility(
        self, base_logits: torch.Tensor, span_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return the eligibility band as a `[B, S]` bool.

        The band is a property of the ANCHOR's confidence, so it is computable at inference and
        needs no label. Bounds are train-only and committed before validation is read
        (`eligibility_band`); the default `(0.0, 1.0)` admits everything, which is the
        no-op state rather than a silent restriction.
        """
        probabilities = torch.softmax(base_logits.float(), dim=-1)
        # Confidence in the arg-max label: the quantity #218 found the harm concentrated in.
        confidence = probabilities.max(dim=-1).values
        if confidence.dim() == 3:  # [B, S, L] -> already reduced; guard for odd shapes
            confidence = confidence.reshape(confidence.shape[0], -1)
        confidence = confidence.reshape(span_mask.shape)
        low, high = self.eligibility_band
        return (
            (confidence >= float(low)) & (confidence <= float(high)) & span_mask.bool()
        )

    def _candidate_link_diagnostics(
        self,
        batch: dict[str, Any],
        span_indices: torch.Tensor,
        span_mask: torch.Tensor,
        span_rep: torch.Tensor,
        flat_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Graph link features for the RECORD only -- never multiplied into the delta.

        Returns zeros when the graph channel is absent: the candidate-owned branch does not
        require it, so a missing key is a legitimately unavailable diagnostic rather than the
        #171 ambiguity (nothing downstream consumes it).
        """
        zeros = torch.zeros(
            span_mask.shape, dtype=span_rep.dtype, device=span_rep.device
        )
        if self.NER_LINK_SCORES_KEY not in batch or self.NER_LINK_MASK_KEY not in batch:
            return zeros, zeros.bool()
        try:
            node_mention = batch[self.NER_LINK_MASK_KEY]
            span_to_mention = self._candidate_to_mention(
                batch,
                span_indices=span_indices,
                span_mask=span_mask,
                mention_width=node_mention.shape[-1],
                device=span_rep.device,
            )
            score, mask = self._candidate_link_features(
                batch, span_to_mention, span_rep.dtype
            )
            score = score.reshape(flat_mask.shape)
            mask = mask.reshape(flat_mask.shape)
        except (ValueError, RuntimeError, IndexError):
            # A diagnostic must never take down a run. The delta does not depend on it.
            return zeros, zeros.bool()
        return score, mask.bool()

    def _encode_packed_text(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode PACKED candidate text through the real evidence encoder.

        packed token ids -> encoder -> masked-mean pool per candidate -> scatter to `[B, M, H]`.

        Only candidates that own text occupy an encoder row, so cost scales with evidence
        actually present rather than with `B * M`. Absent candidates keep an all-zero slot and
        a False mask -- the same NIL state the graph source uses, so the head cannot tell
        "no text" from "text that encoded to zero" by looking at the value alone.
        """
        if self.text_encoder is None:
            # Unreachable via __init__, which refuses `text` without an encoder. Kept because
            # this is the exact state that previously degraded to a hash.
            msg = (
                "evidence_source='text' but no text encoder was built; refusing to fall back "
                "to a hash (see validate_evidence_source)"
            )
            raise ValueError(msg)

        missing = [
            key
            for key in (
                self.EVIDENCE_IDS_KEY,
                self.EVIDENCE_ATTENTION_KEY,
                self.EVIDENCE_OWNER_KEY,
                self.EVIDENCE_MASK_KEY,
            )
            if key not in batch
        ]
        if missing:
            msg = (
                f"packed text evidence requires {missing} in the batch. Zero-filling would be "
                "indistinguishable from a candidate that owns no text -- the #171 ambiguity."
            )
            raise ValueError(msg)

        from knowhyper.ieforge.evidence_encoder import masked_mean_pool

        device = next(self.parameters()).device
        candidate_mask = batch[self.EVIDENCE_MASK_KEY].to(device).bool()
        batch_size, max_candidates = candidate_mask.shape
        input_ids = batch[self.EVIDENCE_IDS_KEY].to(device)
        attention = batch[self.EVIDENCE_ATTENTION_KEY].to(device)
        owner_index = batch[self.EVIDENCE_OWNER_KEY].to(device)

        if input_ids.shape[0] == 0:
            # Legitimate and common: no candidate in this batch owns text. An empty encoder
            # call would error on some backends, so short-circuit to the all-NIL state.
            kv = torch.zeros(
                (batch_size, max_candidates, self.hidden_size), device=device
            )
            ownership = torch.zeros(
                (batch_size, max_candidates, max_candidates),
                dtype=torch.bool,
                device=device,
            )
            return kv, candidate_mask, ownership

        encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention)
        special = batch.get("candidate_evidence_special_tokens_mask")
        pooled = masked_mean_pool(
            encoded.last_hidden_state,
            attention,
            special_tokens_mask=(special.to(device) if special is not None else None),
        )
        kv = self.scatter_evidence(
            pooled=pooled,
            owner_index=owner_index,
            batch_size=batch_size,
            max_candidates=max_candidates,
        )
        # Diagonal ownership: candidate i sees slot i and nothing else. Sharing a slot would
        # let one candidate's text reach another and make `text_swapped` meaningless.
        ownership = torch.diag_embed(candidate_mask)
        return kv, candidate_mask, ownership

    def _encode_lexical(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project surface-match features into the evidence slot. NO text content.

        The control arm. Features arrive as `[B, M, F]` and are projected to `[B, M, H]` so
        the residual head sees the same interface as the text arm -- same shapes, same
        ownership, same availability mask -- differing only in what the evidence encodes.

        Fail-closed for the same #171 reason as the text branch: an absent feature tensor
        must raise rather than become an all-zero evidence tensor, which is what "this
        candidate owns nothing" already looks like.
        """
        missing = [
            key for key in (self.LEXICAL_KEY, self.LEXICAL_MASK_KEY) if key not in batch
        ]
        if missing:
            msg = (
                f"lexical evidence requires {missing} in the batch. Zero-filling would be "
                "indistinguishable from a candidate with no surface match -- the #171 "
                "ambiguity."
            )
            raise ValueError(msg)

        features = batch[self.LEXICAL_KEY].to(next(self.parameters()).device).float()
        mask = batch[self.LEXICAL_MASK_KEY].to(features.device).bool()
        if features.dim() != 3:
            msg = f"lexical features must be [B, M, F], got {tuple(features.shape)}"
            raise ValueError(msg)

        if self._lexical_projection is None:
            msg = (
                "lexical evidence reached the model without an eagerly constructed "
                "projection; creating it here would exclude it from the optimizer"
            )
            raise RuntimeError(msg)
        if features.shape[2] != self._lexical_projection.in_features:
            msg = (
                f"lexical feature width {features.shape[2]} != configured projection width "
                f"{self._lexical_projection.in_features}"
            )
            raise ValueError(msg)
        kv = self._lexical_projection(features)
        # Diagonal ownership, matching the text arm: candidate i owns slot i only.
        width = features.shape[1]
        ownership = torch.eye(
            width, dtype=torch.bool, device=features.device
        ).unsqueeze(0).expand(features.shape[0], width, width) & mask.unsqueeze(1)
        return kv, mask, ownership

    def _late_output(
        self,
        *,
        base_output: SpanClassifierOutput,
        final_logits: torch.Tensor,
        delta: torch.Tensor,
        has_evidence: torch.Tensor,
        link_mask: torch.Tensor,
        reliability: torch.Tensor | None = None,
        attn_norm: torch.Tensor | None = None,
        link_score: torch.Tensor | None = None,
        eligible: torch.Tensor | None = None,
        intervened: torch.Tensor | None = None,
    ) -> SpanClassifierOutput:
        """Package the result, passing the base geometry through unchanged.

        The candidate-owned branch has no graph `reliability` and no cross-attention, so those
        are optional. It supplies `eligible` / `intervened` / `link_score` instead, recorded
        SEPARATELY: `evidence_valid` and `eligible` are what gate the delta, while the graph
        link features are diagnostics only. Collapsing them into one `gate_relevance` field is
        what made the previous zero-delta failure invisible.
        """
        # `evidence_valid` is the candidate-owned branch's gate. For the graph branch the same
        # field carries `has_evidence`, so a report can read one name across both.
        evidence_valid = has_evidence
        if reliability is None:
            # `gate_relevance` must stay populated for `SpanDiagnosticsAccumulator`; for this
            # branch the honest analogue is the intervention mask, not a confidence.
            gate_source = intervened if intervened is not None else has_evidence
            reliability = gate_source.detach().to(final_logits.dtype)
        if attn_norm is None:
            attn_norm = torch.zeros_like(reliability)
        return SpanKnowledgeOutput(
            logits=final_logits,
            span_representations=base_output.span_representations,
            span_indices=base_output.span_indices,
            span_mask=base_output.span_mask,
            # The payload must satisfy `SpanDiagnosticsAccumulator`, which the inherited
            # `SpanDiagnosticsCallback` feeds from a forward hook. Emitting only the
            # variant-specific names cost a whole 5-cell submission: every cell reached the
            # FIRST VALIDATION EPOCH and died with `KeyError: 'residual_norm'`.
            #
            # The shared names are not placeholders here, they are true statements about
            # this architecture:
            #   residual_norm  -- how far the span representation moved. EXACTLY ZERO by
            #                     construction; that is Variant 1's defining property, and
            #                     reporting it keeps #218 and #231 comparable in one field.
            #   logit_delta    -- the late delta itself, which is a logit shift.
            #   attn_norm      -- the evidence summary's norm, same meaning as in #218.
            #   gate_relevance -- reliability: this variant's analogue of the gate, i.e. how
            #                     strongly a candidate is allowed to be rescued.
            knowledge_diagnostics={
                "control": self.knowledge_control,
                "residual_norm": torch.zeros_like(reliability),
                "attn_norm": attn_norm.detach(),
                "logit_delta": delta.detach(),
                "gate_relevance": reliability.detach(),
                # Kept alongside so a #231-specific report can read the precise name.
                "late_delta": delta.detach(),
                "reliability": reliability.detach(),
                "has_evidence": has_evidence.detach(),
                "link_mask": link_mask.detach().bool(),
                # SEPARATED, per the registered decomposition. Merging these into one field is
                # what let an all-zero delta look like a normal run: `evidence_valid` was
                # false everywhere while `link_mask` was true, and only their combination
                # showed the gate was shut.
                "evidence_valid": evidence_valid.detach().bool(),
                "eligible": (
                    eligible.detach().bool()
                    if eligible is not None
                    else torch.ones_like(has_evidence, dtype=torch.bool)
                ),
                "intervened": (
                    intervened.detach().bool()
                    if intervened is not None
                    else has_evidence.detach().bool()
                ),
                # Graph link features: RECORDED, never multiplied into the delta on the
                # candidate-owned branch.
                "graph_link_score": (
                    link_score.detach()
                    if link_score is not None
                    else torch.zeros_like(reliability)
                ),
                "graph_link_mask": link_mask.detach().bool(),
            },
        )

    def score_span_representations(self, rep: torch.Tensor) -> torch.Tensor:
        """Delegate to the frozen inner scorer.

        Variant 1 never re-scores a modified representation, so this exists only for the
        ieforge public contract; the late delta is applied to the scorer's OUTPUT.
        """
        return cast("Any", self.inner).score_span_representations(rep)


class OOFSpanLateLogitKnowledgeModel(SpanLateLogitKnowledgeModel):
    """Route each train row through the anchor that held out its OOF fold.

    The ordinary late-logit model owns one frozen ``inner``.  That is correct for #231 but
    cannot implement Phase A's declared stacking protocol: one full-train anchor is
    in-sample on all 4,560 train rows.  This A0-only subclass keeps the full-train ``inner``
    for validation/test and loads one additional frozen anchor per fold for train rows.

    Fold anchors intentionally live in a plain tuple rather than ``ModuleList``.  They are
    immutable external inputs, not learned state; registering them would write five extra
    BERTs into every A0 checkpoint (roughly 2.2 GB per cell) and make the checkpoint look
    self-contained while omitting the paths/digests that define the cross-fit.  They are
    moved to the one training device on first use, kept in eval mode, excluded from the
    optimizer and checkpoint, and reloaded from the mandatory paths on reconstruction.
    """

    def __init__(
        self,
        *args: Any,
        oof_baseline_paths: list[str] | tuple[str, ...],
        oof_fold_key: str = "a0_oof_fold",
        **kwargs: Any,
    ) -> None:
        paths = tuple(str(path) for path in oof_baseline_paths)
        if len(paths) < 2:
            msg = f"oof_baseline_paths needs at least two folds, got {len(paths)}"
            raise ValueError(msg)
        if len(set(paths)) != len(paths):
            msg = "oof_baseline_paths contains duplicate checkpoints"
            raise ValueError(msg)
        super().__init__(*args, **kwargs)
        if not self.freeze_inner:
            msg = "OOF A0 requires freeze_inner=True"
            raise ValueError(msg)
        self.oof_baseline_paths = paths
        self.oof_fold_key = str(oof_fold_key)

        anchors: list[torch.nn.Module] = []
        for path in paths:
            anchor = copy.deepcopy(self.inner)
            self._load_baseline_into(anchor, path)
            anchor.requires_grad_(False)
            anchor.eval()
            anchors.append(anchor)
        # A tuple is deliberately unregistered; see the class docstring.
        self._oof_anchors = tuple(anchors)
        self._oof_device: torch.device | None = None
        self.inner.eval()

    def train(self, mode: bool = True) -> OOFSpanLateLogitKnowledgeModel:
        """Train the evidence path while every frozen anchor stays deterministic."""
        super().train(mode)
        self.inner.eval()
        for anchor in self._oof_anchors:
            anchor.eval()
        return self

    def _move_oof_anchors(self, device: torch.device) -> None:
        if self._oof_device == device:
            return
        for anchor in self._oof_anchors:
            anchor.to(device=device)
            anchor.eval()
        self._oof_device = device

    @staticmethod
    def _anchor_subbatch(batch: dict[str, Any], rows: torch.Tensor) -> dict[str, Any]:
        """Slice only the public HFSpanClassifier inputs for ``rows``."""
        indices = rows.detach().cpu().tolist()
        return {
            "input_ids": batch["input_ids"].index_select(0, rows),
            "attention_mask": batch["attention_mask"].index_select(0, rows),
            "word_starts": [batch["word_starts"][index] for index in indices],
            "num_words": [batch["num_words"][index] for index in indices],
        }

    def _base_forward_with_span_features(
        self, batch: dict[str, Any]
    ) -> tuple[SpanClassifierOutput, torch.Tensor, torch.Tensor, torch.Tensor]:
        folds = batch.get(self.oof_fold_key)
        if folds is None:
            msg = (
                f"OOF A0 requires batch key {self.oof_fold_key!r}; without it train rows "
                "would silently use the full-train anchor"
            )
            raise ValueError(msg)
        if not isinstance(folds, torch.Tensor) or folds.ndim != 1:
            msg = f"{self.oof_fold_key} must be a rank-1 tensor"
            raise TypeError(msg)
        batch_size = int(batch["input_ids"].shape[0])
        if int(folds.shape[0]) != batch_size:
            msg = (
                f"{self.oof_fold_key} has {folds.shape[0]} rows, expected {batch_size}"
            )
            raise ValueError(msg)
        invalid = (folds < -1) | (folds >= len(self._oof_anchors))
        if bool(invalid.any()):
            values = sorted({int(value) for value in folds[invalid].detach().cpu()})
            msg = f"invalid OOF fold ids {values}; expected -1 or 0..{len(self._oof_anchors) - 1}"
            raise ValueError(msg)

        device = batch["input_ids"].device
        self._move_oof_anchors(device)
        max_words = max(int(value) for value in batch["num_words"])
        max_span_width = int(cast("Any", self.inner).max_span_width)
        hidden = int(cast("Any", self.inner).span_hidden_dim)
        num_labels = len(self.label_list)

        logits: torch.Tensor | None = None
        representations: torch.Tensor | None = None
        indices_out = torch.zeros(
            batch_size,
            max_words,
            max_span_width,
            2,
            dtype=torch.long,
            device=device,
        )
        mask_out = torch.zeros(
            batch_size,
            max_words,
            max_span_width,
            dtype=torch.bool,
            device=device,
        )

        for fold in sorted({int(value) for value in folds.detach().cpu()}):
            rows = (folds == fold).nonzero(as_tuple=False).flatten()
            anchor = self.inner if fold == -1 else self._oof_anchors[fold]
            anchor.eval()
            output = cast("Any", anchor).forward(
                self._anchor_subbatch(batch, rows), return_span_features=True
            )
            if not isinstance(output, SpanClassifierOutput):
                msg = f"OOF anchor {fold} did not return SpanClassifierOutput"
                raise TypeError(msg)
            if (
                output.span_representations is None
                or output.span_indices is None
                or output.span_mask is None
            ):
                msg = f"OOF anchor {fold} omitted public span features"
                raise TypeError(msg)
            group_words = int(output.logits.shape[1])
            if logits is None:
                logits = output.logits.new_zeros(
                    (batch_size, max_words, max_span_width, num_labels)
                )
                representations = output.span_representations.new_zeros(
                    (batch_size, max_words, max_span_width, hidden)
                )
            if representations is None:
                msg = "OOF representation buffer was not initialised with logits"
                raise RuntimeError(msg)
            logits[rows, :group_words] = output.logits
            representations[rows, :group_words] = output.span_representations
            indices_out[rows, :group_words] = output.span_indices
            mask_out[rows, :group_words] = output.span_mask

        if logits is None or representations is None:
            msg = "OOF routing received an empty batch"
            raise ValueError(msg)
        base_output = SpanClassifierOutput(
            logits=logits,
            span_representations=representations,
            span_indices=indices_out,
            span_mask=mask_out,
        )
        return base_output, representations, indices_out, mask_out
