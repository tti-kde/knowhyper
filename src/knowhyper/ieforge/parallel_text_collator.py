"""Parallel-branch evidence collation for the GRAFT-style baseline (#189).

:class:`~knowhyper.ieforge.text_collator.PairAwareTextKnowledgeCollator` rebuilds
source and knowledge into ONE 512-token sequence as BERT segments A/B.  GRAFT
encodes the two separately, so that collator cannot serve this arm: there is no
independent evidence tensor to hand a second encoder, and the source is truncated
to make room for evidence that no longer shares its sequence.

This collator therefore leaves the base collator's ``input_ids`` /
``attention_mask`` **untouched** and emits the evidence as its own fixed-budget
tensor pair.  ``source_truncation`` is consequently NOT part of the
``graft_style_real - graft_style_masked`` contrast, which is the point: that delta
measures retrieved *content* only.  Comparing either arm against ``serialized``
does fold source truncation back in, so the issue records it as secondary.

Arms are produced along TWO independent axes, so one class covers four combinations.

``evidence_mode`` -- what the branch sees:

``real``
    the payload's own token IDs, whole items kept or dropped against the evidence
    budget exactly as the R2b collator does.

``masked``
    the same instance's ``real`` token sequence with every non-special token
    replaced by ``[MASK]``.  Built by masking the real IDs rather than by
    re-tokenizing a masked STRING, so the two arms are matched per instance on
    sequence length, retained item count and special-token positions -- an
    exact-token-count capacity control.  ``real - masked`` then isolates content
    from "a second encoder and a gate were added".

``payload`` -- what the evidence IS:

``serialized``
    R2b's ``[HG] ... [/HG]`` blocks, tokenized verbatim (the serializer already
    emitted role tokens, so no wrapper or redaction pass runs).

``definitions``
    R2a's raw ``node_features_text``.  Added for the #189 follow-up asking whether
    the parallel branch transmits NATURAL TEXT when it does not transmit serialized
    structure.  This payload is prose with no boundary tags, so the R2a wrapper and
    redaction policies DO run -- and because it is licensed upstream text under the
    project's R2a rules, the mention/redaction policy is load-bearing rather than
    cosmetic: DDI masks both drug surfaces in the source, so a surface left in the
    evidence would reintroduce the identity the task hides.

No trainable module lives here (that is Phase 3's model).  Nothing this collator
EMITS carries evidence text -- only token IDs the model already sees, plus
payload-free counts and fingerprints -- but note that on the ``definitions`` path the
INPUT is licensed text, so the R2a text rules in CLAUDE.md apply to what reaches an
artifact from anywhere else in that arm.  All outputs are ordinary CPU leaf tensors,
required by the project's ``num_workers>=4`` setting.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import Any

import torch
from ieforge.knowledge.integration.collators import _InstanceMetadataView

from knowhyper.ieforge.text_collator import _KnowledgeSegment

EVIDENCE_MODES = ("real", "masked")
#: `serialized` = R2b's [HG] blocks. `definitions` = R2a's node_features_text, added
#: for the #189 follow-up that asks whether the parallel branch transmits NATURAL TEXT
#: when it does not transmit serialized structure -- separating "the [HG] format does
#: not carry" from "the fusion design does not carry".
PAYLOADS = ("serialized", "definitions")
MENTION_MODES = ("include", "omit", "redact")
_REDACTION_POLICY = "target-surfaces-role-token-v1"
#: Bumped when the evidence assembly or the masking rule changes, so an edit
#: re-fingerprints instead of silently mixing two policies inside one contrast.
_POLICY_VERSION = 1


def truncate_whole_segments(
    segments: Iterable[_KnowledgeSegment], budget: int
) -> tuple[list[int], list[int]]:
    """Keep whole structural blocks in order; drop the ones that do not fit.

    Same rule as the R2b collator's ``_truncate_whole_segments``: a serialized
    ``[HG] ... [/HG]`` block is not gracefully degradable, and cutting one from the
    right removes the trailing ``polarity=`` field first -- the one typed signal
    this project has positive evidence for. Blocks arrive in ``he_id`` order, so
    dropping from the tail is deterministic.

    Budgeting uses ``budget_cost`` (the TREATED rendering's length) when set, so a
    control whose corrupted values tokenize to a different length still retains the
    same number of blocks.

    Returns ``(token_ids, per_segment_counts)``; a count of 0 marks a dropped block.
    """
    tokens: list[int] = []
    counts: list[int] = []
    used = 0
    for segment in segments:
        item = [*segment.prefix, *segment.body, *segment.suffix]
        cost = segment.budget_cost if segment.budget_cost is not None else len(item)
        if used + cost > budget:
            counts.append(0)
            continue
        used += cost
        tokens.extend(item)
        counts.append(len(item))
    return tokens, counts


def truncate_round_robin(
    segments: Sequence[_KnowledgeSegment], budget: int
) -> tuple[list[int], list[int]]:
    """Split the budget evenly across items, clipping each body from the right.

    R2a's rule (``PairAwareTextKnowledgeCollator._truncate_knowledge``): reserve every
    item's boundary tags, then grow the bodies one token at a time round-robin. EVERY
    item survives with a clipped body; nothing is dropped.

    The definitions payload must use THIS and not ``truncate_whole_segments``, because
    definitions are long prose. Measured on the real R2a cache at the production
    126-token budget, whole-segment dropping discarded so much that ChemProt saw 45.2
    mean evidence tokens against R2a's 102.7, and 25.9% of instances that HAD retrieved
    definitions were pushed to ``present=False`` (BioRED 37.2%). Those rows force
    ``g=0``, i.e. they are the plain baseline in BOTH the real and masked arm and
    contribute a structural zero to ``real - masked`` -- diluting the very contrast the
    arm exists to measure, while the config claimed budget parity with R2a.

    Returns ``(token_ids, per_segment_counts)``; unlike the whole-segment rule a count
    is never 0 unless the item was empty to begin with.
    """
    wrapper_tokens = sum(
        len(segment.prefix) + len(segment.suffix) for segment in segments
    )
    if wrapper_tokens > budget:
        msg = (
            "max_evidence_tokens is too small to preserve definition boundary tags: "
            f"need {wrapper_tokens}, got {budget}"
        )
        raise ValueError(msg)
    body_counts = [0] * len(segments)
    remaining = budget - wrapper_tokens
    while remaining:
        progressed = False
        for index, segment in enumerate(segments):
            if body_counts[index] >= len(segment.body):
                continue
            body_counts[index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    tokens: list[int] = []
    counts: list[int] = []
    for segment, body_count in zip(segments, body_counts, strict=True):
        item = [*segment.prefix, *segment.body[:body_count], *segment.suffix]
        tokens.extend(item)
        counts.append(len(item))
    return tokens, counts


class ParallelEvidenceKnowledgeCollator:
    """Emit retrieved evidence as a SEPARATE encoder input beside the source.

    Wraps the dataset's RE collator, runs the knowledge pipeline once per pair
    carrier, and adds the ``knowledge_evidence_*`` keys
    :class:`~knowhyper.ieforge.models.ParallelTextGatedKnowledgeModel` consumes.
    The wrapped collator's own keys pass through unchanged, so the source branch of
    this arm is byte-identical to the ``base`` arm's input -- which is what makes
    ``g=0`` reproduce baseline logits exactly.

    An instance with no retrieved evidence emits ``[CLS] [SEP]`` and
    ``present=False``; the model then forces its gate to 0 rather than attending to
    a padding-only sequence. Empty evidence is common (46% of BioRED sub-HGs), so
    this is the normal path, not an error.
    """

    def __init__(
        self,
        base_collator: Any,
        pipeline: Any,
        *,
        evidence_mode: str = "real",
        max_evidence_tokens: int = 128,
        payload: str = "serialized",
        # `include` matches PairAwareTextKnowledgeCollator's own default, so a
        # definitions arm that does not set the key renders the SAME payload R2a
        # renders. Defaulting to `omit` here (the first version did) would silently
        # drop the "mention: " prefix on ChemProt and BioRED and turn a route
        # comparison into a payload comparison. Ignored on the serialized path, where
        # the [HG] block already carries its own role tokens.
        mention_mode: str = "include",
        redaction_policy: str = _REDACTION_POLICY,
        instance_metadata: Mapping[str, Any] | None = None,
        **inherited: Any,
    ) -> None:
        """Pin the arm, the evidence budget, and the payload key to consume.

        The GRAFT configs inherit the R2b arm, so a handful of that collator's keys
        (``mode`` / ``mention_mode`` / ``min_source_tokens`` /
        ``max_knowledge_tokens``) arrive here and mean NOTHING on this path -- there
        is no shared sequence to budget and the serializer already emitted complete
        ``[HG]`` blocks. Swallowing them silently would be worse than tolerating
        them: someone tuning ``max_knowledge_tokens`` to change the evidence budget
        would see the run complete with the budget unchanged. So the inherited values
        are accepted (removing them from the parent would fork the config chain) but
        the CONFLICTING one is rejected outright.
        """
        budget_key = "max_knowledge_tokens"
        if inherited.get(budget_key) is not None and int(inherited[budget_key]) != int(
            max_evidence_tokens
        ):
            msg = (
                f"{budget_key}={inherited[budget_key]} is inherited from the R2b arm "
                "and does NOT control this collator; set max_evidence_tokens="
                f"{inherited[budget_key]} instead (currently {max_evidence_tokens})"
            )
            raise ValueError(msg)
        if evidence_mode not in EVIDENCE_MODES:
            msg = (
                f"unsupported evidence_mode: {evidence_mode!r}; "
                f"expected one of {EVIDENCE_MODES}"
            )
            raise ValueError(msg)
        if payload not in PAYLOADS:
            msg = (
                f"unsupported payload: {payload!r}; expected one of {PAYLOADS}. "
                "`serialized` consumes the R2b [HG] blocks verbatim; `definitions` "
                "consumes R2a's node_features_text and applies the mention/redaction "
                "policies that payload requires."
            )
            raise ValueError(msg)
        if payload == "definitions" and mention_mode not in MENTION_MODES:
            msg = (
                f"unsupported mention_mode: {mention_mode!r}; expected one of "
                f"{MENTION_MODES}"
            )
            raise ValueError(msg)
        if redaction_policy != _REDACTION_POLICY:
            # Refuse an unknown policy rather than silently NOT redacting: on DDI both
            # drug surfaces are masked in the source, so leaving a surface in the
            # evidence text would reintroduce the identity the task hides.
            #
            # Validated UNCONDITIONALLY, not only on the definitions path. The value is
            # folded into `evidence_policy_fingerprint` below either way, so accepting
            # an arbitrary string on the serialized path would give two behaviourally
            # IDENTICAL arms two different identities -- a fingerprint collision in
            # reverse, which is just as bad for provenance as two arms sharing one.
            msg = f"unsupported redaction policy: {redaction_policy!r}"
            raise ValueError(msg)
        # 2 = the [CLS] and [SEP] this collator adds around the evidence body.
        if max_evidence_tokens < 2:
            msg = (
                "max_evidence_tokens must leave room for [CLS]/[SEP], got "
                f"{max_evidence_tokens}"
            )
            raise ValueError(msg)
        self.base_collator = base_collator
        self.pipeline = pipeline
        self.evidence_mode = evidence_mode
        self.max_evidence_tokens = int(max_evidence_tokens)
        self.payload = payload
        self.mention_mode = mention_mode
        self.redaction_policy = redaction_policy
        self.instance_metadata = instance_metadata
        # R2a and R2b publish under distinct keys so a misconfigured arm yields EMPTY
        # evidence instead of silently consuming the other arm's payload.
        self._items_key = (
            "serialized_subhg" if payload == "serialized" else "retrieved_definitions"
        )
        self.tokenizer = base_collator.tokenizer
        mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_id is None:
            msg = "the masked evidence arm requires a tokenizer with a [MASK] token"
            raise ValueError(msg)
        self._mask_id = int(mask_id)
        policy = {
            "policy": "parallel_evidence_v1",
            "policy_version": _POLICY_VERSION,
            "evidence_mode": evidence_mode,
            "max_evidence_tokens": self.max_evidence_tokens,
            "payload": payload,
            # Folded in so two arms differing only in the mention policy cannot share
            # provenance. Recorded unconditionally rather than only for `definitions`:
            # a conditional key makes the serialized arms' fingerprints depend on which
            # branch of this dict ran, which is harder to reason about later than one
            # extra constant.
            "mention_mode": mention_mode,
            "redaction_policy": redaction_policy,
        }
        canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"))
        self.evidence_policy_fingerprint = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()[:16]

    def __call__(self, examples: Iterable[Any]) -> MutableMapping[str, Any]:
        """Add the evidence branch to the base collator's batch."""
        example_list = list(examples)
        batch = self.base_collator(example_list)
        if not isinstance(batch, MutableMapping):
            msg = "base_collator must return a mutable mapping"
            raise TypeError(msg)
        carriers = batch.pop("_knowledge_examples", None)
        retrieval_examples = list(carriers) if carriers is not None else example_list
        if self.instance_metadata is not None:
            retrieval_examples = [
                _InstanceMetadataView(item, self.instance_metadata)
                for item in retrieval_examples
            ]
        contexts = [self.pipeline.run(item) for item in retrieval_examples]

        input_ids = batch.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            msg = "base_collator must return tensor input_ids"
            raise TypeError(msg)
        if len(contexts) != input_ids.shape[0]:
            # The same alignment failure the R2b collator guards: a query builder
            # emitting a different number of carrier rows than encoded pairs would
            # pair each row with ANOTHER instance's evidence and still train.
            msg = (
                "knowledge context count must match encoded target-pair rows: "
                f"{len(contexts)} != {input_ids.shape[0]}"
            )
            raise ValueError(msg)
        if input_ids.shape[0] == 0:
            return batch

        rows: list[list[int]] = []
        token_counts: list[int] = []
        dropped: list[int] = []
        retained_blocks: list[int] = []
        present: list[bool] = []
        serializer_fingerprints: list[str] = []
        selection_fingerprints: list[str] = []
        for context in contexts:
            metadata = getattr(context, "metadata", {}) or {}
            items = list(metadata.get(self._items_key, []))
            # Target surfaces come from the QUERY, not the evidence item, and are only
            # read on the definitions path (mention_mode=redact). Fetched per row
            # because each pair has its own head/tail surfaces.
            target_groups: dict[str, list[str]] = {}
            if self.payload == "definitions":
                query_metadata = dict(
                    getattr(getattr(context, "query", None), "metadata", {}) or {}
                )
                target_groups = {
                    str(role): [str(surface) for surface in surfaces]
                    for role, surfaces in dict(
                        query_metadata.get("target_entity_groups", {})
                    ).items()
                }
            row, counts = self._build_evidence_row(items, target_groups)
            rows.append(row)
            token_counts.append(len(row))
            dropped.append(sum(1 for count in counts if count == 0))
            kept = sum(1 for count in counts if count > 0)
            retained_blocks.append(kept)
            present.append(kept > 0)
            serializer_fingerprints.append(
                str(metadata.get("text_serialization_fingerprint", ""))
            )
            selection_fingerprints.append(
                str(metadata.get("subhg_selection_fingerprint", ""))
            )

        pad_id = int(self.tokenizer.pad_token_id or 0)
        width = max(len(row) for row in rows)
        batch["knowledge_evidence_input_ids"] = torch.tensor(
            [row + [pad_id] * (width - len(row)) for row in rows], dtype=torch.long
        )
        batch["knowledge_evidence_attention_mask"] = torch.tensor(
            [[1] * len(row) + [0] * (width - len(row)) for row in rows],
            dtype=torch.long,
        )
        batch["knowledge_evidence_present"] = torch.tensor(present, dtype=torch.bool)
        batch["knowledge_evidence_token_count"] = torch.tensor(
            token_counts, dtype=torch.long
        )
        batch["knowledge_evidence_items_dropped"] = torch.tensor(
            dropped, dtype=torch.long
        )
        batch["knowledge_evidence_retained_blocks"] = torch.tensor(
            retained_blocks, dtype=torch.long
        )
        batch["knowledge_evidence_mode"] = [self.evidence_mode for _ in rows]
        # Payload-free provenance: fingerprints only, never the evidence text. The
        # selection fingerprint is what makes "the graph arm saw the same 3 edges"
        # auditable from a run record.
        batch["knowledge_serializer_fingerprint"] = serializer_fingerprints
        batch["knowledge_selection_fingerprint"] = selection_fingerprints
        batch["knowledge_evidence_policy_fingerprint"] = [
            self.evidence_policy_fingerprint for _ in rows
        ]
        return batch

    def _build_evidence_row(
        self,
        items: list[Mapping[str, Any]],
        target_groups: Mapping[str, list[str]] | None = None,
    ) -> tuple[list[int], list[int]]:
        """Assemble one row of ``[CLS] evidence [SEP]``, masked when so configured."""
        cls_id = int(self.tokenizer.cls_token_id)
        sep_id = int(self.tokenizer.sep_token_id)
        segments = [
            self._tokenize_item(item, target_entity_groups=target_groups)
            for item in items
        ]
        # Payload-aware, mirroring R2a: whole-block drop for [HG] structure (a block
        # cut mid-field loses its trailing polarity= first), round-robin for prose
        # definitions (dropping whole definitions starves the arm -- measured 45.2 vs
        # R2a's 102.7 mean evidence tokens on ChemProt, and 26-37% of instances pushed
        # to present=False on ChemProt/BioRED).
        if self.payload == "definitions":
            body, counts = truncate_round_robin(segments, self.max_evidence_tokens - 2)
        else:
            body, counts = truncate_whole_segments(
                segments, self.max_evidence_tokens - 2
            )
        if self.evidence_mode == "masked":
            # Mask the REAL ids rather than re-tokenizing a masked string: only
            # this keeps `real` and `masked` matched per instance on length,
            # retained-block count and special-token positions. Re-tokenizing
            # "[MASK] [MASK] ..." would drift in length and turn an exact-count
            # capacity control into an approximate one.
            body = [self._mask_id] * len(body)
        return [cls_id, *body, sep_id], counts

    def _tokenize_item(
        self,
        item: Mapping[str, Any],
        *,
        target_entity_groups: Mapping[str, list[str]] | None = None,
    ) -> _KnowledgeSegment:
        """Tokenize one evidence item, per the configured payload.

        The serializer already emitted the complete ``[HG] ... [/HG]`` string,
        including role tokens for target endpoints, so no wrapper or redaction pass
        runs here -- re-applying either would double-wrap the block and rewrite
        structural field values. ``budget_cost`` comes from the treated rendering so
        a shuffled payload retains the same number of blocks.

        The ``definitions`` payload is raw ``node_features_text`` and DOES need the
        R2a policies: it is prose with no boundary tags, and on DDI both target drug
        surfaces are masked in the source, so a surface left in the evidence would
        reintroduce the identity the task hides.
        """
        if self.payload == "definitions":
            return self._tokenize_definition(item, target_entity_groups)
        text = str(item.get("text", ""))
        budget_text = str(item.get("budget_text") or text)
        return _KnowledgeSegment(
            prefix=(),
            body=tuple(self._tokenize(text)),
            suffix=(),
            budget_cost=len(self._tokenize(budget_text)),
        )

    def _tokenize_definition(
        self,
        item: Mapping[str, Any],
        target_entity_groups: Mapping[str, list[str]] | None,
    ) -> _KnowledgeSegment:
        """Tokenize one retrieved definition, wrapped and redacted as R2a does.

        Mirrors ``PairAwareTextKnowledgeCollator._tokenize_item``'s definitions path so
        the two arms differ in the INJECTION ROUTE (parallel branch vs one shared
        sequence) and not in how the payload is rendered. If the rendering differed
        too, `graft_style_text - retrieved_definitions` would confound the route with
        the text itself, which is the whole comparison this arm exists to make.

        ``budget_cost`` is left None so whole-block budgeting uses the real length:
        unlike the shuffled R2b control there is no second rendering to match against.
        """
        role = str(item.get("role", "seed")).upper()
        mention = str(item.get("mention", ""))
        text = str(item.get("text", ""))
        if self.mention_mode == "redact":
            groups = {
                key: list(values)
                for key, values in (target_entity_groups or {}).items()
            }
            groups.setdefault(str(item.get("role", "seed")), []).append(mention)
            body = self._tokenize(self._redact_target_surfaces(text, groups))
        elif self.mention_mode == "omit":
            body = self._tokenize(text)
        else:
            body = self._tokenize(f"{mention}: {text}")
        return _KnowledgeSegment(
            prefix=tuple(self._tokenize(f"[{role}_DEFINITION]")),
            body=tuple(body),
            suffix=tuple(self._tokenize(f"[/{role}_DEFINITION]")),
        )

    def _redact_target_surfaces(
        self, text: str, groups: Mapping[str, list[str]]
    ) -> str:
        """Replace target-entity surfaces with their role token.

        Longest surface first, so a short surface that is a substring of a longer one
        cannot pre-empt it and leave the longer form partly intact.
        """
        import re

        redacted = text
        replacements = [
            (surface, f"[{role.upper()}_ENTITY]")
            for role, surfaces in groups.items()
            for surface in surfaces
            if surface.strip()
        ]
        for surface, token in sorted(replacements, key=lambda pair: -len(pair[0])):
            redacted = re.sub(re.escape(surface), token, redacted, flags=re.IGNORECASE)
        return redacted

    def _tokenize(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        return [int(token) for token in encoded["input_ids"]]
