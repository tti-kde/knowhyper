"""Pair-aware, fixed-budget text knowledge collation for RE baselines."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import torch
from ieforge.knowledge.integration.collators import _InstanceMetadataView


@dataclass(frozen=True)
class _KnowledgeSegment:
    prefix: tuple[int, ...]
    body: tuple[int, ...]
    suffix: tuple[int, ...]
    # Token cost used for keep/drop decisions. Differs from len(body) only on the
    # serialized path, where both arms must be budgeted on the treated rendering
    # so a corrupted value's different length cannot change how many blocks fit.
    budget_cost: int | None = None


class PairAwareTextKnowledgeCollator:
    """Append raw text knowledge to each already markerized RE pair.

    The wrapped RE collator establishes the exact pair rows and their markerized
    source token IDs. The knowledge pipeline then runs once for each corresponding
    pair carrier. Source and knowledge are rebuilt as BERT sequence A/B under one
    total budget, while emitting the source length needed by an OFF-short control.

    All output tensors are ordinary CPU leaves, so this wrapper is safe with the
    project's required ``num_workers>=4`` setting.
    """

    def __init__(
        self,
        base_collator: Any,
        pipeline: Any,
        *,
        max_length: int = 512,
        max_knowledge_tokens: int = 128,
        min_source_tokens: int = 256,
        mode: str = "definitions",
        payload: str = "definitions",
        mention_mode: str = "include",
        redaction_policy: str = "target-surfaces-role-token-v1",
        emit_raw_audit_text: bool = False,
        instance_metadata: Mapping[str, Any] | None = None,
        feature_channels: Iterable[Any] | None = None,
        **_unused: Any,
    ) -> None:
        if max_length < 4:
            msg = "max_length must leave room for BERT special tokens"
            raise ValueError(msg)
        if max_knowledge_tokens < 0 or min_source_tokens < 0:
            msg = "token budgets must be non-negative"
            raise ValueError(msg)
        if mode not in {"definitions", "serialized", "off_short"}:
            msg = f"unsupported text knowledge mode: {mode!r}"
            raise ValueError(msg)
        if payload not in {"definitions", "serialized"}:
            msg = f"unsupported text knowledge payload: {payload!r}"
            raise ValueError(msg)
        if mode not in {"off_short", payload}:
            msg = (
                f"text knowledge mode {mode!r} requires payload {mode!r}, "
                f"got {payload!r}"
            )
            raise ValueError(msg)
        if mention_mode not in {"include", "omit", "redact", "sham"}:
            msg = f"unsupported definition mention mode: {mention_mode!r}"
            raise ValueError(msg)
        self.base_collator = base_collator
        self.pipeline = pipeline
        self.max_length = max_length
        self.max_knowledge_tokens = max_knowledge_tokens
        self.min_source_tokens = min_source_tokens
        self.mode = mode
        self.payload = payload
        self.mention_mode = mention_mode
        self.redaction_policy = redaction_policy
        # R2a and R2b publish under distinct metadata keys so a misconfigured arm
        # yields empty knowledge instead of silently borrowing the other arm's
        # items. `payload` is tracked separately from `mode` because an off_short
        # control must still consume its parent arm's payload: it reproduces that
        # arm's source truncation and only then drops the knowledge tokens.
        self._items_key = (
            "serialized_subhg" if payload == "serialized" else "retrieved_definitions"
        )
        policy_payload = json.dumps(
            {
                "mention_mode": mention_mode,
                "redaction_policy": redaction_policy,
                # Distinguishes the R2a and R2b audit records; the text cache is
                # keyed by the serializer's own fingerprint, not by this value.
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self.text_policy_fingerprint = hashlib.sha256(
            policy_payload.encode("utf-8")
        ).hexdigest()[:16]
        self.emit_raw_audit_text = emit_raw_audit_text
        self.instance_metadata = instance_metadata
        # #220 used no feature channels: it injected text as sequence B and nothing else.
        # The fixed cumulative NER arm keeps that exact input while exposing #231's incidence
        # and candidate-span tensors from the SAME pipeline contexts. Accepting this config
        # through ``**_unused`` used to be a silent no-op, yielding a text-only run wearing a
        # graph label. Store and execute the channels explicitly instead.
        self.feature_channels = tuple(feature_channels or ())
        self.tokenizer = base_collator.tokenizer
        added_vocab = getattr(self.tokenizer, "get_added_vocab", dict)()
        base_vocab_size = int(getattr(self.tokenizer, "vocab_size", 0))
        self._marker_token_ids = frozenset(
            int(token_id)
            for token_id in added_vocab.values()
            if token_id is not None and int(token_id) >= base_vocab_size
        )

    def __call__(self, examples: Iterable[Any]) -> MutableMapping[str, Any]:
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
        attention_mask = batch.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(
            attention_mask, torch.Tensor
        ):
            msg = "base_collator must return tensor input_ids and attention_mask"
            raise TypeError(msg)
        if len(contexts) != input_ids.shape[0]:
            msg = (
                "knowledge context count must match encoded target-pair rows: "
                f"{len(contexts)} != {input_ids.shape[0]}"
            )
            raise ValueError(msg)
        if input_ids.shape[0] == 0:
            return batch

        rows: list[list[int]] = []
        type_rows: list[list[int]] = []
        # Per-row count of SOURCE tokens that survived truncation, excluding the
        # [CLS]/[SEP] this collator adds. Needed to realign TOKEN-LEVEL labels (NER);
        # unused for sequence-level labels (RE).
        kept_source_lengths: list[int] = []
        diagnostics: list[dict[str, Any]] = []
        audit_records: list[dict[str, Any]] = []
        for row_ids, row_mask, context in zip(
            input_ids.tolist(), attention_mask.tolist(), contexts, strict=True
        ):
            source = [
                token for token, keep in zip(row_ids, row_mask, strict=True) if keep
            ]
            source_content = self._strip_special_tokens(source)
            metadata = getattr(context, "metadata", {}) or {}
            # R2a and R2b publish their payloads under distinct keys so a
            # misconfigured arm fails loudly (empty knowledge) instead of
            # silently serializing the other arm's items.
            items = list(metadata.get(self._items_key, []))
            query = getattr(context, "query", None)
            query_metadata = dict(getattr(query, "metadata", {}) or {})
            target_groups = {
                str(role): [str(surface) for surface in surfaces]
                for role, surfaces in dict(
                    query_metadata.get("target_entity_groups", {})
                ).items()
            }
            knowledge_segments = [
                self._tokenize_item(item, target_entity_groups=target_groups)
                for item in items
            ]
            row, token_types, diag, kept_source, kept_knowledge_items = self._build_row(
                source_content, knowledge_segments
            )
            kept_source_lengths.append(len(kept_source))
            diag["roles"] = [str(item.get("role", "seed")) for item in items]
            diag["evidence_provenance"] = [
                {
                    key: item.get(key)
                    for key in (
                        "node_id",
                        "modality",
                        "content_hash",
                        "language",
                        "source",
                        "source_version",
                        "license_tag",
                        "license_id",
                        "license_url",
                        "usage_scope",
                        "redistribution_scope",
                        "payload_class",
                        "source_field",
                        "source_release",
                        "upstream_source",
                        "upstream_release",
                        "reviewed_at",
                        "registry_digest",
                    )
                }
                for item in items
            ]
            diag["mention_mode"] = self.mention_mode
            diag["redaction_policy"] = self.redaction_policy
            rows.append(row)
            type_rows.append(token_types)
            diagnostics.append(diag)
            if self.emit_raw_audit_text:
                audit_records.append(
                    {
                        # Keep the exact retained sequence-A segment separate from
                        # the unprocessed query-builder source.
                        "model_source_text": self._decode(kept_source),
                        # Decoded only in worker memory for the leakage audit. The
                        # audit script hashes/aggregates these strings and never
                        # writes the model-visible knowledge text to artifacts.
                        "model_knowledge_items": [
                            self._decode(token_ids) if token_ids else ""
                            for token_ids in kept_knowledge_items
                        ],
                        "query_source_text": str(getattr(query, "text", "")),
                        "final_input_ids_hash": hashlib.sha256(
                            json.dumps(row, separators=(",", ":")).encode("utf-8")
                        ).hexdigest(),
                        "final_input_length": len(row),
                        "entities": [
                            str(entity)
                            for entity in getattr(query, "entities", ()) or ()
                        ],
                        "target_entity_groups": query_metadata.get(
                            "target_entity_groups", {}
                        ),
                        "metadata": query_metadata,
                        "items": [dict(item) for item in items],
                        "text_policy_fingerprint": self.text_policy_fingerprint,
                    }
                )

        pad_id = int(self.tokenizer.pad_token_id or 0)
        max_row = max(len(row) for row in rows)
        batch["input_ids"] = torch.tensor(
            [row + [pad_id] * (max_row - len(row)) for row in rows], dtype=torch.long
        )
        batch["attention_mask"] = torch.tensor(
            [[1] * len(row) + [0] * (max_row - len(row)) for row in rows],
            dtype=torch.long,
        )
        batch["token_type_ids"] = torch.tensor(
            [row + [0] * (max_row - len(row)) for row in type_rows], dtype=torch.long
        )
        self._realign_token_labels(batch, kept_source_lengths, max_row)
        for name, dtype in (
            ("knowledge_token_count", torch.long),
            ("source_token_count_before", torch.long),
            ("source_token_count_after", torch.long),
            ("source_tokens_removed", torch.long),
            ("off_short_source_tokens", torch.long),
            ("source_truncation_rate", torch.float32),
        ):
            batch[name] = torch.tensor(
                [item[name] for item in diagnostics], dtype=dtype
            )
        batch["knowledge_item_token_counts"] = [
            item["item_counts"] for item in diagnostics
        ]
        batch["knowledge_pair_roles"] = [item["roles"] for item in diagnostics]
        batch["knowledge_evidence_provenance"] = [
            item["evidence_provenance"] for item in diagnostics
        ]
        batch["knowledge_tokens_injected"] = torch.tensor(
            [item["tokens_injected"] for item in diagnostics], dtype=torch.long
        )
        batch["knowledge_items_dropped"] = torch.tensor(
            [item["items_dropped"] for item in diagnostics], dtype=torch.long
        )
        batch["source_marker_count_before"] = torch.tensor(
            [item["marker_before"] for item in diagnostics], dtype=torch.long
        )
        batch["source_marker_count_after"] = torch.tensor(
            [item["marker_after"] for item in diagnostics], dtype=torch.long
        )
        batch["knowledge_mention_mode"] = [item["mention_mode"] for item in diagnostics]
        batch["knowledge_redaction_policy"] = [
            item["redaction_policy"] for item in diagnostics
        ]
        batch["knowledge_text_policy_fingerprint"] = [
            self.text_policy_fingerprint for _ in diagnostics
        ]
        if self.emit_raw_audit_text:
            batch["knowledge_audit_records"] = audit_records
        for channel in self.feature_channels:
            feature_batch = channel.collate(contexts)
            if not isinstance(feature_batch, Mapping):
                msg = (
                    "PairAwareTextKnowledgeCollator expected feature channel to return a "
                    f"mapping, got {type(feature_batch).__name__}"
                )
                raise TypeError(msg)
            collisions = sorted(set(batch).intersection(feature_batch))
            if collisions:
                msg = (
                    "PairAwareTextKnowledgeCollator feature channel key collision: "
                    f"{collisions}. Overwriting a second-segment or graph tensor would "
                    "silently change the cumulative arm."
                )
                raise ValueError(msg)
            batch.update(feature_batch)
        return batch

    def _strip_special_tokens(self, token_ids: list[int]) -> list[int]:
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        start = 1 if token_ids and token_ids[0] == cls_id else 0
        end = -1 if token_ids and token_ids[-1] == sep_id else None
        return token_ids[start:end]

    def _tokenize_item(
        self,
        item: Mapping[str, Any],
        *,
        target_entity_groups: Mapping[str, list[str]] | None = None,
    ) -> _KnowledgeSegment:
        role = str(item.get("role", "seed")).upper()
        mention = str(item.get("mention", ""))
        text = str(item.get("text", ""))
        if self.payload == "serialized":
            # The serializer already emitted the complete "[HG] ... [/HG]" string,
            # including its own role tokens for target endpoints. Re-applying the
            # definition wrapper or the mention/redaction policies here would
            # double-wrap it and rewrite structural field values, so the body is
            # tokenized verbatim and the boundary tags come from the payload.
            budget_text = str(item.get("budget_text") or text)
            return _KnowledgeSegment(
                prefix=(),
                body=tuple(self._tokenize(text)),
                suffix=(),
                budget_cost=len(self._tokenize(budget_text)),
            )
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
        if self.mention_mode == "sham":
            mention_prefix = self._tokenize(f"{mention}:")
            filler_count = max(0, len(body) - len(mention_prefix))
            filler_id = self.tokenizer.mask_token_id
            if filler_id is None:
                filler_id = self.tokenizer.unk_token_id
            if filler_id is None:
                msg = "mention-sham mode requires a tokenizer mask or unknown token"
                raise ValueError(msg)
            body = [*mention_prefix, *([int(filler_id)] * filler_count)]
        return _KnowledgeSegment(
            prefix=tuple(self._tokenize(f"[{role}_DEFINITION]")),
            body=tuple(body),
            suffix=tuple(self._tokenize(f"[/{role}_DEFINITION]")),
        )

    def _redact_target_surfaces(
        self, text: str, groups: Mapping[str, list[str]]
    ) -> str:
        if self.redaction_policy != "target-surfaces-role-token-v1":
            msg = f"unsupported definition redaction policy: {self.redaction_policy!r}"
            raise ValueError(msg)
        redacted = text
        replacements = [
            (surface, f"[{role.upper()}_ENTITY]")
            for role, surfaces in groups.items()
            for surface in surfaces
            if surface.strip()
        ]
        for surface, token in sorted(replacements, key=lambda item: -len(item[0])):
            redacted = re.sub(re.escape(surface), token, redacted, flags=re.IGNORECASE)
        return redacted

    def _tokenize(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        return [int(token) for token in encoded["input_ids"]]

    def _decode(self, token_ids: list[int]) -> str:
        decode = getattr(self.tokenizer, "decode", None)
        if decode is None:
            return " ".join(str(token) for token in token_ids)
        return str(
            decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    def _realign_token_labels(
        self,
        batch: MutableMapping[str, Any],
        kept_source_lengths: list[int],
        max_row: int,
    ) -> None:
        """Rebuild TOKEN-LEVEL labels to match the reassembled sequence.

        This collator was written for RE, whose `labels` are one class per EXAMPLE and
        so are unaffected by re-tokenising the row. NER labels are one class per TOKEN,
        and the rebuilt row is `[CLS] kept_source [SEP] knowledge [SEP]` -- a different
        length from the base collator's `[CLS] source [SEP]`. Leaving `labels` alone
        made every text-arm cell die with
        `ValueError: Expected input batch_size (17920) to match target batch_size
        (8832)` (280 vs 138 positions at bs=64).

        Alignment rules, matching how the row is assembled:
          * position 0 is the [CLS] this collator emits -> ignore index
          * the next `kept_source_lengths[i]` positions are source content, in order,
            so they take the original labels' content prefix. `_truncate_source`
            drops from the TAIL when no markers are present (`source[:capacity]`),
            which is the NER case, so a prefix copy is correct.
          * every remaining position ([SEP], knowledge tokens, [SEP], padding) is
            not a labelled token -> ignore index.

        Uses the loss's own ignore index (-100 by default) so the dropped positions
        contribute nothing rather than being scored as class 0 ("O"), which would
        silently reward predicting O on knowledge tokens.
        """
        labels = batch.get("labels")
        if not isinstance(labels, torch.Tensor) or labels.dim() != 2:
            return  # sequence-level labels (RE) or none: nothing to realign
        ignore_index = -100
        n_rows = labels.shape[0]
        if n_rows != len(kept_source_lengths):
            msg = (
                "label row count does not match rebuilt rows: "
                f"{n_rows} != {len(kept_source_lengths)}"
            )
            raise ValueError(msg)
        new = torch.full((n_rows, max_row), ignore_index, dtype=labels.dtype)
        for i, kept in enumerate(kept_source_lengths):
            # The base collator's row is [CLS] content... [SEP]; content starts at 1.
            content = labels[i][1 : 1 + kept]
            take = min(kept, content.shape[0])
            if take:
                new[i, 1 : 1 + take] = content[:take]
        batch["labels"] = new

    def _build_row(
        self, source: list[int], knowledge_segments: list[_KnowledgeSegment]
    ) -> tuple[list[int], list[int], dict[str, Any], list[int], list[list[int]]]:
        cls_id = int(self.tokenizer.cls_token_id)
        sep_id = int(self.tokenizer.sep_token_id)
        source_before = len(source)
        if not knowledge_segments or self.max_knowledge_tokens == 0:
            kept_source = self._truncate_source(source, self.max_length - 2)
            row = [cls_id, *kept_source, sep_id]
            return (
                row,
                [0] * len(row),
                self._diagnostics(
                    source_before,
                    len(kept_source),
                    [],
                    marker_before=self._marker_count(source),
                    marker_after=self._marker_count(kept_source),
                ),
                kept_source,
                [[] for _ in knowledge_segments],
            )

        reserved_source = min(source_before, self.min_source_tokens)
        knowledge_budget = min(
            self.max_knowledge_tokens,
            max(0, self.max_length - 3 - reserved_source),
        )
        knowledge, item_counts, retained_bodies = self._truncate_knowledge(
            knowledge_segments, knowledge_budget
        )
        source_capacity = self.max_length - 3 - len(knowledge)
        kept_source = self._truncate_source(source, max(0, source_capacity))
        row = [cls_id, *kept_source, sep_id, *knowledge, sep_id]
        first_len = len(kept_source) + 2
        token_types = [0] * first_len + [1] * (len(row) - first_len)
        diagnostics = self._diagnostics(
            source_before,
            len(kept_source),
            item_counts,
            marker_before=self._marker_count(source),
            marker_after=self._marker_count(kept_source),
        )
        if self.mode == "off_short":
            row = [cls_id, *kept_source, sep_id]
            token_types = [0] * len(row)
            diagnostics["tokens_injected"] = 0
            retained_bodies = [[] for _ in retained_bodies]
        else:
            diagnostics["tokens_injected"] = len(knowledge)
        return row, token_types, diagnostics, kept_source, retained_bodies

    def _truncate_knowledge(
        self, segments: list[_KnowledgeSegment], budget: int
    ) -> tuple[list[int], list[int], list[list[int]]]:
        if self.payload == "serialized":
            return self._truncate_whole_segments(segments, budget)
        wrapper_tokens = sum(
            len(segment.prefix) + len(segment.suffix) for segment in segments
        )
        if wrapper_tokens > budget:
            msg = (
                "max_knowledge_tokens is too small to preserve definition boundary "
                f"tags: need {wrapper_tokens}, got {budget}"
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
        knowledge: list[int] = []
        item_counts: list[int] = []
        retained_bodies: list[list[int]] = []
        for segment, body_count in zip(segments, body_counts, strict=True):
            item = [
                *segment.prefix,
                *segment.body[:body_count],
                *segment.suffix,
            ]
            knowledge.extend(item)
            item_counts.append(len(item))
            # Audit the exact retained model-visible content, excluding the
            # structural boundary tags that are not part of the evidence body.
            retained_bodies.append(list(segment.body[:body_count]))
        return knowledge, item_counts, retained_bodies

    def _truncate_whole_segments(
        self, segments: list[_KnowledgeSegment], budget: int
    ) -> tuple[list[int], list[int], list[list[int]]]:
        """Keep whole structural blocks; never cut one mid-field.

        A serialized ``[HG] ... [/HG]`` block is not gracefully degradable the way
        prose is. The round-robin split used for definitions divides the budget
        evenly and then cuts each item from the right, which silently removes the
        trailing ``polarity=`` field first -- the one typed signal this project has
        positive evidence for -- and does so at different points in the treated and
        shuffled arms, so the arms stop being budget-matched exactly where the
        signal lives. Hyperedges are unbounded in arity (no retriever profile sets
        ``max_hyperedge_size``; ChemProt has an arity>2 edge in 43.9% of instances,
        max 410), so a per-block cap alone cannot bound the length.

        Blocks are already ordered by ``he_id``, so dropping from the tail is
        deterministic and identical across arms whenever their block lengths agree.
        ``knowledge_items_dropped`` reports what was dropped, so an arm that had to
        discard structure is visible in the diagnostics rather than silently thinner.

        The keep/drop rule itself lives in
        :func:`~knowhyper.ieforge.parallel_text_collator.truncate_whole_segments`
        because the GRAFT-style parallel arm (#189) must retain exactly the same
        blocks as this one at the same budget; two copies would let the arms carry
        different amounts of structure while both reporting valid diagnostics.
        """
        from knowhyper.ieforge.parallel_text_collator import truncate_whole_segments

        knowledge, item_counts = truncate_whole_segments(segments, budget)
        retained_bodies = [
            list(segment.body) if count else []
            for segment, count in zip(segments, item_counts, strict=True)
        ]
        return knowledge, item_counts, retained_bodies

    def _truncate_source(self, source: list[int], capacity: int) -> list[int]:
        if len(source) <= capacity:
            return source
        marker_positions = [
            index
            for index, token in enumerate(source)
            if token in self._marker_token_ids
        ]
        if not marker_positions:
            return source[:capacity]
        if len(marker_positions) > capacity:
            msg = (
                "source token budget cannot preserve all target-pair markers: "
                f"{len(marker_positions)} markers for capacity {capacity}"
            )
            raise ValueError(msg)
        selected = set(marker_positions)
        candidates = sorted(
            (index for index in range(len(source)) if index not in selected),
            key=lambda index: (
                min(abs(index - marker) for marker in marker_positions),
                index,
            ),
        )
        selected.update(candidates[: capacity - len(selected)])
        return [token for index, token in enumerate(source) if index in selected]

    def _marker_count(self, source: list[int]) -> int:
        return sum(token in self._marker_token_ids for token in source)

    @staticmethod
    def _diagnostics(
        source_before: int,
        source_after: int,
        item_counts: list[int],
        *,
        marker_before: int,
        marker_after: int,
    ) -> dict[str, Any]:
        removed = source_before - source_after
        return {
            "knowledge_token_count": sum(item_counts),
            "source_token_count_before": source_before,
            "source_token_count_after": source_after,
            "source_tokens_removed": removed,
            "off_short_source_tokens": source_after,
            "source_truncation_rate": removed / source_before if source_before else 0.0,
            "item_counts": item_counts,
            # Structural blocks are kept whole or dropped, so a non-zero count
            # means the arm carried less structure than was retrieved.
            "items_dropped": sum(1 for count in item_counts if count == 0),
            "tokens_injected": 0,
            "marker_before": marker_before,
            "marker_after": marker_after,
        }
