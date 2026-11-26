import random
import re
from itertools import combinations
from typing import Any

from ieforge.data.schema import (
    BioREDExample,
    BioREDPairExample,
    BioREDRelation,
    Entity,
    REExample,
    Relation,
    REPairExample,
)
from ieforge.registry import register
from ieforge.tasks.base import IETask

_BIORED_ENTITY_TYPES = (
    "ChemicalEntity",
    "DiseaseOrPhenotypicFeature",
    "GeneOrGeneProduct",
)

_BIORED_ALLOWED_TYPE_PAIRS = frozenset(
    {
        tuple(sorted(("ChemicalEntity", "ChemicalEntity"))),
        tuple(sorted(("ChemicalEntity", "DiseaseOrPhenotypicFeature"))),
        tuple(sorted(("ChemicalEntity", "GeneOrGeneProduct"))),
        tuple(sorted(("DiseaseOrPhenotypicFeature", "GeneOrGeneProduct"))),
        tuple(sorted(("GeneOrGeneProduct", "GeneOrGeneProduct"))),
    }
)


@register("task", name="re")
class RETask(IETask):
    """Sentence-level relation extraction task via entity-pair classification."""

    def __init__(
        self,
        label_list: list[str],
        entity_representation: str = "dummify",
        negative_label: str = "false",
        max_length: int = 512,
        context_strategy: str = "full_text",
        window_sentences: int = 0,
        schema: str = "default",
        train_pair_chunk_size: int = 0,
        val_pair_chunk_size: int = 0,
        train_negative_downsample_ratio: int = 0,
        train_exclude_negative: bool = False,
        name: str = "re",
    ) -> None:
        """Initialize label mappings and encoding settings."""
        self.name = name
        self.label_list = label_list
        self.entity_representation = entity_representation
        self.negative_label = negative_label
        self.max_length = max_length
        self.context_strategy = context_strategy
        self.schema = schema
        if window_sentences < 0:
            msg = "window_sentences must be non-negative"
            raise ValueError(msg)
        if train_pair_chunk_size < 0:
            msg = "train_pair_chunk_size must be non-negative"
            raise ValueError(msg)
        if val_pair_chunk_size < 0:
            msg = "val_pair_chunk_size must be non-negative"
            raise ValueError(msg)
        if train_negative_downsample_ratio < 0:
            msg = "train_negative_downsample_ratio must be non-negative"
            raise ValueError(msg)
        self.window_sentences = window_sentences
        self.train_pair_chunk_size = train_pair_chunk_size
        self.val_pair_chunk_size = val_pair_chunk_size
        self.train_negative_downsample_ratio = train_negative_downsample_ratio
        self.train_exclude_negative = train_exclude_negative
        self.label2id = dict(zip(label_list, range(len(label_list))))
        self.id2label = dict(enumerate(label_list))
        if self.negative_label not in self.label2id:
            msg = f"negative_label {self.negative_label!r} is not in label_list"
            raise ValueError(msg)

    def encode(
        self,
        example: REExample | REPairExample | BioREDExample | BioREDPairExample,
        tokenizer: Any,
        *,
        is_training: bool = False,
    ) -> list[dict[str, Any]]:
        """Encode RE examples into one or more pair-classification instances."""
        if isinstance(example, REPairExample):
            return [self._encode_pair_instance(example, tokenizer)]
        if isinstance(example, BioREDPairExample):
            return [self._encode_biored_pair_instance(example, tokenizer)]
        if isinstance(example, BioREDExample):
            pair_examples = self._iter_biored_pair_examples(example)
            if is_training:
                if self.train_exclude_negative:
                    pair_examples = [
                        p for p in pair_examples if p.label != self.negative_label
                    ]
                pair_examples = self._downsample_train_pairs(pair_examples)
            return [
                self._encode_biored_pair_instance(pair_example, tokenizer)
                for pair_example in pair_examples
            ]

        gold_map: dict[tuple[int, int], str] = {
            (r.head, r.tail): r.label for r in example.relations
        }
        instances = []
        for i, j in combinations(range(len(example.entities)), 2):
            label_str = gold_map.get((i, j), self.negative_label)
            pair_example = REPairExample(
                id=f"{example.id}#{i}-{j}",
                doc_id=example.id,
                text=example.text,
                entities=example.entities,
                relations=example.relations,
                head_idx=i,
                tail_idx=j,
                label=label_str,
            )
            instances.append(self._encode_pair_instance(pair_example, tokenizer))
        return instances

    def _downsample_train_pairs(
        self, pair_examples: list[BioREDPairExample]
    ) -> list[BioREDPairExample]:
        """Keep all positives and cap negatives at ratio * positives per document."""
        if self.train_negative_downsample_ratio == 0:
            return pair_examples
        positives = [p for p in pair_examples if p.label != self.negative_label]
        negatives = [p for p in pair_examples if p.label == self.negative_label]
        if not positives or not negatives:
            return pair_examples
        max_negatives = self.train_negative_downsample_ratio * len(positives)
        if len(negatives) <= max_negatives:
            return pair_examples
        sampled = random.sample(negatives, max_negatives)
        return positives + sampled

    def _encode_pair_instance(
        self, example: REPairExample, tokenizer: Any
    ) -> dict[str, Any]:
        """Encode a single pair candidate into one classifier instance."""
        head = example.entities[example.head_idx]
        tail = example.entities[example.tail_idx]
        return self._encode_pair_payload(
            text=example.text,
            head=head,
            tail=tail,
            label=example.label,
            head_idx=example.head_idx,
            tail_idx=example.tail_idx,
            doc_id=example.doc_id,
            tokenizer=tokenizer,
            all_entities=example.entities,
        )

    def _encode_biored_pair_instance(
        self, example: BioREDPairExample, tokenizer: Any
    ) -> dict[str, Any]:
        """Encode a single BioRED pair candidate into one classifier instance."""
        head = example.entities[example.head_idx]
        tail = example.entities[example.tail_idx]
        # _iter_biored_pair_examples already maps labels; pair-level BioRED rows may not.
        label = (
            example.label
            if example.label in self.label2id
            else self.map_biored_label(example.label, example.novelty)
        )
        return self._encode_pair_payload(
            text=example.text,
            head=head,
            tail=tail,
            label=label,
            head_idx=example.head_idx,
            tail_idx=example.tail_idx,
            doc_id=example.doc_id,
            tokenizer=tokenizer,
            all_entities=example.entities,
        )

    def _encode_pair_payload(
        self,
        *,
        text: str,
        head: Entity,
        tail: Entity,
        label: str,
        head_idx: int,
        tail_idx: int,
        doc_id: str,
        tokenizer: Any,
        all_entities: list[Entity] | None = None,
    ) -> dict[str, Any]:
        """Encode a normalized pair payload shared by RE and BioRED examples."""
        context_text, context_head, context_tail, context_doc_start = (
            self._build_pair_context(text, head, tail)
        )
        if (
            self.entity_representation == "typed_marker_all_mentions"
            and all_entities is not None
        ):
            if context_doc_start == 0 and context_text == text:
                context_all = all_entities
            else:
                context_end = context_doc_start + len(context_text)
                context_all = [
                    self._shift_entity(e, context_doc_start)
                    for e in all_entities
                    if context_doc_start <= e.start and e.end <= context_end
                ]
            modified_text = self._modify_text_all_mentions(
                context_text, context_head, context_tail, context_all
            )
        else:
            modified_text = self._modify_text(context_text, context_head, context_tail)
        tokenized = tokenizer(
            modified_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
        )
        label = self._validate_label(label)
        label_id = self.label2id[label]
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "label": label_id,
            "head_entity": head,
            "tail_entity": tail,
            "head_idx": head_idx,
            "tail_idx": tail_idx,
            "example_id": doc_id,
            "head_mention": head.mention,
            "tail_mention": tail.mention,
        }

    def _validate_label(self, label: str) -> str:
        """Validate that a label belongs to this task vocabulary."""
        if label not in self.label2id:
            msg = f"unknown relation label: {label!r}"
            raise ValueError(msg)
        return label

    def map_biored_label(self, relation_label: str, novelty: str | None) -> str:
        """Map BioRED document truth into this task's label space."""
        if self.schema == "default":
            return self._validate_label(relation_label)
        if self.schema == "biored_entity_pair":
            is_negative = relation_label in (self.negative_label, "None")
            label = "false" if is_negative else "true"
            return self._validate_label(label)
        if self.schema == "biored_entity_pair_plus_type":
            return self._validate_label(relation_label)
        if self.schema == "biored_entity_pair_plus_type_plus_novelty":
            is_negative = relation_label in (self.negative_label, "None")
            if is_negative:
                return self._validate_label(self.negative_label)
            if novelty is None:
                msg = "BioRED novelty schema requires novelty for positive relations"
                raise ValueError(msg)
            return self._validate_label(f"{relation_label}|{novelty}")
        msg = f"Unsupported schema: {self.schema!r}"
        raise ValueError(msg)

    def _iter_biored_pair_examples(
        self, example: BioREDExample
    ) -> list[BioREDPairExample]:
        """Expand a BioRED document into schema-aware pair examples."""
        gold_map: dict[tuple[int, int], BioREDRelation] = {
            (relation.head, relation.tail): relation for relation in example.relations
        }
        pair_examples = []
        for i, j in combinations(range(len(example.entities)), 2):
            head = example.entities[i]
            tail = example.entities[j]
            if not self._is_allowed_biored_type_pair(head, tail):
                continue
            relation = gold_map.get((i, j)) or gold_map.get((j, i))
            raw_label = self.negative_label if relation is None else relation.label
            novelty = None if relation is None else relation.novelty
            pair_examples.append(
                BioREDPairExample(
                    id=f"{example.id}#{i}-{j}",
                    doc_id=example.id,
                    text=example.text,
                    entities=example.entities,
                    head_idx=i,
                    tail_idx=j,
                    label=self.map_biored_label(raw_label, novelty),
                    novelty=novelty,
                )
            )
        return pair_examples

    def _is_allowed_biored_type_pair(self, head: Entity, tail: Entity) -> bool:
        """Return whether a BioRED pair uses an allowed entity-type combination."""
        return tuple(sorted((head.label, tail.label))) in _BIORED_ALLOWED_TYPE_PAIRS

    def _build_pair_context(
        self, text: str, head: Entity, tail: Entity
    ) -> tuple[str, Entity, Entity, int]:
        """Build pair-specific context and remap entity offsets into it.

        Returns (context_text, shifted_head, shifted_tail, context_doc_start)
        where context_doc_start is the offset in the original document at which
        context_text begins; subtract it from any document-coordinate entity to
        obtain context-coordinate offsets.
        """
        if self.context_strategy == "full_text":
            return text, head, tail, 0
        if self.context_strategy != "biored_pair_window":
            msg = f"Unsupported context_strategy: {self.context_strategy}"
            raise ValueError(msg)

        sentence_spans = self._split_sentence_spans(text)
        if not sentence_spans:
            return text, head, tail, 0

        pair_start = min(head.start, tail.start)
        pair_end = max(head.end, tail.end)
        overlapping = [
            idx
            for idx, (start, end) in enumerate(sentence_spans)
            if start < pair_end and pair_start < end
        ]
        if not overlapping:
            msg = (
                "entity pair does not overlap any sentence span: "
                f"head=({head.start},{head.end}) tail=({tail.start},{tail.end})"
            )
            raise ValueError(msg)
        focus_start = overlapping[0]
        focus_end = overlapping[-1]

        window_start = max(0, focus_start - self.window_sentences)
        window_end = min(len(sentence_spans) - 1, focus_end + self.window_sentences)
        context_start = sentence_spans[window_start][0]
        context_end = sentence_spans[window_end][1]
        raw_context = text[context_start:context_end]
        leading_trim = len(raw_context) - len(raw_context.lstrip())
        trailing_trim = len(raw_context) - len(raw_context.rstrip())
        trimmed_end = (
            len(raw_context) - trailing_trim if trailing_trim else len(raw_context)
        )
        context_text = raw_context[leading_trim:trimmed_end]

        shifted_head = self._shift_entity(head, context_start + leading_trim)
        shifted_tail = self._shift_entity(tail, context_start + leading_trim)
        for entity in (shifted_head, shifted_tail):
            if (
                entity.start < 0
                or entity.end > len(context_text)
                or entity.start >= entity.end
            ):
                msg = (
                    "shifted entity span is outside the pair context: "
                    f"span=({entity.start},{entity.end}) context_len={len(context_text)}"
                )
                raise ValueError(msg)

        return context_text, shifted_head, shifted_tail, context_start + leading_trim

    def _split_sentence_spans(self, text: str) -> list[tuple[int, int]]:
        """Split text into coarse sentence spans using punctuation boundaries."""
        spans = []
        for match in re.finditer(r"[^.!?]+[.!?]?", text):
            start, end = match.span()
            sentence = text[start:end]
            if sentence.strip():
                spans.append((start, end))
        return spans or [(0, len(text))]

    def _shift_entity(self, entity: Entity, context_start: int) -> Entity:
        """Translate entity offsets from document coordinates into context coordinates."""
        return Entity(
            start=entity.start - context_start,
            end=entity.end - context_start,
            text=entity.text,
            label=entity.label,
        )

    def _modify_text(self, text: str, head: Entity, tail: Entity) -> str:
        """Return text with entity spans replaced or marked for classification.

        "passthrough" skips modification entirely for pre-dummified datasets.
        """
        if self.entity_representation == "passthrough":
            return text
        if self.entity_representation == "typed_marker_all_mentions":
            return self._modify_text_all_mentions(text, head, tail)
        spans = sorted(
            [("head", head), ("tail", tail)], key=lambda x: x[1].start, reverse=True
        )
        result = text
        for role, entity in spans:
            before = result[: entity.start]
            after = result[entity.end :]
            if self.entity_representation == "dummify":
                replacement = "@ENTITY@"
            elif self.entity_representation == "typed_marker":
                replacement = self._typed_marker(entity, role)
            elif self.entity_representation == "marker":
                if role == "head":
                    replacement = f"[HEAD_START] {entity.text} [HEAD_END]"
                else:
                    replacement = f"[TAIL_START] {entity.text} [TAIL_END]"
            else:
                msg = (
                    f"Unsupported entity_representation: {self.entity_representation!r}"
                )
                raise ValueError(msg)
            result = before + replacement + after
        return result

    def _modify_text_all_mentions(
        self,
        text: str,
        head: Entity,
        tail: Entity,
        all_entities: list[Entity] | None = None,
    ) -> str:
        """Tag all entity spans of head type as Src and all of tail type as Tgt.

        When head.label == tail.label, the gold head and tail are identified by
        their (start, end) span; any additional same-type mention is assigned
        Src if its start is below tail.start, otherwise Tgt.
        """
        if all_entities is None:
            all_entities = [head, tail]

        head_label = head.label
        tail_label = tail.label
        same_type = head_label == tail_label
        head_span = (head.start, head.end)
        tail_span = (tail.start, tail.end)

        replacements: list[tuple[int, int, str]] = []
        head_tagged = False
        tail_tagged = False
        for entity in all_entities:
            span = (entity.start, entity.end)
            if same_type and entity.label == head_label:
                if span == head_span:
                    role = "head"
                elif span == tail_span:
                    role = "tail"
                else:
                    role = "head" if entity.start < tail.start else "tail"
            elif entity.label == head_label:
                role = "head"
            elif entity.label == tail_label:
                role = "tail"
            else:
                continue
            if span == head_span:
                head_tagged = True
            if span == tail_span:
                tail_tagged = True
            replacements.append(
                (entity.start, entity.end, self._typed_marker(entity, role))
            )

        if not head_tagged:
            msg = (
                f"head entity span {head_span} (label={head.label!r}) was not "
                "tagged; head was missing from all_entities"
            )
            raise ValueError(msg)
        if not tail_tagged:
            msg = (
                f"tail entity span {tail_span} (label={tail.label!r}) was not "
                "tagged; tail was missing from all_entities"
            )
            raise ValueError(msg)

        replacements.sort(key=lambda x: x[0], reverse=True)
        result = text
        for start, end, replacement in replacements:
            result = result[:start] + replacement + result[end:]
        return result

    def _typed_marker(self, entity: Entity, role: str) -> str:
        """Return BioRED-style role-aware markers with the entity type in the tag."""
        suffix = "Src" if role == "head" else "Tgt"
        return f"@{entity.label}{suffix}$ {entity.text} @/{entity.label}{suffix}$"

    def special_tokens(self) -> list[str]:
        """Return sorted special token strings for typed marker entity representations."""
        if self.entity_representation not in {
            "typed_marker",
            "typed_marker_all_mentions",
        }:
            return []
        tokens = []
        for etype in _BIORED_ENTITY_TYPES:
            for role in ("Src", "Tgt"):
                tokens.append(f"@{etype}{role}$")
                tokens.append(f"@/{etype}{role}$")
        return sorted(tokens)

    def decode(self, outputs: Any, batch: dict[str, Any]) -> list[list[Relation]]:
        """Convert per-pair logits to per-example Relation lists."""
        import torch

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        pred_ids: list[int] = torch.argmax(logits, dim=-1).tolist()
        example_ids: list[str] = batch["example_ids"]
        head_idxs: list[int] = batch["head_idxs"]
        tail_idxs: list[int] = batch["tail_idxs"]

        grouped: dict[str, list[Relation]] = {}
        seen_order: list[str] = []
        for pred_id, ex_id, head_idx, tail_idx in zip(
            pred_ids, example_ids, head_idxs, tail_idxs
        ):
            if ex_id not in grouped:
                grouped[ex_id] = []
                seen_order.append(ex_id)
            label = self.id2label[pred_id]
            grouped[ex_id].append(Relation(head=head_idx, tail=tail_idx, label=label))
        return [grouped[ex_id] for ex_id in seen_order]
