from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from ieforge.data.schema import (
    BioREDExample,
    BioREDPairExample,
    BioREDRelation,
    DocClassificationExample,
    Entity,
    NERExample,
    PICOExample,
    QAExample,
    REExample,
    Relation,
    REPairExample,
    SentenceSimilarityExample,
)
from ieforge.tasks.biored import BioREDPairTask
from ieforge.tasks.ner import NERTask
from ieforge.tasks.pico import PICOTask
from ieforge.tasks.qa import QATask
from ieforge.tasks.re import RETask

_IGNORE_INDEX = -100


@dataclass(frozen=True)
class _REPairKnowledgeExample:
    """Per-pair carrier exposing the fields a knowledge query builder reads."""

    id: str
    text: str
    head_mention: list[str] | None = None
    tail_mention: list[str] | None = None
    head_type: str | None = None
    tail_type: str | None = None


class NERCollator:
    """Collate NERExample instances into a padded batch tensor dict."""

    def __init__(self, task: NERTask, tokenizer: PreTrainedTokenizerBase) -> None:
        """Store the task and tokenizer used for encoding examples."""
        self.task = task
        self.tokenizer = tokenizer

    def __call__(self, examples: list[NERExample]) -> dict[str, Any]:
        """Encode and pad a list of NERExamples into a model-ready batch."""
        encoded = [self.task.encode(ex, self.tokenizer) for ex in examples]
        input_ids = [e["input_ids"] for e in encoded]
        attention_masks = [e["attention_mask"] for e in encoded]
        labels = [e["labels"] for e in encoded]
        offset_mappings = [e["offset_mapping"] for e in encoded]
        texts = [ex.text for ex in examples]
        entities = [ex.entities for ex in examples]

        max_len = max(len(ids) for ids in input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        input_ids_padded = torch.tensor(
            [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids],
            dtype=torch.long,
        )
        attention_mask_padded = torch.tensor(
            [mask + [0] * (max_len - len(mask)) for mask in attention_masks],
            dtype=torch.long,
        )
        labels_padded = torch.tensor(
            [lab + [_IGNORE_INDEX] * (max_len - len(lab)) for lab in labels],
            dtype=torch.long,
        )
        offset_padded = [
            offsets + [(0, 0)] * (max_len - len(offsets)) for offsets in offset_mappings
        ]

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "offset_mapping": offset_padded,
            "text": texts,
            "entities": entities,
        }


class SpanNERCollator:
    """Collate NERExample instances into a span-based NER batch."""

    def __init__(self, task: Any, tokenizer: PreTrainedTokenizerBase) -> None:
        """Store the task and tokenizer used for encoding examples."""
        self.task = task
        self.tokenizer = tokenizer

    def __call__(self, examples: list[NERExample]) -> dict[str, Any]:
        """Encode examples into word-span candidates and pad into a model-ready batch."""
        encoded = [self.task.encode(ex, self.tokenizer) for ex in examples]
        input_ids = [e["input_ids"] for e in encoded]
        attention_masks = [e["attention_mask"] for e in encoded]
        word_starts = [e["word_starts"] for e in encoded]
        num_words_list = [len(ws) for ws in word_starts]
        max_words = max(num_words_list) if num_words_list else 0
        max_span_width = self.task.max_span_width
        num_spans = max_words * max_span_width
        num_types = len(self.task.entity_types)

        max_len = max(len(ids) for ids in input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        input_ids_padded = torch.tensor(
            [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids],
            dtype=torch.long,
        )
        attention_mask_padded = torch.tensor(
            [mask + [0] * (max_len - len(mask)) for mask in attention_masks],
            dtype=torch.long,
        )

        batch_span_labels = torch.zeros(len(examples), num_spans, num_types)
        batch_span_mask = torch.zeros(len(examples), num_spans, dtype=torch.bool)

        for b, enc in enumerate(encoded):
            src_labels = enc["span_labels"]
            src_idx = enc["span_idx"]
            nw = num_words_list[b]
            for i in range(nw):
                for k in range(max_span_width):
                    src_flat = i * max_span_width + k
                    dst_flat = i * max_span_width + k
                    if src_flat < len(src_labels) and src_idx[src_flat] != (-1, -1):
                        batch_span_labels[b, dst_flat] = torch.tensor(
                            src_labels[src_flat]
                        )
                        batch_span_mask[b, dst_flat] = True

        offset_padded = [e["offset_mapping"] for e in encoded]
        texts = [ex.text for ex in examples]
        entities = [ex.entities for ex in examples]

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "word_starts": word_starts,
            "word_char_spans": [e["word_char_spans"] for e in encoded],
            "num_words": num_words_list,
            "span_labels": batch_span_labels,
            "span_mask": batch_span_mask,
            "offset_mapping": offset_padded,
            "text": texts,
            "entities": entities,
        }


class QACollator:
    """Collate QAExample instances into a padded batch for sequence classification."""

    def __init__(self, task: QATask, tokenizer: PreTrainedTokenizerBase) -> None:
        """Store the task and tokenizer used for encoding examples."""
        self.task = task
        self.tokenizer = tokenizer

    def __call__(self, examples: list[QAExample]) -> dict[str, Any]:
        """Encode and pad QA examples into a model-ready batch."""
        encoded = [self.task.encode(ex, self.tokenizer) for ex in examples]
        input_ids = [e["input_ids"] for e in encoded]
        attention_masks = [e["attention_mask"] for e in encoded]
        labels = [e["label"] for e in encoded]
        questions = [ex.question for ex in examples]
        contexts = [ex.context for ex in examples]

        max_len = max(len(ids) for ids in input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        return {
            "input_ids": torch.tensor(
                [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [mask + [0] * (max_len - len(mask)) for mask in attention_masks],
                dtype=torch.long,
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
            "question": questions,
            "context": contexts,
            "label_text": [ex.label for ex in examples],
        }


class PICOCollator:
    """Collate PICO examples into a padded token-classification batch."""

    def __init__(self, task: PICOTask, tokenizer: Any) -> None:
        """Store the task and tokenizer used for encoding examples."""
        self.task = task
        self.tokenizer = tokenizer

    def __call__(self, examples: list[PICOExample]) -> dict[str, Any]:
        """Encode and pad PICO examples into a model-ready batch."""
        encoded = [self.task.encode(ex, self.tokenizer) for ex in examples]
        input_ids = [e["input_ids"] for e in encoded]
        attention_masks = [e["attention_mask"] for e in encoded]
        labels = [e["labels"] for e in encoded]
        word_ids = [e["word_ids"] for e in encoded]

        max_len = max(len(ids) for ids in input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        return {
            "input_ids": torch.tensor(
                [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [mask + [0] * (max_len - len(mask)) for mask in attention_masks],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [lab + [_IGNORE_INDEX] * (max_len - len(lab)) for lab in labels],
                dtype=torch.long,
            ),
            "word_ids": [ids + [None] * (max_len - len(ids)) for ids in word_ids],
            "label_text": [ex.labels for ex in examples],
            "text": [" ".join(ex.tokens) for ex in examples],
        }


class SentenceSimilarityCollator:
    """Collate sentence-similarity examples into a padded regression batch."""

    def __init__(
        self,
        task: Any,
        tokenizer: Any,
    ) -> None:
        """Store the task and tokenizer used for encoding examples."""
        self.task = task
        self.tokenizer = tokenizer

    def __call__(self, examples: list[SentenceSimilarityExample]) -> dict[str, Any]:
        """Encode and pad sentence-pair regression examples into a batch."""
        encoded = [self.task.encode(ex, self.tokenizer) for ex in examples]
        input_ids = [e["input_ids"] for e in encoded]
        attention_masks = [e["attention_mask"] for e in encoded]
        token_type_ids = [e.get("token_type_ids") for e in encoded]
        labels = [e["label"] for e in encoded]
        sentence1 = [ex.sentence1 for ex in examples]
        sentence2 = [ex.sentence2 for ex in examples]

        max_len = max(len(ids) for ids in input_ids)
        pad_id = self.tokenizer.pad_token_id or 0
        batch = {
            "input_ids": torch.tensor(
                [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [mask + [0] * (max_len - len(mask)) for mask in attention_masks],
                dtype=torch.long,
            ),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "sentence1": sentence1,
            "sentence2": sentence2,
            "label_text": labels,
        }
        if any(token_type_id is not None for token_type_id in token_type_ids):
            batch["token_type_ids"] = torch.tensor(
                [
                    (token_type_id or []) + [0] * (max_len - len(token_type_id or []))
                    for token_type_id in token_type_ids
                ],
                dtype=torch.long,
            )
        return batch


class RECollator:
    """Collate RE examples into a padded batch for sequence classification."""

    def __init__(
        self,
        task: RETask,
        tokenizer: PreTrainedTokenizerBase,
        is_training: bool = False,
    ) -> None:
        """Store the task and tokenizer, and whether this collator runs in train mode."""
        self.task = task
        self.tokenizer = tokenizer
        self.is_training = is_training

    def __call__(
        self,
        examples: list[REExample | REPairExample | BioREDExample | BioREDPairExample],
    ) -> dict[str, Any]:
        """Encode RE examples and pad them into a model-ready batch."""
        all_input_ids: list[list[int]] = []
        all_attention_masks: list[list[int]] = []
        all_labels: list[int] = []
        all_head_entities: list[Entity] = []
        all_tail_entities: list[Entity] = []
        all_head_idxs: list[int] = []
        all_tail_idxs: list[int] = []
        all_example_ids: list[str] = []
        all_label_text: list[Any] = []
        all_novelty_text: list[Any] = []
        all_pmids: list[str] = []
        all_id1s: list[str] = []
        all_id2s: list[str] = []
        all_id1_types: list[str] = []
        all_id2_types: list[str] = []
        relations_by_doc: dict[str, list[Relation]] = {}
        texts_by_doc: dict[str, str] = {}
        knowledge_examples: list[_REPairKnowledgeExample] = []

        for ex in examples:
            instances = self.task.encode(
                ex, self.tokenizer, is_training=self.is_training
            )
            doc_id = (
                ex.doc_id
                if isinstance(ex, (REPairExample, BioREDPairExample))
                else ex.id
            )
            if not instances:
                import logging

                logging.getLogger(__name__).debug(
                    "RECollator: skipping doc_id=%r — no candidate pairs after type filtering",
                    doc_id,
                )
                continue
            relations_by_doc.setdefault(doc_id, self._metric_relations(ex))
            texts_by_doc.setdefault(doc_id, ex.text)
            for inst in instances:
                all_input_ids.append(inst["input_ids"])
                all_attention_masks.append(inst["attention_mask"])
                all_labels.append(inst["label"])
                all_head_entities.append(inst["head_entity"])
                all_tail_entities.append(inst["tail_entity"])
                all_head_idxs.append(inst["head_idx"])
                all_tail_idxs.append(inst["tail_idx"])
                all_example_ids.append(inst.get("example_id", doc_id))
                if "label_text" in inst:
                    all_label_text.append(inst["label_text"])
                if "novelty_text" in inst:
                    all_novelty_text.append(inst["novelty_text"])
                pair_metadata_keys = ("pmid", "id1", "id2", "id1_type", "id2_type")
                if all(inst.get(key) for key in pair_metadata_keys):
                    all_pmids.append(inst["pmid"])
                    all_id1s.append(inst["id1"])
                    all_id2s.append(inst["id2"])
                    all_id1_types.append(inst["id1_type"])
                    all_id2_types.append(inst["id2_type"])
                head_ent = inst.get("head_entity")
                tail_ent = inst.get("tail_entity")
                knowledge_examples.append(
                    _REPairKnowledgeExample(
                        id=inst.get("example_id", doc_id),
                        text=texts_by_doc.get(doc_id, ex.text),
                        head_mention=inst.get("head_mention"),
                        tail_mention=inst.get("tail_mention"),
                        head_type=getattr(head_ent, "label", None),
                        tail_type=getattr(tail_ent, "label", None),
                    )
                )

        if not all_input_ids:
            return {
                "input_ids": torch.zeros(0, 0, dtype=torch.long),
                "attention_mask": torch.zeros(0, 0, dtype=torch.long),
                "labels": torch.zeros(0, dtype=torch.long),
                "head_entities": [],
                "tail_entities": [],
                "head_idxs": [],
                "tail_idxs": [],
                "example_ids": [],
                "relations": [],
                "label_text": [],
                "novelty_text": [],
                "pmids": [],
                "id1s": [],
                "id2s": [],
                "id1_types": [],
                "id2_types": [],
                "text": [],
                "_knowledge_examples": [],
            }

        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        input_ids_padded = torch.tensor(
            [ids + [pad_id] * (max_len - len(ids)) for ids in all_input_ids],
            dtype=torch.long,
        )
        attention_mask_padded = torch.tensor(
            [mask + [0] * (max_len - len(mask)) for mask in all_attention_masks],
            dtype=torch.long,
        )
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        ordered_doc_ids = list(dict.fromkeys(all_example_ids))

        batch = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_tensor,
            "head_entities": all_head_entities,
            "tail_entities": all_tail_entities,
            "head_idxs": all_head_idxs,
            "tail_idxs": all_tail_idxs,
            "example_ids": all_example_ids,
            "relations": [relations_by_doc[doc_id] for doc_id in ordered_doc_ids],
            "label_text": all_label_text,
            "text": [texts_by_doc[doc_id] for doc_id in ordered_doc_ids],
            "_knowledge_examples": knowledge_examples,
        }
        if len(all_novelty_text) == len(all_input_ids):
            batch["novelty_text"] = all_novelty_text
        if len(all_pmids) == len(all_input_ids):
            batch.update(
                {
                    "pmids": all_pmids,
                    "id1s": all_id1s,
                    "id2s": all_id2s,
                    "id1_types": all_id1_types,
                    "id2_types": all_id2_types,
                }
            )
        return batch

    def _metric_relations(
        self, example: REExample | REPairExample | BioREDExample | BioREDPairExample
    ) -> list[Relation]:
        if isinstance(example, BioREDExample):
            if not hasattr(self.task, "map_biored_label"):
                return []
            return [
                self._map_biored_relation(relation) for relation in example.relations
            ]
        if isinstance(example, BioREDPairExample):
            return []
        return list(example.relations)

    def _map_biored_relation(self, relation: BioREDRelation) -> Relation:
        head = min(relation.head, relation.tail)
        tail = max(relation.head, relation.tail)
        return Relation(
            head=head,
            tail=tail,
            label=self.task.map_biored_label(relation.label, relation.novelty),
        )


class BioREDPairCollator:
    """Collate BioRED pair examples into a padded sequence-classification batch."""

    def __init__(
        self, task: BioREDPairTask, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        """Store the task and tokenizer used for encoding examples."""
        self.task = task
        self.tokenizer = tokenizer

    def __call__(self, examples: list[BioREDPairExample]) -> dict[str, Any]:
        """Encode and pad BioRED pair rows while preserving PubTator metadata."""
        encoded = [self.task.encode(ex, self.tokenizer) for ex in examples]
        input_ids = [e["input_ids"] for e in encoded]
        attention_masks = [e["attention_mask"] for e in encoded]
        labels = [e["label"] for e in encoded]

        max_len = max(len(ids) for ids in input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        return {
            "input_ids": torch.tensor(
                [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [mask + [0] * (max_len - len(mask)) for mask in attention_masks],
                dtype=torch.long,
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pmids": [e["pmid"] for e in encoded],
            "id1s": [e["id1"] for e in encoded],
            "id2s": [e["id2"] for e in encoded],
            "id1_types": [e["id1_type"] for e in encoded],
            "id2_types": [e["id2_type"] for e in encoded],
            "label_text": [e["label_text"] for e in encoded],
            "novelty_text": [e["novelty_text"] for e in encoded],
            "is_in_same_sent": [e["is_in_same_sent"] for e in encoded],
            "min_sents_window": [e["min_sents_window"] for e in encoded],
            "text": [e["text"] for e in encoded],
        }


class DocClassificationCollator:
    """Collate document-classification examples into a padded batch."""

    def __init__(self, task: Any, tokenizer: PreTrainedTokenizerBase) -> None:
        """Store the task and tokenizer used for encoding examples."""
        self.task = task
        self.tokenizer = tokenizer

    def __call__(self, examples: list[DocClassificationExample]) -> dict[str, Any]:
        """Encode and pad document-classification examples into a batch."""
        encoded = [self.task.encode(ex, self.tokenizer) for ex in examples]
        input_ids = [e["input_ids"] for e in encoded]
        attention_masks = [e["attention_mask"] for e in encoded]
        labels = [e["label"] for e in encoded]
        texts = [ex.text for ex in examples]
        label_text = [ex.labels for ex in examples]

        max_len = max(len(ids) for ids in input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        return {
            "input_ids": torch.tensor(
                [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [mask + [0] * (max_len - len(mask)) for mask in attention_masks],
                dtype=torch.long,
            ),
            "labels": torch.tensor(labels, dtype=torch.float),
            "text": texts,
            "label_text": label_text,
        }


class BioREDirectCollator:
    """Collate BioREDirect encodings into chunked, marker-indexed batches."""

    def __init__(
        self,
        task: Any,
        tokenizer: Any,
        num_chunks: int = 3,
        soft_prompt_len: int = 0,
    ) -> None:
        """Store the task, tokenizer, chunk count, and marker token ids."""
        self.task = task
        self.tokenizer = tokenizer
        self.num_chunks = num_chunks
        self.soft_prompt_len = soft_prompt_len
        self._rel_id = tokenizer.convert_tokens_to_ids("[REL]")
        self._dir_id = tokenizer.convert_tokens_to_ids("[DIR]")
        self._nov_id = tokenizer.convert_tokens_to_ids("[NOV]")

    def _marker_index(self, ids: list[int], marker_id: int) -> int:
        return ids.index(marker_id) if marker_id in ids else 0

    def __call__(self, examples: list[Any]) -> dict[str, Any]:
        """Encode raw examples then build a (B, num_chunks, T) marker-indexed batch.

        Accepts either raw ``BioREDirectPairExample`` objects (the datamodule
        path) or already-encoded dicts (encode is idempotent on dicts via the
        guard below), so a row is normalized to its encoded dict first.
        """
        import torch

        examples = [
            ex if isinstance(ex, dict) else self.task.encode(ex, self.tokenizer)
            for ex in examples
        ]
        pad = getattr(self.tokenizer, "pad_token_id", 0) or 0
        per_example_ids = [ex["input_ids"] for ex in examples]
        max_len = max(len(ids) for ids in per_example_ids)
        b = len(examples)

        input_ids = torch.full((b, self.num_chunks, max_len), pad, dtype=torch.long)
        attention = torch.zeros((b, self.num_chunks, max_len), dtype=torch.long)
        rel_idx = torch.zeros((b, self.num_chunks), dtype=torch.long)
        dir_idx = torch.zeros((b, self.num_chunks), dtype=torch.long)
        nov_idx = torch.zeros((b, self.num_chunks), dtype=torch.long)
        prompt_ids = torch.zeros((b, self.num_chunks), dtype=torch.long)

        for i, ex in enumerate(examples):
            ids = ex["input_ids"]
            am = ex["attention_mask"]
            r = self._marker_index(ids, self._rel_id)
            d = self._marker_index(ids, self._dir_id)
            n = self._marker_index(ids, self._nov_id)
            for c in range(self.num_chunks):
                input_ids[i, c, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention[i, c, : len(am)] = torch.tensor(am, dtype=torch.long)
                rel_idx[i, c] = r
                dir_idx[i, c] = d
                nov_idx[i, c] = n

        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "relation_token_index": rel_idx,
            "direction_token_index": dir_idx,
            "novelty_token_index": nov_idx,
            "pair_prompt_ids": prompt_ids,
            "relation_labels": torch.tensor(
                [ex["relation_labels"] for ex in examples], dtype=torch.float
            ),
            "novelty_labels": torch.tensor(
                [ex["novelty_labels"] for ex in examples], dtype=torch.float
            ),
            "direction_labels": torch.tensor(
                [ex["direction_labels"] for ex in examples], dtype=torch.float
            ),
            "pmids": [ex["pmid"] for ex in examples],
            "id1s": [ex["id1"] for ex in examples],
            "id2s": [ex["id2"] for ex in examples],
            "id1_types": [ex["id1_type"] for ex in examples],
            "id2_types": [ex["id2_type"] for ex in examples],
            "label_text": [ex["label_text"] for ex in examples],
            "novelty_text": [ex["novelty_text"] for ex in examples],
            "direction_text": [ex["direction_text"] for ex in examples],
            "text": [ex["text"] for ex in examples],
        }
