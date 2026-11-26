from typing import Any

from ieforge.data.schema import Entity, NERExample
from ieforge.registry import register
from ieforge.tasks.base import IETask

_IGNORE_INDEX = -100


@register("task", name="ner")
class NERTask(IETask):
    """BIO-based named entity recognition task."""

    def __init__(
        self,
        label_list: list[str] | None = None,
        label_scheme: str = "BIO",
        max_length: int = 512,
        name: str = "ner",
    ) -> None:
        """Initialize label mappings and tokenization settings."""
        self.name = name
        self.label_list: list[str] = label_list if label_list is not None else []
        self.label_scheme = label_scheme
        self.max_length = max_length
        self.label2id = dict(zip(self.label_list, range(len(self.label_list))))
        self.id2label = dict(enumerate(self.label_list))

    def encode(self, example: NERExample, tokenizer: Any) -> dict[str, Any]:
        """Tokenize text and align entity spans to BIO token labels."""
        tokenized = tokenizer(
            example.text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
        )
        offset_mapping: list[tuple[int, int]] = tokenized["offset_mapping"]
        labels = self._align_labels(example.entities, offset_mapping, tokenized)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "offset_mapping": offset_mapping,
            "labels": labels,
        }

    def _align_labels(
        self,
        entities: list[Entity],
        offset_mapping: list[tuple[int, int]],
        tokenized: Any,
    ) -> list[int]:
        """Map character-level entity spans onto token-level BIO label ids."""
        labels = []
        special_token_mask = tokenized.get("special_tokens_mask") or [
            1 if (start == 0 and end == 0) else 0 for start, end in offset_mapping
        ]

        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if special_token_mask[idx] == 1 or tok_start == tok_end:
                labels.append(_IGNORE_INDEX)
                continue
            bio_label = "O"
            for entity in entities:
                if tok_start < entity.end and tok_end > entity.start:
                    # B- when entity starts within this token's span
                    prefix = "B" if entity.start >= tok_start else "I"
                    bio_label = f"{prefix}-{entity.label}"
                    break
            labels.append(self.label2id.get(bio_label, self.label2id["O"]))

        return labels

    def decode(self, outputs: Any, batch: dict[str, Any]) -> list[list[Entity]]:
        """Convert token-level logits to lists of Entity spans."""
        import torch

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        predictions = torch.argmax(logits, dim=-1).tolist()
        offset_mappings = batch["offset_mapping"]
        texts = batch["text"]

        results = []
        for pred_seq, offsets, text in zip(predictions, offset_mappings, texts):
            entities = self._decode_bio(pred_seq, offsets, text)
            results.append(entities)
        return results

    def _decode_bio(
        self,
        pred_seq: list[int],
        offsets: list[tuple[int, int]],
        text: str,
    ) -> list[Entity]:
        """Decode a BIO label sequence into entity spans using character offsets."""
        entities: list[Entity] = []
        current_label: str | None = None
        current_start: int | None = None
        current_end: int | None = None

        for token_id, (start, end) in zip(pred_seq, offsets):
            if start == end:
                continue
            label_name = self.id2label.get(token_id, "O")
            if label_name in ("O", _IGNORE_INDEX):
                if current_label is not None:
                    assert current_start is not None and current_end is not None
                    entities.append(
                        Entity(
                            start=current_start,
                            end=current_end,
                            text=text[current_start:current_end],
                            label=current_label,
                        )
                    )
                    current_label = None
                continue
            prefix, entity_type = label_name.split("-", 1)
            # Skip leading whitespace in token span (e.g. ModernBERT prepends spaces)
            effective_start = (
                start + len(text[start:end]) - len(text[start:end].lstrip())
            )
            if prefix == "B":
                if current_label is not None:
                    assert current_start is not None and current_end is not None
                    entities.append(
                        Entity(
                            start=current_start,
                            end=current_end,
                            text=text[current_start:current_end],
                            label=current_label,
                        )
                    )
                current_label = entity_type
                current_start = effective_start
                current_end = end
            elif prefix == "I" and current_label == entity_type:
                current_end = end
            else:
                if current_label is not None:
                    assert current_start is not None and current_end is not None
                    entities.append(
                        Entity(
                            start=current_start,
                            end=current_end,
                            text=text[current_start:current_end],
                            label=current_label,
                        )
                    )
                current_label = entity_type
                current_start = effective_start
                current_end = end

        if current_label is not None:
            assert current_start is not None and current_end is not None
            entities.append(
                Entity(
                    start=current_start,
                    end=current_end,
                    text=text[current_start:current_end],
                    label=current_label,
                )
            )
        return entities
