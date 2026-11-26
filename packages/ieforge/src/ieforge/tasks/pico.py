from typing import Any

import torch

from ieforge.data.schema import PICOExample
from ieforge.registry import register
from ieforge.tasks.base import IETask

_IGNORE_INDEX = -100


@register("task", name="pico")
class PICOTask(IETask):
    """Token classification task for BLURB EBM PICO labels."""

    def __init__(
        self,
        label_list: list[str],
        max_length: int = 512,
        name: str = "pico",
    ) -> None:
        """Initialize label mappings and tokenization settings."""
        self.name = name
        self.label_list = list(label_list)
        self.max_length = max_length
        self.label2id = dict(zip(self.label_list, range(len(self.label_list))))
        self.id2label = dict(enumerate(self.label_list))

    def encode(self, example: PICOExample, tokenizer: Any) -> dict[str, Any]:
        """Tokenize one labeled word sequence and align labels to subwords."""
        tokenized = tokenizer(
            list(example.tokens),
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        word_ids = list(tokenized.word_ids())
        labels: list[int] = []
        for word_id in word_ids:
            if word_id is None:
                labels.append(_IGNORE_INDEX)
                continue
            labels.append(self.label2id[example.labels[word_id]])
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "word_ids": word_ids,
        }

    def decode(self, outputs: Any, batch: dict[str, Any]) -> list[tuple[str, ...]]:
        """Collapse token-level predictions back to one label per word."""
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        predictions = torch.argmax(logits, dim=-1).tolist()
        decoded: list[tuple[str, ...]] = []
        for pred_seq, word_ids in zip(predictions, batch["word_ids"]):
            words: list[str] = []
            seen: set[int] = set()
            for label_id, word_id in zip(pred_seq, word_ids):
                if word_id is None or word_id in seen:
                    continue
                seen.add(word_id)
                words.append(self.id2label[label_id])
            decoded.append(tuple(words))
        return decoded
