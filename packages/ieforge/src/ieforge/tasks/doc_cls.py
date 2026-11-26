from typing import Any

import torch

from ieforge.data.schema import DocClassificationExample
from ieforge.registry import register
from ieforge.tasks.base import IETask


@register("task", name="doc_cls")
class DocClassificationTask(IETask):
    """Document classification with multi-label targets."""

    def __init__(
        self,
        label_list: list[str],
        threshold: float = 0.5,
        max_length: int = 512,
        name: str = "doc_cls",
    ) -> None:
        """Initialize label mappings and tokenization settings."""
        if not label_list:
            msg = "DocClassificationTask requires at least one label"
            raise ValueError(msg)
        if len(set(label_list)) != len(label_list):
            msg = f"DocClassificationTask labels must be unique: {label_list!r}"
            raise ValueError(msg)
        self.name = name
        self.label_list = list(label_list)
        self.threshold = threshold
        self.max_length = max_length
        self.label2id = dict(zip(self.label_list, range(len(self.label_list))))
        self.id2label = dict(enumerate(self.label_list))

    def encode(
        self, example: DocClassificationExample, tokenizer: Any
    ) -> dict[str, Any]:
        """Encode one document and its multi-label target vector."""
        tokenized = tokenizer(
            example.text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "label": self.encode_labels(example.labels).tolist(),
        }

    def encode_labels(self, labels: tuple[str, ...]) -> torch.Tensor:
        """Convert a label tuple into a multi-hot float target vector."""
        encoded = torch.zeros(len(self.label_list), dtype=torch.float)
        for label in labels:
            if label not in self.label2id:
                msg = (
                    f"Unknown document classification label {label!r}; "
                    f"expected one of {self.label_list!r}"
                )
                raise ValueError(msg)
            encoded[self.label2id[label]] = 1.0
        return encoded

    def decode(self, outputs: Any, batch: dict[str, Any]) -> list[tuple[str, ...]]:
        """Convert logits to thresholded label tuples."""
        del batch
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = torch.sigmoid(logits)
        predictions: list[tuple[str, ...]] = []
        for row in probs:
            labels = tuple(
                self.id2label[index]
                for index, value in enumerate(row.tolist())
                if value >= self.threshold
            )
            predictions.append(labels)
        return predictions
