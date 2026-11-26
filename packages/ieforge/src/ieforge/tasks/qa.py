from typing import Any

from ieforge.data.schema import QAExample
from ieforge.registry import register
from ieforge.tasks.base import IETask


@register("task", name="qa")
class QATask(IETask):
    """Question answering as question/context sequence classification."""

    def __init__(
        self,
        label_list: list[str],
        max_length: int = 512,
        name: str = "qa",
    ) -> None:
        """Initialize label mappings and encoding settings."""
        if not label_list:
            msg = "QATask requires at least one label"
            raise ValueError(msg)
        if len(set(label_list)) != len(label_list):
            msg = f"QATask labels must be unique: {label_list!r}"
            raise ValueError(msg)
        self.name = name
        self.label_list = list(label_list)
        self.max_length = max_length
        self.label2id = dict(zip(self.label_list, range(len(self.label_list))))
        self.id2label = dict(enumerate(self.label_list))

    def encode(self, example: QAExample, tokenizer: Any) -> dict[str, Any]:
        """Encode one question/context pair for sequence classification."""
        tokenized = tokenizer(
            example.question,
            example.context,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        if example.label not in self.label2id:
            msg = (
                f"Unknown QA label {example.label!r}; "
                f"expected one of {self.label_list!r}"
            )
            raise ValueError(msg)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "label": self.label2id[example.label],
        }

    def decode(self, outputs: Any, batch: dict[str, Any]) -> list[str]:
        """Convert logits to predicted label strings."""
        del batch
        import torch

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        return [self.id2label[pred_id] for pred_id in pred_ids]
