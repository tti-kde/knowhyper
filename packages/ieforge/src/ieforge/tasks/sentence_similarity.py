from typing import Any

import torch

from ieforge.data.schema import SentenceSimilarityExample
from ieforge.registry import register
from ieforge.tasks.base import IETask


@register("task", name="sentence_similarity")
class SentenceSimilarityTask(IETask):
    """Sentence-pair regression for semantic similarity."""

    def __init__(
        self,
        max_length: int = 512,
        name: str = "sentence_similarity",
        label_list: list[str] | None = None,
    ) -> None:
        """Store tokenization settings and a nominal label list for runtime wiring."""
        self.name = name
        self.max_length = max_length
        self.label_list = list(label_list or ["score"])

    def encode(
        self, example: SentenceSimilarityExample, tokenizer: Any
    ) -> dict[str, Any]:
        """Encode one sentence pair for scalar regression."""
        tokenized = tokenizer(
            example.sentence1,
            example.sentence2,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        encoded = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "label": example.score,
        }
        token_type_ids = tokenized.get("token_type_ids")
        if token_type_ids is not None:
            encoded["token_type_ids"] = token_type_ids
        return encoded

    def decode(self, outputs: Any, batch: dict[str, Any]) -> list[float]:
        """Convert regression logits to python floats."""
        del batch
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if logits.ndim == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        return [float(value) for value in cast_tensor(logits).tolist()]


def cast_tensor(value: Any) -> torch.Tensor:
    """Normalize decoded logits to a tensor."""
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value, dtype=torch.float32)
