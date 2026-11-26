from typing import Any, cast

from transformers import (
    AutoModelForTokenClassification,
    BertConfig,
    BertForTokenClassification,
)

from ieforge.models.base import IEModel
from ieforge.registry import register


@register("model", name="hf_token_classifier")
class HFTokenClassifier(IEModel):
    """Token classifier backed by a Hugging Face pretrained encoder."""

    def __init__(
        self,
        model_name: str,
        label_list: list[str],
        name: str = "hf_token_classifier",
        extra_token_count: int = 0,
    ) -> None:
        """Load a pretrained encoder and add a token classification head."""
        super().__init__()
        self.name = name
        id2label = dict(enumerate(label_list))
        label2id = {label: i for i, label in enumerate(label_list)}
        num_labels = len(label_list)
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
        except ValueError:
            # Fallback for checkpoints without model_type (e.g. prajjwal1/bert-tiny)
            config = BertConfig.from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
            self.model = BertForTokenClassification.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )
        if extra_token_count < 0:
            msg = f"extra_token_count must be non-negative, got {extra_token_count}"
            raise ValueError(msg)
        if extra_token_count > 0:
            embeddings = self.model.get_input_embeddings()
            current_size = cast("int", embeddings.num_embeddings)
            self.model.resize_token_embeddings(current_size + extra_token_count)

    def forward(self, batch: dict[str, Any]) -> Any:
        """Run the encoder and return token classification outputs."""
        kwargs: dict[str, Any] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch.get("labels"),
        }
        for key in ("token_type_ids", "position_ids"):
            value = batch.get(key)
            if value is not None:
                kwargs[key] = value
        return self.model(**kwargs)
