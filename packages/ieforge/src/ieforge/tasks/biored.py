from typing import Any

from ieforge.data.schema import BioREDPairExample, BioREDPairPrediction
from ieforge.registry import register
from ieforge.tasks.base import IETask

_BIORED_MARKER_SPECIAL_TOKENS = (
    "@ChemicalEntitySrc$",
    "@ChemicalEntityTgt$",
    "@DiseaseOrPhenotypicFeatureSrc$",
    "@DiseaseOrPhenotypicFeatureTgt$",
    "@GeneOrGeneProductSrc$",
    "@GeneOrGeneProductTgt$",
)


@register("task", name="biored_pair")
class BioREDPairTask(IETask):
    """BioRED PubTator pair-level relation classification."""

    def __init__(
        self,
        label_list: list[str],
        negative_label: str = "None",
        max_length: int = 512,
        name: str = "biored_pair",
    ) -> None:
        """Initialize relation labels and tokenization settings."""
        if not label_list:
            msg = "BioREDPairTask requires at least one label"
            raise ValueError(msg)
        if len(set(label_list)) != len(label_list):
            msg = f"BioREDPairTask labels must be unique: {label_list!r}"
            raise ValueError(msg)
        self.name = name
        self.label_list = list(label_list)
        self.negative_label = negative_label
        self.max_length = max_length
        self.label2id = dict(zip(self.label_list, range(len(self.label_list))))
        self.id2label = dict(enumerate(self.label_list))
        if self.negative_label not in self.label2id:
            msg = f"negative_label {self.negative_label!r} is not in label_list"
            raise ValueError(msg)

    def special_tokens(self) -> list[str]:
        """Return BioRED source/target marker tokens used by published training."""
        return list(_BIORED_MARKER_SPECIAL_TOKENS)

    def encode(self, example: BioREDPairExample, tokenizer: Any) -> dict[str, Any]:
        """Encode one pre-markerized BioRED pair row without modifying its text."""
        if example.label in self.label2id:
            label_text = example.label
        elif example.label == self.negative_label:
            label_text = self.negative_label
        else:
            composite_label = f"{example.label}|{example.novelty}"
            if composite_label not in self.label2id:
                msg = f"unknown relation label: {example.label!r}"
                raise ValueError(msg)
            label_text = composite_label

        tokenized = tokenizer(
            example.text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "label": self.label2id[label_text],
            "pmid": example.doc_id,
            "id1": example.id1,
            "id2": example.id2,
            "id1_type": example.id1_type,
            "id2_type": example.id2_type,
            "label_text": label_text,
            "novelty_text": example.novelty,
            "is_in_same_sent": example.is_in_same_sent,
            "min_sents_window": example.min_sents_window,
            "text": example.text,
        }

    def decode(self, outputs: Any, batch: dict[str, Any]) -> list[BioREDPairPrediction]:
        """Convert pair logits to BioRED predictions with PubTator metadata."""
        import torch

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        pred_ids: list[int] = torch.argmax(logits, dim=-1).tolist()
        novelty_text = batch.get("novelty_text", [None] * len(pred_ids))
        return [
            BioREDPairPrediction(
                pmid=pmid,
                id1=id1,
                id2=id2,
                id1_type=id1_type,
                id2_type=id2_type,
                gold_label=gold_label,
                pred_label=self.id2label[pred_id],
                novelty=novelty,
            )
            for pred_id, pmid, id1, id2, id1_type, id2_type, gold_label, novelty in zip(
                pred_ids,
                batch["pmids"],
                batch["id1s"],
                batch["id2s"],
                batch["id1_types"],
                batch["id2_types"],
                batch["label_text"],
                novelty_text,
            )
        ]
