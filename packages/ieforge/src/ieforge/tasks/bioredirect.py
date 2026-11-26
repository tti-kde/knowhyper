"""BioREDirect multi-task (relation / novelty / direction) pair task."""

from typing import Any

from ieforge.data.schema import BioREDirectPairExample, BioREDirectPairPrediction
from ieforge.registry import register
from ieforge.tasks.base import IETask

_DIRECTION_LABELS = ("None", "Left_to_Right", "Right_to_Left", "No_Direct")

_BIOREDIRECT_SPECIAL_TOKENS = ("[REL]", "[DIR]", "[NOV]")


def derive_direction(relation: str, subject_id: str, id1: str, id2: str) -> str:
    """Derive the directionality label from the annotated subject id.

    Mirrors the official BioREDirect rule (data_processor.py:345-353):
    negative relation -> ``None``; no subject -> ``No_Direct``;
    subject is the first entity -> ``Left_to_Right``; second -> ``Right_to_Left``.
    """
    if relation == "None":
        return "None"
    if subject_id == "None":
        return "No_Direct"
    if subject_id == id1:
        return "Left_to_Right"
    if subject_id == id2:
        return "Right_to_Left"
    return "No_Direct"


@register("task", name="bioredirect_pair")
class BioREDirectPairTask(IETask):
    """BioREDirect multi-task pair classification (relation/novelty/direction)."""

    def __init__(
        self,
        relation_labels: list[str],
        novelty_labels: list[str],
        direction_labels: list[str] | None = None,
        negative_label: str = "None",
        max_length: int = 512,
        name: str = "bioredirect_pair",
    ) -> None:
        """Initialize relation/novelty/direction label mappings and tokenization settings."""
        if not relation_labels:
            msg = "BioREDirectPairTask requires relation_labels"
            raise ValueError(msg)
        self.name = name
        self.relation_labels = list(relation_labels)
        self.novelty_labels = list(novelty_labels)
        self.direction_labels = list(direction_labels or _DIRECTION_LABELS)
        self.negative_label = negative_label
        self.max_length = max_length
        self.label_list = self.relation_labels
        self._rel2id = {label: i for i, label in enumerate(self.relation_labels)}
        self._nov2id = {label: i for i, label in enumerate(self.novelty_labels)}
        self._dir2id = {label: i for i, label in enumerate(self.direction_labels)}
        if negative_label not in self._rel2id:
            msg = f"negative_label {negative_label!r} not in relation_labels"
            raise ValueError(msg)

    def special_tokens(self) -> list[str]:
        """Return the three BioREDirect task-indicator special tokens."""
        return list(_BIOREDIRECT_SPECIAL_TOKENS)

    @staticmethod
    def _multihot(label: str, mapping: dict[str, int]) -> list[float]:
        vec = [0.0] * len(mapping)
        if label in mapping:
            vec[mapping[label]] = 1.0
        return vec

    def encode(self, example: BioREDirectPairExample, tokenizer: Any) -> dict[str, Any]:
        """Tokenize text and build multi-hot target vectors for all three heads."""
        novelty = example.novelty if example.novelty is not None else "None"
        tokenized = tokenizer(
            example.text, max_length=self.max_length, truncation=True, padding=False
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "relation_labels": self._multihot(example.label, self._rel2id),
            "novelty_labels": self._multihot(novelty, self._nov2id),
            "direction_labels": self._multihot(example.direction, self._dir2id),
            "pmid": example.doc_id,
            "id1": example.id1,
            "id2": example.id2,
            "id1_type": example.id1_type,
            "id2_type": example.id2_type,
            "label_text": example.label,
            "novelty_text": novelty,
            "direction_text": example.direction,
            "text": example.text,
        }

    def decode(
        self, outputs: Any, batch: dict[str, Any]
    ) -> list[BioREDirectPairPrediction]:
        """Apply argmax per head and return predictions with metadata from batch."""
        import torch

        rel = torch.argmax(outputs.relation_logits, dim=-1).tolist()
        nov = torch.argmax(outputs.novelty_logits, dim=-1).tolist()
        direction = torch.argmax(outputs.direction_logits, dim=-1).tolist()
        n = len(rel)
        return [
            BioREDirectPairPrediction(
                pmid=pmid,
                id1=id1,
                id2=id2,
                id1_type=id1_type,
                id2_type=id2_type,
                gold_label=gold_label,
                pred_label=self.relation_labels[r],
                novelty=self.novelty_labels[nv],
                direction=self.direction_labels[d],
                gold_direction=gold_dir,
            )
            for r, nv, d, pmid, id1, id2, id1_type, id2_type, gold_label, gold_dir in zip(
                rel,
                nov,
                direction,
                batch["pmids"],
                batch["id1s"],
                batch["id2s"],
                batch["id1_types"],
                batch["id2_types"],
                batch["label_text"],
                batch.get("direction_text", ["None"] * n),
                strict=True,
            )
        ]
