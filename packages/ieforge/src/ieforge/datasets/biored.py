import os
from dataclasses import dataclass
from typing import Any

from ieforge.data.schema import BioREDirectPairExample, BioREDPairExample, Entity
from ieforge.datasets.base import IEDataset
from ieforge.registry import register
from ieforge.tasks.bioredirect import derive_direction

_HF_BIORED_REPO = "tti-kde/biored"


@dataclass(frozen=True)
class _BioREDTaskSpec:
    hf_config: str
    label_list: tuple[str, ...]
    negative_label: str
    label_field: str
    exclude_negative_train_dev: bool = False
    produces_direction: bool = False


_BIORED_RELATION_LABELS = (
    "None",
    "Association",
    "Bind",
    "Comparison",
    "Conversion",
    "Cotreatment",
    "Drug_Interaction",
    "Negative_Correlation",
    "Positive_Correlation",
)
_BIORED_NOVELTY_LABELS = ("None", "No", "Novel")

_BIORED_RELATION_SPEC = _BioREDTaskSpec(
    hf_config="biored",
    label_list=_BIORED_RELATION_LABELS,
    negative_label="None",
    label_field="relation_label",
)

_BIORED_TASK_REGISTRY: dict[str, _BioREDTaskSpec] = {
    "biored_pair": _BioREDTaskSpec(
        hf_config="biored",
        label_list=("false", "true"),
        negative_label="false",
        label_field="binary",
    ),
    "biored": _BIORED_RELATION_SPEC,
    "biored_relation": _BIORED_RELATION_SPEC,
    "bioredirect": _BioREDTaskSpec(
        hf_config="biored",
        label_list=_BIORED_RELATION_LABELS,
        negative_label="None",
        label_field="relation_label",
        produces_direction=True,
    ),
    "biored_novelty": _BioREDTaskSpec(
        hf_config="biored",
        label_list=_BIORED_NOVELTY_LABELS,
        negative_label="None",
        label_field="novelty_label",
        exclude_negative_train_dev=True,
    ),
}


def load_dataset(repo: str, config: str, split: str, token: str | None) -> Any:
    """Load from the HF Hub, or from ``IEFORGE_DATA_DIR`` when set (mockable in tests)."""
    from ieforge.datasets.local import load_hf_dataset

    return load_hf_dataset(repo, config, split=split, token=token)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    return bool(value)


@register("dataset", name="biored")
class BioREDDataset(IEDataset):
    """Canonical BioRED pair-level relation-extraction dataset wrapper."""

    def __init__(self, task: str, split: str = "train", name: str = "biored") -> None:
        """Load the requested BioRED split."""
        if task not in _BIORED_TASK_REGISTRY:
            msg = f"Unknown BioRED task {task!r}. Choose from: {list(_BIORED_TASK_REGISTRY)}"
            raise ValueError(msg)
        self._task = task
        self.name = name
        spec = _BIORED_TASK_REGISTRY[task]
        token = os.environ.get("HF_TOKEN")
        hf = load_dataset(_HF_BIORED_REPO, spec.hf_config, split=split, token=token)
        examples = [self._parse_record(record) for record in hf]
        if spec.exclude_negative_train_dev and split in {"train", "validation", "dev"}:
            examples = [
                example for example in examples if example.label != spec.negative_label
            ]
        self._examples = examples

    def _parse_record(
        self, record: dict[str, Any]
    ) -> BioREDPairExample | BioREDirectPairExample:
        relation_label = str(record["relation_label"])
        novelty = record.get("novelty_label")
        label = self._map_label(relation_label, novelty)
        id1_type = str(record["id1_type"])
        id2_type = str(record["id2_type"])
        id1 = str(record["id1"])
        id2 = str(record["id2"])
        if _BIORED_TASK_REGISTRY[self._task].produces_direction:
            subject_id = str(record.get("subject_id", "None"))
            direction = derive_direction(relation_label, subject_id, id1, id2)
            return BioREDirectPairExample(
                id=str(record["id"]),
                doc_id=str(record["pmid"]),
                text=str(record["text"]),
                label=label,
                novelty=None if novelty is None else str(novelty),
                direction=direction,
                subject_id=subject_id,
                id1=id1,
                id2=id2,
                id1_type=id1_type,
                id2_type=id2_type,
                is_in_same_sent=_coerce_bool(record["is_in_same_sent"]),
                min_sents_window=int(record["min_sents_window"]),
            )
        id1_mentions = record.get("id1_mentions") or None
        id2_mentions = record.get("id2_mentions") or None
        return BioREDPairExample(
            id=str(record["id"]),
            doc_id=str(record["pmid"]),
            text=str(record["text"]),
            entities=[
                Entity(
                    start=0,
                    end=0,
                    text=str(record["id1"]),
                    label=id1_type,
                    mention=list(id1_mentions) if id1_mentions else None,
                ),
                Entity(
                    start=0,
                    end=0,
                    text=str(record["id2"]),
                    label=id2_type,
                    mention=list(id2_mentions) if id2_mentions else None,
                ),
            ],
            head_idx=0,
            tail_idx=1,
            label=label,
            novelty=None if novelty is None else str(novelty),
            id1=str(record["id1"]),
            id2=str(record["id2"]),
            id1_type=id1_type,
            id2_type=id2_type,
            is_in_same_sent=_coerce_bool(record["is_in_same_sent"]),
            min_sents_window=int(record["min_sents_window"]),
        )

    def _map_label(self, relation_label: str, novelty: Any) -> str:
        spec = _BIORED_TASK_REGISTRY[self._task]
        if spec.label_field == "binary":
            return "false" if relation_label == "None" else "true"
        if spec.label_field == "relation_label":
            return relation_label
        if spec.label_field == "novelty_label":
            return "None" if novelty is None else str(novelty)
        msg = f"unsupported BioRED label field: {spec.label_field!r}"
        raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self._examples)

    def __getitem__(self, index: int) -> BioREDPairExample | BioREDirectPairExample:
        """Return the BioRED pair example at the given index."""
        return self._examples[index]

    @property
    def label_list(self) -> list[str]:
        """Return the label vocabulary for this task."""
        return list(_BIORED_TASK_REGISTRY[self._task].label_list)

    @property
    def negative_label(self) -> str:
        """Return the negative label string."""
        return _BIORED_TASK_REGISTRY[self._task].negative_label
