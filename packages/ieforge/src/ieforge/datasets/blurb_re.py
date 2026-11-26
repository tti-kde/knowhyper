import os
from dataclasses import dataclass
from typing import Any

from ieforge.data.schema import Entity, REExample, Relation
from ieforge.datasets.base import IEDataset
from ieforge.registry import register

_HF_RE_REPO = "tti-kde/blurb-re"


@dataclass(frozen=True)
class _RETaskSpec:
    hf_config: str
    label_list: tuple[str, ...]
    negative_label: str


_RE_TASK_REGISTRY: dict[str, _RETaskSpec] = {
    "chemprot": _RETaskSpec(
        hf_config="chemprot",
        label_list=("CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "false"),
        negative_label="false",
    ),
    "ddi": _RETaskSpec(
        hf_config="ddi",
        label_list=(
            "DDI-mechanism",
            "DDI-effect",
            "DDI-advise",
            "DDI-int",
            "DDI-false",
        ),
        negative_label="DDI-false",
    ),
    "gad": _RETaskSpec(
        hf_config="gad",
        label_list=("0", "1"),
        negative_label="0",
    ),
}


def load_dataset(repo: str, config: str, split: str, token: str | None) -> Any:
    """Load from the HF Hub, or from ``IEFORGE_DATA_DIR`` when set (mockable in tests)."""
    from ieforge.datasets.local import load_hf_dataset

    return load_hf_dataset(repo, config, split=split, token=token)


def _normalize_mention(value: object) -> list[str] | None:
    """Normalize a HF `mention` field to ``list[str] | None``.

    ChemProt stores ``list[str]``, DDI stores ``str``, GAD stores ``None``.
    Empty values collapse to ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [value] if value else None
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value if v]
        return items or None
    return None


def _parse_record(record: dict) -> REExample:
    """Convert a Hugging Face `blurb_re` record into the shared RE schema."""
    entities = [
        Entity(
            start=e.get("start", 0),
            end=e.get("end", 0),
            text=e["text"],
            label=e["label"],
            mention=_normalize_mention(e.get("mention")),
        )
        for e in record["entities"]
    ]
    relations = [
        Relation(head=r["head"], tail=r["tail"], label=r["label"])
        for r in record["relations"]
    ]
    return REExample(
        id=record["id"], text=record["text"], entities=entities, relations=relations
    )


@register("dataset", name="blurb_re")
class BLURBREDataset(IEDataset):
    """Canonical `blurb_re` dataset wrapper for BLURB RE tasks.

    Loads ChemProt, DDI, and GAD from `tti-kde/blurb-re` on Hugging Face.
    """

    def __init__(self, task: str, split: str = "train", name: str = "blurb_re") -> None:
        """Load the requested BLURB RE task split under the canonical `blurb_re` name."""
        if task not in _RE_TASK_REGISTRY:
            msg = f"Unknown BLURB RE task {task!r}. Choose from: {list(_RE_TASK_REGISTRY)}"
            raise ValueError(msg)
        self._task = task
        self.name = name
        spec = _RE_TASK_REGISTRY[task]
        token = os.environ.get("HF_TOKEN")
        hf = load_dataset(_HF_RE_REPO, spec.hf_config, split=split, token=token)
        self._examples: list[REExample] = [_parse_record(rec) for rec in hf]

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self._examples)

    def __getitem__(self, index: int) -> REExample:
        """Return the REExample at the given index."""
        return self._examples[index]

    @property
    def label_list(self) -> list[str]:
        """Return the relation label vocabulary for this task."""
        return list(_RE_TASK_REGISTRY[self._task].label_list)

    @property
    def negative_label(self) -> str:
        """Return the negative (no-relation) label string."""
        return _RE_TASK_REGISTRY[self._task].negative_label
