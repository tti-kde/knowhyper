import os
from dataclasses import dataclass
from typing import Any

from ieforge.data.schema import QAExample
from ieforge.datasets.base import IEDataset
from ieforge.registry import register

_HF_QA_REPO = "tti-kde/blurb-qa"


@dataclass(frozen=True)
class _QATaskSpec:
    hf_config: str
    label_list: tuple[str, ...]


_QA_TASK_REGISTRY: dict[str, _QATaskSpec] = {
    "pubmedqa": _QATaskSpec(
        hf_config="pubmedqa",
        label_list=("yes", "maybe", "no"),
    ),
    "bioasq": _QATaskSpec(
        hf_config="bioasq",
        label_list=("yes", "no"),
    ),
}


def load_dataset(repo: str, config: str, split: str, token: str | None) -> Any:
    """Load from the HF Hub, or from ``IEFORGE_DATA_DIR`` when set (mockable in tests)."""
    from ieforge.datasets.local import load_hf_dataset

    return load_hf_dataset(repo, config, split=split, token=token)


@register("dataset", name="blurb_qa")
class BLURBQADataset(IEDataset):
    """Canonical `blurb_qa` dataset wrapper for BLURB QA tasks."""

    def __init__(self, task: str, split: str = "train", name: str = "blurb_qa") -> None:
        """Load the requested BLURB QA task split from Hugging Face."""
        if task not in _QA_TASK_REGISTRY:
            msg = f"Unknown BLURB QA task {task!r}. Choose from: {list(_QA_TASK_REGISTRY)}"
            raise ValueError(msg)
        self._task = task
        self.name = name
        spec = _QA_TASK_REGISTRY[task]
        token = os.environ.get("HF_TOKEN")
        hf = load_dataset(_HF_QA_REPO, spec.hf_config, split=split, token=token)
        self._examples = [
            QAExample(
                id=str(record["id"]),
                question=record["question"],
                context=record["context"],
                label=record["label"],
            )
            for record in hf
        ]

    def __len__(self) -> int:
        """Return the number of QA examples."""
        return len(self._examples)

    def __getitem__(self, index: int) -> QAExample:
        """Return the QA example at the given index."""
        return self._examples[index]

    @property
    def label_list(self) -> tuple[str, ...]:
        """Return the immutable label vocabulary for the selected QA task."""
        return _QA_TASK_REGISTRY[self._task].label_list
