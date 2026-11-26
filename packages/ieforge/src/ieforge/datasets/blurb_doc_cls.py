import os
from dataclasses import dataclass
from typing import Any

from ieforge.data.schema import DocClassificationExample
from ieforge.datasets.base import IEDataset
from ieforge.registry import register

_HF_DOC_CLS_REPO = "tti-kde/blurb-hoc"


@dataclass(frozen=True)
class _DocClsTaskSpec:
    hf_config: str
    label_list: tuple[str, ...]


_DOC_CLS_TASK_REGISTRY: dict[str, _DocClsTaskSpec] = {
    "hoc": _DocClsTaskSpec(
        hf_config="default",
        label_list=(
            "activating invasion and metastasis",
            "avoiding immune destruction",
            "cellular energetics",
            "enabling replicative immortality",
            "evading growth suppressors",
            "genome instability and mutation",
            "inducing angiogenesis",
            "resisting cell death",
            "sustaining proliferative signaling",
            "tumor promoting inflammation",
        ),
    )
}


def load_dataset(repo: str, config: str, split: str, token: str | None) -> Any:
    """Load from the HF Hub, or from ``IEFORGE_DATA_DIR`` when set (mockable in tests)."""
    from ieforge.datasets.local import load_hf_dataset

    return load_hf_dataset(repo, config, split=split, token=token)


@register("dataset", name="blurb_doc_cls")
class BLURBDocClsDataset(IEDataset):
    """Canonical `blurb_doc_cls` dataset wrapper for BLURB document classification."""

    def __init__(
        self, task: str, split: str = "train", name: str = "blurb_doc_cls"
    ) -> None:
        """Load the requested BLURB document-classification task split."""
        if task not in _DOC_CLS_TASK_REGISTRY:
            msg = (
                f"Unknown BLURB doc classification task {task!r}. "
                f"Choose from: {list(_DOC_CLS_TASK_REGISTRY)}"
            )
            raise ValueError(msg)
        self._task = task
        self.name = name
        spec = _DOC_CLS_TASK_REGISTRY[task]
        token = os.environ.get("HF_TOKEN")
        hf = load_dataset(_HF_DOC_CLS_REPO, spec.hf_config, split=split, token=token)
        self._examples = [
            DocClassificationExample(
                id=str(record["id"]),
                text=record["text"],
                labels=tuple(record["labels"]),
            )
            for record in hf
        ]

    def __len__(self) -> int:
        """Return the number of document-classification examples."""
        return len(self._examples)

    def __getitem__(self, index: int) -> DocClassificationExample:
        """Return the document-classification example at the given index."""
        return self._examples[index]

    @property
    def label_list(self) -> tuple[str, ...]:
        """Return the immutable label vocabulary for the selected task."""
        return _DOC_CLS_TASK_REGISTRY[self._task].label_list
