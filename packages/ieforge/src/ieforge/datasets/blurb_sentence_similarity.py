import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ieforge.data.schema import SentenceSimilarityExample
from ieforge.datasets.base import IEDataset
from ieforge.registry import register

_HF_SENTENCE_SIMILARITY_REPO = "tti-kde/blurb-sentence-similarity"


@dataclass(frozen=True)
class _SentenceSimilarityTaskSpec:
    hf_config: str
    label_list: tuple[str, ...]


_SENTENCE_SIMILARITY_TASK_REGISTRY: dict[str, _SentenceSimilarityTaskSpec] = {
    "biosses": _SentenceSimilarityTaskSpec(
        hf_config="biosses",
        label_list=("score",),
    )
}

_DEFAULT_SPLIT_COUNTS = {"train": 64, "validation": 16, "test": 20}


def _load_local_biosses_examples(
    csv_path: Path,
    *,
    split: str,
    split_counts: dict[str, int],
) -> list[SentenceSimilarityExample]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    expected_total = sum(split_counts.values())
    if len(rows) != expected_total:
        msg = (
            f"BIOSSES row count mismatch for {csv_path}: "
            f"expected {expected_total}, found {len(rows)}"
        )
        raise ValueError(msg)

    train_end = split_counts["train"]
    validation_end = train_end + split_counts["validation"]
    split_rows = {
        "train": rows[:train_end],
        "validation": rows[train_end:validation_end],
        "test": rows[validation_end : validation_end + split_counts["test"]],
    }

    return [
        SentenceSimilarityExample(
            id=str(record["id"]),
            sentence1=str(record["sentence1"]),
            sentence2=str(record["sentence2"]),
            score=float(record["score"]),
        )
        for record in split_rows[split]
    ]


def load_dataset(repo: str, config: str, split: str, token: str | None) -> Any:
    """Load from the HF Hub, or from ``IEFORGE_DATA_DIR`` when set (mockable in tests)."""
    from ieforge.datasets.local import load_hf_dataset

    return load_hf_dataset(repo, config, split=split, token=token)


@register("dataset", name="blurb_sentence_similarity")
class BLURBSentenceSimilarityDataset(IEDataset):
    """Canonical BLURB sentence-similarity dataset wrapper."""

    def __init__(
        self,
        task: str,
        split: str = "train",
        name: str = "blurb_sentence_similarity",
        data_root: str | Path | None = None,
        split_counts: dict[str, int] | None = None,
    ) -> None:
        """Load the requested BLURB sentence-similarity task split."""
        if task not in _SENTENCE_SIMILARITY_TASK_REGISTRY:
            msg = (
                f"Unknown BLURB sentence similarity task {task!r}. "
                f"Choose from: {list(_SENTENCE_SIMILARITY_TASK_REGISTRY)}"
            )
            raise ValueError(msg)
        self._task = task
        self.name = name
        spec = _SENTENCE_SIMILARITY_TASK_REGISTRY[task]
        if data_root is not None:
            csv_path = Path(data_root) / task / f"{task}.csv"
            self._examples = _load_local_biosses_examples(
                csv_path,
                split=split,
                split_counts=split_counts or _DEFAULT_SPLIT_COUNTS,
            )
            return

        token = os.environ.get("HF_TOKEN")
        hf = load_dataset(
            _HF_SENTENCE_SIMILARITY_REPO,
            spec.hf_config,
            split=split,
            token=token,
        )
        self._examples = [
            SentenceSimilarityExample(
                id=str(record["id"]),
                sentence1=str(record["sentence1"]),
                sentence2=str(record["sentence2"]),
                score=float(record["score"]),
            )
            for record in hf
        ]

    def __len__(self) -> int:
        """Return the number of sentence-similarity examples."""
        return len(self._examples)

    def __getitem__(self, index: int) -> SentenceSimilarityExample:
        """Return the sentence-similarity example at the given index."""
        return self._examples[index]

    @property
    def label_list(self) -> tuple[str, ...]:
        """Return the immutable label vocabulary for the selected task."""
        return _SENTENCE_SIMILARITY_TASK_REGISTRY[self._task].label_list
