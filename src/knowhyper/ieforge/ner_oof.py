"""Deterministic cross-fitting splits for first-pass NER predictions."""

from __future__ import annotations

import random
from typing import Any, Literal

from ieforge.datasets.base import IEDataset
from ieforge.datasets.blurb_ner import BLURBNERDataset


class NERFoldDataset(IEDataset):
    """Expose the train or holdout side of one deterministic NER fold.

    Original example objects and IDs are preserved so predictions from all
    holdout folds can be merged into a train JSONL consumed by the second pass.
    """

    def __init__(
        self,
        task: str,
        split: str = "train",
        *,
        num_folds: int = 5,
        fold: int = 0,
        fold_seed: int = 42,
        role: Literal["train", "holdout"] = "train",
    ) -> None:
        if split != "train":
            msg = f"NERFoldDataset only partitions split='train', got {split!r}"
            raise ValueError(msg)
        if num_folds < 2:
            msg = f"num_folds must be at least 2, got {num_folds}"
            raise ValueError(msg)
        if not 0 <= fold < num_folds:
            msg = f"fold must satisfy 0 <= fold < {num_folds}, got {fold}"
            raise ValueError(msg)
        if role not in {"train", "holdout"}:
            msg = f"role must be 'train' or 'holdout', got {role!r}"
            raise ValueError(msg)

        base = BLURBNERDataset(task=task, split=split)
        shuffled = list(range(len(base)))
        random.Random(fold_seed).shuffle(shuffled)  # noqa: S311 - deterministic fold assignment
        holdout = frozenset(shuffled[fold::num_folds])
        if role == "holdout":
            selected = [index for index in range(len(base)) if index in holdout]
        else:
            selected = [index for index in range(len(base)) if index not in holdout]

        self._examples = [base[index] for index in selected]
        self._label_list = list(base.label_list)

    def __len__(self) -> int:
        """Return the selected fold size."""
        return len(self._examples)

    def __getitem__(self, index: int) -> Any:
        """Return one selected example with its original stable ID."""
        return self._examples[index]

    @property
    def label_list(self) -> list[str]:
        """Return the base dataset's BIO label vocabulary."""
        return list(self._label_list)
