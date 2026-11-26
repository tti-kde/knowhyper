"""Dataset wrappers that apply optional example augmentation (e.g. KnowForge)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ieforge.datasets.base import IEDataset


@runtime_checkable
class ExampleAugmentor(Protocol):
    """Protocol for callables that map one training example to another."""

    def augment(self, example: Any) -> Any:
        """Return a (possibly) transformed copy of ``example``."""
        ...


class AugmentedDataset(IEDataset):
    """Wraps an `IEDataset` and runs `augmentor.augment` on each `__getitem__`."""

    def __init__(
        self,
        dataset: IEDataset,
        augmentor: ExampleAugmentor,
        cache: dict[int, Any] | None = None,
    ) -> None:
        """Build a dataset that augments every example drawn from ``dataset``."""
        self._dataset = dataset
        self._augmentor = augmentor
        self._cache = cache

    def __len__(self) -> int:
        """Return the same length as the wrapped dataset."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        """Return ``augmentor.augment(inner[index])``, with optional per-index cache."""
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        ex = self._dataset[index]
        aug = self._augmentor.augment(ex)
        if self._cache is not None:
            self._cache[index] = aug
        return aug

    def __getattr__(self, name: str) -> Any:
        """Delegate missing attributes (e.g. ``label_list``) to the inner dataset."""
        if name.startswith("_"):
            msg = f"{type(self).__name__!r} object has no attribute {name!r}"
            raise AttributeError(msg)
        return getattr(self._dataset, name)
