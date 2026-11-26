from abc import abstractmethod
from typing import Any

from torch.utils.data import Dataset


class IEDataset(Dataset):  # type: ignore[type-arg]
    """Abstract base class for information extraction datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Return the example at the given index."""
        ...
