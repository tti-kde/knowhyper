from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ieforge.datasets.base import IEDataset


class IEDataModule(pl.LightningDataModule):
    """Lightning DataModule that wraps train/val/test IEDataset instances."""

    def __init__(
        self,
        train_dataset: IEDataset,
        val_dataset: IEDataset,
        test_dataset: IEDataset,
        collator: Any,
        batch_size: int = 32,
        num_workers: int = 4,
        train_collator: Any = None,
        train_batch_size: int | None = None,
        test_collator: Any = None,
    ) -> None:
        """Initialize the data module with datasets and up to three collators.

        ``collator`` is used for validation (and test when ``test_collator`` is
        ``None``).  ``train_collator`` overrides it for the training loader when
        provided.  ``test_collator`` overrides it for the test loader, allowing
        split-specific metadata (e.g. knowledge cache namespace) to differ between
        validation and test.

        ``train_batch_size`` overrides ``batch_size`` for the training DataLoader only.
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.collator = collator
        self.train_collator = train_collator if train_collator is not None else collator
        self.test_collator = test_collator if test_collator is not None else collator
        self.batch_size = batch_size
        self.train_batch_size = (
            train_batch_size if train_batch_size is not None else batch_size
        )
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Return the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_collator,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Return the test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_collator,
        )


class IEEvalDataModule(pl.LightningDataModule):
    """Test-only DataModule for evaluation runs.

    Unlike ``IEDataModule`` this holds only the test dataset; there is no
    train/val split to reuse. Use it from the eval CLI path.
    """

    def __init__(
        self,
        test_dataset: IEDataset,
        collator: Any,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        """Initialize with the single test dataset and its collator."""
        super().__init__()
        self.test_dataset = test_dataset
        self.collator = collator
        self.batch_size = batch_size
        self.num_workers = num_workers

    def test_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Return the test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )
