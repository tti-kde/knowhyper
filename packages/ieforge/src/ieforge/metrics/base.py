from abc import ABC, abstractmethod


class IEMetric(ABC):
    """Abstract base class for information extraction evaluation metrics."""

    @abstractmethod
    def update(self, predictions: list, references: list, texts: list[str]) -> None:
        """Accumulate predictions and references for a batch."""
        ...

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Compute and return the aggregated metric scores."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all accumulated state."""
        ...
