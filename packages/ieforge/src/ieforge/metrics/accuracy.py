from ieforge.metrics.base import IEMetric
from ieforge.registry import register


@register("metric", name="accuracy")
class AccuracyMetric(IEMetric):
    """Simple classification accuracy metric."""

    def __init__(self, name: str = "accuracy") -> None:
        """Initialize running accuracy counters."""
        self.name = name
        self.correct = 0
        self.total = 0

    def update(self, predictions: list, references: list, texts: list[str]) -> None:
        """Accumulate prediction/reference pairs for accuracy."""
        del texts
        for prediction, reference in zip(predictions, references):
            if prediction == reference:
                self.correct += 1
            self.total += 1

    def compute(self) -> dict[str, float]:
        """Return the aggregated accuracy score."""
        accuracy = self.correct / self.total if self.total else 0.0
        return {"accuracy": accuracy}

    def reset(self) -> None:
        """Clear accumulated state."""
        self.correct = 0
        self.total = 0
