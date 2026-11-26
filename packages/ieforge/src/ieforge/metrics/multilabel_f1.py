from ieforge.metrics.base import IEMetric
from ieforge.registry import register


@register("metric", name="multilabel_f1")
class MultiLabelF1(IEMetric):
    """Micro-F1 over multi-label document classification predictions."""

    def __init__(self, threshold: float = 0.5, name: str = "multilabel_f1") -> None:
        """Initialize counters for micro-averaged F1."""
        self.name = name
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(
        self,
        predictions: list[tuple[str, ...]],
        references: list[tuple[str, ...]],
        texts: list[str],
    ) -> None:
        """Accumulate true positives, false positives, and false negatives."""
        del texts
        for predicted, reference in zip(predictions, references):
            predicted_set = set(predicted)
            reference_set = set(reference)
            self.tp += len(predicted_set & reference_set)
            self.fp += len(predicted_set - reference_set)
            self.fn += len(reference_set - predicted_set)

    def compute(self) -> dict[str, float]:
        """Return the micro-F1 score."""
        denom = (2 * self.tp) + self.fp + self.fn
        f1 = (2 * self.tp) / denom if denom else 0.0
        return {"f1": f1}

    def reset(self) -> None:
        """Clear accumulated state."""
        self.tp = 0
        self.fp = 0
        self.fn = 0
