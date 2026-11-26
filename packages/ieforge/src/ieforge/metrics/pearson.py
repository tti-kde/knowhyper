from math import isnan

from ieforge.metrics.base import IEMetric
from ieforge.registry import register


@register("metric", name="pearson")
class PearsonMetric(IEMetric):
    """Pearson correlation for scalar regression targets."""

    def __init__(self, name: str = "pearson") -> None:
        """Initialize metric state."""
        self.name = name
        self.predictions: list[float] = []
        self.references: list[float] = []

    def update(self, predictions: list, references: list, texts: list[str]) -> None:
        """Accumulate scalar predictions and references."""
        del texts
        self.predictions.extend(float(value) for value in predictions)
        self.references.extend(float(value) for value in references)

    def compute(self) -> dict[str, float]:
        """Return the aggregated Pearson correlation."""
        if len(self.predictions) < 2:
            return {"pearson": 0.0}

        mean_pred = sum(self.predictions) / len(self.predictions)
        mean_ref = sum(self.references) / len(self.references)
        centered_pred = [value - mean_pred for value in self.predictions]
        centered_ref = [value - mean_ref for value in self.references]

        numerator = sum(pred * ref for pred, ref in zip(centered_pred, centered_ref))
        pred_norm = sum(pred * pred for pred in centered_pred) ** 0.5
        ref_norm = sum(ref * ref for ref in centered_ref) ** 0.5
        if pred_norm == 0.0 or ref_norm == 0.0:
            return {"pearson": 0.0}

        score = numerator / (pred_norm * ref_norm)
        return {"pearson": 0.0 if isnan(score) else score}

    def reset(self) -> None:
        """Clear accumulated state."""
        self.predictions.clear()
        self.references.clear()
