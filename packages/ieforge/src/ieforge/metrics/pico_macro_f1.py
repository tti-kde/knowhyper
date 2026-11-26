from ieforge.metrics.base import IEMetric
from ieforge.registry import register


@register("metric", name="pico_macro_f1")
class PICOMacroF1(IEMetric):
    """Macro F1 over PICO word labels excluding background."""

    def __init__(self, label_list: list[str], name: str = "pico_macro_f1") -> None:
        """Initialize the non-background label set and accumulators."""
        self.name = name
        self.label_list = [label for label in label_list if label != "O"]
        self._predictions: list[tuple[str, ...]] = []
        self._references: list[tuple[str, ...]] = []

    def update(
        self,
        predictions: list[tuple[str, ...]],
        references: list[tuple[str, ...]],
        texts: list[str],
    ) -> None:
        """Accumulate word-level predictions and references for a batch."""
        del texts
        self._predictions.extend(predictions)
        self._references.extend(references)

    def compute(self) -> dict[str, float]:
        """Compute the macro-averaged F1 across PICO labels."""
        if not self._predictions:
            return {"f1": 0.0}

        scores: list[float] = []
        for label in self.label_list:
            tp = fp = fn = 0
            for pred_seq, ref_seq in zip(self._predictions, self._references):
                max_len = max(len(pred_seq), len(ref_seq))
                for idx in range(max_len):
                    pred = pred_seq[idx] if idx < len(pred_seq) else None
                    ref = ref_seq[idx] if idx < len(ref_seq) else None
                    if pred == label and ref == label:
                        tp += 1
                    elif pred == label and ref != label:
                        fp += 1
                    elif pred != label and ref == label:
                        fn += 1
            denom = (2 * tp) + fp + fn
            scores.append(0.0 if denom == 0 else (2 * tp) / denom)
        return {"f1": sum(scores) / len(scores)}

    def reset(self) -> None:
        """Clear all accumulated metric state."""
        self._predictions = []
        self._references = []
