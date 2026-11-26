import math
from collections import Counter
from typing import cast

from ieforge.data.schema import BioREDPairPrediction, Relation
from ieforge.metrics.base import IEMetric
from ieforge.registry import register


@register("metric", name="relation_f1")
class RelationF1(IEMetric):
    """Relation-level precision, recall, and F1 using exact (head, tail, label) match."""

    def __init__(
        self,
        negative_label: str = "false",
        exclude_negative: bool = True,
        average: str = "micro",
        name: str = "relation_f1",
    ) -> None:
        """Initialise accumulators."""
        self.name = name
        self.negative_label = negative_label
        self.exclude_negative = exclude_negative
        self.average = average
        self._pred_tuples: list[tuple[object, ...]] = []
        self._ref_tuples: list[tuple[object, ...]] = []
        self._n_examples: int = 0

    def update(
        self,
        predictions: list[list[Relation]] | list[BioREDPairPrediction],
        references: list[list[Relation]] | list[BioREDPairPrediction],
        texts: list[str],  # noqa: ARG002
    ) -> None:
        """Accumulate exact labeled relation or pair tuples from a batch.

        Relation batches are tagged with a globally unique example index so that
        Counter-based set matching remains correct when head/tail indices repeat
        across examples (e.g. all pairs have head=0, tail=1 in dummified data).
        BioRED pair batches use PubTator IDs as their stable key.
        """
        if self._is_pair_batch(predictions, references):
            self._update_pair_batch(
                cast("list[BioREDPairPrediction]", predictions),
                cast("list[BioREDPairPrediction]", references),
            )
            self._n_examples += len(predictions)
            return

        base = self._n_examples
        for ex_idx, (preds, refs) in enumerate(zip(predictions, references)):
            uid = base + ex_idx
            for r in preds:
                self._pred_tuples.append((uid, r.head, r.tail, r.label))
            for r in refs:
                self._ref_tuples.append((uid, r.head, r.tail, r.label))
        self._n_examples += len(predictions)

    @staticmethod
    def _is_pair_batch(predictions: list, references: list) -> bool:
        return bool(predictions or references) and all(
            isinstance(item, BioREDPairPrediction) for item in predictions + references
        )

    def _update_pair_batch(
        self,
        predictions: list[BioREDPairPrediction],
        references: list[BioREDPairPrediction],
    ) -> None:
        for pred in predictions:
            self._pred_tuples.append((pred.pmid, pred.id1, pred.id2, pred.pred_label))
        for ref in references:
            self._ref_tuples.append((ref.pmid, ref.id1, ref.id2, ref.gold_label))

    def compute(self) -> dict[str, float]:
        """Compute precision, recall, and F1 via exact set matching."""
        pred_tuples = self._pred_tuples
        ref_tuples = self._ref_tuples

        if self.exclude_negative:
            pred_tuples = [t for t in pred_tuples if t[-1] != self.negative_label]
            ref_tuples = [t for t in ref_tuples if t[-1] != self.negative_label]

        if not ref_tuples and not pred_tuples:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        pred_counter = Counter(pred_tuples)
        ref_counter = Counter(ref_tuples)

        true_pos = sum((pred_counter & ref_counter).values())
        false_pos = sum((pred_counter - ref_counter).values())
        false_neg = sum((ref_counter - pred_counter).values())

        precision = (
            true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        )
        recall = (
            true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        def _safe(v: float) -> float:
            return 0.0 if math.isnan(v) else v

        return {"precision": _safe(precision), "recall": _safe(recall), "f1": _safe(f1)}

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._pred_tuples = []
        self._ref_tuples = []
        self._n_examples = 0
