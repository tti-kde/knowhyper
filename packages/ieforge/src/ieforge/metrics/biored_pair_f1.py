from collections import Counter
from typing import cast

from ieforge.data.schema import BioREDPairPrediction, Relation
from ieforge.metrics.base import IEMetric
from ieforge.registry import register


@register("metric", name="biored_pair_f1")
class BioREDPairF1(IEMetric):
    """Pair-level F1 using exact (head, tail) match only."""

    def __init__(
        self,
        negative_label: str = "false",
        exclude_negative: bool = True,
        name: str = "biored_pair_f1",
    ) -> None:
        """Initialise accumulators."""
        self.name = name
        self.negative_label = negative_label
        self.exclude_negative = exclude_negative
        self._pred_tuples: list[tuple[object, ...]] = []
        self._ref_tuples: list[tuple[object, ...]] = []
        self._n_examples = 0

    def update(
        self,
        predictions: list[list[Relation]] | list[BioREDPairPrediction],
        references: list[list[Relation]] | list[BioREDPairPrediction],
        texts: list[str],
    ) -> None:
        """Accumulate exact pair tuples from a batch."""
        if len(predictions) != len(references) or len(predictions) != len(texts):
            msg = (
                "BioREDPairF1 batch length mismatch: "
                f"predictions={len(predictions)} references={len(references)} texts={len(texts)}"
            )
            raise ValueError(msg)
        if self._is_pair_batch(predictions, references):
            self._update_pair_batch(
                cast("list[BioREDPairPrediction]", predictions),
                cast("list[BioREDPairPrediction]", references),
            )
            self._n_examples += len(predictions)
            return

        base = self._n_examples
        for ex_idx, (preds, refs) in enumerate(
            zip(predictions, references, strict=True)
        ):
            uid = base + ex_idx
            for relation in preds:
                if self.exclude_negative and relation.label == self.negative_label:
                    continue
                self._pred_tuples.append((uid, relation.head, relation.tail))
            for relation in refs:
                if self.exclude_negative and relation.label == self.negative_label:
                    continue
                self._ref_tuples.append((uid, relation.head, relation.tail))
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
            if self.exclude_negative and pred.pred_label == self.negative_label:
                continue
            self._pred_tuples.append((pred.pmid, pred.id1, pred.id2))
        for ref in references:
            if self.exclude_negative and ref.gold_label == self.negative_label:
                continue
            self._ref_tuples.append((ref.pmid, ref.id1, ref.id2))

    def compute(self) -> dict[str, float]:
        """Compute precision, recall, and F1 via exact pair matching."""
        if not self._pred_tuples and not self._ref_tuples:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        pred_counter = Counter(self._pred_tuples)
        ref_counter = Counter(self._ref_tuples)
        true_pos = sum((pred_counter & ref_counter).values())
        false_pos = sum((pred_counter - ref_counter).values())
        false_neg = sum((ref_counter - pred_counter).values())

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._pred_tuples = []
        self._ref_tuples = []
        self._n_examples = 0
