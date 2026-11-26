"""Faithful BioREDirect F1 metric (per-instance position-wise counting).

This is a line-for-line port of the official scorer in
``BioREDirect/src/evaluation.py`` (``f1_score_with_details``,
``f1_score_novelty_with_details`` and their relaxed variants), assembled into
the ten official breakdowns plus an ``f1`` alias for the ``all`` level used by
the training monitor.

Unlike a set-based de-duplicating scorer, the official metric counts each
candidate instance independently (position-wise), so the same key appearing
twice contributes twice. ``update`` therefore receives two *position-aligned*
lists: ``predictions`` carry the model output (``pred_label`` / ``direction`` /
``novelty``) and ``references`` carry the gold (``gold_label`` /
``gold_direction`` / ``novelty``).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ieforge.metrics.base import IEMetric
from ieforge.registry import register

if TYPE_CHECKING:
    from ieforge.data.schema import BioREDirectPairPrediction

_POS_LABEL = "Positive_Correlation"
_NEG_LABEL = "Negative_Correlation"
_ASS_LABEL = "Association"


def _f1_from_counts(tp: float, fp: float, fn: float) -> float:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def f1_score(y_true: Sequence, y_pred: Sequence, labels: set) -> float:
    """Plain per-instance F1 (evaluation.py:15-40)."""
    tp = fp = fn = 0.0
    for gold, pred in zip(y_true, y_pred, strict=True):
        if gold == pred and gold in labels:
            tp += 1
        elif gold != pred:
            if gold in labels:
                fn += 1
            if pred in labels:
                fp += 1
    return _f1_from_counts(tp, fp, fn)


def relaxed_f1_score(
    y_true: Sequence,
    y_pred: Sequence,
    labels: set,
    pos_label: str,
    neg_label: str,
    ass_label: str,
) -> float:
    """Relaxed per-instance F1 (evaluation.py:42-69).

    Gold Positive/Negative-Correlation predicted as Association counts as TP.
    """
    tp = fp = fn = 0.0
    for gold, pred in zip(y_true, y_pred, strict=True):
        relaxed_tp = gold in (pos_label, neg_label) and pred == ass_label
        if relaxed_tp or (gold == pred and gold in labels):
            tp += 1
        elif gold != pred:
            if gold in labels:
                fn += 1
            if pred in labels:
                fp += 1
    return _f1_from_counts(tp, fp, fn)


def f1_score_novelty(
    y_true: Sequence, y_pred: Sequence, labels: set, none_label: str
) -> float:
    """Tuple per-instance F1 (evaluation.py:71-100).

    ``gold``/``pred`` are 2-tuples ``(rel, other)`` where ``other`` is novelty,
    direction or a composite ``(nov, dir)`` string depending on the breakdown.
    """
    tp = fp = fn = 0.0
    for gold, pred in zip(y_true, y_pred, strict=True):
        if gold[0] == pred[0] and gold[1] == pred[1] and gold[0] != none_label:
            tp += 1
        elif gold[0] != pred[0] or gold[1] != pred[1]:
            if gold[0] in labels:
                fn += 1
            if pred[0] in labels:
                fp += 1
    return _f1_from_counts(tp, fp, fn)


def relaxed_f1_score_novelty(
    y_true: Sequence,
    y_pred: Sequence,
    labels: set,
    none_label: str,
    pos_label: str,
    neg_label: str,
    ass_label: str,
) -> float:
    """Relaxed tuple per-instance F1 (evaluation.py:102-136)."""
    tp = fp = fn = 0.0
    for gold, pred in zip(y_true, y_pred, strict=True):
        if gold[0] in (pos_label, neg_label) and pred[0] == ass_label:
            if gold[1] == pred[1]:
                tp += 1
            else:
                fn += 1
                fp += 1
        elif gold[0] == pred[0] and gold[1] == pred[1] and gold[0] != none_label:
            tp += 1
        elif gold[0] != pred[0] or gold[1] != pred[1]:
            if gold[0] in labels:
                fn += 1
            if pred[0] in labels:
                fp += 1
    return _f1_from_counts(tp, fp, fn)


@register("metric", name="bioredirect_f1")
class BioREDirectF1(IEMetric):
    """Per-instance micro-F1 at the ten official BioREDirect breakdowns."""

    def __init__(
        self, negative_label: str = "None", name: str = "bioredirect_f1"
    ) -> None:
        """Initialise prediction/reference accumulators."""
        self.name = name
        self.negative_label = negative_label
        self._preds: list[BioREDirectPairPrediction] = []
        self._refs: list[BioREDirectPairPrediction] = []

    def update(self, predictions: list, references: list, texts: list[str]) -> None:
        """Accumulate a batch of position-aligned predictions and references."""
        del texts
        self._preds.extend(predictions)
        self._refs.extend(references)

    def _triples(self) -> tuple[list[tuple], list[tuple]]:
        """Return position-aligned ``(rel, novelty, direction)`` gold/pred triples.

        Gold values come from the reference objects (``gold_label`` /
        ``novelty`` / ``gold_direction`` built in ``_metric_inputs``); predicted
        values come from the prediction objects (``pred_label`` / ``novelty`` /
        ``direction`` produced by ``decode``).
        """
        gold = [(r.gold_label, str(r.novelty), r.gold_direction) for r in self._refs]
        pred = [(p.pred_label, str(p.novelty), p.direction) for p in self._preds]
        return gold, pred

    def compute(self) -> dict[str, float]:
        """Return per-instance F1 at every official breakdown plus ``f1`` alias."""
        gold, pred = self._triples()

        # Official ``unique_labels``: relation labels present in gold minus None.
        labels = {g[0] for g in gold}
        labels.discard(self.negative_label)

        none_label = self.negative_label

        rel_gold = [g[0] for g in gold]
        rel_pred = [p[0] for p in pred]

        # rel (RT): plain F1 over relation labels.
        rel = f1_score(rel_gold, rel_pred, labels)
        relaxed_rel = relaxed_f1_score(
            rel_gold, rel_pred, labels, _POS_LABEL, _NEG_LABEL, _ASS_LABEL
        )

        # rel_nov (RT+NV): tuple (rel, novelty).
        rel_nov_gold = [(g[0], g[1]) for g in gold]
        rel_nov_pred = [(p[0], p[1]) for p in pred]
        rel_nov = f1_score_novelty(rel_nov_gold, rel_nov_pred, labels, none_label)

        # rel_dir (RT+DI): tuple (rel, direction).
        rel_dir_gold = [(g[0], g[2]) for g in gold]
        rel_dir_pred = [(p[0], p[2]) for p in pred]
        rel_dir = f1_score_novelty(rel_dir_gold, rel_dir_pred, labels, none_label)

        # all: tuple (rel, "<novelty>_<direction>").
        all_gold = [(g[0], f"{g[1]}_{g[2]}") for g in gold]
        all_pred = [(p[0], f"{p[1]}_{p[2]}") for p in pred]
        all_f1 = f1_score_novelty(all_gold, all_pred, labels, none_label)

        # Relaxed novelty variants operate on the full (rel, nov, dir) triples
        # (the official code passes gold_labels/pred_labels directly), where the
        # tuple comparison uses index [1] as novelty.
        relaxed_rel_nov = relaxed_f1_score_novelty(
            gold, pred, labels, none_label, _POS_LABEL, _NEG_LABEL, _ASS_LABEL
        )
        relaxed_rel_dir = relaxed_f1_score_novelty(
            gold, pred, labels, none_label, _POS_LABEL, _NEG_LABEL, _ASS_LABEL
        )
        relaxed_all = relaxed_f1_score_novelty(
            gold, pred, labels, none_label, _POS_LABEL, _NEG_LABEL, _ASS_LABEL
        )

        # pair (EP): merge all positive relation labels to "Association",
        # then plain F1 over [Association].
        merged_gold = [_ASS_LABEL if g in labels else g for g in rel_gold]
        merged_pred = [_ASS_LABEL if p in labels else p for p in rel_pred]
        pair = f1_score(merged_gold, merged_pred, {_ASS_LABEL})

        # pair_nov: EP-merged on (rel, novelty) tuples, novelty-tuple F1.
        merged_nov_gold = [
            (_ASS_LABEL, g[1]) if g[0] in labels else (g[0], g[1]) for g in gold
        ]
        merged_nov_pred = [
            (_ASS_LABEL, p[1]) if p[0] in labels else (p[0], p[1]) for p in pred
        ]
        pair_nov = f1_score_novelty(
            merged_nov_gold, merged_nov_pred, labels, none_label
        )

        scores = {
            "pair": pair,
            "pair_nov": pair_nov,
            "rel": rel,
            "rel_nov": rel_nov,
            "rel_dir": rel_dir,
            "all": all_f1,
            "relaxed_rel": relaxed_rel,
            "relaxed_rel_nov": relaxed_rel_nov,
            "relaxed_rel_dir": relaxed_rel_dir,
            "relaxed_all": relaxed_all,
        }
        scores["f1"] = scores["all"]
        return scores

    def reset(self) -> None:
        """Clear accumulators."""
        self._preds = []
        self._refs = []
