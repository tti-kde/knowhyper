import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import MISSING, asdict, fields, replace
from pathlib import Path

from ieforge.data.schema import BioREDPairPrediction
from ieforge.metrics.base import IEMetric
from ieforge.registry import register

_NEGATIVE_RELATION_LABELS = {"None", "false"}
_BINARY_POSITIVE_LABEL = "true"
_DEFAULT_RELATION_LABEL = "Association"
_DEFAULT_NOVELTY = "Novel"


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Return precision, recall, and F1 for count totals."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _split_label_and_novelty(label: str, novelty: str | None) -> tuple[str, str | None]:
    if "|" in label:
        relation, label_novelty = label.split("|", 1)
        return relation, label_novelty
    if label == _BINARY_POSITIVE_LABEL:
        return _DEFAULT_RELATION_LABEL, novelty
    return label, novelty


def _normalize_pubtator_novelty(novelty: str | None) -> str:
    if not novelty or novelty == "None":
        return _DEFAULT_NOVELTY
    return novelty


def _pair_key(prediction: BioREDPairPrediction) -> tuple[str, tuple[str, str]]:
    first_id, second_id = sorted((prediction.id1, prediction.id2))
    return prediction.pmid, (first_id, second_id)


def merge_relation_and_novelty_predictions(
    relation_predictions: Iterable[BioREDPairPrediction],
    novelty_predictions: Iterable[BioREDPairPrediction],
) -> list[BioREDPairPrediction]:
    """Attach novelty predictions to relation predictions by unordered PubTator pair."""
    novelty_by_pair = {
        _pair_key(prediction): prediction.pred_label
        for prediction in novelty_predictions
    }
    return [
        replace(
            prediction,
            novelty=novelty_by_pair.get(_pair_key(prediction), _DEFAULT_NOVELTY),
        )
        for prediction in relation_predictions
    ]


def write_biored_predictions_jsonl(
    predictions: Iterable[BioREDPairPrediction], path: str | Path
) -> None:
    """Write BioRED pair predictions as JSONL dataclass-field records."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(asdict(prediction), sort_keys=True) + "\n")


def read_biored_predictions_jsonl(path: str | Path) -> list[BioREDPairPrediction]:
    """Read BioRED pair predictions from JSONL dataclass-field records."""
    prediction_fields = fields(BioREDPairPrediction)
    predictions = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            values = {}
            for field in prediction_fields:
                if field.name in row:
                    values[field.name] = row[field.name]
                elif field.default is not MISSING:
                    values[field.name] = field.default
                elif field.default_factory is not MISSING:
                    values[field.name] = field.default_factory()
                else:
                    raise KeyError(field.name)
            predictions.append(BioREDPairPrediction(**values))
    return predictions


def _relation_predictions_to_pubtator_lines(
    predictions: Iterable[BioREDPairPrediction],
    negative_labels: set[str],
) -> list[str]:
    lines = []
    for prediction in predictions:
        relation_label, novelty = _split_label_and_novelty(
            prediction.pred_label, prediction.novelty
        )
        if relation_label in negative_labels:
            continue
        lines.append(
            "\t".join(
                (
                    prediction.pmid,
                    relation_label,
                    prediction.id1,
                    prediction.id2,
                    _normalize_pubtator_novelty(novelty),
                )
            )
        )
    return sorted(lines)


def relation_predictions_to_pubtator_lines(
    predictions: Iterable[BioREDPairPrediction],
) -> list[str]:
    """Convert non-negative pair predictions to sorted PubTator-style lines."""
    return _relation_predictions_to_pubtator_lines(
        predictions, set(_NEGATIVE_RELATION_LABELS)
    )


def _pubtator_key(line: str) -> tuple[str, str, str, str, str]:
    parts = line.strip().split()
    if len(parts) != 5:
        msg = f"BioRED PubTator relation line must have 5 fields: {line!r}"
        raise ValueError(msg)
    pmid, relation, id1, id2, novelty = parts
    first_id, second_id = sorted((id1, id2))
    return pmid, relation, first_id, second_id, novelty


def biored_pubtator_score(
    gold_lines: Sequence[str], pred_lines: Sequence[str]
) -> dict[str, float]:
    """Score exact BioRED relation matches with unordered entity pairs."""
    gold_counter = Counter(_pubtator_key(line) for line in gold_lines)
    pred_counter = Counter(_pubtator_key(line) for line in pred_lines)
    tp = sum((gold_counter & pred_counter).values())
    fp = sum((pred_counter - gold_counter).values())
    fn = sum((gold_counter - pred_counter).values())
    return _prf(tp, fp, fn)


def _references_to_pubtator_lines(
    references: Iterable[BioREDPairPrediction],
    negative_labels: set[str],
) -> list[str]:
    return _relation_predictions_to_pubtator_lines(
        (
            BioREDPairPrediction(
                pmid=reference.pmid,
                id1=reference.id1,
                id2=reference.id2,
                id1_type=reference.id1_type,
                id2_type=reference.id2_type,
                gold_label=reference.gold_label,
                pred_label=reference.gold_label,
                novelty=reference.novelty,
            )
            for reference in references
        ),
        negative_labels,
    )


@register("metric", name="biored_f1")
class BioREDF1(IEMetric):
    """BioRED pair/type/novelty F1 using PubTator-style exact matching."""

    def __init__(
        self,
        negative_label: str = "None",
        name: str = "biored_f1",
    ) -> None:
        """Initialise BioRED exact-match accumulators."""
        self.name = name
        self.negative_label = negative_label
        self._predictions: list[BioREDPairPrediction] = []
        self._references: list[BioREDPairPrediction] = []
        self.last_predictions: list[BioREDPairPrediction] = []

    def update(
        self,
        predictions: list[BioREDPairPrediction],
        references: list[BioREDPairPrediction],
        texts: list[str],
    ) -> None:
        """Accumulate BioRED pair predictions and gold references."""
        if len(predictions) != len(references) or len(predictions) != len(texts):
            msg = (
                "BioREDF1 batch length mismatch: "
                f"predictions={len(predictions)} "
                f"references={len(references)} texts={len(texts)}"
            )
            raise ValueError(msg)
        if not all(
            isinstance(item, BioREDPairPrediction) for item in predictions + references
        ):
            msg = "BioREDF1 expects flat BioREDPairPrediction inputs"
            raise TypeError(msg)
        self._predictions.extend(predictions)
        self._references.extend(references)

    def compute(self) -> dict[str, float]:
        """Compute BioRED precision, recall, and F1."""
        self.last_predictions = list(self._predictions)
        negative_labels = set(_NEGATIVE_RELATION_LABELS) | {self.negative_label}
        pred_lines = _relation_predictions_to_pubtator_lines(
            self._predictions, negative_labels
        )
        gold_lines = _references_to_pubtator_lines(self._references, negative_labels)
        return biored_pubtator_score(gold_lines, pred_lines)

    def reset(self) -> None:
        """Clear accumulated state."""
        self._predictions = []
        self._references = []
