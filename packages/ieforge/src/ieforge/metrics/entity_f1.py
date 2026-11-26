from ieforge.data.schema import Entity
from ieforge.metrics.base import IEMetric
from ieforge.registry import register


@register("metric", name="entity_f1")
class EntityF1(IEMetric):
    """Entity-level precision, recall, and F1 via seqeval on whitespace-token BIO sequences."""

    def __init__(self, average: str = "micro", name: str = "entity_f1") -> None:
        """Initialise accumulators."""
        self.name = name
        self.average = average
        self._pred_seqs: list[list[str]] = []
        self._ref_seqs: list[list[str]] = []

    def update(
        self,
        predictions: list[list[Entity]],
        references: list[list[Entity]],
        texts: list[str],
    ) -> None:
        """Accumulate BIO sequences for a batch of predictions and references."""
        for preds, refs, text in zip(predictions, references, texts):
            self._pred_seqs.append(_entities_to_bio(preds, text))
            self._ref_seqs.append(_entities_to_bio(refs, text))

    def compute(self) -> dict[str, float]:
        """Compute and return precision, recall, and F1."""
        if not self._pred_seqs:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        import math

        from seqeval.metrics import f1_score, precision_score, recall_score

        p = precision_score(
            self._ref_seqs, self._pred_seqs, average=self.average, zero_division="warn"
        )
        r = recall_score(
            self._ref_seqs, self._pred_seqs, average=self.average, zero_division="warn"
        )
        f = f1_score(
            self._ref_seqs, self._pred_seqs, average=self.average, zero_division="warn"
        )

        def _safe(v: float) -> float:
            return 0.0 if math.isnan(v) else v

        return {
            "precision": _safe(float(p)),
            "recall": _safe(float(r)),
            "f1": _safe(float(f)),
        }

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._pred_seqs = []
        self._ref_seqs = []


def _entities_to_bio(entities: list[Entity], text: str) -> list[str]:
    """Convert character-span entities to a whitespace-token BIO sequence.

    Splits text on whitespace, computes each token's character span,
    then assigns B-/I-/O tags based on overlap with entity spans.
    Matches BLURB's entity-level seqeval evaluation convention.
    """
    tokens = text.split()
    tags = ["O"] * len(tokens)
    pos = 0
    token_spans: list[tuple[int, int]] = []
    for tok in tokens:
        start = text.index(tok, pos)
        end = start + len(tok)
        token_spans.append((start, end))
        pos = end

    for entity in entities:
        for i, (tok_start, tok_end) in enumerate(token_spans):
            if tok_end <= entity.start or tok_start >= entity.end:
                continue
            tags[i] = (
                f"B-{entity.label}"
                if tok_start <= entity.start
                else f"I-{entity.label}"
            )

    return tags
