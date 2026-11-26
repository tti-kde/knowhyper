import bisect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ieforge.data.schema import Entity, NERExample
from ieforge.registry import register
from ieforge.tasks.base import IETask


def _bio_to_entity_types(bio_labels: list[str]) -> list[str]:
    """Extract unique entity type names from a BIO label list."""
    types: list[str] = []
    seen: set[str] = set()
    for label in bio_labels:
        if label.startswith("B-"):
            entity_type = label[2:]
            if entity_type not in seen:
                types.append(entity_type)
                seen.add(entity_type)
    return types


def _offset_gap_word_ids(
    offset_mapping: list[tuple[int, int]],
) -> list[int | None]:
    """Derive word ids from offset gaps, for encodings without `word_ids()`.

    Only used as a fallback when the tokenizer output exposes no `word_ids()`
    (e.g. a plain dict from a stub tokenizer). Assumes WordPiece style offsets
    where a gap between tokens marks a new word.
    """
    word_ids: list[int | None] = []
    current = -1
    prev_end = -1
    for start, end in offset_mapping:
        if start == end:
            word_ids.append(None)
            continue
        if start > prev_end:
            current += 1
        word_ids.append(current)
        prev_end = end
    return word_ids


def _resolve_word_ids(
    tokenized: Any,
    offset_mapping: list[tuple[int, int]],
) -> list[int | None]:
    """Return per-token word ids, preferring the tokenizer's own grouping."""
    word_ids_fn = getattr(tokenized, "word_ids", None)
    if callable(word_ids_fn):
        try:
            word_ids = word_ids_fn()
        except (ValueError, TypeError):
            word_ids = None
        if word_ids is not None:
            return list(word_ids)
    return _offset_gap_word_ids(offset_mapping)


def _build_words(
    word_ids: list[int | None],
    offset_mapping: list[tuple[int, int]],
    text: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Group tokens into words, returning start token indices and char spans.

    Grouping is driven by the tokenizer's own `word_ids()` rather than inferred
    from offset arithmetic. Offset gaps only mark word boundaries for WordPiece
    style tokenizers; byte-level BPE tokenizers (e.g. ModernBERT) emit
    contiguous offsets because the leading space belongs to the token, so any
    gap-based heuristic collapses the whole sequence into a single word.

    Char spans are trimmed against `text` so leading whitespace absorbed into a
    byte-level token is not reported as part of the word.
    """
    word_starts: list[int] = []
    word_char_spans: list[tuple[int, int]] = []
    prev_word_id: int | None = None

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        char_start, char_end = offset_mapping[idx]
        if char_start == char_end:
            continue
        if word_id != prev_word_id:
            word_starts.append(idx)
            word_char_spans.append((char_start, char_end))
        else:
            span_start, _ = word_char_spans[-1]
            word_char_spans[-1] = (span_start, char_end)
        prev_word_id = word_id

    trimmed: list[tuple[int, int]] = []
    for char_start, char_end in word_char_spans:
        start = char_start
        while start < char_end and text[start].isspace():
            start += 1
        end = char_end
        while end > start and text[end - 1].isspace():
            end -= 1
        trimmed.append((start, end))

    return word_starts, trimmed


@dataclass(frozen=True)
class _SpanCandidate:
    """An above-threshold span candidate in character offsets."""

    start: int
    end: int
    label: str
    score: float


def _select_greedy(candidates: list[_SpanCandidate]) -> list[_SpanCandidate]:
    """Take candidates in descending score order, skipping any that overlap a kept span.

    Locally optimal per pick, but a high-scoring span can block a set of lower
    scoring spans whose combined score is larger.
    """
    chosen: list[_SpanCandidate] = []
    used: list[tuple[int, int]] = []
    for cand in sorted(candidates, key=lambda c: -c.score):
        if any(not (cand.end <= us or cand.start >= ue) for us, ue in used):
            continue
        chosen.append(cand)
        used.append((cand.start, cand.end))
    return chosen


def _select_max_total_score(candidates: list[_SpanCandidate]) -> list[_SpanCandidate]:
    """Choose the non-overlapping subset maximising total score.

    Weighted interval scheduling over character offsets: sort by end offset and
    keep, for each prefix, the best of "skip this candidate" and "take it plus
    the best solution ending at or before its start". Unlike the greedy rule
    this cannot be blocked by a single high-scoring span.
    """
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: (c.end, c.start))
    ends = [c.end for c in ordered]
    best_score = [0.0] * (len(ordered) + 1)
    kept: list[list[int]] = [[] for _ in range(len(ordered) + 1)]

    for i, cand in enumerate(ordered, start=1):
        # Rightmost prefix whose candidates all end at or before this start.
        j = bisect.bisect_right(ends, cand.start)
        take = best_score[j] + cand.score
        if take > best_score[i - 1]:
            best_score[i] = take
            kept[i] = [*kept[j], i - 1]
        else:
            best_score[i] = best_score[i - 1]
            kept[i] = kept[i - 1]

    return [ordered[idx] for idx in kept[len(ordered)]]


_SPAN_DECODERS: dict[str, Callable[[list[_SpanCandidate]], list[_SpanCandidate]]] = {
    "greedy": _select_greedy,
    "max_total_score": _select_max_total_score,
}


@register("task", name="span_ner")
class SpanNERTask(IETask):
    """Span-based named entity recognition task."""

    def __init__(
        self,
        label_list: list[str] | None = None,
        max_length: int = 512,
        max_span_width: int = 12,
        threshold: float = 0.5,
        decoder: str = "greedy",
        name: str = "span_ner",
    ) -> None:
        """Derive entity types from a BIO label list and set span-encoding options.

        `decoder` picks how overlapping candidates are resolved into a flat set:
        `"greedy"` takes them in descending score order, `"max_total_score"`
        maximises the summed score via weighted interval scheduling. Both emit
        non-overlapping spans; neither can produce nested entities.
        """
        self.name = name
        bio_labels = label_list if label_list is not None else []
        self.entity_types = _bio_to_entity_types(bio_labels)
        self.label_list: list[str] = list(self.entity_types)
        self.max_length = max_length
        self.max_span_width = max_span_width
        self.threshold = threshold
        if decoder not in _SPAN_DECODERS:
            msg = (
                f"Unknown span decoder {decoder!r}; "
                f"expected one of {sorted(_SPAN_DECODERS)}"
            )
            raise ValueError(msg)
        self.decoder = decoder
        self._type2id = {t: i for i, t in enumerate(self.entity_types)}

    def encode(self, example: NERExample, tokenizer: Any) -> dict[str, Any]:
        """Tokenize text and build candidate word-spans with one-hot type labels."""
        tokenized = tokenizer(
            example.text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
        )
        offset_mapping: list[tuple[int, int]] = tokenized["offset_mapping"]
        word_starts, word_char_spans = _build_words(
            _resolve_word_ids(tokenized, offset_mapping),
            offset_mapping,
            example.text,
        )
        num_words = len(word_starts)

        span_idx: list[tuple[int, int]] = []
        for i in range(num_words):
            for w in range(self.max_span_width):
                j = i + w
                if j < num_words:
                    span_idx.append((i, j))
                else:
                    span_idx.append((-1, -1))

        num_types = len(self.entity_types)
        span_labels: list[list[float]] = [
            [0.0] * num_types for _ in range(len(span_idx))
        ]

        for entity in example.entities:
            ent_word_start = None
            ent_word_end = None
            for wi, (wc_start, wc_end) in enumerate(word_char_spans):
                if wc_start <= entity.start < wc_end:
                    ent_word_start = wi
                if wc_start < entity.end <= wc_end:
                    ent_word_end = wi
            if ent_word_start is not None and ent_word_end is not None:
                width = ent_word_end - ent_word_start
                if width < self.max_span_width:
                    flat_idx = ent_word_start * self.max_span_width + width
                    if flat_idx < len(span_labels):
                        type_id = self._type2id.get(entity.label)
                        if type_id is not None:
                            span_labels[flat_idx][type_id] = 1.0

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "offset_mapping": offset_mapping,
            "word_starts": word_starts,
            "word_char_spans": word_char_spans,
            "span_idx": span_idx,
            "span_labels": span_labels,
            "entities": example.entities,
        }

    def _candidates(
        self,
        prob_b: torch.Tensor,
        num_words: int,
        word_char_spans: list[tuple[int, int]],
    ) -> list[_SpanCandidate]:
        """Collect above-threshold spans for one example as character-offset candidates."""
        candidates: list[_SpanCandidate] = []
        for wi in range(num_words):
            for k in range(prob_b.shape[1]):
                wj = wi + k
                if wj >= num_words:
                    break
                for c in range(prob_b.shape[2]):
                    score = prob_b[wi, k, c].item()
                    if score > self.threshold:
                        candidates.append(
                            _SpanCandidate(
                                start=word_char_spans[wi][0],
                                end=word_char_spans[wj][1],
                                label=self.entity_types[c],
                                score=score,
                            )
                        )
        return candidates

    def decode(self, outputs: Any, batch: dict[str, Any]) -> list[list[Entity]]:
        """Select a non-overlapping set of spans per example and return Entity objects."""
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = torch.sigmoid(logits)
        select = _SPAN_DECODERS[self.decoder]
        results: list[list[Entity]] = []

        for b in range(probs.shape[0]):
            text = batch["text"][b]
            candidates = self._candidates(
                probs[b], batch["num_words"][b], batch["word_char_spans"][b]
            )
            selected = [
                Entity(
                    start=cand.start,
                    end=cand.end,
                    text=text[cand.start : cand.end],
                    label=cand.label,
                )
                for cand in select(candidates)
            ]
            selected.sort(key=lambda e: e.start)
            results.append(selected)

        return results
