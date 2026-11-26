"""Exportable candidate lattice utilities for ieforge span NER."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from knowhyper.ieforge.ner_candidates import NERSpanCandidate

if TYPE_CHECKING:
    from ieforge.data.schema import Entity


def span_candidates_from_logits(
    logits: torch.Tensor,
    *,
    text: str,
    word_char_spans: list[tuple[int, int]],
    entity_types: list[str],
    decoded_entities: list[Entity],
    top_k: int = 16,
) -> list[NERSpanCandidate]:
    """Return top-scoring valid spans plus every hard-decoded entity.

    ``logits`` has shape ``[max_words, max_span_width, num_types]``. Candidate
    scores and uncertainty come from the same span head used for decoding, so
    the JSONL consumed by the knowledge arms does not reconstruct a BIO lattice.
    """
    if top_k <= 0:
        msg = "top_k must be positive"
        raise ValueError(msg)
    if logits.dim() != 3 or logits.shape[2] != len(entity_types):
        msg = (
            "span logits must have shape [words, width, entity_types], got "
            f"{tuple(logits.shape)} for {len(entity_types)} types"
        )
        raise ValueError(msg)

    probs = torch.sigmoid(logits.detach()).cpu()
    lattice: list[NERSpanCandidate] = []
    for start_word in range(min(len(word_char_spans), probs.shape[0])):
        for width in range(probs.shape[1]):
            end_word = start_word + width
            if end_word >= len(word_char_spans):
                break
            start = int(word_char_spans[start_word][0])
            end = int(word_char_spans[end_word][1])
            for type_id, label in enumerate(entity_types):
                confidence = float(probs[start_word, width, type_id])
                clipped = min(max(confidence, 1e-7), 1.0 - 1e-7)
                entropy = -(
                    clipped * math.log2(clipped)
                    + (1.0 - clipped) * math.log2(1.0 - clipped)
                )
                lattice.append(
                    NERSpanCandidate(
                        start=start,
                        end=end,
                        text=text[start:end],
                        label=label,
                        confidence=confidence,
                        entropy=entropy,
                        source="lattice",
                    )
                )

    selected = sorted(
        lattice,
        key=lambda candidate: (
            -candidate.confidence,
            candidate.start,
            candidate.end,
            candidate.label,
        ),
    )[:top_k]
    by_key = {
        (candidate.start, candidate.end, candidate.label): candidate
        for candidate in selected
    }
    score_by_key = {
        (candidate.start, candidate.end, candidate.label): candidate
        for candidate in lattice
    }
    for entity in decoded_entities:
        key = (entity.start, entity.end, entity.label)
        scored = score_by_key.get(key)
        by_key[key] = NERSpanCandidate(
            start=entity.start,
            end=entity.end,
            text=entity.text,
            label=entity.label,
            confidence=scored.confidence if scored is not None else 1.0,
            entropy=scored.entropy if scored is not None else 0.0,
            source="decoded",
        )
    return sorted(
        by_key.values(),
        key=lambda candidate: (
            candidate.start,
            candidate.end,
            candidate.label,
        ),
    )
