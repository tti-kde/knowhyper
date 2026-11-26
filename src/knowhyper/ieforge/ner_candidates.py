"""High-recall span candidates derived exclusively from first-pass NER logits.

The generator deliberately has no access to gold entities. It turns calibrated
BIO probabilities and tokenizer offsets into a bounded span lattice which can be
stored with out-of-fold first-pass predictions and consumed by the second pass.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ieforge.data.schema import Entity


@dataclass(frozen=True)
class NERSpanCandidate:
    """One proposed mention span and its first-pass uncertainty features."""

    start: int
    end: int
    text: str
    label: str
    confidence: float
    entropy: float
    source: str = "lattice"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def generate_span_candidates(
    logits: torch.Tensor,
    offsets: Sequence[Sequence[int]],
    text: str,
    label_list: Sequence[str],
    *,
    decoded_entities: Sequence[Entity] = (),
    start_threshold: float = 0.05,
    top_k: int = 32,
    max_span_tokens: int = 12,
) -> list[NERSpanCandidate]:
    """Build a bounded high-recall candidate lattice from one BIO logit matrix.

    A candidate starts wherever P(B-type) clears start_threshold and is extended
    for up to max_span_tokens using matching I-type probabilities. Its confidence
    is the geometric mean of the token decisions plus the probability of ending
    before the next token. Normalised predictive entropy is retained for the
    downstream conditional gate. Hard-decoded spans are always included.
    """
    if logits.ndim != 2 or logits.shape[0] != len(offsets):
        msg = (
            "logits must have shape [tokens, labels] aligned with offsets, got "
            f"{tuple(logits.shape)} and {len(offsets)} offsets"
        )
        raise ValueError(msg)
    if logits.shape[1] != len(label_list):
        msg = "logit label dimension does not match label_list"
        raise ValueError(msg)
    if not 0.0 <= start_threshold <= 1.0:
        msg = "start_threshold must be in [0, 1]"
        raise ValueError(msg)
    if top_k <= 0 or max_span_tokens <= 0:
        msg = "top_k and max_span_tokens must be positive"
        raise ValueError(msg)

    probabilities = torch.softmax(logits.detach().float(), dim=-1).cpu()
    safe_probabilities = probabilities.clamp_min(1e-12)
    entropy_denominator = math.log(max(len(label_list), 2))
    entropies = (
        -(safe_probabilities * safe_probabilities.log()).sum(dim=-1)
        / entropy_denominator
    )
    valid = [
        index
        for index, raw in enumerate(offsets)
        if int(raw[0]) < int(raw[1]) and int(raw[1]) <= len(text)
    ]
    label_to_id = {label: index for index, label in enumerate(label_list)}
    begin_labels = [
        (index, label.removeprefix("B-"))
        for index, label in enumerate(label_list)
        if label.startswith("B-")
    ]

    generated: dict[tuple[int, int, str], NERSpanCandidate] = {}
    for begin_id, entity_type in begin_labels:
        inside_id = label_to_id.get(f"I-{entity_type}")
        for start_pos, token_index in enumerate(valid):
            begin_probability = float(probabilities[token_index, begin_id])
            if begin_probability < start_threshold:
                continue
            token_log_probabilities = [math.log(max(begin_probability, 1e-12))]
            token_entropies: list[float] = []
            for end_pos in range(
                start_pos, min(len(valid), start_pos + max_span_tokens)
            ):
                end_token = valid[end_pos]
                if end_pos > start_pos:
                    if inside_id is None:
                        break
                    token_log_probabilities.append(
                        math.log(max(float(probabilities[end_token, inside_id]), 1e-12))
                    )
                token_entropies.append(float(entropies[end_token]))
                next_pos = end_pos + 1
                continue_probability = (
                    float(probabilities[valid[next_pos], inside_id])
                    if inside_id is not None and next_pos < len(valid)
                    else 0.0
                )
                end_probability = max(1.0 - continue_probability, 1e-12)
                log_terms = [*token_log_probabilities, math.log(end_probability)]
                confidence = math.exp(sum(log_terms) / len(log_terms))
                raw_start, raw_end = offsets[token_index][0], offsets[end_token][1]
                start = int(raw_start)
                end = int(raw_end)
                start += len(text[start:end]) - len(text[start:end].lstrip())
                if start >= end:
                    continue
                candidate = NERSpanCandidate(
                    start=start,
                    end=end,
                    text=text[start:end],
                    label=entity_type,
                    confidence=confidence,
                    entropy=sum(token_entropies) / len(token_entropies),
                )
                key = (start, end, entity_type)
                current = generated.get(key)
                if current is None or candidate.confidence > current.confidence:
                    generated[key] = candidate

    ranked = sorted(
        generated.values(),
        key=lambda item: (-item.confidence, item.start, item.end, item.label),
    )
    # Mention provenance is keyed by character span. Keep only the strongest type
    # hypothesis per span so multi-type tasks remain unambiguous downstream.
    selected: dict[tuple[int, int], NERSpanCandidate] = {}
    for item in ranked:
        span = (item.start, item.end)
        if span not in selected:
            selected[span] = item
        if len(selected) == top_k:
            break

    # Hard predictions must never be removed by the lattice budget.
    for entity in decoded_entities:
        span = (int(entity.start), int(entity.end))
        key = (*span, str(entity.label))
        existing = generated.get(key)
        if existing is not None:
            selected[span] = NERSpanCandidate(
                start=existing.start,
                end=existing.end,
                text=existing.text,
                label=existing.label,
                confidence=existing.confidence,
                entropy=existing.entropy,
                source="decoded",
            )
            continue
        covered = [
            token_index
            for token_index in valid
            if int(offsets[token_index][0]) < int(entity.end)
            and int(offsets[token_index][1]) > int(entity.start)
        ]
        confidence = (
            sum(float(probabilities[index].max()) for index in covered) / len(covered)
            if covered
            else 0.0
        )
        entropy = (
            sum(float(entropies[index]) for index in covered) / len(covered)
            if covered
            else 1.0
        )
        selected[span] = NERSpanCandidate(
            start=int(entity.start),
            end=int(entity.end),
            text=str(entity.text),
            label=str(entity.label),
            confidence=confidence,
            entropy=entropy,
            source="decoded",
        )

    return sorted(
        selected.values(),
        key=lambda item: (item.start, item.end, item.label),
    )
