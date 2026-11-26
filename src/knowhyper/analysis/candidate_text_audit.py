"""#233 Phase T0 accounting: coverage and discriminability of candidate-owned text.

Protocol: `docs/reports/data/ner_candidate_correction_2026_08_02/phase_t0_protocol.md`.

This module holds only the **accounting**. Selection is deliberately NOT reimplemented here:
the driver calls `RetrievedDefinitionsSerializer.serialize()` itself with `cache_root=None`,
which suppresses persistence while `_targets` -> `max_targets` truncation -> `_select` run
exactly as they will at training time. Reimplementing any of that would let T0 count text the
serializer never selects, and the GPU experiment would then under-deliver for a reason
invisible in the audit.

Three accounting rules, each preventing a specific misreading:

* **Denominators stay separate.** A single pooled "coverage" number is how Phase 0's "2.16%
  evidence coverage" got read as a ceiling on rescuable gold when it was coverage over all
  candidates. `correctable_gold` is the denominator that decides whether a correction head has
  material to work with, and it is reported on its own.
* **0/0 is `None`, not `0.0`.** Zero-over-zero printed as 0.0 reads as "measured, and nothing
  was covered", which is a different claim from "there was nothing to measure".
* **Length is in tokens.** 295 characters says nothing about truncation; only the text
  encoder's own tokenizer does. The budget is recorded alongside the rate so a later reader
  cannot mistake which budget it refers to.

Raw text never enters a stats object or the artifact -- only counts and content digests. See
CLAUDE.md: "Never write raw evidence text to a publishable artifact."
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


class TextAuditError(RuntimeError):
    """The audit cannot produce a trustworthy figure."""


def normalise_surface(value: str) -> str:
    """Casefold and collapse whitespace.

    ONE normaliser, shared by the coverage figure and the lexical control. If they used
    different rules the control could "explain" a gain the coverage number never measured, and
    nothing would error. Idempotent by construction, so a re-normalised join cannot silently
    miss.
    """
    return " ".join(value.split()).casefold()


@dataclass
class CandidateTextStats:
    """Counts for one scope (backbone x split). No text, ever -- digests only."""

    candidates: int = 0
    linked_candidates: int = 0
    candidates_with_text: int = 0
    correctable_gold: int = 0
    correctable_gold_with_text: int = 0
    non_gold: int = 0
    non_gold_with_text: int = 0
    instances: int = 0
    instances_with_any_text: int = 0
    multi_node_candidates: int = 0
    # `_select` returns AT MOST ONE evidence row per node (`text_serialization.py:738`), so
    # this is 0 by construction under the current serializer -- kept because the protocol asks
    # for it and a future multi-select serializer would make it non-zero. Do not read 0 here
    # as "no node had several candidate texts": several are AVAILABLE, one is SELECTED.
    multi_text_nodes: int = 0
    lexical_exact: int = 0
    lexical_normalised: int = 0
    token_lengths: list[int] = field(default_factory=list)
    by_source_field: Counter[str] = field(default_factory=Counter)
    by_payload_class: Counter[str] = field(default_factory=Counter)
    by_modality: Counter[str] = field(default_factory=Counter)
    anomalies: Counter[str] = field(default_factory=Counter)
    content_hashes: set[str] = field(default_factory=set)

    def coverage(self, denominator: str) -> float | None:
        """Fraction of `denominator` carrying text, or `None` when the denominator is 0."""
        numerator_of = {
            "candidates": self.candidates_with_text,
            "correctable_gold": self.correctable_gold_with_text,
            "non_gold": self.non_gold_with_text,
            "instances": self.instances_with_any_text,
        }
        if denominator not in numerator_of:
            msg = f"unknown denominator {denominator!r}"
            raise TextAuditError(msg)
        total = int(getattr(self, denominator))
        if total == 0:
            return None
        return numerator_of[denominator] / total

    def as_dict(self) -> dict[str, Any]:
        """Return serialisable counts, deliberately excluding anything text-shaped."""
        return {
            "candidates": self.candidates,
            "linked_candidates": self.linked_candidates,
            "candidates_with_text": self.candidates_with_text,
            "correctable_gold": self.correctable_gold,
            "correctable_gold_with_text": self.correctable_gold_with_text,
            "non_gold": self.non_gold,
            "non_gold_with_text": self.non_gold_with_text,
            "instances": self.instances,
            "instances_with_any_text": self.instances_with_any_text,
            "multi_node_candidates": self.multi_node_candidates,
            "multi_text_nodes": self.multi_text_nodes,
            "lexical_exact": self.lexical_exact,
            "lexical_normalised": self.lexical_normalised,
            "coverage": {
                name: self.coverage(name)
                for name in ("candidates", "correctable_gold", "non_gold", "instances")
            },
            "by_source_field": dict(self.by_source_field.most_common()),
            # UNAVAILABLE under `audience: legacy`, not missing data: `license_registry:
            # null` makes `license_info` None, so the serializer emits payload_class="" for
            # every item (`text_serialization.py:694`). Reported as-is rather than dropped, so
            # the empty breakdown is visibly explained instead of looking like a gap.
            "by_payload_class": dict(self.by_payload_class.most_common()),
            "by_modality": dict(self.by_modality.most_common()),
            "anomalies": dict(self.anomalies.most_common()),
            # Count, not the digests themselves: the count answers "how many distinct texts"
            # without inviting anyone to treat the artifact as a text store.
            "content_hash_count": len(self.content_hashes),
        }


def token_length_stats(lengths: list[int], *, budget: int) -> dict[str, Any]:
    """Distribution of tokenised lengths plus the truncation rate against `budget`."""
    if budget <= 0:
        msg = f"token budget must be positive, got {budget}"
        raise TextAuditError(msg)
    if not lengths:
        # NOT 0.0: "nothing was truncated" and "there was nothing to truncate" differ.
        return {
            "n": 0,
            "median": None,
            "p90": None,
            "max": None,
            "over_budget": 0,
            "truncation_rate": None,
        }
    ordered = sorted(lengths)
    over = sum(1 for value in ordered if value > budget)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
        "over_budget": over,
        "truncation_rate": over / len(ordered),
    }


def summarise(
    scopes: dict[str, CandidateTextStats],
    *,
    budget: int,
) -> dict[str, Any]:
    """One payload per scope, kept separate.

    `train` and `validation` must not be merged: they use different anchors (OOF vs
    full-train) and different proposals, so a combined figure describes neither.
    """
    return {
        "token_budget": budget,
        "scopes": {
            name: {
                **stats.as_dict(),
                "token_lengths": token_length_stats(stats.token_lengths, budget=budget),
            }
            for name, stats in sorted(scopes.items())
        },
    }
