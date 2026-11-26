"""Built-in ``LinkResolver`` implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from primehkg.linking.fuzzy import combined_fuzzy_score
from primehkg.linking.protocols import LinkCandidate, LinkMatchKind
from primehkg.linking.xref_index import XrefIndex, normalize_label

if TYPE_CHECKING:
    from collections.abc import Collection


def _candidate(
    *,
    node_id: str,
    score: float,
    match_kind: LinkMatchKind,
    matched_surface: str,
    matched_field: str,
    score_breakdown: dict[str, float],
    audit_flags: list[str] | None = None,
) -> LinkCandidate:
    return {
        "node_id": node_id,
        "score": score,
        "match_kind": match_kind,
        "matched_surface": matched_surface,
        "matched_field": matched_field,
        "score_breakdown": score_breakdown,
        "audit_flags": list(audit_flags or []),
    }


class ExactNodeIdResolver:
    def __init__(self, index: XrefIndex) -> None:
        self._index = index

    def resolve(
        self,
        query: str,
        *,
        namespace: str | None = None,
        node_type_hint: str | Collection[str] | None = None,
        allowed_node_types: Collection[str] | None = None,
        top_k: int = 5,
    ) -> list[LinkCandidate]:
        del namespace, node_type_hint, allowed_node_types, top_k
        hits = self._index.lookup_node_id(query)
        return [
            _candidate(
                node_id=nid,
                score=1.0,
                match_kind=LinkMatchKind.EXACT_NODE_ID,
                matched_surface=query,
                matched_field="node_id",
                score_breakdown={"base": 1.0},
            )
            for nid in hits
        ]


class ExactXrefResolver:
    def __init__(self, index: XrefIndex) -> None:
        self._index = index

    def resolve(
        self,
        query: str,
        *,
        namespace: str | None = None,
        node_type_hint: str | Collection[str] | None = None,
        allowed_node_types: Collection[str] | None = None,
        top_k: int = 5,
    ) -> list[LinkCandidate]:
        del node_type_hint, allowed_node_types, top_k
        if namespace is None:
            return []
        hits = self._index.lookup_xref(namespace, query)
        out: list[LinkCandidate] = []
        flags = ["ambiguous_xref"] if len(hits) > 1 else []
        for nid in hits:
            out.append(
                _candidate(
                    node_id=nid,
                    score=0.95,
                    match_kind=LinkMatchKind.EXACT_XREF,
                    matched_surface=query,
                    matched_field=f"xrefs.{namespace}",
                    score_breakdown={"base": 0.95},
                    audit_flags=flags,
                )
            )
        return out


class ExactLabelResolver:
    def __init__(self, index: XrefIndex) -> None:
        self._index = index

    def resolve(
        self,
        query: str,
        *,
        namespace: str | None = None,
        node_type_hint: str | Collection[str] | None = None,
        allowed_node_types: Collection[str] | None = None,
        top_k: int = 5,
    ) -> list[LinkCandidate]:
        del namespace, node_type_hint, allowed_node_types, top_k
        hits = self._index.lookup_label(query)
        out: list[LinkCandidate] = []
        flags = ["ambiguous_label"] if len(hits) > 1 else []
        for nid in hits:
            out.append(
                _candidate(
                    node_id=nid,
                    score=0.90,
                    match_kind=LinkMatchKind.EXACT_LABEL,
                    matched_surface=query,
                    matched_field="name|synonym",
                    score_breakdown={"base": 0.90},
                    audit_flags=flags,
                )
            )
        return out


class FuzzyLabelResolver:
    def __init__(
        self,
        index: XrefIndex,
        *,
        w_jaccard: float = 0.3,
        w_levenshtein: float = 0.7,
        min_prefix_len: int = 2,
    ) -> None:
        self._index = index
        self._w_j = w_jaccard
        self._w_l = w_levenshtein
        self._min_prefix = min_prefix_len

    def resolve(
        self,
        query: str,
        *,
        namespace: str | None = None,
        node_type_hint: str | Collection[str] | None = None,
        allowed_node_types: Collection[str] | None = None,
        top_k: int = 5,
    ) -> list[LinkCandidate]:
        del namespace, node_type_hint, allowed_node_types
        qnorm = normalize_label(query)
        if not qnorm:
            return []
        prefix = qnorm[: self._min_prefix] if len(qnorm) >= self._min_prefix else qnorm
        scored: list[LinkCandidate] = []
        for label, node_ids in self._index.all_labels():
            if prefix and not label.startswith(prefix) and prefix not in label:
                continue
            score, breakdown = combined_fuzzy_score(
                qnorm, label, w_jaccard=self._w_j, w_levenshtein=self._w_l
            )
            for nid in node_ids:
                scored.append(
                    _candidate(
                        node_id=nid,
                        score=score,
                        match_kind=LinkMatchKind.FUZZY_LABEL,
                        matched_surface=query,
                        matched_field=f"label:{label}",
                        score_breakdown=breakdown,
                    )
                )
        scored.sort(key=lambda c: (-float(c["score"]), str(c["node_id"])))
        return scored[: max(0, int(top_k))]
