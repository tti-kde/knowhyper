"""Multi-stage link pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from primehkg.linking.build import (
    build_link_resolvers,
    insert_extra_resolvers,
    is_lexical_resolver,
)
from primehkg.linking.chemical_xref_resolver import ChemicalExternalIdResolver
from primehkg.linking.presets import LinkingPreset
from primehkg.linking.protocols import LinkCandidate, LinkMatchKind, LinkResult

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from primehkg.core.dataset import CoreDataset
    from primehkg.linking.protocols import LinkResolver

LINK_DETERMINISM_VERSION = "link:2026-05-19-v1"

# When node_type_hint is set, widen the per-resolver candidate request by this
# factor so a correct-type candidate ranked below top_k can still be promoted.
_HINT_POOL_FACTOR = 10

_KIND_RANK: dict[LinkMatchKind, int] = {
    LinkMatchKind.EXACT_NODE_ID: 4,
    LinkMatchKind.EXACT_XREF: 3,
    LinkMatchKind.EXACT_LABEL: 2,
    LinkMatchKind.DENSE_LABEL: 2,
    LinkMatchKind.FUZZY_LABEL: 1,
}


def node_type_of(node_id: str) -> str | None:
    """Extract the node type from a ``node_id``.

    ``primekg:<type>:<id>`` -> ``<type>`` (e.g. ``drug``, ``gene/protein``); a
    namespaced id like ``chebi:29101`` -> its namespace (``chebi``), which acts as
    the type for non-primekg ids. Returns ``None`` for a bare id with no separator.
    """
    parts = node_id.split(":")
    if parts[0] == "primekg" and len(parts) >= 3:
        return parts[1]
    return parts[0] if len(parts) >= 2 else None


def _apply_node_type_hint(
    candidates: list[LinkCandidate], node_type_hint: str | Collection[str] | None
) -> list[LinkCandidate]:
    """Soft re-rank: stable-partition type-matching candidates first.

    Candidates whose :func:`node_type_of` is in the hint are moved ahead of the
    rest, each group keeping its incoming (score) order. Nothing is dropped, so a
    wrong hint cannot zero out linking and a ``None`` hint is a no-op (output is
    byte-identical to the input order).
    """
    if node_type_hint is None:
        return candidates
    allowed = (
        {node_type_hint} if isinstance(node_type_hint, str) else set(node_type_hint)
    )
    if not allowed:
        return candidates
    matching = [c for c in candidates if node_type_of(str(c["node_id"])) in allowed]
    rest = [c for c in candidates if node_type_of(str(c["node_id"])) not in allowed]
    return matching + rest


def _link_candidate_rank(candidate: LinkCandidate) -> tuple[int, float, str]:
    kind = candidate["match_kind"]
    if not isinstance(kind, LinkMatchKind):
        kind = LinkMatchKind(str(kind))
    return (
        _KIND_RANK.get(kind, 0),
        float(candidate["score"]),
        str(candidate["node_id"]),
    )


def _prefer_link_candidate(new: LinkCandidate, prev: LinkCandidate) -> bool:
    return _link_candidate_rank(new) > _link_candidate_rank(prev)


class LinkPipeline:
    def __init__(
        self,
        dataset: CoreDataset,
        *,
        resolvers: Sequence[LinkResolver] | None = None,
        fuzzy_weights: tuple[float, float] = (0.3, 0.7),
        enable_fuzzy: bool = True,
        preset_min_score: float | None = None,
        sapbert_sidecar_missing: bool = False,
    ) -> None:
        self._dataset = dataset
        if resolvers is not None:
            self._resolvers = list(resolvers)
        else:
            built, info = build_link_resolvers(
                dataset,
                {
                    "enable_fuzzy": enable_fuzzy,
                    "fuzzy_weights": fuzzy_weights,
                },
            )
            self._resolvers = built
            sapbert_sidecar_missing = info.sapbert_sidecar_missing
        self._enable_fuzzy = enable_fuzzy
        self._preset_min_score = preset_min_score
        self._sapbert_sidecar_missing = sapbert_sidecar_missing

    @classmethod
    def from_preset(
        cls,
        dataset: CoreDataset,
        preset: LinkingPreset,
        *,
        extra_resolvers: Sequence[LinkResolver] = (),
        resolvers: Sequence[LinkResolver] | None = None,
    ) -> LinkPipeline:
        if resolvers is not None:
            return cls(dataset, resolvers=resolvers)
        built, info = build_link_resolvers(dataset, preset)
        if extra_resolvers:
            built = insert_extra_resolvers(built, list(extra_resolvers))
        return cls(
            dataset,
            resolvers=built,
            enable_fuzzy=bool(preset.get("enable_fuzzy", True)),
            preset_min_score=float(preset.get("min_score", 0.0)),
            sapbert_sidecar_missing=info.sapbert_sidecar_missing,
        )

    def link(
        self,
        query: str,
        *,
        namespace: str | None = None,
        node_type_hint: str | Collection[str] | None = None,
        allowed_node_types: Collection[str] | None = None,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> LinkResult:
        q = (query or "").strip()
        audit: dict[str, object] = {}
        effective_min_score = (
            float(min_score)
            if min_score is not None
            else float(self._preset_min_score or 0.0)
        )
        if not q:
            return {
                "query": query,
                "namespace_hint": namespace,
                "candidates": [],
                "determinism_version": LINK_DETERMINISM_VERSION,
                "audit": {"empty_query": True},
            }

        if not self._dataset.is_enriched:
            audit["minimal_build"] = True
            resolvers = [self._resolvers[0]] if self._resolvers else []
        else:
            resolvers = self._resolvers
            if namespace is not None:
                resolvers = [r for r in resolvers if not is_lexical_resolver(r)]

        if self._sapbert_sidecar_missing:
            audit["sapbert_sidecar_missing"] = True

        # When a type hint is set, ask resolvers for a wider pool so a correct-type
        # candidate ranked below top_k by raw score still reaches the bias step
        # below (resolvers truncate to the top_k they are given). No effect when
        # the hint is None.
        resolver_top_k = (
            top_k if node_type_hint is None else max(top_k, top_k * _HINT_POOL_FACTOR)
        )

        merged: dict[str, LinkCandidate] = {}
        for resolver in resolvers:
            for cand in resolver.resolve(
                q,
                namespace=namespace,
                node_type_hint=node_type_hint,
                allowed_node_types=allowed_node_types,
                top_k=resolver_top_k,
            ):
                nid = cand["node_id"]
                prev = merged.get(nid)
                if prev is None or _prefer_link_candidate(cand, prev):
                    merged[nid] = cand
            if isinstance(resolver, ChemicalExternalIdResolver):
                audit.update(resolver.last_resolve_audit)

        candidates = sorted(merged.values(), key=_link_candidate_rank, reverse=True)
        candidates = [c for c in candidates if float(c["score"]) >= effective_min_score]
        if self._enable_fuzzy and len(candidates) < top_k and self._dataset.is_enriched:
            pass
        # Soft node-type bias BEFORE truncation: a correct-type candidate ranked
        # below top_k by score is promoted into the kept window (the whole point —
        # the dense resolver otherwise returns the nearest embedding of any type).
        candidates = _apply_node_type_hint(candidates, node_type_hint)
        candidates = candidates[: max(0, int(top_k))]

        if not candidates and audit.get("minimal_build"):
            audit["no_match"] = True

        return {
            "query": q,
            "namespace_hint": namespace,
            "candidates": candidates,
            "determinism_version": LINK_DETERMINISM_VERSION,
            "audit": audit,
        }


def __getattr__(name: str) -> type[LinkPipeline]:
    if name == "DefaultLinkPipeline":
        import warnings

        warnings.warn(
            "DefaultLinkPipeline is deprecated; use LinkPipeline (removed in 0.2.0)",
            DeprecationWarning,
            stacklevel=2,
        )
        return LinkPipeline
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
