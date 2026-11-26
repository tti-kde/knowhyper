"""Conversions between upstream (primehkg/ieforge) shapes and KnowHyper types.

This module is the single version-absorption boundary. If an upstream API at the
pinned rev changes shape, the adapter tests fail here first.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ieforge.knowledge.core.base import IEKnowledgeQueryBuilder
from ieforge.knowledge.core.schema import Query
from ieforge.registry import register

from knowhyper.ieforge.biored_query_builder import id_namespace, node_id_for
from knowhyper.ieforge.schwartz_hearst import extract_abbreviations
from knowhyper.pipeline.types import HyperedgeRef, LinkedEntity, Mention, SubHypergraph


def mentions_from_query_text(
    text: str,
    entities: list[str],
    *,
    types: list[str | None] | None = None,
    type_map: Mapping[str, str] | None = None,
    expand_abbreviations: bool = False,
    derive_id_namespaces: bool = False,
    pair_roles: list[str | None] | None = None,
) -> list[Mention]:
    """Prefer explicit entity spans; fall back to the whole query text.

    ``types`` (optional) is the per-entity type label aligned positionally with
    ``entities`` (the dataset label, e.g. ``CHEMICAL``/``GENE-Y``). ``type_map``
    maps that label to the value stored on ``Mention.entity_type`` — set it to the
    primekg node type (``drug``/``gene/protein``) so it flows to the linker's
    ``node_type_hint``. An unmapped label is carried through unchanged; a missing
    or short ``types`` list leaves the corresponding ``entity_type`` ``None``. The
    whole-text fallback is always untyped.

    ``expand_abbreviations`` (default False): when True, build the document's
    Schwartz-Hearst abbreviation map once and stamp the long form onto
    ``Mention.expansion`` for any entity whose surface form is a defined short form
    (a linking aid for the expansion fallback). False keeps ``expansion`` None.

    ``derive_id_namespaces`` (default False): when True, derive each mention's
    xref namespace from its id shape via ``id_namespace`` and normalize the
    mention text to the prefixed id form. A numeric NCBIGene id under the
    ``protein`` type instead becomes the primekg protein node_id (via
    ``node_id_for``) with namespace left None, so the exact-node-id resolver links
    it directly; default False leaves namespace None and text untouched.
    """
    if not entities:
        return [Mention(text=text)]
    types = types or []
    pair_roles = pair_roles or []
    abbrevs = extract_abbreviations(text) if expand_abbreviations else {}

    def _etype(i: int) -> str | None:
        raw = types[i] if i < len(types) else None
        if raw is None:
            return None
        return type_map.get(raw, raw) if type_map is not None else raw

    def _mention(i: int, e: str) -> Mention:
        etype = _etype(i)
        text_e, namespace = e, None
        if derive_id_namespaces:
            ns = id_namespace(e)
            if ns is not None:
                namespace, text_e = ns[0], ns[1]
            else:
                node_id = node_id_for(e, etype)
                if node_id is not None:
                    text_e = node_id
        return Mention(
            text=text_e,
            entity_type=etype,
            expansion=abbrevs.get(e),
            namespace=namespace,
            pair_role=pair_roles[i] if i < len(pair_roles) else None,
        )

    return [_mention(i, e) for i, e in enumerate(entities)]


def link_result_to_linked_entities(
    mention: Mention, link_result: dict[str, Any]
) -> LinkedEntity | None:
    """Take the top candidate of a primehkg LinkResult for one mention."""
    candidates = link_result.get("candidates") or []
    if not candidates:
        return None
    top = candidates[0]
    return LinkedEntity(
        mention=mention,
        node_id=top["node_id"],
        score=float(top["score"]),
        match_kind=str(top["match_kind"]),
        candidates=tuple(c["node_id"] for c in candidates[1:]),
    )


def _normalize_roles(raw: Any) -> dict[str, str]:
    """Coerce an upstream ``roles`` value into a ``{node_id: role}`` dict.

    primehkg's schema types ``roles`` as ``dict[str, str]``, but after a round-trip
    through the HF Arrow/parquet hyperedge table it comes back as a JSON-encoded
    **string** (e.g. ``'{"primekg:exposure:D001241":"exposure"}'``). This absorbs
    both shapes so the adapter never crashes on a real build:

    - a plain mapping (the in-memory / fixture shape);
    - a JSON-object string (the real HF-Arrow round-trip shape).

    Anything else yields an empty dict rather than raising — roles are advisory
    metadata on the hyperedge, not load-bearing for retrieval.
    """
    if isinstance(raw, Mapping):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return {str(k): str(v) for k, v in parsed.items()}
    return {}


def retrieval_output_to_subhypergraph(output: dict[str, Any]) -> SubHypergraph:
    """Convert a primehkg RetrievalPipelineOutput into a KnowHyper SubHypergraph."""
    seeds = tuple(output.get("seed_node_ids", []))
    hyperedges: list[HyperedgeRef] = []
    node_id_set: set[str] = set(seeds)
    for result in output.get("results", []):
        he = result["hyperedge"]
        he_node_ids = tuple(he.get("node_ids", []))
        node_id_set.update(he_node_ids)
        # Upstream ScoreBreakdown is nested: it contains a scalar `final_score`,
        # scalar weights, AND a nested `ewo` EwoBreakdown dict. HyperedgeRef.
        # score_breakdown is typed Mapping[str, float], so keep only the scalar
        # (float) top-level entries and drop nested ones (e.g. `ewo`) to keep the
        # annotation truthful and robust to the upstream shape.
        raw_breakdown = result.get("score_breakdown", {})
        scalar_breakdown = {
            key: float(value)
            for key, value in raw_breakdown.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        hyperedges.append(
            HyperedgeRef(
                he_id=result["he_id"],
                semantics=str(he.get("semantics", "")),
                node_ids=he_node_ids,
                roles=_normalize_roles(he.get("roles")),
                polarity=str(he.get("polarity", "unknown")),
                score=float(result.get("score", 0.0)),
                score_breakdown=scalar_breakdown,
            )
        )
    return SubHypergraph(
        seed_node_ids=seeds,
        node_ids=tuple(sorted(node_id_set)),
        hyperedges=tuple(hyperedges),
        provenance={"candidate_mode": output.get("candidate_mode")},
    )


def mentions_from_ner_query(
    text: str,
    entities: list[str],
    entity_types: list[str] | None = None,
    entity_spans: list[tuple[int, int]] | None = None,
    *,
    type_map: Mapping[str, str] | None = None,
    expand_abbreviations: bool = False,
) -> list[Mention]:
    """Build Mention list from NER predicted entities.

    Unlike ``mentions_from_query_text`` which reads positional head/tail pairs,
    this reads a flat list of entity surface forms + their type labels. An empty
    first-pass prediction produces no mentions (and therefore no knowledge), rather
    than retrieving from the whole document or falling back to gold annotations.
    """
    if not entities:
        return []
    entity_types = entity_types or []
    entity_spans = entity_spans or []
    if entity_spans and len(entity_spans) != len(entities):
        msg = (
            "entity_spans must align positionally with entities: "
            f"{len(entity_spans)} spans for {len(entities)} entities"
        )
        raise ValueError(msg)
    abbrevs = extract_abbreviations(text) if expand_abbreviations else {}

    def _etype(i: int) -> str | None:
        raw = entity_types[i] if i < len(entity_types) else None
        if raw is None:
            return None
        return type_map.get(raw, raw) if type_map is not None else raw

    return [
        Mention(
            text=e,
            char_span=entity_spans[i] if entity_spans else None,
            entity_type=_etype(i),
            expansion=abbrevs.get(e),
        )
        for i, e in enumerate(entities)
    ]


def _ner_get(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


@register("knowledge_query_builder", name="ner_attribute")
class NERAttributeQueryBuilder(IEKnowledgeQueryBuilder):
    """Query builder for NER: extracts entity texts and types from Entity objects."""

    def __init__(
        self,
        *,
        field: str = "text",
        task: str | None = None,
        metadata_fields: list[str] | None = None,
    ) -> None:
        self._field = field
        self._task = task
        self._metadata_fields = list(metadata_fields or [])

    def build(self, example: Any) -> Query:
        text = str(_ner_get(example, self._field))
        entities_raw = _ner_get(example, "entities")
        entity_texts = [str(getattr(e, "text", e)) for e in entities_raw]
        entity_types = [getattr(e, "label", None) for e in entities_raw]

        metadata: dict[str, Any] = {}
        for f in self._metadata_fields:
            metadata[f] = _ner_get(example, f)
        metadata["entity_types"] = entity_types

        return Query(
            text=text,
            entities=entity_texts,
            task=self._task,
            metadata=metadata,
        )
