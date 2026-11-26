"""KnowHyper's ieforge knowledge injector: runs the 4 stages into a KnowledgeContext."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from ieforge.knowledge.core.base import IEKnowledgeInjector
from ieforge.knowledge.core.schema import KnowledgeContext
from ieforge.registry import register

from knowhyper.cache.keys import InstanceKey
from knowhyper.ieforge.adapters import mentions_from_ner_query, mentions_from_query_text
from knowhyper.ieforge.ner_predictions import prediction_scoped_instance_key
from knowhyper.ieforge.primehkg_backend import LINKED_MENTION_PROVENANCE_SCHEMA

if TYPE_CHECKING:
    from ieforge.knowledge.core.schema import Query, RetrievalResult

    from knowhyper.pipeline.entity_view_augmenter import EntityViewAugmenter
    from knowhyper.pipeline.protocols import (
        EntityLinker,
        Integrator,
        SubHGEncoder,
        SubHGRetriever,
    )


def _text_serialized_context(
    *,
    serializer: Any,
    query: Query,
    sub: Any,
    facts: Any,
) -> KnowledgeContext:
    """Build the text-arm KnowledgeContext, shared by the RE and NER injectors.

    Extracted rather than duplicated: the NER text arm needs byte-identical
    behaviour here, and a second copy would let the two drift -- particularly the
    `items_key` check, whose whole purpose is to stop one arm publishing under
    another arm's key and training with silently empty knowledge.
    """
    from dataclasses import asdict

    serialized = serializer.serialize(query, sub)
    # R2a (definitions) and R2b (typed sub-HG) publish under distinct keys so the
    # collator cannot silently consume the wrong arm's payload. Required rather than
    # defaulted: a serializer that forgot to declare a key would publish under
    # another arm's, and the collator would find nothing and train with no knowledge
    # while still reporting a valid fingerprint (see pipeline.protocols.TextSerializer).
    items_key = getattr(serializer, "items_key", None)
    if not items_key:
        msg = (
            f"{type(serializer).__name__} must declare items_key; "
            "an undeclared key silently publishes under another arm's key"
        )
        raise TypeError(msg)
    metadata: dict[str, Any] = {
        "knowhyper_subhypergraph_size": len(sub.hyperedges),
        "text_serialization_fingerprint": serialized.fingerprint,
        items_key: [asdict(item) for item in serialized.items],
    }
    # Forward the SELECTION provenance a `subhypergraph_transform` recorded, so a text
    # arm's run record can show which hyperedges it actually saw. Without this the
    # collator's `knowledge_selection_fingerprint` is always "" -- it reads
    # `KnowledgeContext.metadata`, while the selector writes to
    # `SubHypergraph.provenance`, and the two were never connected. Copied key-by-key
    # rather than merging all of `sub.provenance`: that dict also holds per-mention
    # linked records, which are payload and must not travel into batch diagnostics.
    for key in (
        "subhg_selection_policy",
        "subhg_selection_fingerprint",
        "subhg_selected_edge_ids_hash",
        "subhg_hyperedges_before_selection",
    ):
        if key in sub.provenance:
            metadata[key] = sub.provenance[key]
    return KnowledgeContext(
        query=query, facts=facts, text_context=serialized.text, metadata=metadata
    )


@register("knowledge_injector", name="knowhyper")
class KnowHyperInjector(IEKnowledgeInjector):
    """Run the KnowHyper stage pipeline and pack the result into a KnowledgeContext."""

    def __init__(
        self,
        linker: EntityLinker,
        retriever: SubHGRetriever,
        encoder: SubHGEncoder,
        integrator: Integrator,
        top_k: int = 64,
        instance_metadata_fields: tuple[str, str, str] | None = None,
        mention_type_map: dict[str, str] | None = None,
        expand_abbreviations: bool = False,
        derive_id_namespaces: bool = False,
        entity_view_augmenter: EntityViewAugmenter | None = None,
        text_serializer: Any | None = None,
        subhypergraph_transform: Any | None = None,
    ) -> None:
        """Store the four pipeline stages, retrieval budget, and instance fields.

        ``mention_type_map`` (optional) maps a dataset entity label (read from
        ``query.metadata['head_type']``/``['tail_type']``) to the primekg node type
        passed as the linker's ``node_type_hint`` — e.g.
        ``{"CHEMICAL": "drug", "GENE-Y": "gene/protein"}``. ``None`` leaves mentions
        untyped (the linker biases nothing), preserving prior behavior.

        ``expand_abbreviations`` (default False) stamps each mention's in-document
        full form (Schwartz-Hearst) onto ``Mention.expansion``; it pairs with the
        linker's ``expansion_fallback`` and both must be true to enable the feature.

        ``derive_id_namespaces`` (default False) derives each mention's xref
        namespace from its id shape and normalizes its text to the prefixed id
        form (BioRED); False leaves namespace None and text untouched.

        ``entity_view_augmenter`` (optional) attaches ``entity_attribute``
        hyperedges and private attribute nodes to the retrieved sub-hypergraph
        before encoding (Lever B); ``None`` leaves the sub-HG untouched.

        ``subhypergraph_transform`` (optional) rewrites the retrieved sub-HG after
        cache lookup and **before** entity-view augmentation --
        :class:`~knowhyper.pipeline.subhg_selection.TopRankedHyperedgeSelector` uses
        it to restrict the graph arm to the same top-3 hyperedges the R2b text arm
        serializes (#189). It runs before augmentation so attribute hyperedges
        cannot compete for those 3 slots, and it must return a new object: the raw
        cache entry is shared with the full-budget arms. ``None`` is the default and
        leaves every existing arm byte-identical.
        """
        self.linker = linker
        self.retriever = retriever
        self.encoder = encoder
        self.integrator = integrator
        # The endpoint shuffled control needs the SubHG *cache* fingerprint to
        # find its substitution pool, and only the caching retriever knows it --
        # it is derived from the resolved profile, not configurable. The injector
        # is the one object holding both, so the hand-off happens here rather
        # than by duplicating the derivation inside the encoder.
        bind_pool = getattr(encoder, "bind_pool", None)
        if callable(bind_pool):
            bind_pool(retriever)
        self.top_k = top_k
        self.instance_metadata_fields = instance_metadata_fields
        self.expand_abbreviations = expand_abbreviations
        self.derive_id_namespaces = derive_id_namespaces
        self.entity_view_augmenter = entity_view_augmenter
        self.text_serializer = text_serializer
        self.subhypergraph_transform = subhypergraph_transform
        # Drop None-valued entries: a config that overrides the map (e.g. a DDI
        # experiment setting {Drug: drug}) leaves the inherited chemprot keys as
        # None via Hydra dict-merge; a None mapping is a no-op, so strip it to keep
        # the effective map clean (and out of the linker_id fingerprint).
        self.mention_type_map = (
            {k: v for k, v in mention_type_map.items() if v is not None}
            if mention_type_map is not None
            else None
        )

    def inject(self, query: Query, retrieval: RetrievalResult) -> KnowledgeContext:
        """Link mentions, retrieve a sub-hypergraph, encode, and integrate.

        On a cache hit the linker is skipped entirely: a cached sub-hypergraph is
        addressed by the instance key alone (the retriever ignores ``linked`` when
        it hits), and entity linking is the dominant per-instance cost during
        training (~118 ms/instance). If the retriever exposes ``get_cached`` and
        returns a sub-HG for this instance, link/retrieve are bypassed.
        """
        instance = None
        if self.instance_metadata_fields is not None:
            instance = InstanceKey.from_metadata(
                query.metadata,
                fields=self.instance_metadata_fields,
                fallback_text=query.text,
            )

        sub = None
        get_cached = getattr(self.retriever, "get_cached", None)
        if instance is not None and callable(get_cached):
            sub = get_cached(instance)

        if sub is None:
            types: list[str | None] | None = None
            if self.mention_type_map is not None:
                # head/tail types align positionally with entities (head, tail).
                # A builder that emits MORE than two entities (e.g. BioRED with
                # surface_fallback, which appends each endpoint's surface forms)
                # must publish the full aligned list itself, or the extra forms
                # would silently lose their node-type hint.
                published = query.metadata.get("entity_types")
                types = (
                    list(published)
                    if published
                    else [
                        query.metadata.get("head_type"),
                        query.metadata.get("tail_type"),
                    ]
                )
            mentions = mentions_from_query_text(
                query.text,
                list(query.entities),
                types=types,
                type_map=self.mention_type_map,
                expand_abbreviations=self.expand_abbreviations,
                derive_id_namespaces=self.derive_id_namespaces,
                pair_roles=list(query.metadata.get("target_entity_roles", [])),
            )
            linked = self.linker.link(mentions)
            sub = self.retriever.retrieve(linked, top_k=self.top_k, instance=instance)
        # Before augmentation: the top-3 selector must choose among RETRIEVED
        # hyperedges only, or entity-attribute edges would take the slots and the
        # matched graph arm would stop being structure-only (#189).
        if self.subhypergraph_transform is not None:
            sub = self.subhypergraph_transform(sub)
        if self.entity_view_augmenter is not None:
            sub = self.entity_view_augmenter.augment(sub)
        if self.text_serializer is not None:
            return _text_serialized_context(
                serializer=self.text_serializer,
                query=query,
                sub=sub,
                facts=retrieval.facts,
            )
        if getattr(self.integrator, "needs_incidence", False):
            from knowhyper.pipeline.protocols import IncidenceEncoder
            from knowhyper.pipeline.types import KnowledgeRepr

            if not isinstance(self.encoder, IncidenceEncoder):
                msg = (
                    "integrator needs incidence but encoder "
                    f"{type(self.encoder).__name__} has no build_incidence"
                )
                raise TypeError(msg)
            repr_ = KnowledgeRepr(incidence=self.encoder.build_incidence(sub))
        else:
            repr_ = self.encoder.encode(sub)
        payload = self.integrator.to_payload(repr_)
        return KnowledgeContext(
            query=query,
            facts=retrieval.facts,
            text_context=payload.text_context,
            prompt_embeds=payload.prompt_embeds,
            prompt_attention_mask=payload.prompt_attention_mask,
            graph_features=payload.graph_features,
            metadata={"knowhyper_subhypergraph_size": len(sub.hyperedges)},
        )


def _mention_seed_features(
    *, query: Query, sub: Any, required_schema: str
) -> dict[str, torch.Tensor]:
    """Map predicted mentions to their linked seed-node rows and scores."""
    provenance = sub.provenance
    actual_schema = provenance.get("linked_mentions_schema")
    if actual_schema != required_schema:
        msg = (
            "NER mention-specific fusion requires cache provenance schema "
            f"{required_schema!r}, got {actual_schema!r}"
        )
        raise ValueError(msg)
    raw_spans = list(query.metadata.get("entity_spans", []))
    spans = [(int(span[0]), int(span[1])) for span in raw_spans]
    span_to_index: dict[tuple[int, int], int] = {}
    for index, span in enumerate(spans):
        if span in span_to_index:
            msg = f"duplicate predicted mention span is ambiguous: {span}"
            raise ValueError(msg)
        span_to_index[span] = index
    node_to_index = {node_id: index for index, node_id in enumerate(sub.node_ids)}
    ownership = torch.zeros((len(sub.node_ids), len(spans)), dtype=torch.bool)
    local_ownership = torch.zeros_like(ownership)
    hyperedge_ownership = torch.zeros(
        (len(sub.hyperedges), len(spans)), dtype=torch.bool
    )
    scores = torch.zeros(len(spans), dtype=torch.float)
    linked_mask = torch.zeros(len(spans), dtype=torch.bool)
    for record in provenance.get("linked_mentions", []):
        raw_span = record.get("char_span")
        if raw_span is None:
            msg = "linked mention provenance is missing char_span"
            raise ValueError(msg)
        span = (int(raw_span[0]), int(raw_span[1]))
        mention_index = span_to_index.get(span)
        if mention_index is None:
            msg = f"linked provenance span is not a predicted mention: {span}"
            raise ValueError(msg)
        node_id = str(record["node_id"])
        node_index = node_to_index.get(node_id)
        if node_index is None:
            msg = f"linked seed node {node_id!r} is absent from the sub-hypergraph"
            raise ValueError(msg)
        ownership[node_index, mention_index] = True
        local_ownership[node_index, mention_index] = True
        for edge_index, hyperedge in enumerate(sub.hyperedges):
            if node_id not in hyperedge.node_ids:
                continue
            hyperedge_ownership[edge_index, mention_index] = True
            for adjacent_id in hyperedge.node_ids:
                adjacent_index = node_to_index.get(adjacent_id)
                if adjacent_index is not None:
                    local_ownership[adjacent_index, mention_index] = True
        scores[mention_index] = max(
            float(scores[mention_index]), float(record["score"])
        )
        linked_mask[mention_index] = True
    return {
        "incidence_node_mention_mask": ownership,
        "incidence_local_node_mention_mask": local_ownership,
        "incidence_hyperedge_mention_mask": hyperedge_ownership,
        "ner_mention_link_scores": scores,
        "ner_mention_link_mask": linked_mask,
    }


@register("knowledge_injector", name="knowhyper_ner")
class NERKnowHyperInjector(IEKnowledgeInjector):
    """KnowHyper injector for NER: reads flat entity list instead of head/tail pair."""

    def __init__(
        self,
        linker: EntityLinker,
        retriever: SubHGRetriever,
        encoder: SubHGEncoder,
        integrator: Integrator,
        top_k: int = 64,
        instance_metadata_fields: tuple[str, str, str] | None = None,
        mention_type_map: dict[str, str] | None = None,
        expand_abbreviations: bool = False,
        entity_view_augmenter: EntityViewAugmenter | None = None,
        mention_provenance_schema: str | None = None,
        # #233 Phase A0. OFF by default: `khp_span_late_*` covers 400 recorded cells and
        # must keep an unchanged context, so the extra keys appear only when an arm opts in.
        emit_a0_metadata: bool = False,
        # Set for the NER TEXT arm, which serializes retrieved node text into the
        # input instead of encoding a graph. When set, `encoder`/`integrator` are
        # unused in the default ``replace_graph`` mode -- the arm has no graph branch.
        text_serializer: Any | None = None,
        # ``graph_sidecar`` is the fixed cumulative #220 + #231 contract: serialize the
        # selected text into context metadata AND retain graph encoding. It is opt-in so the
        # completed #220 text-only and #231 graph-only contexts stay byte-for-byte unchanged.
        text_serializer_mode: str = "replace_graph",
    ) -> None:
        self.linker = linker
        self.retriever = retriever
        self.encoder = encoder
        self.integrator = integrator
        self.top_k = top_k
        self.instance_metadata_fields = instance_metadata_fields
        self.expand_abbreviations = expand_abbreviations
        self.entity_view_augmenter = entity_view_augmenter
        self.mention_provenance_schema = mention_provenance_schema
        self.emit_a0_metadata = bool(emit_a0_metadata)
        self.text_serializer = text_serializer
        if text_serializer_mode not in {"replace_graph", "graph_sidecar"}:
            msg = (
                "text_serializer_mode must be 'replace_graph' or 'graph_sidecar', got "
                f"{text_serializer_mode!r}"
            )
            raise ValueError(msg)
        self.text_serializer_mode = text_serializer_mode
        if self.emit_a0_metadata and self.text_serializer is None:
            msg = (
                "emit_a0_metadata=True requires text_serializer. Treating an absent "
                "serializer as an empty selection would make every candidate look like an "
                "explicit NIL and let the A0 text arm train with zero text."
            )
            raise ValueError(msg)
        if self.emit_a0_metadata and self.text_serializer_mode != "replace_graph":
            msg = (
                "emit_a0_metadata and text_serializer_mode='graph_sidecar' are separate "
                "contracts and cannot be enabled together"
            )
            raise ValueError(msg)
        if (
            self.text_serializer_mode == "graph_sidecar"
            and self.text_serializer is None
        ):
            msg = "text_serializer_mode='graph_sidecar' requires text_serializer"
            raise ValueError(msg)
        if mention_provenance_schema not in (None, LINKED_MENTION_PROVENANCE_SCHEMA):
            msg = (
                f"unsupported mention provenance schema: {mention_provenance_schema!r}"
            )
            raise ValueError(msg)
        self.mention_type_map = (
            {k: v for k, v in mention_type_map.items() if v is not None}
            if mention_type_map is not None
            else None
        )

    def inject(self, query: Query, retrieval: RetrievalResult) -> KnowledgeContext:
        instance = None
        if self.instance_metadata_fields is not None:
            instance = InstanceKey.from_metadata(
                query.metadata,
                fields=self.instance_metadata_fields,
                fallback_text=query.text,
            )
            instance = prediction_scoped_instance_key(
                instance, query.metadata.get("prediction_fingerprint")
            )

        sub = None
        get_cached = getattr(self.retriever, "get_cached", None)
        if instance is not None and callable(get_cached):
            sub = get_cached(instance)

        if sub is None:
            entity_types: list[str] | None = None
            raw_types = query.metadata.get("entity_types")
            if raw_types is not None:
                entity_types = list(raw_types)
            raw_spans = query.metadata.get("entity_spans")
            entity_spans = (
                [(int(span[0]), int(span[1])) for span in raw_spans]
                if raw_spans is not None
                else None
            )
            mentions = mentions_from_ner_query(
                query.text,
                list(query.entities),
                entity_types,
                entity_spans,
                type_map=self.mention_type_map,
                expand_abbreviations=self.expand_abbreviations,
            )
            linked = self.linker.link(mentions)
            sub = self.retriever.retrieve(linked, top_k=self.top_k, instance=instance)
        if self.entity_view_augmenter is not None:
            sub = self.entity_view_augmenter.augment(sub)
        # Ordinary text arms replace graph encoding completely. A0 is different: it needs the
        # serializer's selected items as candidate-owned SIDE-CAR evidence while retaining the
        # incidence payload used for span geometry and graph-link diagnostics. The opt-in flag
        # distinguishes those contracts; without it, adding the serializer would silently
        # switch A0 back to the instance-level #220 path.
        serialized_for_a0 = None
        serialized_text_context = None
        if (
            self.text_serializer is not None
            and not self.emit_a0_metadata
            and self.text_serializer_mode == "replace_graph"
        ):
            return _text_serialized_context(
                serializer=self.text_serializer,
                query=query,
                sub=sub,
                facts=retrieval.facts,
            )
        if self.text_serializer_mode == "graph_sidecar":
            # Build through the exact #220 helper so items_key, selected-item ordering,
            # fingerprint, and selection provenance cannot drift between the standalone and
            # cumulative routes. Only its metadata is merged below; graph payload construction
            # continues normally.
            serializer = self.text_serializer
            if serializer is None:  # Defensive against mutation after construction.
                msg = "graph-sidecar serializer disappeared after injector construction"
                raise RuntimeError(msg)
            serialized_text_context = _text_serialized_context(
                serializer=serializer,
                query=query,
                sub=sub,
                facts=retrieval.facts,
            )
        if self.emit_a0_metadata:
            # Constructor validation above guarantees this is real. Serialize exactly once;
            # T1 populated the restricted cache, so this normally reads the selected items
            # from that cache while preserving the serializer's target ordering/max_targets.
            serializer = self.text_serializer
            if (
                serializer is None
            ):  # Defensive against state mutation after construction.
                msg = "A0 serializer disappeared after injector construction"
                raise RuntimeError(msg)
            serialized_for_a0 = serializer.serialize(query, sub)
        if getattr(self.integrator, "needs_incidence", False):
            from knowhyper.pipeline.protocols import IncidenceEncoder
            from knowhyper.pipeline.types import KnowledgeRepr

            if not isinstance(self.encoder, IncidenceEncoder):
                msg = (
                    "integrator needs incidence but encoder "
                    f"{type(self.encoder).__name__} has no build_incidence"
                )
                raise TypeError(msg)
            repr_ = KnowledgeRepr(incidence=self.encoder.build_incidence(sub))
        else:
            repr_ = self.encoder.encode(sub)
        payload = self.integrator.to_payload(repr_)
        if self.mention_provenance_schema is not None:
            payload.graph_features.update(
                _mention_seed_features(
                    query=query,
                    sub=sub,
                    required_schema=self.mention_provenance_schema,
                )
            )
        context_metadata: dict[str, Any] = {
            "knowhyper_subhypergraph_size": len(sub.hyperedges)
        }
        if serialized_text_context is not None:
            context_metadata.update(serialized_text_context.metadata)
        if self.emit_a0_metadata:
            from knowhyper.ieforge.a0_metadata import build_a0_metadata

            # `serialized` is what the text serializer selected for this instance; A0 forwards
            # only those items, because that is what T0/T1 measured.
            context_metadata.update(
                build_a0_metadata(
                    query=query,
                    sub=sub,
                    serialized=serialized_for_a0,
                    enabled=True,
                )
            )
        return KnowledgeContext(
            query=query,
            facts=retrieval.facts,
            text_context=payload.text_context,
            prompt_embeds=payload.prompt_embeds,
            prompt_attention_mask=payload.prompt_attention_mask,
            graph_features=payload.graph_features,
            metadata=context_metadata,
        )
