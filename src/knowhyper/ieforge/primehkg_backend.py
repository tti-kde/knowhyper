"""Stage implementations backed by primehkg.

These classes adapt primehkg's ``LinkPipeline`` and retrieval pipeline to the
KnowHyper stage protocols (``EntityLinker``/``SubHGRetriever``). All shape
conversion goes through :mod:`knowhyper.ieforge.adapters`, so this module only
wires upstream calls and never reshapes upstream dicts inline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from primehkg import LinkPipeline, make_retrieval_pipeline
from primehkg.retrieval.defaults import EwoRanker, SeedIncidentCandidate, UnionCandidate
from primehkg.retrieval.pipeline import RetrievalPipeline
from primehkg.retrieval.presets import DEFAULT_RETRIEVAL
from primehkg.retrieval.vector_index import VectorIndex

from knowhyper.features.modality_index import ModalityIndex
from knowhyper.ieforge.adapters import (
    link_result_to_linked_entities,
    retrieval_output_to_subhypergraph,
)
from knowhyper.ieforge.typed_vector_candidate import TypedVectorLocalCandidate
from knowhyper.ieforge.vector_boost_ranker import VectorBoostRanker
from knowhyper.pipeline.types import LinkedEntity, SubHypergraph

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from knowhyper.pipeline.types import Mention

logger = logging.getLogger(__name__)


LINKED_MENTION_PROVENANCE_SCHEMA = "linked_mentions_v1"
LINKED_MENTION_ROLE_PROVENANCE_SCHEMA = "linked_mentions_role_v1"


def _with_linked_mention_provenance(
    sub: SubHypergraph,
    linked: Sequence[LinkedEntity],
    *,
    provenance_schema: str | None = None,
) -> SubHypergraph:
    """Attach JSON-safe mention-to-seed provenance before cache persistence."""
    linked_mentions = []
    for entity in linked:
        span = entity.mention.char_span
        linked_mentions.append(
            {
                "text": entity.mention.text,
                "char_span": list(span) if span is not None else None,
                "entity_type": entity.mention.entity_type,
                "node_id": entity.node_id,
                "score": entity.score,
                "match_kind": entity.match_kind,
                "candidates": list(entity.candidates),
                "pair_role": entity.mention.pair_role,
            }
        )
    if provenance_schema == LINKED_MENTION_ROLE_PROVENANCE_SCHEMA:
        invalid = [
            record
            for record in linked_mentions
            if record["pair_role"] not in {"head", "tail"}
        ]
        if invalid:
            msg = "role-aware provenance requires head/tail on every linked mention"
            raise ValueError(msg)
        schema = LINKED_MENTION_ROLE_PROVENANCE_SCHEMA
    elif provenance_schema in {None, LINKED_MENTION_PROVENANCE_SCHEMA}:
        schema = (
            LINKED_MENTION_ROLE_PROVENANCE_SCHEMA
            if any(record["pair_role"] is not None for record in linked_mentions)
            else LINKED_MENTION_PROVENANCE_SCHEMA
        )
    else:
        msg = f"unsupported linked-mention provenance schema: {provenance_schema!r}"
        raise ValueError(msg)
    provenance = dict(sub.provenance)
    provenance.update(
        {
            "linked_mentions_schema": schema,
            "linked_mentions": linked_mentions,
        }
    )
    return replace(sub, provenance=provenance)


class PrimeHKGLinker:
    """``EntityLinker`` backed by a primehkg ``LinkPipeline``.

    ``preset=None`` uses the default resolver chain (exact node id / xref / label
    plus fuzzy), which works without SapBERT sidecars and is suitable for
    fixtures and CI. Pass a preset such as ``primehkg.SAPBERT_LINKING`` for dense
    linking when the build ships label embeddings.

    Heavy initialization (loading SapBERT, building the LinkPipeline) is deferred
    until the first ``link()`` call, so Hydra-instantiated linkers that are never
    used (cache-hit path) pay zero startup cost.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        preset: Any = None,
        top_k: int = 5,
        recall_min_score: float | None = None,
        expansion_fallback: bool = False,
        strict_type_match: bool = False,
    ) -> None:
        """Store params; pipeline construction is deferred to first ``link()``."""
        self._top_k = top_k
        self._recall_min_score = recall_min_score
        self._expansion_fallback = expansion_fallback
        self._strict_type_match = strict_type_match
        self._dataset = dataset
        self._preset = preset
        self._pipeline: LinkPipeline | None = None

    def _ensure_pipeline(self) -> LinkPipeline:
        if self._pipeline is not None:
            return self._pipeline
        logger.info("PrimeHKGLinker: lazy-initializing LinkPipeline (first link call)")
        if self._preset is None:
            self._pipeline = LinkPipeline(self._dataset)
        else:
            from knowhyper.ieforge.sapbert_cache import install_sapbert_cache

            install_sapbert_cache()
            ensure_label_embedding_sidecar(self._dataset)
            self._pipeline = LinkPipeline.from_preset(self._dataset, self._preset)
        return self._pipeline

    def link(self, mentions: Sequence[Mention]) -> list[LinkedEntity]:
        """Link each mention to its top primehkg candidate (drops misses).

        On a primary miss, if a type-guarded recall fallback is enabled
        (``recall_min_score``) and the mention has a type hint, retry at the lower
        threshold and accept the top candidate only when its node type matches.
        """
        out: list[LinkedEntity] = []
        pipeline = self._ensure_pipeline()
        for mention in mentions:
            result = pipeline.link(
                mention.text,
                namespace=mention.namespace,
                node_type_hint=mention.entity_type,
                top_k=self._top_k,
            )
            if self._strict_type_match and mention.entity_type:
                linked = self._pick_typed_top(mention, dict(result))
            else:
                linked = link_result_to_linked_entities(mention, dict(result))
            if linked is None and self._should_recall(mention):
                linked = self._recall_link(mention)
            if linked is None and self._should_expand(mention):
                linked = self._expansion_link(mention)
            if linked is not None:
                out.append(linked)
        return out

    def _should_recall(self, mention: Mention) -> bool:
        return self._recall_min_score is not None and bool(mention.entity_type)

    def _recall_link(self, mention: Mention) -> LinkedEntity | None:
        """Second-pass link at the lower threshold; accept only on a type match."""
        from primehkg.linking.pipeline import node_type_of

        result = self._ensure_pipeline().link(
            mention.text,
            node_type_hint=mention.entity_type,
            top_k=self._top_k,
            min_score=self._recall_min_score,
        )
        linked = link_result_to_linked_entities(mention, dict(result))
        if linked is None:
            return None
        if node_type_of(str(linked.node_id)) != mention.entity_type:
            return None
        return linked

    def _should_expand(self, mention: Mention) -> bool:
        return (
            self._expansion_fallback
            and bool(mention.expansion)
            and bool(mention.entity_type)
        )

    def _expansion_link(self, mention: Mention) -> LinkedEntity | None:
        """Link the mention's full form; accept only on a type match.

        The returned ``LinkedEntity`` keeps the ORIGINAL mention (the expansion is a
        linking aid, not a relabel).
        """
        from primehkg.linking.pipeline import node_type_of

        if mention.expansion is None:
            return None
        result = self._ensure_pipeline().link(
            mention.expansion,
            node_type_hint=mention.entity_type,
            top_k=self._top_k,
        )
        linked = link_result_to_linked_entities(mention, dict(result))
        if linked is None:
            return None
        if node_type_of(str(linked.node_id)) != mention.entity_type:
            return None
        return linked

    def _pick_typed_top(
        self, mention: Mention, link_result: dict[str, Any]
    ) -> LinkedEntity | None:
        """Top candidate whose node type matches the mention's entity_type.

        Returns None when the pool has no correct-type candidate, so the mention
        falls through to the type-guarded recall/expansion fallbacks rather than
        accepting a wrong-type node. The candidates field lists the other node ids
        (the picked one excluded), matching link_result_to_linked_entities.
        """
        from primehkg.linking.pipeline import node_type_of

        candidates = link_result.get("candidates") or []
        for c in candidates:
            if node_type_of(str(c["node_id"])) == mention.entity_type:
                others = tuple(o["node_id"] for o in candidates if o is not c)
                return LinkedEntity(
                    mention=mention,
                    node_id=c["node_id"],
                    score=float(c["score"]),
                    match_kind=str(c["match_kind"]),
                    candidates=others,
                )
        return None


class PrimeHKGRetriever:
    """``SubHGRetriever`` over a primehkg retrieval pipeline, configured by a profile.

    A "profile" is the bundle of primehkg knobs passed to the constructor:
    ``candidate_mode``, ``top_k``, and the EWO scoring overrides. ``post`` is the
    one custom-component seam primehkg's ``make_retrieval_pipeline`` accepts today
    and is forwarded when present. ``candidate``/``ranker`` are reserved for a
    future seam (e.g. a learned ranker) that upstream does not yet support, so a
    non-``None`` value raises ``NotImplementedError`` rather than being silently
    ignored.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        candidate_mode: str = "seed_incident",
        top_k: int = 64,
        hub_penalty: float | None = None,
        hub_degree_flag_threshold: int | None = None,
        semantics_weight: dict[str, float] | None = None,
        evidence_weight: dict[str, float] | None = None,
        max_hyperedge_size: int | None = None,
        khop_budget: int | None = None,
        candidate: Any = None,
        ranker: Any = None,
        post: Any = None,
        vector_candidate_sidecars: list[dict[str, Any]] | None = None,
        vector_boost: float = 1.0,
        provenance_schema: str | None = None,
    ) -> None:
        """Store params; retrieval pipeline construction is deferred to first use.

        ``khop_budget`` caps the k-hop candidate set primehkg gathers before its
        (expensive, SHA-256-keyed) truncation+ranking. The final sub-hypergraph
        keeps only ``top_k`` hyperedges, so any budget >= the per-seed candidate
        count yields the identical result while a smaller budget cuts wasted
        candidate-gathering. ``None`` keeps primehkg's own default (500).
        """
        if candidate is not None or ranker is not None:
            msg = (
                "PrimeHKGRetriever custom candidate/ranker components are not "
                "supported yet (primehkg's make_retrieval_pipeline accepts only "
                "post); leave them None or extend this wiring first."
            )
            raise NotImplementedError(msg)
        self.profile: dict[str, Any] = {
            "candidate_mode": candidate_mode,
            "top_k": top_k,
            "hub_penalty": hub_penalty,
            "hub_degree_flag_threshold": hub_degree_flag_threshold,
            "semantics_weight": semantics_weight,
            "evidence_weight": evidence_weight,
            "max_hyperedge_size": max_hyperedge_size,
            "khop_budget": khop_budget,
        }
        # Sorted so the cache fingerprint is content-based, not input-order-based:
        # two configs with the same sidecars listed in a different order must not
        # produce different fingerprints (which would force a spurious recompute).
        self.profile["vector_candidate_sidecars"] = tuple(
            sorted(
                (
                    str(d["sidecar_dir"]),
                    str(d["seed_prefix"]),
                    int(d["top_k_nodes"]),
                )
                for d in (vector_candidate_sidecars or [])
            )
        )
        self.profile["vector_boost"] = float(vector_boost)
        if provenance_schema is not None:
            self.profile["provenance_schema"] = provenance_schema
        self._vector_candidate_sidecars = list(vector_candidate_sidecars or [])
        self._vector_boost = float(vector_boost)
        self._provenance_schema = provenance_schema
        self._dataset = dataset
        self._post = post
        self._pipeline = None

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        logger.info(
            "PrimeHKGRetriever: lazy-initializing retrieval pipeline (first retrieve call)"
        )
        if not self._vector_candidate_sidecars:
            kwargs: dict[str, Any] = {"candidate_mode": self.profile["candidate_mode"]}
            if self._post is not None:
                kwargs["post"] = self._post
            khop_budget = self.profile["khop_budget"]
            if khop_budget is not None:
                kwargs["subhypergraph_preset"] = {
                    "max_he_level1": khop_budget,
                    "max_nodes_level2": khop_budget,
                    "max_he_level3": khop_budget,
                }
            self._pipeline = make_retrieval_pipeline(self._dataset, **kwargs)
            return self._pipeline

        # Vector-augmented path: union seed-incident with typed vector-local
        # candidates and rank with an EWO+similarity boost so vector-only
        # (near-zero-EWO) hyperedges survive the top_k cut. NOTE: this path does
        # not build via ``make_retrieval_pipeline``/``from_preset``, so it does
        # not attach the CSR sub-hypergraph sampler and ``khop_budget`` is not
        # applied here — see spec §7 for the follow-up.
        typed_candidates = [
            TypedVectorLocalCandidate(
                node_index=VectorIndex(d["sidecar_dir"]),
                seed_prefix=str(d["seed_prefix"]),
                top_k_nodes=int(d["top_k_nodes"]),
            )
            for d in self._vector_candidate_sidecars
        ]
        candidate = UnionCandidate([SeedIncidentCandidate(), *typed_candidates])
        ranker = VectorBoostRanker(
            EwoRanker(), candidates=typed_candidates, boost=self._vector_boost
        )
        self._pipeline = RetrievalPipeline(
            dataset=self._dataset,
            candidate=candidate,
            ranker=ranker,
            candidate_mode="seed_incident",
            default_scoring_preset=DEFAULT_RETRIEVAL,
        )
        return self._pipeline

    def retrieve(
        self,
        linked: Sequence[LinkedEntity],
        *,
        top_k: int | None = None,
        instance: Any = None,
    ) -> SubHypergraph:
        """Retrieve the seed-incident sub-hypergraph for the linked entities."""
        del instance  # non-caching retriever ignores the instance key
        seed_ids = [le.node_id for le in linked]
        if not seed_ids:
            empty = SubHypergraph(seed_node_ids=(), node_ids=(), hyperedges=())
            return _with_linked_mention_provenance(
                empty, linked, provenance_schema=self._provenance_schema
            )
        output = self._ensure_pipeline().retrieve(
            seed_node_ids=seed_ids,
            top_k=top_k if top_k is not None else self.profile["top_k"],
            hub_penalty=self.profile["hub_penalty"],
            hub_degree_flag_threshold=self.profile["hub_degree_flag_threshold"],
            semantics_weight=self.profile["semantics_weight"],
            evidence_weight=self.profile["evidence_weight"],
            max_hyperedge_size=self.profile["max_hyperedge_size"],
        )
        sub = retrieval_output_to_subhypergraph(dict(output))
        return _with_linked_mention_provenance(
            sub, linked, provenance_schema=self._provenance_schema
        )


class RuleSubHGRetriever(PrimeHKGRetriever):
    """Default rule-based seed-incident retriever (primehkg defaults).

    Thin alias preserving the original name/behavior: ``seed_incident`` candidates
    and primehkg's default EWO scoring, with the original top-k budget.
    """


def ensure_label_embedding_sidecar(dataset: Any) -> bool:
    """Register a present-but-unlisted ``label_embedding_v1`` sidecar.

    primehkg v0.12.x ships the sidecar but records it as a dict
    ``{"path": ..., "schema_version": ...}`` in the manifest, while
    ``_try_load_label_embedding_index`` expects a bare string. Rewrite the
    manifest entry to the string form so the property resolves.
    """
    if getattr(dataset, "label_embedding_index", None) is not None:
        return False
    root = getattr(dataset, "sidecar_root", None)
    manifest = getattr(dataset, "manifest", None)
    if root is None or manifest is None:
        return False
    sidecars = manifest.get("sidecars") or {}
    entry = sidecars.get("label_embedding_v1")
    if isinstance(entry, dict):
        rel = entry.get("path")
    elif isinstance(entry, str):
        rel = entry
    else:
        return False
    if rel is None:
        return False
    sidecar_dir = Path(root) / rel
    if not sidecar_dir.is_dir():
        return False
    sidecars["label_embedding_v1"] = rel
    cache = getattr(dataset, "_cache", None)
    if cache is not None:
        if hasattr(cache, "label_embedding_index"):
            cache.label_embedding_index = None
        if hasattr(cache, "label_embedding_index_checked"):
            cache.label_embedding_index_checked = False
    logger.info("Registered label_embedding_v1 sidecar at %s", sidecar_dir)
    return True


def ensure_vector_node_sidecar(dataset: Any) -> bool:
    """Register a present-but-unlisted ``vector_node_v1`` sidecar, fingerprint-checked.

    The published primehkg v0.10.2 build ships the node-embedding sidecar on disk
    but does not list it in the manifest ``sidecars`` dict, so ``vector_node_index``
    is ``None`` by default. If the sidecar dir is present and its
    ``inputs_fingerprint`` matches the dataset manifest, register it so the real
    embeddings load. A mismatched fingerprint is left unregistered (never attach a
    stale sidecar to a different build). Returns True if it registered.
    """
    if getattr(dataset, "vector_node_index", None) is not None:
        return False
    root = getattr(dataset, "sidecar_root", None)
    manifest = getattr(dataset, "manifest", None)
    if root is None or manifest is None:
        return False
    meta_path = Path(root) / "sidecars" / "vector_node_v1" / "meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("inputs_fingerprint") != manifest.get("inputs_fingerprint"):
        logger.warning(
            "vector_node_v1 sidecar fingerprint mismatch; not registering (stale build)."
        )
        return False
    manifest.setdefault("sidecars", {})["vector_node_v1"] = "sidecars/vector_node_v1"
    # Reset BOTH the cached value AND the "checked" flag. The vector_node_index
    # property short-circuits on `vector_node_index_checked`, which the guard above
    # already set True (caching a None) when it probed the property. Clearing only
    # the value would leave the flag True, so the next access would return the
    # stale None and the registration would be inert.
    cache = getattr(dataset, "_cache", None)
    if cache is not None:
        if hasattr(cache, "vector_node_index"):
            cache.vector_node_index = None
        if hasattr(cache, "vector_node_index_checked"):
            cache.vector_node_index_checked = False
    return True


class PrimeHKGFeatureStore:
    """Resolve per-node embeddings from primehkg's ``vector_node_index``.

    Looks up each node id in the dataset's ``vector_node_v1`` sidecar. Nodes the
    index misses get a zero row flagged by :meth:`missing`, so the encoder can
    mask them rather than treat a hole as a real feature. When the sidecar is
    absent (``vector_node_index is None``) every row is zero and flagged missing,
    and the encoder degrades to structure+role+semantics only.
    """

    def __init__(self, dataset: Any, *, dim: int = 16) -> None:
        """Bind ``dataset``; ``dim`` follows the index when present, else default."""
        ensure_vector_node_sidecar(dataset)
        self._index = getattr(dataset, "vector_node_index", None)
        if self._index is not None:
            self.dim = int(self._index.embedding_dim)
        else:
            self.dim = dim
            logger.warning(
                "PrimeHKGFeatureStore: no vector_node_index sidecar; "
                "returning zero node features (encoder runs structure-only)."
            )

    def _missing_flags(self, node_ids: Sequence[str]) -> list[bool]:
        if self._index is None:
            return [True] * len(node_ids)
        return [self._index.lookup(nid) is None for nid in node_ids]

    def features(self, node_ids: Sequence[str]) -> dict[str, Any]:
        """Return ``{"node_embeds": (len, dim) tensor}``; misses are zero rows."""
        out = torch.zeros(len(node_ids), self.dim)
        if self._index is not None:
            for i, nid in enumerate(node_ids):
                vec = self._index.lookup(nid)
                if vec is not None:
                    out[i] = torch.as_tensor(vec, dtype=torch.float32)
        return {"node_embeds": out}

    def missing(self, node_ids: Sequence[str]) -> torch.Tensor:
        """Bool tensor, True where the node has no resolved embedding."""
        return torch.tensor(self._missing_flags(node_ids), dtype=torch.bool)


PAD_DIM = 768  # zero-pad transport width (max modality dim)


class MultiModalFeatureStore:
    """Resolve per-(node, source_modality) embeddings into a zero-padded per-node stack.

    Unlike :class:`PrimeHKGFeatureStore` (one vector per node), this serves a
    fixed-order stack ``(N, max_m, PAD_DIM)`` of modality vectors + a boolean mask
    ``(N, max_m)`` + long slot-ids ``(N, max_m)`` that encode which modality
    occupies each slot. Vectors are zero-padded from their real dim to ``PAD_DIM``
    so the stack is rectangular; the learnable ``PerModalityAttentionFusion`` reads
    only the real ``dim_m`` columns and ignores the pad. ``display_name`` is always
    slot 0 in the per-type ordering. The slot-id is the index into
    ``sorted(modality_dims)`` (= ``modality_vocab``), matching the convention used by
    :class:`PerModalityAttentionFusion` so slot-ids are consistent end-to-end.
    """

    def __init__(
        self,
        *,
        features_dir: str,
        modalities_by_type: Mapping[str, list[str]],
    ) -> None:
        """Build per-type ordered modality lists, load the index, probe real dims."""
        self.modalities_by_type: dict[str, list[str]] = {}
        union: set[str] = {"display_name"}
        for ntype, mods in modalities_by_type.items():
            ordered = ["display_name", *[m for m in mods if m != "display_name"]]
            self.modalities_by_type[ntype] = ordered
            union |= set(ordered)
        self._index = ModalityIndex(features_dir, modalities=sorted(union))
        self.max_m = max(len(v) for v in self.modalities_by_type.values())

        # Probe real dim for each modality in the union by finding any stored vector.
        # Falls back to PAD_DIM with a warning when a modality has no vectors at all.
        self.modality_dims: dict[str, int] = {}
        for modality in union:
            real_dim: int | None = None
            for (_nid, sm), vec in self._index._map.items():  # noqa: SLF001
                if sm == modality and vec is not None:
                    real_dim = len(vec)
                    break
            if real_dim is None:
                logger.error(
                    "MultiModalFeatureStore: no vectors found for modality %r; "
                    "defaulting real_dim to PAD_DIM=%d. "
                    "W_m will project pad-zeros for this modality — "
                    "check that the features sidecar contains this modality.",
                    modality,
                    PAD_DIM,
                )
                real_dim = PAD_DIM
            self.modality_dims[modality] = real_dim

        self.modality_vocab: list[str] = sorted(self.modality_dims)
        self._slot_of: dict[str, int] = {
            m: i for i, m in enumerate(self.modality_vocab)
        }
        # Flattened-stack width (max_m * PAD_DIM) that the injector-side AllSet
        # encoder's proj reads at construction. The model-side fusion (Task 6)
        # replaces these feats with a fused OUT_DIM vector before the store-free
        # encoder runs, so this proj is unused on the multimodal path — but .dim
        # must exist for AllSetHyperEncoder construction to succeed.
        self.dim = self.max_m * PAD_DIM

    def node_type(self, node_id: str) -> str:
        """Return the type token from the ``primekg:<type>:<id>`` prefix."""
        parts = node_id.split(":")
        return parts[1] if len(parts) >= 3 else (parts[0] if parts else "")

    def features(self, node_ids: Sequence[str]) -> dict[str, Any]:
        """Return the padded modality stack, mask, and slot-ids for ``node_ids``.

        Returns a dict with:

        - ``"modality_embeds"``: ``(N, max_m, PAD_DIM)`` float32 tensor.
        - ``"modality_mask"``: ``(N, max_m)`` float32 tensor (1.0 = present).
        - ``"slot_modality"``: ``(N, max_m)`` long tensor (vocab index; -1 = unused).
        """
        n = len(node_ids)
        embeds = torch.zeros(n, self.max_m, PAD_DIM)
        mask = torch.zeros(n, self.max_m)
        slot_modality = torch.full((n, self.max_m), -1, dtype=torch.long)
        for i, nid in enumerate(node_ids):
            mods = self.modalities_by_type.get(self.node_type(nid))
            if mods is None:
                continue
            for j, modality in enumerate(mods):
                vec = self._index.lookup(nid, modality)
                if vec is not None:
                    real_dim = self.modality_dims.get(modality, PAD_DIM)
                    embeds[i, j, :real_dim] = torch.as_tensor(vec, dtype=torch.float32)
                    mask[i, j] = 1.0
                    slot_modality[i, j] = self._slot_of[modality]
        return {
            "modality_embeds": embeds,
            "modality_mask": mask,
            "slot_modality": slot_modality,
        }

    def missing(self, node_ids: Sequence[str]) -> torch.Tensor:
        """Bool tensor ``(N,)``, True where the node has no present modality."""
        return self.features(node_ids)["modality_mask"].sum(dim=1) == 0
