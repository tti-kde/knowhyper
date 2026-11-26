"""Endpoint-only encoder: the head and tail vectors, with no neighborhood.

This is the matched control for the claim that KnowHyper's gain comes from the
retrieved neighborhood rather than from entity identity. Asada et al. and
Sänger & Leser both attach KG information at the endpoints only; this encoder
reproduces that shape inside our own retrieval, backbone, and evaluator, so the
comparison isolates the neighborhood instead of confounding it with a different
knowledge source.

It reads ``SubHypergraph.seed_node_ids`` -- the linked head and tail -- from the
SAME retrieved sub-hypergraph and the SAME feature store the graph arms use, and
emits their projected vectors as the only knowledge tokens. No hyperedges, no
neighbors, no message passing. Retrieval still runs and the cache fingerprint is
unchanged, so ``endpoint`` and ``ev_cold`` differ in exactly one thing: whether
anything beyond the endpoints reaches the model.

``shuffle_pool`` turns the arm into its own control. See ``EndpointOnlyEncoder``
for why a batch roll was not reused.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from knowhyper.encoders.endpoint_pool import node_type_of
from knowhyper.pipeline.types import KnowledgeRepr, SubHypergraph

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _stable_index(key: str, modulus: int) -> int:
    """Deterministic index in ``[0, modulus)`` from a string key.

    ``hash()`` is salted per process, so using it here would make the shuffled
    control differ between the training and evaluation runs of the same cell --
    a control that is not reproducible is not a control.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


class EndpointOnlyEncoder(nn.Module):
    """Project the seed (head/tail) node vectors and emit them as the tokens.

    Args:
        feature_store: the store the graph arms use; must expose
            ``features(node_ids) -> {"node_embeds": Tensor}`` and ``dim``.
        hidden_size: LM hidden size to project into.
        shuffle_pool: when set, replaces each seed's vector with that of a
            different entity **of the same node type**, chosen deterministically.
            Maps a type key to the pool of node ids of that type.
        node_types: maps node id to its type key, for pool selection.

    The shuffled control is type- and coverage-matched by construction rather
    than by a batch roll. ``model.knowledge_control=shuffled`` rolls the whole
    knowledge batch by one, but (a) it exists only on
    ``TokenCrossAttentionKnowledgeModel``, not on the RE model this arm uses, and
    (b) a roll is not matched: it can hand a drug--drug instance a gene's vector,
    and can hand an instance with two resolvable endpoints one with zero. The
    contrast would then confound wrong-identity with wrong-type and
    different-coverage. Here only identity changes: the substitute is drawn from
    the same type, and a seed that resolves to no vector stays unresolved in both
    arms.
    """

    def __init__(
        self,
        feature_store: Any,
        hidden_size: int = 768,
        shuffle_pool: Mapping[str, Sequence[str]] | None = None,
        node_types: Mapping[str, str] | None = None,
        pending_pool: tuple[str, str] | None = None,
    ) -> None:
        """Store the feature store and build the projection to hidden_size."""
        super().__init__()
        self.feature_store = feature_store
        self.hidden_size = hidden_size
        self.proj = nn.Linear(feature_store.dim, hidden_size)
        self.shuffle_pool = shuffle_pool
        self.node_types = node_types
        #: ``(dataset, splits)`` awaiting a retriever to supply the cache
        #: fingerprint; resolved by ``bind_pool``.
        self.pending_pool = pending_pool
        #: How often a substitution was attempted, and how often it had to pass
        #: the id through because its type had no other member. A nonzero
        #: fallback rate means the control is partly unshuffled and the contrast
        #: is diluted; report it rather than discovering it later.
        self.substitution_attempts = 0
        self.substitution_fallbacks = 0
        if shuffle_pool is not None and node_types is None:
            msg = (
                "shuffle_pool given without node_types: the substitute cannot be "
                "type-matched without knowing each seed's type, and an untyped "
                "substitution is the confounded control this arm exists to avoid"
            )
            raise ValueError(msg)

    def bind_pool(self, retriever: Any) -> None:
        """Resolve the substitution pool from the live retriever's fingerprint.

        Called by the injector, which is the only object holding both the encoder
        and the retriever. Fails loudly when a shuffled arm cannot reach a
        fingerprint: a silently unbuilt pool makes every substitution a no-op, so
        the "shuffled" arm becomes byte-identical to the treatment and the primary
        contrast reports a perfect null that means "the control never ran".
        """
        if self.pending_pool is None:
            return
        dataset, splits = self.pending_pool
        fingerprint = getattr(retriever, "fingerprint", None)
        if not isinstance(fingerprint, str) or not fingerprint:
            msg = (
                f"{type(retriever).__name__} exposes no cache `fingerprint`, so the "
                "endpoint shuffled pool cannot be located. This control requires a "
                "caching retriever; running without the pool would silently make "
                "the control identical to the treatment."
            )
            raise ValueError(msg)

        from knowhyper.encoders.endpoint_pool import (
            build_endpoint_pool,
            resolve_pool_dir,
        )

        pool, node_types = build_endpoint_pool(
            *(
                resolve_pool_dir(dataset, fingerprint, split)
                for split in splits.split(",")
            )
        )
        self.shuffle_pool = pool
        self.node_types = node_types
        self.pending_pool = None

    def _substitute(self, node_id: str) -> str:
        """Return a same-type stand-in for ``node_id``, or the id unchanged.

        The type is derived from the id rather than looked up in ``node_types``,
        so a seed the pool never saw is still shuffled within its own type. Only
        when that type has no other member does the id pass through unchanged;
        substituting across types would break the matching this control depends
        on, which is worse than leaving one endpoint real.

        Every pass-through is counted. A control that quietly degraded into the
        treatment for some fraction of endpoints would report a null that means
        "partly unshuffled" -- so the rate is recorded and asserted on rather
        than assumed to be zero.
        """
        if self.shuffle_pool is None:
            return node_id
        type_key = node_type_of(node_id)
        pool = [
            candidate
            for candidate in self.shuffle_pool.get(type_key, ())
            if candidate != node_id
        ]
        self.substitution_attempts += 1
        if not pool:
            self.substitution_fallbacks += 1
            return node_id
        return pool[_stable_index(node_id, len(pool))]

    def _empty(self) -> KnowledgeRepr:
        device = self.proj.weight.device
        dtype = self.proj.weight.dtype
        return KnowledgeRepr(
            node_embeds=torch.zeros(0, self.hidden_size, device=device, dtype=dtype),
            node_mask=torch.zeros(0, device=device),
            graph_embed=torch.zeros(self.hidden_size, device=device, dtype=dtype),
        )

    def encode(self, sub: SubHypergraph) -> KnowledgeRepr:
        """Project the seed node vectors; ignore every non-seed node and edge."""
        seeds = tuple(dict.fromkeys(sub.seed_node_ids))
        if not seeds:
            return self._empty()

        lookup_ids = [self._substitute(seed) for seed in seeds]
        device = self.proj.weight.device
        feats = self.feature_store.features(lookup_ids)["node_embeds"].to(device)
        if feats.shape[0] == 0:
            return self._empty()

        # `encode` runs inside DataLoader workers, and a tensor carrying
        # requires_grad cannot cross a process boundary -- torch raises
        # "Cowardly refusing to serialize non-leaf tensor which requires_grad".
        # Detaching is also the correct semantics, not just a workaround: a
        # projection applied in a worker would never receive gradient from the
        # main process, so a "learned" projection here would silently stay at its
        # initialisation. The endpoint vectors are FIXED PrimeHKG features by
        # design -- the same fixed features the graph arms consume -- and all
        # learning happens in the fusion layer downstream.
        with torch.no_grad():
            node_embeds = self.proj(feats).detach()
        node_mask = torch.ones(node_embeds.shape[0], device=device)
        return KnowledgeRepr(
            node_embeds=node_embeds,
            node_mask=node_mask,
            graph_embed=node_embeds.mean(dim=0),
            meta={
                "endpoint_seed_count": len(seeds),
                "endpoint_shuffled": self.shuffle_pool is not None,
            },
        )


def build_endpoint_encoder(
    feature_store: Any,
    hidden_size: int = 768,
    shuffled: bool = False,
    dataset: str | None = None,
    pool_split: str = "train,validation,test",
) -> EndpointOnlyEncoder:
    """Hydra entry point: build the real or the shuffled endpoint encoder.

    The SubHG cache fingerprint is NOT taken from config. ``knowhyper.dataset_
    fingerprint`` is the PrimeHKG *dataset* fingerprint (a 64-char hash), a
    different value from the *cache* fingerprint that names the SubHG directory
    (e.g. ``45caf65e8a07ea90``); using the former pointed the pool at a path that
    does not exist. Reproducing the cache-fingerprint derivation here would
    duplicate the retriever's profile assembly, so the shuffled encoder instead
    resolves its pool lazily from the live retriever -- see
    ``EndpointOnlyEncoder.bind_pool``.

    ``shuffled=False`` builds no pool at all, so the treatment arm never pays for
    it.
    """
    if not shuffled:
        return EndpointOnlyEncoder(feature_store, hidden_size=hidden_size)

    if dataset is None:
        msg = (
            "shuffled endpoint encoder needs `dataset` to locate its substitution "
            "pool; without it the shuffle would be a no-op and the control "
            "identical to the treatment"
        )
        raise ValueError(msg)

    return EndpointOnlyEncoder(
        feature_store,
        hidden_size=hidden_size,
        pending_pool=(dataset, pool_split),
    )
