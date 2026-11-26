"""A SubHGRetriever decorator that caches retrieved sub-hypergraphs per instance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from knowhyper.cache.keys import (
    RETRIEVAL_SEMANTICS_VERSION,
    SUBHG_CACHE_SCHEMA_VERSION,
    method_fingerprint,
    method_identity,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from knowhyper.cache.keys import InstanceKey
    from knowhyper.cache.store import SubHGCache
    from knowhyper.pipeline.protocols import SubHGRetriever
    from knowhyper.pipeline.types import LinkedEntity, SubHypergraph


class CachingSubHGRetriever:
    """Cache-decorate any ``SubHGRetriever`` by ``(fingerprint, instance)``.

    On a cache hit the stored sub-hypergraph is returned; on a miss the inner
    retriever runs and its result is stored. When no ``instance`` is supplied the
    call cannot be keyed, so it is delegated without caching.

    The cache fingerprint identifies the retrieval METHOD. It is derived from the
    inner retriever's resolved ``profile`` (its knob bundle), the pinned
    ``primehkg_rev``, concrete dataset fingerprint, cache/semantics versions,
    AND an optional ``linker_id``, so changing any of them forces a recompute.

    ``linker_id`` matters because the retrieved sub-hypergraph depends on the SEED
    nodes, which are produced by a linker that lives UPSTREAM of this decorator
    (in the injector). If the linker changes (e.g. exact/fuzzy -> SapBERT) the
    seeds change even though the retriever profile does not; pass the linker's
    identity here so that swap also changes the fingerprint. Omitting it means the
    cache assumes a fixed linker — document that in the config when you do.

    An explicit ``fingerprint`` override is accepted for inner retrievers that do
    not expose a ``profile`` (e.g. test doubles); when given, derivation and the
    provenance-manifest write are skipped. On the derivation path the resolved
    profile + rev + linker_id are recorded to the cache manifest for provenance.
    """

    def __init__(
        self,
        inner: SubHGRetriever,
        cache: SubHGCache,
        *,
        primehkg_rev: str | None = None,
        dataset_fingerprint: str | None = None,
        cache_schema_version: int = SUBHG_CACHE_SCHEMA_VERSION,
        retrieval_semantics_version: int = RETRIEVAL_SEMANTICS_VERSION,
        linker_id: str | None = None,
        provenance_schema: str | None = None,
        preparation_schema: str | None = None,
        fingerprint: str | None = None,
    ) -> None:
        """Wrap ``inner`` with ``cache``, deriving the method fingerprint.

        Provide ``primehkg_rev`` (the pin from ``pyproject.toml``, e.g. ``"v0.10.2"``)
        to derive the fingerprint from ``inner.profile`` + ``linker_id``; or pass an
        explicit ``fingerprint`` to bypass derivation. Exactly one path applies: an
        explicit ``fingerprint`` takes precedence and skips manifest writing.
        """
        self._inner = inner
        self._cache = cache
        if fingerprint is not None:
            # Explicit override: caller owns the identity; no derivation/manifest.
            self._fingerprint = fingerprint
        elif primehkg_rev is not None:
            profile: dict[str, Any] = dict(getattr(inner, "profile", {}))
            profile["linker_id"] = linker_id
            if provenance_schema is not None:
                profile["provenance_schema"] = provenance_schema
            if preparation_schema is not None:
                profile["preparation_schema"] = preparation_schema
            identity = method_identity(
                profile,
                primehkg_rev=primehkg_rev,
                dataset_fingerprint=dataset_fingerprint,
                cache_schema_version=cache_schema_version,
                retrieval_semantics_version=retrieval_semantics_version,
            )
            self._fingerprint = method_fingerprint(
                profile,
                primehkg_rev=primehkg_rev,
                dataset_fingerprint=dataset_fingerprint,
                cache_schema_version=cache_schema_version,
                retrieval_semantics_version=retrieval_semantics_version,
            )
            # Record the complete identity beside the shared cache entries.
            self._cache.write_manifest(
                self._fingerprint,
                profile,
                primehkg_rev=primehkg_rev,
                cache_identity=identity,
            )
        else:
            msg = "CachingSubHGRetriever needs either primehkg_rev or an explicit fingerprint"
            raise ValueError(msg)

    @property
    def fingerprint(self) -> str:
        """Return the resolved method fingerprint used for cache addressing."""
        return self._fingerprint

    def put_cached(self, instance: InstanceKey, sub: SubHypergraph) -> None:
        """Persist a validated precomputed sub-hypergraph under this method."""
        self._cache.put(self._fingerprint, instance, sub)

    def get_cached(self, instance: InstanceKey) -> SubHypergraph | None:
        """Return the cached sub-hypergraph for ``instance``, or None on a miss.

        Lets a caller (the injector) skip entity linking entirely on a hit: a
        cached sub-HG is addressed by the instance key alone, and ``retrieve``
        ignores ``linked`` on a hit anyway. Linking is the dominant per-instance
        cost during training, so bypassing it on warm-cache reads is a large win.
        """
        return self._cache.get(self._fingerprint, instance)

    def retrieve(
        self,
        linked: Sequence[LinkedEntity],
        *,
        top_k: int,
        instance: InstanceKey | None = None,
    ) -> SubHypergraph:
        """Return a cached sub-hypergraph or compute, store, and return it."""
        if instance is None:
            return self._inner.retrieve(linked, top_k=top_k, instance=instance)
        cached = self._cache.get(self._fingerprint, instance)
        if cached is not None:
            return cached
        sub = self._inner.retrieve(linked, top_k=top_k, instance=instance)
        self._cache.put(self._fingerprint, instance, sub)
        return sub
