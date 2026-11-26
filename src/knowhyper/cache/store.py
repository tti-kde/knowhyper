"""Persistence for retrieved sub-hypergraphs.

Defines the ``SubHGCache`` interface and a first JSON-on-disk backend, plus the
``SubHypergraph`` <-> dict (de)serialization used by every backend. Storage is
behind the interface so a columnar/embedded backend can replace the JSON one
without touching the caching decorator or injector.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from knowhyper.cache.keys import to_plain
from knowhyper.pipeline.types import HyperedgeRef, SubHypergraph

if TYPE_CHECKING:
    from knowhyper.cache.keys import InstanceKey

logger = logging.getLogger(__name__)


def subhypergraph_to_dict(sub: SubHypergraph) -> dict[str, Any]:
    """Serialize a SubHypergraph to a JSON-friendly dict."""
    return {
        "seed_node_ids": list(sub.seed_node_ids),
        "node_ids": list(sub.node_ids),
        "hyperedges": [
            {
                "he_id": he.he_id,
                "semantics": he.semantics,
                "node_ids": list(he.node_ids),
                "roles": dict(he.roles),
                "polarity": he.polarity,
                "score": he.score,
                "score_breakdown": dict(he.score_breakdown),
            }
            for he in sub.hyperedges
        ],
        "provenance": dict(sub.provenance),
    }


def subhypergraph_from_dict(data: dict[str, Any]) -> SubHypergraph:
    """Reconstruct a SubHypergraph from a dict produced by ``subhypergraph_to_dict``."""
    hyperedges = tuple(
        HyperedgeRef(
            he_id=he["he_id"],
            semantics=he["semantics"],
            node_ids=tuple(he["node_ids"]),
            roles=dict(he["roles"]),
            polarity=he.get("polarity", "unknown"),
            score=float(he["score"]),
            score_breakdown=dict(he["score_breakdown"]),
        )
        for he in data["hyperedges"]
    )
    return SubHypergraph(
        seed_node_ids=tuple(data["seed_node_ids"]),
        node_ids=tuple(data["node_ids"]),
        hyperedges=hyperedges,
        provenance=dict(data["provenance"]),
    )


@runtime_checkable
class SubHGCache(Protocol):
    """Persist and load retrieved sub-hypergraphs by ``(fingerprint, instance)``."""

    def get(self, fingerprint: str, instance: InstanceKey) -> SubHypergraph | None:
        """Return the cached sub-hypergraph, or None on miss."""
        ...

    def put(self, fingerprint: str, instance: InstanceKey, sub: SubHypergraph) -> None:
        """Store the sub-hypergraph for ``(fingerprint, instance)``."""
        ...

    def write_manifest(
        self,
        fingerprint: str,
        resolved_profile: dict[str, Any],
        *,
        primehkg_rev: str,
        cache_identity: dict[str, Any] | None = None,
    ) -> None:
        """Record the resolved profile + primehkg rev for a fingerprint dir."""
        ...


class JsonSubHGCache:
    """JSON-on-disk SubHGCache.

    Layout: ``root/<fingerprint>/<dataset>/<split>/<instance_id>.json`` plus
    ``root/<fingerprint>/manifest.json``. Writes are atomic (temp file + replace);
    unreadable/corrupt entries are treated as misses.
    """

    def __init__(self, root: Path | str) -> None:
        """Build the cache rooted at ``root`` (created lazily on write)."""
        self._root = Path(root)

    def _path(self, fingerprint: str, instance: InstanceKey) -> Path:
        return (
            self._root
            / fingerprint
            / instance.dataset
            / instance.split
            / f"{instance.instance_id}.json"
        )

    def get(self, fingerprint: str, instance: InstanceKey) -> SubHypergraph | None:
        """Return the cached sub-hypergraph, or None on miss/corruption."""
        path = self._path(fingerprint, instance)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return subhypergraph_from_dict(data)
        # JSONDecodeError: not valid JSON; KeyError: missing field; ValueError/
        # TypeError: present-but-wrong-typed field (e.g. non-numeric score);
        # OSError: unreadable file. All degrade to a miss + recompute.
        except (json.JSONDecodeError, KeyError, ValueError, TypeError, OSError):
            logger.warning("Corrupt sub-hypergraph cache entry, recomputing: %s", path)
            return None

    def put(self, fingerprint: str, instance: InstanceKey, sub: SubHypergraph) -> None:
        """Atomically store the sub-hypergraph for ``(fingerprint, instance)``."""
        path = self._path(fingerprint, instance)
        _atomic_write_json(path, subhypergraph_to_dict(sub))

    def write_manifest(
        self,
        fingerprint: str,
        resolved_profile: dict[str, Any],
        *,
        primehkg_rev: str,
        cache_identity: dict[str, Any] | None = None,
    ) -> None:
        """Write ``manifest.json`` recording the resolved profile + primehkg rev."""
        path = self._root / fingerprint / "manifest.json"
        manifest = {"resolved_profile": resolved_profile, "primehkg_rev": primehkg_rev}
        if cache_identity is not None:
            manifest["cache_identity"] = cache_identity
        _atomic_write_json(path, manifest, sort_keys=True)


def _atomic_write_json(path: Path, payload: Any, *, sort_keys: bool = False) -> None:
    """Write ``payload`` as JSON to ``path`` atomically and concurrency-safely.

    Writes to a process+uuid-unique temp file in the SAME directory (so the final
    ``os.replace`` is atomic on the same filesystem), then renames. The unique temp
    name keeps concurrent writers of the same ``path`` (e.g. multiple dataloader
    workers) from clobbering each other's in-flight temp file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    # Coerce OmegaConf containers (e.g. a Hydra-instantiated profile's dict knob)
    # to plain Python so the manifest is JSON-serializable and stable.
    tmp.write_text(json.dumps(to_plain(payload), sort_keys=sort_keys), encoding="utf-8")
    tmp.replace(path)
