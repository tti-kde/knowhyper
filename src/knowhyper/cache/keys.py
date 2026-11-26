"""Cache identity: method fingerprint and per-instance keys.

The fingerprint captures the retrieval METHOD, cache/semantics versions, and the
concrete PrimeHKG build so that meaningful changes force a recompute while
unrelated code edits do not. The instance key captures WHICH
dataset example a sub-hypergraph belongs to.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


# Bump these only when the persisted JSON contract or retrieval semantics change.
# They deliberately avoid using the git commit: unrelated source edits should not
# invalidate a large shared cache, while meaningful cache changes must do so.
SUBHG_CACHE_SCHEMA_VERSION = 2
RETRIEVAL_SEMANTICS_VERSION = 1


def to_plain(value: Any) -> Any:
    """Recursively coerce a value to plain JSON-serializable Python.

    Hydra/OmegaConf deliver config values as ``DictConfig``/``ListConfig``, which
    are not JSON-serializable and would otherwise be ``str()``-repr'd by a lenient
    encoder — making any fingerprint or manifest non-deterministic and
    representation-dependent. Resolving them to plain ``dict``/``list`` here keeps
    a profile's fingerprint identical whether its knobs came from Hydra or a plain
    dict. Falls back to ``OmegaConf.to_container`` when available, else duck-types
    mappings/sequences.
    """
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


def method_identity(
    resolved_profile: Mapping[str, object],
    *,
    primehkg_rev: str,
    dataset_fingerprint: str | None = None,
    cache_schema_version: int = SUBHG_CACHE_SCHEMA_VERSION,
    retrieval_semantics_version: int = RETRIEVAL_SEMANTICS_VERSION,
) -> dict[str, Any]:
    """Return the complete, JSON-serializable identity of a SubHG method.

    The dataset fingerprint is intentionally distinct from primehkg_rev. The
    former identifies the concrete published build; the latter identifies the
    source/release pin. Keeping both prevents a Hydra override that changes only
    the dataset build from silently reading entries produced from another build.
    """
    return {
        "cache_schema_version": int(cache_schema_version),
        "retrieval_semantics_version": int(retrieval_semantics_version),
        "profile": to_plain(dict(sorted(resolved_profile.items()))),
        "primehkg_rev": primehkg_rev,
        "dataset_fingerprint": dataset_fingerprint,
    }


def method_fingerprint(
    resolved_profile: Mapping[str, object],
    *,
    primehkg_rev: str,
    dataset_fingerprint: str | None = None,
    cache_schema_version: int = SUBHG_CACHE_SCHEMA_VERSION,
    retrieval_semantics_version: int = RETRIEVAL_SEMANTICS_VERSION,
) -> str:
    """Return a short, stable hex fingerprint of the retrieval method.

    ``primehkg_rev`` is the pin string from ``pyproject.toml`` (e.g. ``"v0.10.2"``),
    threaded in via config. It MUST NOT be sourced from ``primehkg.__version__``,
    whose upstream value is stale and unreliable.

    The profile is coerced to plain Python (see :func:`to_plain`) and serialized
    without a lenient ``default=`` fallback, so a non-serializable knob fails loudly
    rather than silently producing a representation-dependent hash.
    """
    payload = method_identity(
        resolved_profile,
        primehkg_rev=primehkg_rev,
        dataset_fingerprint=dataset_fingerprint,
        cache_schema_version=cache_schema_version,
        retrieval_semantics_version=retrieval_semantics_version,
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class InstanceKey:
    """Identifies one dataset instance for cache addressing."""

    dataset: str
    split: str
    instance_id: str

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, object],
        *,
        fields: Sequence[str],
        fallback_text: str,
    ) -> InstanceKey:
        """Build a key from ``metadata`` using ``(dataset, split, id)`` field names.

        ``fields`` is the triple of metadata key names for dataset, split, and
        instance id. When the id field is absent, a deterministic content hash of
        ``fallback_text`` is used (prefixed ``h:``) so caching still works, keyed
        by content rather than a dataset id.
        """
        dataset_field, split_field, id_field = fields
        dataset = str(metadata.get(dataset_field, "unknown"))
        split = str(metadata.get(split_field, "unknown"))
        raw_id = metadata.get(id_field)
        if raw_id is None:
            digest = hashlib.sha256(fallback_text.encode("utf-8")).hexdigest()[:16]
            instance_id = f"h:{digest}"
        else:
            instance_id = str(raw_id)
        return cls(dataset=dataset, split=split, instance_id=instance_id)
