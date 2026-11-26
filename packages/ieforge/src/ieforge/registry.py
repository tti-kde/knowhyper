"""Central component registry for IEForge.

Allows registering loss, model, metric, task, dataset, and knowledge
component classes by short name, then looking them up by name at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import builtins
    from collections.abc import Callable

__all__ = [
    "KINDS",
    "ComponentRegistry",
    "get",
    "list_components",
    "load_entrypoints",
    "register",
]

KINDS = frozenset(
    {
        "dataset",
        "knowledge_builder",
        "knowledge_channel",
        "knowledge_collator",
        "knowledge_embedder",
        "knowledge_injector",
        "knowledge_query_builder",
        "knowledge_retriever",
        "knowledge_source",
        "loss",
        "metric",
        "model",
        "task",
    }
)


class ComponentRegistry:
    """Registry mapping (kind, name) -> class."""

    def __init__(self) -> None:
        """Initialise empty stores for each kind."""
        self._store: dict[str, dict[str, type]] = {k: {} for k in KINDS}

    def register(self, kind: str, name: str | None = None) -> Callable:
        """Return a decorator that registers the decorated class under (kind, name)."""
        if kind not in KINDS:
            msg = f"unknown kind {kind!r}. Must be one of {sorted(KINDS)}"
            raise ValueError(msg)

        def decorator(cls: type) -> type:
            key = name if name is not None else cls.__name__.lower()
            self._store[kind][key] = cls
            return cls

        return decorator

    def get(self, kind: str, name: str) -> type:
        """Return the class registered under (kind, name).

        Raises KeyError if the name is not registered.
        """
        if kind not in KINDS:
            msg = f"unknown kind {kind!r}"
            raise ValueError(msg)
        store = self._store[kind]
        if name not in store:
            available = sorted(store)
            msg = f"{name!r} not found in {kind!r} registry. Available: {available}"
            raise KeyError(msg)
        return store[name]

    def list(self, kind: str) -> builtins.list[str]:
        """Return a sorted list of registered names for the given kind."""
        if kind not in KINDS:
            msg = f"unknown kind {kind!r}"
            raise ValueError(msg)
        return sorted(self._store[kind])

    def load_entrypoints(self) -> None:
        """Register components declared via package entry points.

        Each installed package can declare components in its pyproject.toml:

            [project.entry-points."ieforge.losses"]
            focal = "my_pkg.losses:FocalLoss"

        Each kind (e.g. "loss") is pluralized to form the entry-point group
        name (e.g. "ieforge.losses"). Broken entry points are skipped with a
        warning so one bad plugin cannot prevent others from loading.
        """
        import logging
        from importlib.metadata import entry_points

        log = logging.getLogger(__name__)
        for kind in KINDS:
            group = f"ieforge.{kind}s"
            for ep in entry_points(group=group):
                try:
                    cls = ep.load()
                except Exception:
                    log.warning(
                        "Failed to load entry point %r from group %r", ep.name, group
                    )
                else:
                    self._store[kind][ep.name] = cls


_REGISTRY = ComponentRegistry()
register = _REGISTRY.register
get = _REGISTRY.get
list_components = _REGISTRY.list
load_entrypoints = _REGISTRY.load_entrypoints
