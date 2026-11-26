"""Helpers for preparing knowledge configs before Hydra instantiation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def strip_names(node: Any) -> Any:
    """Recursively drop ``name`` registry-reference keys from a resolved config tree.

    ``name`` is registry metadata (it matches the ``@register`` name), not a
    constructor argument, so it must be removed before ``hydra.utils.instantiate``
    recurses over the knowledge component tree. Operates on plain resolved
    objects (from ``OmegaConf.to_container(..., resolve=True)``) and returns new
    objects without mutating the input.
    """
    if isinstance(node, Mapping):
        return {key: strip_names(value) for key, value in node.items() if key != "name"}
    if isinstance(node, list):
        return [strip_names(item) for item in node]
    return node
