"""Text-channel knowledge injection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from typing import TYPE_CHECKING, Any

from ieforge.knowledge.core.base import IEKnowledgeChannel
from ieforge.registry import register

if TYPE_CHECKING:
    from ieforge.knowledge.core.schema import KnowledgeContext


@register("knowledge_channel", name="text")
class TextChannel(IEKnowledgeChannel):
    """Prepends text knowledge context to a target example field."""

    def __init__(
        self,
        target_field: str,
        *,
        separator: str = "\n\n",
        mode: str = "prepend",
    ) -> None:
        """Create a text channel for a dataclass or mapping field."""
        if mode != "prepend":
            msg = f"unsupported TextChannel mode {mode!r}; only 'prepend' is supported"
            raise ValueError(msg)
        self._target_field = target_field
        self._separator = separator

    @property
    def target_field(self) -> str:
        """Example field that receives the text context."""
        return self._target_field

    def apply(self, example: Any, context: KnowledgeContext) -> Any:
        """Return a copied example with text context prepended."""
        if not context.text_context:
            return example
        old_value = _read_target(example, self._target_field)
        if not isinstance(old_value, str):
            msg = (
                f"TextChannel target field {self._target_field!r} must be str, "
                f"got {type(old_value).__name__}"
            )
            raise TypeError(msg)
        new_value = f"{context.text_context}{self._separator}{old_value}"
        return _copy_with(example, self._target_field, new_value)


def _read_target(example: Any, field: str) -> Any:
    if isinstance(example, Mapping):
        if field not in example:
            msg = f"TextChannel target field {field!r} missing from mapping example"
            raise AttributeError(msg)
        return example[field]
    if not hasattr(example, field):
        msg = (
            f"TextChannel target field {field!r} missing from {type(example).__name__}"
        )
        raise AttributeError(msg)
    return getattr(example, field)


def _copy_with(example: Any, field: str, value: str) -> Any:
    if isinstance(example, Mapping):
        copied = dict(example)
        copied[field] = value
        return copied
    if is_dataclass(example):
        return replace(example, **{field: value})
    msg = (
        f"TextChannel can only copy dataclass or mapping examples, got "
        f"{type(example).__name__}"
    )
    raise TypeError(msg)
