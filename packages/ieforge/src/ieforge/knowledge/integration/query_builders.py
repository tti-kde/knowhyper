"""Generic config-driven query builders for knowledge retrieval."""

from __future__ import annotations

import string
from collections.abc import Mapping
from typing import Any

from ieforge.knowledge.core.base import IEKnowledgeQueryBuilder
from ieforge.knowledge.core.schema import Query
from ieforge.registry import register


@register("knowledge_query_builder", name="attribute")
class AttributeQueryBuilder(IEKnowledgeQueryBuilder):
    """Builds a query from one field on an example."""

    def __init__(
        self,
        field: str,
        *,
        task: str | None = None,
        entities_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
    ) -> None:
        """Create a field-based query builder."""
        self._field = field
        self._task = task
        self._entities_fields = list(entities_fields or [])
        self._metadata_fields = list(metadata_fields or [])

    def build(self, example: Any) -> Query:
        """Build a query from an example field."""
        text = _require_str(_read_field(example, self._field), self._field)
        return Query(
            text=text,
            entities=_read_entities(example, self._entities_fields),
            task=self._task,
            metadata=_read_metadata(example, self._metadata_fields),
        )


@register("knowledge_query_builder", name="template")
class TemplateQueryBuilder(IEKnowledgeQueryBuilder):
    """Builds a query by formatting fields through a template."""

    def __init__(
        self,
        template: str,
        *,
        task: str | None = None,
        entities_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
    ) -> None:
        """Create a template-based query builder."""
        self._template = template
        self._task = task
        self._entities_fields = list(entities_fields or [])
        self._metadata_fields = list(metadata_fields or [])

    def build(self, example: Any) -> Query:
        """Build a query by formatting example fields."""
        values = {
            field: _read_field(example, field)
            for field in _template_fields(self._template)
        }
        text = _require_str(self._template.format(**values), self._template)
        return Query(
            text=text,
            entities=_read_entities(example, self._entities_fields),
            task=self._task,
            metadata=_read_metadata(example, self._metadata_fields),
        )


def _template_fields(template: str) -> list[str]:
    fields: list[str] = []
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name is not None and field_name:
            fields.append(field_name)
    return fields


def _read_field(example: Any, field: str) -> Any:
    if isinstance(example, Mapping):
        if field not in example:
            msg = f"query builder field {field!r} missing from mapping example"
            raise AttributeError(msg)
        return example[field]
    if not hasattr(example, field):
        msg = f"query builder field {field!r} missing from {type(example).__name__}"
        raise AttributeError(msg)
    return getattr(example, field)


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        msg = (
            f"query builder field {field!r} must resolve to str, "
            f"got {type(value).__name__}"
        )
        raise TypeError(msg)
    return value


def _read_entities(example: Any, fields: list[str]) -> list[str]:
    entities: list[str] = []
    for field in fields:
        value = _read_field(example, field)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            entities.extend(str(item) for item in value if item is not None)
        else:
            entities.append(_require_str(value, field))
    return entities


def _read_metadata(example: Any, fields: list[str]) -> dict[str, Any]:
    return {field: _read_field(example, field) for field in fields}
