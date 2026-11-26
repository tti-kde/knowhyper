"""Role-aware query builder for target-pair text knowledge baselines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ieforge.knowledge.core.base import IEKnowledgeQueryBuilder
from ieforge.knowledge.core.schema import Query
from ieforge.registry import register


def _read(example: Any, field: str) -> Any:
    if isinstance(example, Mapping):
        return example[field]
    return getattr(example, field)


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


@register("knowledge_query_builder", name="target_pair")
class TargetPairQueryBuilder(IEKnowledgeQueryBuilder):
    """Build a query with explicit head/tail ownership for every mention."""

    def __init__(
        self,
        field: str = "text",
        *,
        head_field: str = "head_mention",
        tail_field: str = "tail_mention",
        task: str | None = None,
        metadata_fields: list[str] | None = None,
        **_kwargs: object,
    ) -> None:
        # The base ``knowhyper`` knowledge config injects generic query-builder
        # keys (e.g. ``entities_fields``) that Hydra dict-merges into this override.
        # Accept and ignore them, matching ``BioREDQueryBuilder``.
        self._field = field
        self._head_field = head_field
        self._tail_field = tail_field
        self._task = task
        self._metadata_fields = list(metadata_fields or [])

    def build(self, example: Any) -> Query:
        """Keep flattened entities and an aligned role vector in query metadata."""
        head = _strings(_read(example, self._head_field))
        tail = _strings(_read(example, self._tail_field))
        metadata = {name: _read(example, name) for name in self._metadata_fields}
        metadata["target_entity_groups"] = {"head": head, "tail": tail}
        metadata["target_entity_roles"] = [
            *(["head"] * len(head)),
            *(["tail"] * len(tail)),
        ]
        return Query(
            text=str(_read(example, self._field)),
            entities=[*head, *tail],
            task=self._task,
            metadata=metadata,
        )
