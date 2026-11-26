"""Knowledge query builder for the BioRED pair task.

BioRED uses ``BioREDPairCollator``, which (unlike ``RECollator``) emits no
``_knowledge_examples``; the knowledge pipeline therefore runs on the raw
``BioREDPairExample``. That dataclass carries entity IDs/types as
``entities[head_idx]``/``entities[tail_idx]`` (``Entity.text`` is the
normalized DB id, ``Entity.label`` is the BioRED type string) rather than the
``head_mention``/``head_type`` fields the generic ``AttributeQueryBuilder``
reads. This builder bridges that gap so the linker resolves seeds by DB id.
"""

from __future__ import annotations

import re
from typing import Any

from ieforge.knowledge.core.base import IEKnowledgeQueryBuilder
from ieforge.knowledge.core.schema import Query
from ieforge.registry import register

_MESH_RE = re.compile(r"^(?:MESH:)?([DC]\d+)$")
_OMIM_RE = re.compile(r"^OMIM:\d+$")
_NCBIGENE_RE = re.compile(r"^\d+$")

# primekg gene/protein nodes are keyed `primekg:protein:<NCBIGene id>` (the node
# type segment is "protein", NOT "gene/protein"), so a bare numeric NCBIGene id
# links via ExactNodeId once formatted to this node_id.
_PROTEIN_NODE_PREFIX = "primekg:protein:"


def id_namespace(entity_id: str) -> tuple[str, str] | None:
    """Map a BioRED entity DB-id to its primehkg xref ``(namespace, id)``.

    BioRED disease/chemical ids are MeSH (``D*``/``C*``, optionally ``MESH:``-
    prefixed) and some OMIM (``OMIM:*``). primehkg's ``ExactXrefResolver`` keys on
    a lowercase namespace and a prefixed id (``mesh``/``MESH:D...``,
    ``omim``/``OMIM:...``). Numeric NCBI gene ids and variant strings have no xref
    namespace in primehkg, so they return ``None`` (the linker falls back to its
    existing type-hinted / fuzzy path).
    """
    if not entity_id:
        return None
    m = _MESH_RE.match(entity_id)
    if m:
        return ("mesh", "MESH:" + m.group(1))
    if _OMIM_RE.match(entity_id):
        return ("omim", entity_id)
    return None


def node_id_for(entity_id: str, entity_type: str | None) -> str | None:
    """Map a BioRED gene endpoint to its primekg protein ``node_id``.

    BioRED ``GeneOrGeneProduct`` endpoints arrive as bare numeric NCBIGene ids
    (e.g. ``"7157"``), and primekg stores gene/protein nodes as
    ``primekg:protein:<NCBIGene id>``. Formatting the id to that node_id lets
    primehkg's ``ExactNodeIdResolver`` link it directly; the caller leaves
    ``namespace`` ``None`` so the exact-node-id resolver fires and the xref
    resolver no-ops. The ``protein`` gate is also what ``node_type_of`` returns for
    these nodes, so ``strict_type_match`` accepts the exact-id hit.

    Returns ``None`` for anything that is not a numeric id under the ``protein``
    type — MeSH/OMIM ids, variant-string garbage, wrong types, and untyped
    mentions all fall back to the existing xref / fuzzy path.
    """
    if entity_type != "protein" or not entity_id:
        return None
    if not _NCBIGENE_RE.match(entity_id):
        return None
    return _PROTEIN_NODE_PREFIX + entity_id


@register("knowledge_query_builder", name="biored_attribute")
class BioREDQueryBuilder(IEKnowledgeQueryBuilder):
    """Build a knowledge ``Query`` from a raw BioRED pair example."""

    def __init__(
        self,
        field: str,
        *,
        task: str | None = None,
        metadata_fields: list[str] | None = None,
        include_target_roles: bool = False,
        surface_fallback: bool = False,
        **_kwargs: object,
    ) -> None:
        """Create the builder reading ``field`` for query text.

        ``surface_fallback`` (default False) additionally emits the entity's
        surface mentions, so an endpoint whose DB id is not in PrimeHKG can still
        link the way ChemProt and DDI do.

        Why this is not the default: it changes the seeds, hence the retrieval
        cache fingerprint, hence every recorded BioRED result. Off by default
        keeps the existing runs reproducible; turn it on in a new arm.
        """
        self._field = field
        self._task = task
        self._metadata_fields = list(metadata_fields or [])
        self._include_target_roles = include_target_roles
        self._surface_fallback = surface_fallback

    def build(self, example: Any) -> Query:
        """Build a query from the example's head/tail entities and text."""
        entities = example.entities
        n_needed = max(example.head_idx, example.tail_idx) + 1
        if len(entities) < n_needed:
            msg = (
                f"BioREDQueryBuilder: example {getattr(example, 'id', '<unknown>')!r} "
                f"has {len(entities)} entities but needs head_idx={example.head_idx}/"
                f"tail_idx={example.tail_idx}"
            )
            raise ValueError(msg)
        head = entities[example.head_idx]
        tail = entities[example.tail_idx]
        metadata: dict[str, Any] = {
            "head_type": head.label,
            "tail_type": tail.label,
        }
        if self._include_target_roles:
            metadata["target_entity_groups"] = {
                "head": [head.text],
                "tail": [tail.text],
            }
            metadata["target_entity_roles"] = ["head", "tail"]
        for name in self._metadata_fields:
            metadata[name] = getattr(example, name)
        text = getattr(example, self._field)
        entities = [head.text, tail.text]
        if self._surface_fallback:
            # Append each endpoint's surface mentions after its DB id. The id is
            # tried first, so an id that resolves keeps its exact hit; the surface
            # form only matters when it does not. Measured on the BioRED
            # validation split: 3,079 instances retrieve an EMPTY sub-hypergraph
            # under id-only linking, and 100% of them do carry usable surface
            # mentions ("iodide", "L-thyroxine", ...) -- the same strings
            # ChemProt and DDI link successfully.
            head_forms = [head.text, *(head.mention or [])]
            tail_forms = [tail.text, *(tail.mention or [])]
            entities = [*head_forms, *tail_forms]
            # The injector builds `types` as exactly [head_type, tail_type],
            # aligned POSITIONALLY with entities. Extra surface forms would slide
            # out of that alignment and lose their node-type hint (and with
            # strict_type_match on, an untyped mention links to the wrong type or
            # not at all). Publish the full aligned list for the injector to use.
            metadata["entity_types"] = [
                *([head.label] * len(head_forms)),
                *([tail.label] * len(tail_forms)),
            ]
            if self._include_target_roles:
                metadata["target_entity_groups"] = {
                    "head": head_forms,
                    "tail": tail_forms,
                }
        return Query(
            text=text,
            entities=entities,
            task=self._task,
            metadata=metadata,
        )
