"""Post-retrieval sub-HG augmentation with primehkg entity_attribute hyperedges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from knowhyper.pipeline.types import HyperedgeRef, SubHypergraph

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True)
class EntityViewEntry:
    """Pre-indexed entity_attribute data for one anchor node."""

    attr_node_ids: tuple[str, ...]
    modalities: tuple[str, ...]
    he_id: str
    roles: dict[str, str]


class EntityViewAugmenter:
    """Attach entity_attribute HEs and private attribute nodes to a SubHypergraph."""

    def __init__(self, index: dict[str, EntityViewEntry]) -> None:
        self._index = index

    def augment(self, sub: SubHypergraph) -> SubHypergraph:
        """Return a new SubHypergraph with entity_attribute edges attached."""
        if not sub.node_ids:
            return sub

        extra_node_ids: list[str] = []
        extra_hes: list[HyperedgeRef] = []
        seen_anchors: set[str] = set()

        for nid in sub.node_ids:
            if nid in seen_anchors:
                continue
            entry = self._index.get(nid)
            if entry is None:
                continue
            seen_anchors.add(nid)
            extra_node_ids.extend(entry.attr_node_ids)
            all_he_node_ids = (nid, *entry.attr_node_ids)
            extra_hes.append(
                HyperedgeRef(
                    he_id=entry.he_id,
                    semantics="entity_attribute",
                    node_ids=all_he_node_ids,
                    roles=entry.roles,
                )
            )

        if not extra_hes:
            return sub

        # Deduplicate node_ids (attr nodes could overlap if same anchor listed twice)
        seen_nodes: set[str] = set()
        merged_nodes: list[str] = []
        for nid in [*sub.node_ids, *extra_node_ids]:
            if nid not in seen_nodes:
                seen_nodes.add(nid)
                merged_nodes.append(nid)

        return SubHypergraph(
            seed_node_ids=sub.seed_node_ids,
            node_ids=tuple(merged_nodes),
            hyperedges=(*sub.hyperedges, *extra_hes),
            provenance=sub.provenance,
        )


def load_core_nodes_iter(
    hub_repo_id: str, revision: str, expected_fingerprint: str
) -> list[dict[str, Any]]:
    """Load core nodes as a list for Hydra instantiation.

    Hydra can't instantiate a generator expression as a config leaf, so this
    wraps ``primehkg.load_core_dataset`` + the ``nodes`` table into a plain
    list that ``build_entity_view_index``'s ``nodes_iter`` param accepts.
    """
    from primehkg import load_core_dataset

    core = load_core_dataset(
        hub_repo_id=hub_repo_id,
        revision=revision,
        expected_fingerprint=expected_fingerprint,
    )
    nodes_ds = core.nodes if hasattr(core, "nodes") else core["nodes"]
    return [dict(row) for row in nodes_ds]


def build_entity_view_index(
    nodes_iter: Iterable[dict[str, Any]],
    *,
    source_fingerprint: str,
    skip_modalities: Sequence[str] | None = None,
) -> dict[str, EntityViewEntry]:
    """Build an in-memory anchor→EntityViewEntry index from primehkg node rows.

    ``skip_modalities`` drops attr nodes for the listed umbrella modalities
    (e.g. ``["name"]`` when the anchor already carries display_name).
    """
    from primehkg.views.entity_view import build_entity_view_tables

    tables = build_entity_view_tables(nodes_iter, source_fingerprint=source_fingerprint)
    ev_nodes = tables["entity_view_nodes"]
    ev_edges = tables["entity_view_hyperedges"]

    _skip = frozenset(skip_modalities) if skip_modalities is not None else frozenset()

    # Index attr nodes by anchor, filtering out skipped modalities
    anchor_attrs: dict[str, list[dict[str, Any]]] = {}
    for row in ev_nodes:
        if row["modality"] in _skip:
            continue
        anchor_attrs.setdefault(row["entity_node_id"], []).append(row)

    index: dict[str, EntityViewEntry] = {}
    for he_row in ev_edges:
        anchor_nid = he_row["node_ids"][0]  # first is always anchor
        attrs = anchor_attrs.get(anchor_nid, [])
        if not attrs:
            continue
        surviving_ids = {anchor_nid} | {a["attribute_node_id"] for a in attrs}
        index[anchor_nid] = EntityViewEntry(
            attr_node_ids=tuple(a["attribute_node_id"] for a in attrs),
            modalities=tuple(a["modality"] for a in attrs),
            he_id=he_row["he_id"],
            roles={k: v for k, v in he_row["roles"].items() if k in surviving_ids},
        )

    return index
