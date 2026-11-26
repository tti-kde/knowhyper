from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from datasets import load_from_disk

if TYPE_CHECKING:
    from pathlib import Path

_log = logging.getLogger(__name__)


class PygHyperIncidence(TypedDict):
    """A minimal, deterministic hypergraph incidence representation.

    This is intended as a thin adapter layer for downstream learning code (e.g. pyg-hyper).
    It intentionally avoids importing external learning frameworks.
    """

    node_ids: list[str]
    hyperedge_ids: list[str]
    incidence: list[tuple[int, int]]
    roles: list[str | None]
    hyperedge_semantics: list[str | None]
    hyperedge_index: NotRequired[list[list[int]]]
    node_types: NotRequired[list[str | None]]
    attribute_feature_refs: NotRequired[list[dict[str, Any]]]


def build_pyg_hyper_incidence(
    hyperedges: list[dict[str, Any]],
    *,
    include_hyperedge_index: bool = False,
    extra_hyperedges: list[dict[str, Any]] | None = None,
    node_types: dict[str, str] | None = None,
) -> PygHyperIncidence:
    """Convert PrimeHKG hyperedges into a deterministic incidence list.

    - `node_ids`: sorted unique node ids across all hyperedges
    - `hyperedge_ids`: sorted unique he_id values
    - `incidence`: list of `(node_index, hyperedge_index)` tuples sorted by (hyperedge_index, node_index)
    - `roles`: parallel to `incidence`; role string if available, else None
    - `extra_hyperedges`: optional additional hyperedges (e.g. entity-attribute view HEs) merged with `hyperedges` before incidence construction; the caller's input list is not mutated. Note: an extra HE sharing an `he_id` with a core HE will override it.
    """
    if extra_hyperedges:
        hyperedges = [*hyperedges, *extra_hyperedges]
    node_set: set[str] = set()
    he_set: set[str] = set()
    he_nodes: dict[str, list[str]] = {}
    he_roles: dict[str, dict[str, str]] = {}
    he_semantics: dict[str, str | None] = {}

    for he in hyperedges:
        hid = he.get("he_id")
        node_ids_obj = he.get("node_ids")
        if not isinstance(hid, str) or not isinstance(node_ids_obj, list):
            continue
        nodes = [n for n in node_ids_obj if isinstance(n, str)]
        if not nodes:
            continue
        he_set.add(hid)
        he_nodes[hid] = nodes
        node_set.update(nodes)

        sem_obj = he.get("semantics")
        if isinstance(sem_obj, str) and sem_obj.strip():
            he_semantics[hid] = sem_obj
        elif hid not in he_semantics:
            he_semantics[hid] = None

        roles_obj = he.get("roles")
        if isinstance(roles_obj, dict):
            he_roles[hid] = {
                k: v
                for k, v in roles_obj.items()
                if isinstance(k, str) and isinstance(v, str)
            }

    node_ids = sorted(node_set)
    hyperedge_ids = sorted(he_set)
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    he_index = {hid: j for j, hid in enumerate(hyperedge_ids)}

    incidence: list[tuple[int, int]] = []
    roles: list[str | None] = []

    for hid in hyperedge_ids:
        j = he_index[hid]
        nodes = sorted(set(he_nodes.get(hid, [])))
        rmap = he_roles.get(hid, {})
        for nid in nodes:
            i = node_index[nid]
            incidence.append((i, j))
            roles.append(rmap.get(nid))

    out: PygHyperIncidence = {
        "node_ids": node_ids,
        "hyperedge_ids": hyperedge_ids,
        "incidence": incidence,
        "roles": roles,
        "hyperedge_semantics": [he_semantics.get(hid) for hid in hyperedge_ids],
    }

    if node_types is not None:
        out["node_types"] = [node_types.get(nid) for nid in node_ids]

    if include_hyperedge_index:
        # PyG-style COO for bipartite node-hyperedge incidence:
        # shape [2, num_connections], rows are (node_index, hyperedge_index).
        out["hyperedge_index"] = [
            [i for (i, _j) in incidence],
            [_j for (_i, _j) in incidence],
        ]

    return out


def build_pyg_hyper_incidence_from_dataset_dir(
    dataset_dir: str | Path,
    *,
    include_hyperedge_index: bool = False,
    entity_view_dir: str | Path | None = None,
) -> PygHyperIncidence:
    """Build incidence from a PrimeHKG DatasetDict saved via `save_to_disk`.

    This reads the `hyperedges` table, and tolerates JSON-encoded `roles` values that
    are produced by `primehkg.io.hf_dataset.write_hf_tables_to_disk`.
    """
    dd = load_from_disk(str(dataset_dir))
    he_ds = dd["hyperedges"]
    hyperedges: list[dict[str, Any]] = []
    for row in he_ds:
        he = dict(row)
        roles_obj = he.get("roles")
        if isinstance(roles_obj, str):
            try:
                decoded = json.loads(roles_obj)
            except json.JSONDecodeError:
                _log.warning(
                    "he_id=%s: roles is a non-JSON string; role labels dropped",
                    he.get("he_id"),
                )
                decoded = None
            if isinstance(decoded, dict):
                he["roles"] = decoded
            elif decoded is not None:
                _log.warning(
                    "he_id=%s: decoded roles is %s, not a dict; role labels dropped",
                    he.get("he_id"),
                    type(decoded).__name__,
                )
        hyperedges.append(he)
    extra: list[dict[str, Any]] | None = None
    node_types: dict[str, str] | None = None
    attr_rows: list[dict[str, Any]] = []
    if entity_view_dir is not None:
        extra = load_entity_view_hyperedges(entity_view_dir)
        attr_rows = load_entity_view_nodes(entity_view_dir)
        node_types = {}
        if "nodes" in dd:
            for row in dd["nodes"]:
                nid = row.get("node_id")
                ntype = row.get("node_type")
                if isinstance(nid, str) and isinstance(ntype, str):
                    node_types[nid] = ntype
        for arow in attr_rows:
            anid = arow.get("attribute_node_id")
            if not isinstance(anid, str):
                continue
            ntype = arow.get("node_type")
            node_types[anid] = (
                ntype if isinstance(ntype, str) and ntype else "attribute"
            )
    result = build_pyg_hyper_incidence(
        hyperedges,
        include_hyperedge_index=include_hyperedge_index,
        extra_hyperedges=extra,
        node_types=node_types,
    )
    if entity_view_dir is not None:
        result["attribute_feature_refs"] = _build_attribute_feature_refs(
            attr_rows, result["node_ids"]
        )
    return result


def load_entity_view_hyperedges(view_dir: str | Path) -> list[dict[str, Any]]:
    """Load ``entity_view_hyperedges`` rows from a sidecar DatasetDir.

    Decodes JSON-encoded ``roles`` the same way ``build_pyg_hyper_incidence_from_dataset_dir``
    does, so view HEs merge cleanly with core HEs.
    """
    dd = load_from_disk(str(view_dir))
    he_ds = dd["entity_view_hyperedges"]
    out: list[dict[str, Any]] = []
    for row in he_ds:
        he = dict(row)
        roles_obj = he.get("roles")
        if isinstance(roles_obj, str):
            try:
                decoded = json.loads(roles_obj)
            except json.JSONDecodeError:
                _log.warning(
                    "he_id=%s: roles is a non-JSON string; role labels dropped",
                    he.get("he_id"),
                )
                decoded = None
            if isinstance(decoded, dict):
                he["roles"] = decoded
            elif decoded is not None:
                _log.warning(
                    "he_id=%s: decoded roles is %s, not a dict; role labels dropped",
                    he.get("he_id"),
                    type(decoded).__name__,
                )
        out.append(he)
    return out


def load_entity_view_nodes(view_dir: str | Path) -> list[dict[str, Any]]:
    """Load ``entity_view_nodes`` rows (attribute nodes) from a sidecar DatasetDir."""
    dd = load_from_disk(str(view_dir))
    if "entity_view_nodes" not in dd:
        return []
    return [dict(row) for row in dd["entity_view_nodes"]]


def _build_attribute_feature_refs(
    attribute_nodes: list[dict[str, Any]],
    node_ids: list[str],
) -> list[dict[str, Any]]:
    """Build reference rows joining attribute nodes to ``node_features_embedding``.

    The embedding join key is ``(entity_node_id, source_modality)`` — NOT the attribute
    node id — because attribute nodes carry no payload and the embedding table is keyed by
    the entity ``node_id``. One attribute node may carry several refs (e.g. text with both
    ``definition`` and ``description``), so each ref yields its own row. ``node_index``
    indexes into the incidence ``node_ids`` array.
    """
    from primehkg.views.entity_view.modalities import embedding_source_modality

    node_index = {nid: i for i, nid in enumerate(node_ids)}
    refs: list[dict[str, Any]] = []
    for arow in attribute_nodes:
        anid = arow.get("attribute_node_id")
        entity_node_id = arow.get("entity_node_id")
        if not (isinstance(anid, str) and isinstance(entity_node_id, str)):
            continue
        feature_refs = arow.get("feature_refs")
        if isinstance(feature_refs, str):
            try:
                feature_refs = json.loads(feature_refs)
            except json.JSONDecodeError:
                _log.warning(
                    "attribute_node_id=%s: feature_refs is non-JSON; feature refs dropped",
                    anid,
                )
                continue
        if not isinstance(feature_refs, dict):
            continue
        for modality, ref_list in feature_refs.items():
            if not isinstance(modality, str) or not isinstance(ref_list, list):
                continue
            for ref in ref_list:
                if not isinstance(ref, dict):
                    continue
                refs.append(
                    {
                        "attribute_node_id": anid,
                        "node_index": node_index.get(anid),
                        "entity_node_id": entity_node_id,
                        "modality": modality,
                        "source_modality": embedding_source_modality(modality, ref),
                    }
                )
    refs.sort(key=lambda r: (r["attribute_node_id"], r["source_modality"]))
    return refs
