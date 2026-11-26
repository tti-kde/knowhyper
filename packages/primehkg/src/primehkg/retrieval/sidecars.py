from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from primehkg.ids import normalize_node_id

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def _nodes_from_he(he: Mapping[str, object]) -> list[str]:
    node_ids_obj = he.get("node_ids")
    if not isinstance(node_ids_obj, list):
        return []

    nodes: set[str] = set()
    for n in node_ids_obj:
        if not isinstance(n, str):
            continue
        nn = normalize_node_id(n)
        if not nn:
            continue
        nodes.add(nn)

    return sorted(nodes)


def build_node_to_he_ids(
    hyperedges: Iterable[Mapping[str, object]],
) -> dict[str, list[str]]:
    idx: dict[str, set[str]] = defaultdict(set)
    for he in hyperedges:
        hid = he.get("he_id")
        if not isinstance(hid, str):
            continue
        for nid in _nodes_from_he(he):
            idx[nid].add(hid)

    return {nid: sorted(hids) for nid, hids in sorted(idx.items())}


def build_he_id_to_node_ids(
    hyperedges: Iterable[Mapping[str, object]],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for he in hyperedges:
        hid = he.get("he_id")
        if not isinstance(hid, str):
            continue

        out[hid] = _nodes_from_he(he)

    return {hid: out[hid] for hid in sorted(out)}
