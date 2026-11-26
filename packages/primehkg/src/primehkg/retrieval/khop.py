from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from primehkg.ids import normalize_node_id
from primehkg.retrieval.graph_csr.query import (
    he_neighbors_for_node,
    node_neighbors_for_he,
)
from primehkg.retrieval.sidecars import build_he_id_to_node_ids, build_node_to_he_ids

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from primehkg.retrieval.graph_csr.types import GraphCsrInMemory
    from primehkg.retrieval.truncate import Truncator


@dataclass(frozen=True)
class BudgetConfig:
    """Budget configuration for k-hop candidate expansion."""

    max_he_level1: int
    max_nodes_level2: int
    max_he_level3: int


def khop_determinism_version(
    *,
    budgets: BudgetConfig,
    truncator: Truncator,
    k: int,
    inputs_fingerprint: str | None,
) -> str:
    fp = inputs_fingerprint or ""
    truncator_version = getattr(truncator, "version", "")
    if not isinstance(truncator_version, str):
        truncator_version = str(truncator_version)
    payload = (
        "khop_v1\n"
        f"k={int(k)}\n"
        f"he1={int(budgets.max_he_level1)}\n"
        f"n2={int(budgets.max_nodes_level2)}\n"
        f"he2={int(budgets.max_he_level3)}\n"
        f"trunc={truncator_version}\n"
        f"fp={fp}"
    ).encode()
    return "khop:" + hashlib.sha256(payload).hexdigest()[:16]


def _norm_ids(ids: list[str]) -> list[str]:
    out: list[str] = []
    for x in ids:
        if not isinstance(x, str):
            continue
        nx = normalize_node_id(x)
        if not nx:
            continue
        out.append(nx)
    return sorted(set(out))


def khop_candidates_k2(
    *,
    seed_node_ids: list[str],
    hyperedges: Sequence[Mapping[str, object]],
    budgets: BudgetConfig,
    truncator: Truncator,
    query_id: str,
    graph_csr: GraphCsrInMemory | None = None,
) -> list[str]:
    seeds = _norm_ids(list(seed_node_ids))
    if graph_csr is not None:
        node_vocab = graph_csr["node_vocab"]
        he_vocab = graph_csr["he_vocab"]
        node_id_to_idx = {nid: i for i, nid in enumerate(node_vocab)}
        he_id_to_idx = {hid: j for j, hid in enumerate(he_vocab)}

        he1_set: set[str] = set()
        for s in seeds:
            node_idx = node_id_to_idx.get(s)
            if node_idx is None:
                continue
            for he_idx in he_neighbors_for_node(graph_csr, node_idx):
                he1_set.add(he_vocab[he_idx])
        he1 = truncator.truncate(
            sorted(he1_set),
            budget=int(budgets.max_he_level1),
            context={"query_id": query_id, "level": "he1"},
        )

        n2_set: set[str] = set()
        for hid in he1:
            he_idx = he_id_to_idx.get(hid)
            if he_idx is None:
                continue
            for node_idx in node_neighbors_for_he(graph_csr, he_idx):
                n2_set.add(node_vocab[node_idx])
        n2_set.difference_update(seeds)
        n2 = truncator.truncate(
            sorted(n2_set),
            budget=int(budgets.max_nodes_level2),
            context={"query_id": query_id, "level": "n2"},
        )

        he2_set: set[str] = set()
        for nid in n2:
            node_idx = node_id_to_idx.get(nid)
            if node_idx is None:
                continue
            for he_idx in he_neighbors_for_node(graph_csr, node_idx):
                he2_set.add(he_vocab[he_idx])
        return truncator.truncate(
            sorted(he2_set),
            budget=int(budgets.max_he_level3),
            context={"query_id": query_id, "level": "he2"},
        )

    n2h = build_node_to_he_ids(hyperedges)
    h2n = build_he_id_to_node_ids(hyperedges)

    he1_set: set[str] = set()
    for s in seeds:
        he1_set.update(n2h.get(s, []))
    he1 = truncator.truncate(
        sorted(he1_set),
        budget=int(budgets.max_he_level1),
        context={"query_id": query_id, "level": "he1"},
    )

    n2_set: set[str] = set()
    for hid in he1:
        n2_set.update(h2n.get(hid, []))
    # Prevent a k=2 expansion from "snapping back" to all seed-incident edges.
    # This keeps budgets meaningful: seed -> HE1 -> N2(non-seed) -> HE2.
    n2_set.difference_update(seeds)
    n2 = truncator.truncate(
        sorted(n2_set),
        budget=int(budgets.max_nodes_level2),
        context={"query_id": query_id, "level": "n2"},
    )

    he2_set: set[str] = set()
    for nid in n2:
        he2_set.update(n2h.get(nid, []))
    return truncator.truncate(
        sorted(he2_set),
        budget=int(budgets.max_he_level3),
        context={"query_id": query_id, "level": "he2"},
    )


class Subhypergraph(TypedDict):
    """Materialized subhypergraph slice."""

    node_ids: list[str]
    hyperedge_ids: list[str]
    hyperedges: list[dict[str, object]]


def materialize_subhypergraph(
    *,
    candidate_hyperedge_ids: list[str],
    hyperedges: Sequence[Mapping[str, object]] | None = None,
    hyperedge_lookup: Callable[[str], Mapping[str, object] | None] | None = None,
) -> Subhypergraph:
    if hyperedges is None and hyperedge_lookup is None:
        msg = "materialize_subhypergraph requires hyperedges or hyperedge_lookup"
        raise ValueError(msg)

    want = sorted({x for x in candidate_hyperedge_ids if isinstance(x, str) and x})

    by_id: dict[str, Mapping[str, object]] = {}
    if hyperedges is not None:
        for he in hyperedges:
            hid = he.get("he_id")
            if isinstance(hid, str) and hid:
                by_id[hid] = he

    chosen_hyperedge_ids: list[str] = []
    chosen_hyperedges: list[dict[str, object]] = []
    node_set: set[str] = set()

    for hid in want:
        he = hyperedge_lookup(hid) if hyperedge_lookup is not None else by_id.get(hid)
        if he is None:
            continue

        node_ids_obj = he.get("node_ids")
        if isinstance(node_ids_obj, list):
            node_ids = [n for n in node_ids_obj if isinstance(n, str) and n]
        else:
            node_ids = []

        node_set.update(node_ids)
        chosen_hyperedge_ids.append(hid)
        chosen_hyperedges.append({"he_id": hid, "node_ids": node_ids})

    return {
        "node_ids": sorted(node_set),
        "hyperedge_ids": chosen_hyperedge_ids,
        "hyperedges": chosen_hyperedges,
    }
