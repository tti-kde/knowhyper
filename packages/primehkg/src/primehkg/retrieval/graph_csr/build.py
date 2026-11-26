from __future__ import annotations

from typing import TYPE_CHECKING

from primehkg.ids import normalize_node_id
from primehkg.retrieval.graph_csr.io import GRAPH_CSR_FILE_VERSION

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from primehkg.retrieval.graph_csr.types import GraphCsrInMemory


def build_vocab_from_hyperedges(
    hyperedges: Iterable[Mapping[str, object]],
) -> tuple[list[str], list[str]]:
    nodes: set[str] = set()
    hes: set[str] = set()
    for he in hyperedges:
        hid = he.get("he_id")
        if isinstance(hid, str) and hid:
            hes.add(hid)
        nobj = he.get("node_ids")
        if isinstance(nobj, list):
            for n in nobj:
                if not isinstance(n, str):
                    continue
                nn = normalize_node_id(n)
                if nn:
                    nodes.add(nn)
    return sorted(nodes), sorted(hes)


def build_graph_csr_from_hyperedges(
    hyperedges: Iterable[Mapping[str, object]],
    *,
    inputs_fingerprint: str,
) -> GraphCsrInMemory:
    node_vocab, he_vocab = build_vocab_from_hyperedges(hyperedges)
    node_index = {nid: i for i, nid in enumerate(node_vocab)}
    he_index = {hid: j for j, hid in enumerate(he_vocab)}

    # We need a second pass; keep this toy implementation simple by materializing
    # unique node<->hyperedge connections as index pairs.
    pairs: list[tuple[int, int]] = []
    for he in hyperedges:
        hid = he.get("he_id")
        if not isinstance(hid, str) or hid not in he_index:
            continue
        he_idx = he_index[hid]

        nobj = he.get("node_ids")
        if not isinstance(nobj, list):
            continue

        node_idxs: set[int] = set()
        for n in nobj:
            if not isinstance(n, str):
                continue
            nn = normalize_node_id(n)
            if nn and nn in node_index:
                node_idxs.add(node_index[nn])

        pairs.extend((node_idx, he_idx) for node_idx in sorted(node_idxs))

    num_nodes = len(node_vocab)
    num_hes = len(he_vocab)

    deg_n = [0] * num_nodes
    deg_h = [0] * num_hes
    for node_idx, he_idx in pairs:
        deg_n[node_idx] += 1
        deg_h[he_idx] += 1

    node_ptr = [0] * (num_nodes + 1)
    he_ptr = [0] * (num_hes + 1)
    for i in range(num_nodes):
        node_ptr[i + 1] = node_ptr[i] + deg_n[i]
    for j in range(num_hes):
        he_ptr[j + 1] = he_ptr[j] + deg_h[j]

    node_adj_he = [0] * node_ptr[-1]
    he_adj_node = [0] * he_ptr[-1]

    cur_n = node_ptr[:-1].copy()
    cur_h = he_ptr[:-1].copy()
    for node_idx, he_idx in pairs:
        node_adj_he[cur_n[node_idx]] = he_idx
        cur_n[node_idx] += 1

        he_adj_node[cur_h[he_idx]] = node_idx
        cur_h[he_idx] += 1

    # Sort adjacency per row for determinism.
    for i in range(num_nodes):
        a, b = node_ptr[i], node_ptr[i + 1]
        node_adj_he[a:b] = sorted(node_adj_he[a:b])
    for j in range(num_hes):
        a, b = he_ptr[j], he_ptr[j + 1]
        he_adj_node[a:b] = sorted(he_adj_node[a:b])

    return {
        "file_version": GRAPH_CSR_FILE_VERSION,
        "inputs_fingerprint": str(inputs_fingerprint),
        "num_nodes": num_nodes,
        "num_hyperedges": num_hes,
        "num_connections": len(pairs),
        "node_vocab": node_vocab,
        "he_vocab": he_vocab,
        "node_ptr": node_ptr,
        "node_adj_he": node_adj_he,
        "he_ptr": he_ptr,
        "he_adj_node": he_adj_node,
    }
