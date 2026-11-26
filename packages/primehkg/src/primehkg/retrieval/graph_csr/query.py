from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from primehkg.retrieval.graph_csr.types import GraphCsrInMemory


def he_neighbors_for_node(g: GraphCsrInMemory, node_idx: int) -> list[int]:
    a = int(g["node_ptr"][node_idx])
    b = int(g["node_ptr"][node_idx + 1])
    return [int(x) for x in g["node_adj_he"][a:b]]


def node_neighbors_for_he(g: GraphCsrInMemory, he_idx: int) -> list[int]:
    a = int(g["he_ptr"][he_idx])
    b = int(g["he_ptr"][he_idx + 1])
    return [int(x) for x in g["he_adj_node"][a:b]]
