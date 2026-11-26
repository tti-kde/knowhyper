from __future__ import annotations

from typing import NotRequired, TypedDict


class GraphCsrMeta(TypedDict):
    """On-disk metadata shared by all graph CSR variants."""

    file_version: str
    inputs_fingerprint: str
    num_nodes: int
    num_hyperedges: int
    num_connections: int


class GraphCsrInMemory(GraphCsrMeta, total=False):
    """In-memory representation used for sidecar writing and queries."""

    node_vocab: NotRequired[list[str]]
    he_vocab: NotRequired[list[str]]
    node_ptr: NotRequired[list[int]]
    node_adj_he: NotRequired[list[int]]
    he_ptr: NotRequired[list[int]]
    he_adj_node: NotRequired[list[int]]
