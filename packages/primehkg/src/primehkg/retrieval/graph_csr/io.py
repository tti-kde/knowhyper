from __future__ import annotations

import datetime
import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from primehkg.retrieval.graph_csr.types import GraphCsrInMemory

GRAPH_CSR_FILE_VERSION = "graph_csr_v1"


class GraphCsrMetaJsonError(TypeError):
    """Raised when `meta.json` has an unexpected shape."""

    def __init__(self) -> None:
        """Create the exception with a fixed message."""
        super().__init__("meta.json must be an object")


def write_meta_json(
    out_dir: str | Path,
    *,
    inputs_fingerprint: str,
    num_nodes: int,
    num_hyperedges: int,
    num_connections: int,
) -> None:
    p = Path(out_dir)
    payload: dict[str, Any] = {
        "file_version": GRAPH_CSR_FILE_VERSION,
        "inputs_fingerprint": str(inputs_fingerprint),
        "num_nodes": int(num_nodes),
        "num_hyperedges": int(num_hyperedges),
        "num_connections": int(num_connections),
        "dtype": {"ptr": "uint64", "adj": "uint32"},
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    (p / "meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_vocab_txt(path: Path, vocab: list[str]) -> None:
    path.write_text("".join(f"{x}\n" for x in vocab), encoding="utf-8")


def read_vocab_txt(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if not text:
        return []
    return [line for line in text.splitlines() if line]


def read_meta_json(out_dir: str | Path) -> dict[str, Any]:
    p = Path(out_dir) / "meta.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise GraphCsrMetaJsonError
    return raw


def write_u64_file(path: Path, arr: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(arr)}Q", *[int(x) for x in arr]))


def write_u32_file(path: Path, arr: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(arr)}I", *[int(x) for x in arr]))


def write_graph_csr_sidecar(out_dir: str | Path, g: GraphCsrInMemory) -> None:
    p = Path(out_dir)
    write_vocab_txt(p / "node_vocab.txt", [str(x) for x in g.get("node_vocab", [])])
    write_vocab_txt(p / "he_vocab.txt", [str(x) for x in g.get("he_vocab", [])])

    write_u64_file(p / "node_ptr.u64", [int(x) for x in g["node_ptr"]])
    write_u32_file(
        p / "node_adj_he.u32",
        [int(x) for x in g["node_adj_he"]],
    )
    write_u64_file(p / "he_ptr.u64", [int(x) for x in g["he_ptr"]])
    write_u32_file(
        p / "he_adj_node.u32",
        [int(x) for x in g["he_adj_node"]],
    )

    write_meta_json(
        p,
        inputs_fingerprint=str(g["inputs_fingerprint"]),
        num_nodes=g["num_nodes"],
        num_hyperedges=g["num_hyperedges"],
        num_connections=g["num_connections"],
    )


def load_graph_csr_sidecar(
    out_dir: str | Path,
    *,
    expected_fingerprint: str | None = None,
    strict_fingerprint: bool = True,
) -> GraphCsrInMemory:
    """Load a mmap-backed CSR sidecar for k-hop queries."""
    from primehkg.retrieval.graph_csr.mmap import mmap_u32_array, mmap_u64_array
    from primehkg.retrieval.graph_csr.verify import require_matching_fingerprint

    p = Path(out_dir)
    meta = read_meta_json(p)
    fp = str(meta.get("inputs_fingerprint", ""))
    if expected_fingerprint is not None or strict_fingerprint:
        require_matching_fingerprint(
            loaded_fingerprint=fp,
            expected_fingerprint=expected_fingerprint,
            strict=strict_fingerprint,
        )
    loaded: GraphCsrInMemory = cast(
        "GraphCsrInMemory",
        {
            "file_version": str(meta.get("file_version", GRAPH_CSR_FILE_VERSION)),
            "inputs_fingerprint": fp,
            "num_nodes": int(meta["num_nodes"]),
            "num_hyperedges": int(meta["num_hyperedges"]),
            "num_connections": int(meta["num_connections"]),
            "node_vocab": read_vocab_txt(p / "node_vocab.txt"),
            "he_vocab": read_vocab_txt(p / "he_vocab.txt"),
            "node_ptr": mmap_u64_array(p / "node_ptr.u64"),
            "node_adj_he": mmap_u32_array(p / "node_adj_he.u32"),
            "he_ptr": mmap_u64_array(p / "he_ptr.u64"),
            "he_adj_node": mmap_u32_array(p / "he_adj_node.u32"),
        },
    )
    return loaded
