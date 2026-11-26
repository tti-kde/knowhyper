"""``CoreDataset`` handle and ``load_core_dataset`` entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from datasets import Dataset, load_from_disk

from primehkg.core.exceptions import CoreDatasetLoadError, FingerprintMismatchError
from primehkg.core.manifest import (
    BuildManifest,
    is_enriched,
    nodes_have_linking_fields,
    read_build_manifest,
)
from primehkg.retrieval.he_row_index import build_he_row_index_from_dataset
from primehkg.retrieval.he_row_index_io import load_he_row_index_json_validated
from primehkg.retrieval.index import build_inverted_index_from_dataset
from primehkg.retrieval.index_io import load_inverted_index_json_validated

if TYPE_CHECKING:
    from primehkg.linking.chemical_xref_index import ChemicalXrefIndex
    from primehkg.linking.embedding_index import LabelEmbeddingIndex
    from primehkg.linking.xref_index import XrefIndex
    from primehkg.retrieval.graph_csr.types import GraphCsrInMemory
    from primehkg.retrieval.vector_index import VectorIndex

DatasetSource = Literal["disk", "hub"]
_REQUIRED_SPLITS = ("nodes", "hyperedges", "support")


@dataclass
class _LazyCache:
    inverted_index: dict[str, list[str]] | None = None
    xref_index: XrefIndex | None = None
    graph_csr: GraphCsrInMemory | None = None
    hyperedges_list: list[dict[str, object]] | None = None
    he_row_index: dict[str, int] | None = None
    node_row_index: dict[str, int] | None = None
    label_embedding_index: LabelEmbeddingIndex | None = None
    label_embedding_index_checked: bool = False
    graph_csr_checked: bool = False
    vector_node_index: VectorIndex | None = None
    vector_node_index_checked: bool = False
    vector_hyperedge_index: VectorIndex | None = None
    vector_hyperedge_index_checked: bool = False
    chemical_xref_index: ChemicalXrefIndex | None = None
    chemical_xref_index_checked: bool = False


@dataclass
class CoreDataset:
    """Loaded PrimeHKG core tables plus lazy derived indexes."""

    nodes: Dataset
    hyperedges: Dataset
    support: Dataset
    manifest: BuildManifest
    source: DatasetSource
    root: Path | None = None
    sidecar_root: Path | None = None
    _linking_enriched: bool = field(default=False, repr=False)
    _cache: _LazyCache = field(default_factory=_LazyCache, repr=False)

    @property
    def inputs_fingerprint(self) -> str:
        return str(self.manifest["inputs_fingerprint"])

    @property
    def is_enriched(self) -> bool:
        if is_enriched(self.manifest):
            return True
        return self._linking_enriched

    @property
    def inverted_index(self) -> dict[str, list[str]]:
        if self._cache.inverted_index is None:
            self._cache.inverted_index = self._build_inverted_index()
        return self._cache.inverted_index

    @property
    def xref_index(self) -> XrefIndex:
        if self._cache.xref_index is None:
            from primehkg.linking.xref_index import XrefIndex

            self._cache.xref_index = XrefIndex.from_nodes(self.nodes)
        return self._cache.xref_index

    @property
    def label_embedding_index(self) -> LabelEmbeddingIndex | None:
        if not self._cache.label_embedding_index_checked:
            self._cache.label_embedding_index_checked = True
            self._cache.label_embedding_index = self._try_load_label_embedding_index()
        return self._cache.label_embedding_index

    @property
    def chemical_xref_index(self) -> ChemicalXrefIndex | None:
        if not self._cache.chemical_xref_index_checked:
            self._cache.chemical_xref_index_checked = True
            self._cache.chemical_xref_index = self._try_load_chemical_xref_index()
        return self._cache.chemical_xref_index

    def _try_load_chemical_xref_index(self) -> ChemicalXrefIndex | None:
        sidecars = self.manifest.get("sidecars") or {}
        rel = self._manifest_sidecar_rel(sidecars.get("chemical_xref_v1"))
        if rel is None or self.sidecar_root is None:
            return None
        path = self.sidecar_root / rel
        if not path.is_dir():
            return None
        from primehkg.io.chemical_xref_sidecar import ChemicalXrefFingerprintError
        from primehkg.linking.chemical_xref_index import ChemicalXrefIndex

        try:
            return ChemicalXrefIndex.from_sidecar_dir(
                path,
                expected_fingerprint=self.inputs_fingerprint,
            )
        except ChemicalXrefFingerprintError:
            raise
        except (OSError, KeyError, FileNotFoundError):
            return None

    def _try_load_label_embedding_index(self) -> LabelEmbeddingIndex | None:
        sidecars = self.manifest.get("sidecars") or {}
        rel = sidecars.get("label_embedding_v1")
        if not isinstance(rel, str) or self.sidecar_root is None:
            return None
        path = self.sidecar_root / rel
        if not path.is_dir():
            return None
        from primehkg.linking.embedding_index import LabelEmbeddingIndex
        from primehkg.linking.embedding_index_io import load_label_embedding_sidecar

        loaded = load_label_embedding_sidecar(
            path,
            expected_fingerprint=self.inputs_fingerprint,
        )
        return LabelEmbeddingIndex(
            surfaces=loaded.surfaces,
            embeddings=loaded.embeddings,
        )

    @property
    def graph_csr(self) -> GraphCsrInMemory | None:
        if not self._cache.graph_csr_checked:
            self._cache.graph_csr_checked = True
            self._cache.graph_csr = self._try_load_graph_csr()
        return self._cache.graph_csr

    @property
    def vector_node_index(self) -> VectorIndex | None:
        if not self._cache.vector_node_index_checked:
            self._cache.vector_node_index_checked = True
            self._cache.vector_node_index = self._try_load_vector_index(
                "vector_node_v1"
            )
        return self._cache.vector_node_index

    @property
    def vector_hyperedge_index(self) -> VectorIndex | None:
        if not self._cache.vector_hyperedge_index_checked:
            self._cache.vector_hyperedge_index_checked = True
            self._cache.vector_hyperedge_index = self._try_load_vector_index(
                "vector_hyperedge_v1"
            )
        return self._cache.vector_hyperedge_index

    def _try_load_vector_index(self, sidecar_key: str) -> VectorIndex | None:
        sidecars = self.manifest.get("sidecars") or {}
        rel = self._manifest_sidecar_rel(sidecars.get(sidecar_key))
        if rel is None or self.sidecar_root is None:
            return None
        path = self.sidecar_root / rel
        if not path.is_dir():
            return None
        from primehkg.retrieval.vector_index import VectorIndex

        try:
            loaded = VectorIndex(path)
        except (OSError, ValueError, KeyError):
            return None
        if (
            loaded.inputs_fingerprint
            and loaded.inputs_fingerprint != self.inputs_fingerprint
        ):
            return None
        return loaded

    def hyperedges_as_mappings(self) -> list[dict[str, object]]:
        if self._cache.hyperedges_list is None:
            self._cache.hyperedges_list = [dict(row) for row in self.hyperedges]
        return self._cache.hyperedges_list

    def get_hyperedge(self, he_id: str) -> dict[str, object] | None:
        pos = self._ensure_he_row_index().get(he_id)
        if pos is None:
            return None
        return dict(self.hyperedges[pos])

    def _ensure_he_row_index(self) -> dict[str, int]:
        if self._cache.he_row_index is None:
            self._cache.he_row_index = self._load_he_row_index()
        return self._cache.he_row_index

    def _load_he_row_index(self) -> dict[str, int]:
        sidecars = self.manifest.get("sidecars") or {}
        rel = sidecars.get("he_row_index")
        if isinstance(rel, str) and self.sidecar_root is not None:
            path = self.sidecar_root / rel
            if path.is_file():
                return load_he_row_index_json_validated(
                    path,
                    expected_fingerprint=self.inputs_fingerprint,
                    reproducibility="strict",
                )
        return build_he_row_index_from_dataset(self.hyperedges)

    def get_node(self, node_id: str) -> dict[str, object] | None:
        pos = self._ensure_node_row_index().get(node_id)
        if pos is None:
            return None
        return dict(self.nodes[pos])

    def _ensure_node_row_index(self) -> dict[str, int]:
        if self._cache.node_row_index is None:
            self._cache.node_row_index = {
                str(nid): i for i, nid in enumerate(self.nodes["node_id"])
            }
        return self._cache.node_row_index

    def warm(self) -> None:
        """Populate lazy lookup indexes used on the read path.

        Warms the inverted index, hyperedge and node row indexes, graph CSR
        sidecar (if present), and, for enriched builds, the xref index. Call
        once before serving so the first request pays no cold-build cost.
        """
        _ = self.inverted_index
        _ = self.graph_csr
        self._ensure_he_row_index()
        self._ensure_node_row_index()
        if self.is_enriched:
            _ = self.xref_index

    def _build_inverted_index(self) -> dict[str, list[str]]:
        sidecars = self.manifest.get("sidecars") or {}
        rel = sidecars.get("inverted_index")
        if isinstance(rel, str) and self.sidecar_root is not None:
            path = self.sidecar_root / rel
            if path.is_file():
                return load_inverted_index_json_validated(
                    path,
                    expected_fingerprint=self.inputs_fingerprint,
                    reproducibility="strict",
                )
        return build_inverted_index_from_dataset(self.hyperedges)

    def _manifest_sidecar_rel(self, entry: object) -> str | None:
        if isinstance(entry, str) and entry.strip():
            return entry
        if isinstance(entry, dict):
            rel = cast("dict[str, object]", entry).get("dataset_dir")
            if isinstance(rel, str) and rel.strip():
                return rel
        return None

    def _try_load_graph_csr(self) -> GraphCsrInMemory | None:
        sidecars = self.manifest.get("sidecars") or {}
        rel = self._manifest_sidecar_rel(sidecars.get("graph_csr_v1"))
        if rel is None or self.sidecar_root is None:
            return None
        csr_dir = self.sidecar_root / rel
        if not csr_dir.is_dir():
            return None
        from primehkg.retrieval.graph_csr.io import load_graph_csr_sidecar

        return load_graph_csr_sidecar(
            csr_dir,
            expected_fingerprint=self.inputs_fingerprint,
        )


def _validate_fingerprint(
    manifest: BuildManifest, expected_fingerprint: str | None
) -> None:
    if expected_fingerprint is None:
        return
    actual = manifest.get("inputs_fingerprint")
    if actual != expected_fingerprint:
        msg = (
            f"inputs_fingerprint mismatch: manifest has {actual!r}, "
            f"expected {expected_fingerprint!r}"
        )
        raise FingerprintMismatchError(msg)


def _datasetdict_from_disk(path: Path) -> tuple[Dataset, Dataset, Dataset]:
    dd = load_from_disk(str(path))
    missing = [s for s in _REQUIRED_SPLITS if s not in dd]
    if missing:
        msg = f"DatasetDict at {path!r} missing splits: {missing}"
        raise CoreDatasetLoadError(msg)
    return (
        cast("Dataset", dd["nodes"]),
        cast("Dataset", dd["hyperedges"]),
        cast("Dataset", dd["support"]),
    )


def load_core_dataset(
    path: str | Path | None = None,
    *,
    hub_repo_id: str | None = None,
    revision: str | None = None,
    expected_fingerprint: str | None = None,
    sidecar_dir: Path | None = None,
    index_mode: Literal["lazy", "eager"] = "lazy",
    hub_fallback: Literal["disk_cache", "error"] = "disk_cache",
) -> CoreDataset:
    """Load a PrimeHKG core build from disk or Hugging Face Hub.

    ``path=`` reads a ``DatasetDict.save_to_disk`` build directory (the direct output of
    ``primehkg-build``).

    ``hub_repo_id=`` reads the Hub (Parquet-shard) layout: either a Hub dataset repo
    id (``org/primehkg-core``, pinned with ``revision=``), or an existing **local
    directory** with the same layout as a Hub snapshot (for example one written by
    ``primehkg-hf-push --out-dir``). A local directory is read without contacting the
    Hub, ``revision`` is ignored (logged at INFO), it must contain
    ``build_manifest.json``, and ``expected_fingerprint`` is checked against that
    manifest exactly as for a Hub snapshot. Sidecars resolve against the directory
    unless ``sidecar_dir=`` is given.
    """
    del hub_fallback  # reserved for future Hub cache policy

    if path is not None and hub_repo_id is not None:
        msg = "Provide either path= or hub_repo_id=, not both"
        raise CoreDatasetLoadError(msg)
    if path is None and hub_repo_id is None:
        msg = "Provide path= for local load or hub_repo_id= for Hub load"
        raise CoreDatasetLoadError(msg)

    source: DatasetSource
    root: Path | None
    linking_enriched = False
    if path is not None:
        root = Path(path)
        manifest = read_build_manifest(root)
        _validate_fingerprint(manifest, expected_fingerprint)
        nodes, hyperedges, support = _datasetdict_from_disk(root)
        source = "disk"
        effective_sidecar = sidecar_dir if sidecar_dir is not None else root
    else:
        from primehkg.core.hub_loader import (
            load_hub_datasetdict,
            resolve_local_hub_dir,
            try_fetch_hub_build_manifest,
        )

        assert hub_repo_id is not None
        local_hub_dir = resolve_local_hub_dir(hub_repo_id)
        if (
            local_hub_dir is not None
            and not (local_hub_dir / "build_manifest.json").is_file()
        ):
            msg = (
                f"local Hub-layout directory {str(local_hub_dir)!r} has no "
                "build_manifest.json; cannot verify inputs_fingerprint"
            )
            raise CoreDatasetLoadError(msg)
        dd, hub_root = load_hub_datasetdict(
            hub_repo_id=hub_repo_id,
            revision=revision,
        )
        missing = [s for s in _REQUIRED_SPLITS if s not in dd]
        if missing:
            msg = f"Hub dataset {hub_repo_id!r} missing splits: {missing}"
            raise CoreDatasetLoadError(msg)
        nodes = dd["nodes"]
        hyperedges = dd["hyperedges"]
        support = dd["support"]
        root = hub_root
        manifest: BuildManifest | None = None
        if root is not None and (root / "build_manifest.json").is_file():
            manifest = read_build_manifest(root)
        if manifest is None:
            manifest = try_fetch_hub_build_manifest(
                hub_repo_id,
                revision,
            )
        if manifest is None:
            if expected_fingerprint is None:
                msg = "Hub load without local manifest requires expected_fingerprint="
                raise CoreDatasetLoadError(msg)
            manifest = cast(
                "BuildManifest",
                {
                    "schema_version": "0.0.0",
                    "inputs_fingerprint": expected_fingerprint,
                    "builders": [],
                    "sources": [],
                    "outputs": {},
                },
            )
        _validate_fingerprint(manifest, expected_fingerprint)
        linking_enriched = not is_enriched(manifest) and nodes_have_linking_fields(
            nodes
        )
        source = "hub"
        effective_sidecar = sidecar_dir if sidecar_dir is not None else root

    ds = CoreDataset(
        nodes=nodes,
        hyperedges=hyperedges,
        support=support,
        manifest=manifest,
        source=source,
        root=root,
        sidecar_root=effective_sidecar,
        _linking_enriched=linking_enriched,
    )
    if index_mode == "eager":
        _ = ds.inverted_index
        if ds.is_enriched:
            _ = ds.xref_index
        _ = ds.graph_csr
    return ds
