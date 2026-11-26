"""Resolve per-(node, source_modality) embeddings from the primehkg-features repo.

The multimodal payload lives in the ``node_features_embedding`` table of the
``primehkg-features`` dataset, keyed by ``(node_id, source_modality)``. Different
modalities are encoded by different upstream models at DIFFERENT dims: text-family
(display_name / description / mechanism_of_action / indication / ...) = 768-d
PubMedBERT, ``smiles`` = 384-d ChemBERTa, ``protein`` (sequence) = 320-d ESM2. This
index reads that table once and serves vectors by (node, modality); the dim follows
whatever the stored vector is. No encoding happens here; vectors are precomputed
upstream.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Sequence

FEATURES_REPO = os.environ.get("PRIMEHKG_FEATURES", "nishide-dev/primehkg-features")
EMBED_TABLE = "node_features_embedding"
TEXT_TABLE = "node_features_text"


def _local_features_dir() -> str | None:
    """``FEATURES_REPO`` when it names an existing local directory, else None.

    A locally rebuilt features dataset (README, "Using your own PrimeHKG build") is
    read in place; the Hub ``revision`` does not apply to it.
    """
    path = Path(FEATURES_REPO).expanduser()
    return str(path) if path.is_dir() else None


def resolve_features_dir(revision: str) -> str:
    """Return the local snapshot dir for primehkg-features at ``revision``.

    Uses ``snapshot_download`` so it honors ``HF_HUB_OFFLINE`` and raises if the
    embedding table is absent.
    """
    snap = _local_features_dir()
    if snap is None:
        from huggingface_hub import snapshot_download

        snap = snapshot_download(
            repo_id=FEATURES_REPO, repo_type="dataset", revision=revision
        )
    if not list(Path(snap, EMBED_TABLE).glob("*.parquet")):
        msg = (
            f"{FEATURES_REPO}@{revision}: no {EMBED_TABLE}/ parquet in snapshot {snap}"
        )
        raise FileNotFoundError(msg)
    return snap


def resolve_text_features_dir(revision: str) -> str:
    """Return a snapshot containing raw ``node_features_text`` at ``revision``.

    Retrieved-definition baselines use a newer provenance-bearing text snapshot
    than the embedding sidecars used by EV experiments.  Restricting the download
    to the raw text table and its manifest keeps those two immutable inputs
    independent and avoids fetching unrelated feature tables.
    """
    snap = _local_features_dir()
    if snap is None:
        from huggingface_hub import snapshot_download

        snap = snapshot_download(
            repo_id=FEATURES_REPO,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{TEXT_TABLE}/*.parquet", "build_manifest.json"],
        )
    if not list(Path(snap, TEXT_TABLE).glob("*.parquet")):
        msg = f"{FEATURES_REPO}@{revision}: no {TEXT_TABLE}/ parquet in snapshot {snap}"
        raise FileNotFoundError(msg)
    return snap


class ModalityIndex:
    """In-memory ``(node_id, source_modality) -> vector`` map from the features repo."""

    def __init__(
        self, features_dir: str | Path, *, modalities: Sequence[str] | None = None
    ) -> None:
        """Load the embedding table; ``modalities`` restricts which rows are kept (None=all)."""
        want = set(modalities) if modalities is not None else None
        self._map: dict[tuple[str, str], list[float]] = {}
        self._by_node: dict[str, set[str]] = {}
        self.dim = 0
        files = sorted(Path(features_dir, EMBED_TABLE).glob("*.parquet"))
        if not files:
            msg = f"no {EMBED_TABLE} parquet under {features_dir}"
            raise FileNotFoundError(msg)
        for f in files:
            t = pq.read_table(f, columns=["node_id", "source_modality", "vector"])
            for nid, sm, vec in zip(
                t.column("node_id").to_pylist(),
                t.column("source_modality").to_pylist(),
                t.column("vector").to_pylist(),
                strict=True,
            ):
                if want is not None and sm not in want:
                    continue
                self._map[(nid, sm)] = vec
                self._by_node.setdefault(nid, set()).add(sm)
                if self.dim == 0 and vec is not None:
                    self.dim = len(vec)

    def lookup(self, node_id: str, modality: str) -> list[float] | None:
        """Return the vector for ``(node_id, modality)`` or None if absent."""
        return self._map.get((node_id, modality))

    def modalities_for(self, node_id: str) -> set[str]:
        """Return the set of source_modalities present for ``node_id``."""
        return self._by_node.get(node_id, set())
