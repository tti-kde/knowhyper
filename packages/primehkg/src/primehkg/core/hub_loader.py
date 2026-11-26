"""Hugging Face Hub loading helpers for ``CoreDataset``.

``hub_repo_id`` may also name an existing **local directory** laid out like a Hub
snapshot (``nodes/``, ``hyperedges/``, ``support/`` Parquet shards plus
``build_manifest.json`` and optional ``sidecars/``), e.g. a directory written by
``primehkg-hf-push --out-dir``. Such a directory is read in place: the Hub is never
contacted and ``revision`` is ignored (logged).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from primehkg.core.exceptions import HubLoadError

if TYPE_CHECKING:
    from datasets import (
        Dataset,
        DatasetDict,
        IterableDataset,
        IterableDatasetDict,
    )

    from primehkg.core.manifest import BuildManifest

    _LoadDatasetResult = DatasetDict | Dataset | IterableDatasetDict | IterableDataset

logger = logging.getLogger(__name__)

_SPLIT_NAMES = ("nodes", "hyperedges", "support")


def _snapshot_download(
    *,
    repo_id: str,
    revision: str | None,
    repo_type: str,
) -> str:
    from huggingface_hub import snapshot_download as _dl

    return _dl(repo_id=repo_id, revision=revision, repo_type=repo_type)


# Module-attr seam: tests monkeypatch this; the real huggingface_hub import
# stays function-local in _snapshot_download.
snapshot_download = _snapshot_download


def resolve_local_hub_dir(hub_repo_id: str) -> Path | None:
    """Return ``hub_repo_id`` as a local directory when it names an existing one.

    A Hub repo id (``org/name``) that does not exist on disk returns ``None`` and is
    resolved against the Hugging Face Hub as before.
    """
    candidate = Path(hub_repo_id).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return None


def _log_revision_ignored(local_dir: Path, revision: str | None) -> None:
    if revision is not None:
        logger.info(
            "hub_repo_id %s is a local directory; ignoring revision=%r",
            local_dir,
            revision,
        )


def _load_dataset(
    path: str,
    *,
    name: str | None = None,
    revision: str | None = None,
    split: str | None = None,
) -> _LoadDatasetResult:
    from datasets import load_dataset as _ld

    if split is not None:
        return _ld(path, name=name, revision=revision, split=split)
    return _ld(path, name=name, revision=revision)


# Module-attr seam: tests monkeypatch this; the real datasets import stays
# function-local in _load_dataset.
load_dataset = _load_dataset


def load_hub_via_snapshot(
    *,
    hub_repo_id: str,
    revision: str | None,
) -> tuple[DatasetDict, Path]:
    """Load PrimeHKG core by snapshotting the repo and reading split shards directly.

    Bypasses ``datasets.load_dataset`` auto-generation (which fails on PrimeHKG's
    heterogeneous nested columns) by reading each split's Parquet shards with
    ``Dataset.from_parquet`` + ``concatenate_datasets``. Returns the loaded
    ``DatasetDict`` and the local snapshot directory (used as the dataset root and,
    by default, the sidecar root).
    """
    from datasets import (
        Dataset,
        concatenate_datasets,
    )
    from datasets import DatasetDict as HFDatasetDict

    local_dir = resolve_local_hub_dir(hub_repo_id)
    if local_dir is not None:
        _log_revision_ignored(local_dir, revision)
        snap = local_dir
    else:
        snap = Path(
            snapshot_download(
                repo_id=hub_repo_id, revision=revision, repo_type="dataset"
            )
        )

    splits: dict[str, Dataset] = {}
    for name in _SPLIT_NAMES:
        shards = sorted(str(p) for p in (snap / name).glob(f"{name}-*.parquet"))
        if not shards:
            shards = sorted(str(p) for p in (snap / name).glob("*.parquet"))
        if not shards:
            msg = (
                f"Hub repo {hub_repo_id!r} snapshot {snap} has no Parquet shards "
                f"for required split {name!r}"
            )
            raise HubLoadError(msg)
        try:
            parts = [Dataset.from_parquet(s) for s in shards]
            splits[name] = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        except Exception as exc:
            msg = f"Failed to read split {name!r} from Hub snapshot {snap}: {exc}"
            raise HubLoadError(msg) from exc

    return HFDatasetDict(dict(splits)), snap


def load_hub_datasetdict(
    *,
    hub_repo_id: str,
    revision: str | None,
) -> tuple[DatasetDict, Path | None]:
    """Load PrimeHKG core splits from the Hub.

    Returns ``(datasetdict, optional_local_root_with_manifest)``.

    Tries the snapshot path first (``load_hub_via_snapshot``), returning the
    local snapshot root as the second element so callers can resolve the build
    manifest and sidecar root. On any failure falls back to ``load_dataset``
    (per-split configs, then a single ``default`` config), returning ``None``.

    When ``hub_repo_id`` is an existing local directory (see
    ``resolve_local_hub_dir``) only the snapshot-layout reader is used; there is no
    Hub fallback, and a malformed directory raises ``HubLoadError``.
    """
    if resolve_local_hub_dir(hub_repo_id) is not None:
        return load_hub_via_snapshot(hub_repo_id=hub_repo_id, revision=revision)

    try:
        dd_snap, snap_root = load_hub_via_snapshot(
            hub_repo_id=hub_repo_id, revision=revision
        )
    except Exception:
        logger.info(
            "Snapshot-based load failed for Hub repo %s (revision=%s); "
            "falling back to load_dataset",
            hub_repo_id,
            revision,
            exc_info=True,
        )
    else:
        return dd_snap, snap_root

    from typing import cast

    from datasets import Dataset as HFDataset
    from datasets import DatasetDict

    splits: dict[str, Dataset] = {}
    per_split_ok = True
    for name in _SPLIT_NAMES:
        try:
            # The publisher pushes each split as config_name=<split> with
            # split=<split> (see io/hf_hub.py), so the split key equals the name
            # — not "train".
            loaded_split = load_dataset(
                hub_repo_id,
                name=name,
                revision=revision,
                split=name,
            )
        except Exception:
            logger.info(
                "Per-split fallback load failed for %s config %r (revision=%s)",
                hub_repo_id,
                name,
                revision,
                exc_info=True,
            )
            per_split_ok = False
            break
        splits[name] = cast("HFDataset", loaded_split)

    if per_split_ok and len(splits) == len(_SPLIT_NAMES):
        from datasets import DatasetDict as HFDatasetDict

        dd_out = HFDatasetDict(dict(splits))
        return dd_out, None

    try:
        loaded = load_dataset(hub_repo_id, revision=revision)
    except Exception as exc:
        msg = f"Failed to load Hub dataset {hub_repo_id!r}: {exc}"
        raise HubLoadError(msg) from exc

    if isinstance(loaded, DatasetDict):
        missing = [s for s in _SPLIT_NAMES if s not in loaded]
        if not missing:
            return loaded, None
        logger.warning(
            "Hub DatasetDict for %s missing splits %s; available: %s",
            hub_repo_id,
            missing,
            list(loaded.keys()),
        )

    msg = (
        f"Could not load nodes/hyperedges/support from Hub repo {hub_repo_id!r}. "
        "Prefer load_from_disk on a maintainer build directory."
    )
    raise HubLoadError(msg)


def try_fetch_hub_build_manifest(
    hub_repo_id: str,
    revision: str | None,
) -> BuildManifest | None:
    """Download ``build_manifest.json`` from a Hub dataset repo when published.

    For a local snapshot-layout directory, reads ``<dir>/build_manifest.json``
    (``None`` when absent or invalid) without contacting the Hub.
    """
    import json

    local_dir = resolve_local_hub_dir(hub_repo_id)
    if local_dir is not None:
        path = local_dir / "build_manifest.json"
        if not path.is_file():
            return None
    else:
        path = _hf_hub_download_manifest(hub_repo_id, revision)
        if path is None:
            return None

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        logger.warning(
            "Invalid build_manifest.json on Hub repo %s (revision=%s)",
            hub_repo_id,
            revision,
        )
        return None

    if not isinstance(raw, dict):
        return None
    fp = raw.get("inputs_fingerprint")
    if not isinstance(fp, str) or not fp.strip():
        return None
    return raw  # type: ignore[return-value]


def _hf_hub_download_manifest(hub_repo_id: str, revision: str | None) -> str | None:
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(
            repo_id=hub_repo_id,
            filename="build_manifest.json",
            repo_type="dataset",
            revision=revision,
        )
    except Exception:
        logger.debug(
            "No build_manifest.json on Hub repo %s (revision=%s)",
            hub_repo_id,
            revision,
            exc_info=True,
        )
        return None
