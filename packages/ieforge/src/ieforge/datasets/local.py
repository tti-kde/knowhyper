"""Resolve a benchmark dataset from the Hugging Face Hub or a local parquet copy.

Every ieforge dataset loader goes through :func:`load_hf_dataset`.

* ``IEFORGE_DATA_DIR`` unset (or empty): the call is exactly
  ``datasets.load_dataset(repo, config, split=split, token=token)``.
* ``IEFORGE_DATA_DIR`` set: the dataset is read from
  ``$IEFORGE_DATA_DIR/<repo name after the slash>/`` instead of the Hub. The
  directory must use the layout ``datasets.DatasetDict.push_to_hub`` writes::

      <root>/<config>/<split>-00000-of-00001.parquet

  where the ``default`` config lives under ``data/`` (``push_to_hub``'s
  convention when no ``config_name`` is given). The build scripts' ``--out-dir``
  mode writes this layout through :func:`save_parquet_layout`.

The local files are read with the ``parquet`` builder, so rows, column order
and features (stored in the parquet schema metadata by ``datasets``) are the
same as loading the Hub repo that holds the same files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DATA_DIR_ENV = "IEFORGE_DATA_DIR"
_DEFAULT_CONFIG = "default"
_DEFAULT_CONFIG_DIR = "data"
_SHARD_SUFFIX = "-00000-of-00001.parquet"


def _config_dir_name(config: str | None) -> str:
    if config is None or config == _DEFAULT_CONFIG:
        return _DEFAULT_CONFIG_DIR
    return config


def local_dataset_root(repo: str) -> Path | None:
    """Return the local directory that replaces ``repo``, or None if unset."""
    data_dir = os.environ.get(DATA_DIR_ENV)
    if not data_dir:
        return None
    return Path(data_dir).expanduser() / repo.rsplit("/", 1)[-1]


def local_split_files(root: Path, config: str | None, split: str) -> list[str]:
    """Return the sorted parquet shards of ``config``/``split`` under ``root``."""
    candidates = [root / config] if config else []
    if config is None or config == _DEFAULT_CONFIG:
        candidates.append(root / _DEFAULT_CONFIG_DIR)
    for config_dir in candidates:
        files = sorted(config_dir.glob(f"{split}-*.parquet"))
        if files:
            return [str(path) for path in files]
    searched = ", ".join(str(path / f"{split}-*.parquet") for path in candidates)
    msg = (
        f"{DATA_DIR_ENV} is set but no parquet file was found for "
        f"config={config!r} split={split!r} (searched: {searched}). Build the "
        "dataset locally with the scripts/build_*.py --out-dir mode."
    )
    raise FileNotFoundError(msg)


def load_hf_dataset(
    repo: str, config: str | None, split: str, token: str | None
) -> Any:
    """Load one split of ``repo``/``config`` from the Hub or ``IEFORGE_DATA_DIR``."""
    from datasets import load_dataset

    root = local_dataset_root(repo)
    if root is None:
        return load_dataset(repo, config, split=split, token=token)
    files = local_split_files(root, config, split)
    return load_dataset("parquet", data_files={split: files}, split=split)


def save_parquet_layout(
    dataset_dict: Any, out_dir: Path | str, config: str | None
) -> list[Path]:
    """Write a ``DatasetDict`` as ``<out_dir>/<config>/<split>-00000-of-00001.parquet``.

    ``out_dir`` is the per-repo directory (e.g. ``$IEFORGE_DATA_DIR/blurb-re``).
    Existing files of the same splits are overwritten.
    """
    config_dir = Path(out_dir) / _config_dir_name(config)
    config_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for split, dataset in dataset_dict.items():
        for stale in config_dir.glob(f"{split}-*.parquet"):
            stale.unlink()
        path = config_dir / f"{split}{_SHARD_SUFFIX}"
        dataset.to_parquet(str(path))
        written.append(path)
    return written
