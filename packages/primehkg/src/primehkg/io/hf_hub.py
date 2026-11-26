from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from huggingface_hub import HfApi


class MultiSplitCreatePrNotSupportedError(ValueError):
    """Raised when `create_pr=True` is requested for a multi-split `DatasetDict` push."""


def resolve_hf_token(*, token: str | None) -> str | None:
    if token is not None:
        stripped = token.strip()
        if stripped:
            return stripped

    env = (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    ).strip()
    return env or None


def upload_hf_dataset_readme(
    readme_path: str | os.PathLike[str],
    repo_id: str,
    *,
    revision: str | None = None,
    commit_message: str | None = None,
    commit_description: str | None = None,
    create_pr: bool = False,
    token: str | None = None,
) -> Any:
    """Upload a Markdown dataset card to a Hugging Face *dataset* repo as `README.md`."""
    resolved = resolve_hf_token(token=token)
    if not resolved:
        msg = (
            "Missing Hugging Face token. Set `HF_TOKEN` (preferred) or "
            "`HUGGING_FACE_HUB_TOKEN`, or pass `token=` explicitly."
        )
        raise RuntimeError(msg)

    path = Path(readme_path)
    if not path.is_file():
        msg = f"Dataset README path is not a file: {str(path)!r}"
        raise FileNotFoundError(msg)

    api = HfApi(token=resolved)
    return api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        commit_message=commit_message,
        commit_description=commit_description,
        create_pr=create_pr,
    )


def upload_hf_dataset_file(
    local_path: str | os.PathLike[str],
    repo_id: str,
    path_in_repo: str,
    *,
    revision: str | None = None,
    commit_message: str | None = None,
    commit_description: str | None = None,
    create_pr: bool = False,
    token: str | None = None,
) -> Any:
    """Upload a single local file to a HF *dataset* repo at ``path_in_repo``.

    Token-resolved via ``resolve_hf_token`` (HF_TOKEN / HUGGING_FACE_HUB_TOKEN / token=).
    Raises ``FileNotFoundError`` if ``local_path`` is not a file; ``RuntimeError`` if no token.
    """
    resolved = resolve_hf_token(token=token)
    if not resolved:
        msg = (
            "Missing Hugging Face token. Set `HF_TOKEN` (preferred) or "
            "`HUGGING_FACE_HUB_TOKEN`, or pass `token=` explicitly."
        )
        raise RuntimeError(msg)

    path = Path(local_path)
    if not path.is_file():
        msg = f"Upload path is not a file: {str(path)!r}"
        raise FileNotFoundError(msg)

    api = HfApi(token=resolved)
    return api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        commit_message=commit_message,
        commit_description=commit_description,
        create_pr=create_pr,
    )


def create_hf_dataset_release_tag(
    repo_id: str,
    tag: str,
    *,
    revision: str = "main",
    tag_message: str | None = None,
    move_existing: bool = True,
    token: str | None = None,
) -> None:
    """Create a semver git tag on a Hugging Face *dataset* repo."""
    resolved = resolve_hf_token(token=token)
    if not resolved:
        msg = (
            "Missing Hugging Face token. Set `HF_TOKEN` (preferred) or "
            "`HUGGING_FACE_HUB_TOKEN`, or pass `token=` explicitly."
        )
        raise RuntimeError(msg)

    api = HfApi(token=resolved)
    if move_existing:
        try:
            api.delete_tag(repo_id, tag=tag, repo_type="dataset")
        except Exception:
            pass
    api.create_tag(
        repo_id,
        tag=tag,
        repo_type="dataset",
        revision=revision,
        tag_message=tag_message,
        exist_ok=move_existing,
    )


def push_hf_datasetdict_from_disk_to_hub(
    dataset_dir: str | os.PathLike[str],
    repo_id: str,
    *,
    config_name: str = "default",
    private: bool | None = None,
    revision: str | None = None,
    commit_message: str | None = None,
    commit_description: str | None = None,
    create_pr: bool = False,
    max_shard_size: str | int | None = None,
    num_shards: dict[str, int] | None = None,
    embed_external_files: bool = True,
    num_proc: int | None = None,
    token: str | None = None,
) -> list[Any]:
    """Load a PrimeHKG on-disk `DatasetDict` and push its splits to the Hugging Face Hub.

    PrimeHKG stores three **heterogeneous** tables (`nodes`, `hyperedges`, `support`). Hugging Face
    `DatasetDict.push_to_hub` requires identical features across splits, so this function pushes
    each split using `Dataset.push_to_hub` with:

    - a shared `repo_id`
    - `split=<split_name>` (always the split key in the on-disk `DatasetDict`)
    - `config_name` is `split_name` when `config_name=="default"`; otherwise the same non-default
      `config_name` is reused for every split (advanced / usually not what you want for PrimeHKG)

    This matches the intended `primehkg-core` layout (`primehkg.distribution.PRIMEHKG_CORE_HF_CONFIGS`)
    better than forcing a single shared schema.

    Authentication:
    - Pass `token=` explicitly, or set `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` in the environment.

    Notes:
    - `create_pr=True` is not supported when pushing multiple heterogeneous splits from a single
      `DatasetDict` directory (it tends to create multiple inconsistent PRs). Use direct commits
      (`create_pr=False`) or a manual Hub branch workflow.
    """
    resolved = resolve_hf_token(token=token)
    if not resolved:
        msg = (
            "Missing Hugging Face token. Set `HF_TOKEN` (preferred) or "
            "`HUGGING_FACE_HUB_TOKEN`, or pass `token=` explicitly."
        )
        raise RuntimeError(msg)

    dd = load_from_disk(str(dataset_dir))
    if not isinstance(dd, DatasetDict):
        msg = f"Expected a DatasetDict at {str(dataset_dir)!r}, got {type(dd).__name__}"
        raise TypeError(msg)

    if create_pr and len(dd) > 1:
        msg = (
            "`create_pr=True` is not supported for multi-split DatasetDict pushes (PrimeHKG tables). "
            "Each split is pushed independently; Hub PR creation would not compose into one coherent PR. "
            "Re-run with `create_pr=False` (default) or use a manual branch/PR workflow on the Hub."
        )
        raise MultiSplitCreatePrNotSupportedError(msg)

    commits: list[Any] = []
    for split_name, ds in dd.items():
        if not isinstance(ds, Dataset):
            msg = f"Expected split {split_name!r} to be a Dataset, got {type(ds).__name__}"
            raise TypeError(msg)

        per_split_cfg = config_name if config_name != "default" else str(split_name)

        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "config_name": per_split_cfg,
            "split": str(split_name),
            "private": private,
            "token": resolved,
            "create_pr": create_pr,
            "embed_external_files": embed_external_files,
        }
        if revision is not None:
            kwargs["revision"] = revision
        if commit_message is not None:
            kwargs["commit_message"] = commit_message
        if commit_description is not None:
            kwargs["commit_description"] = commit_description
        if max_shard_size is not None:
            kwargs["max_shard_size"] = max_shard_size
        if num_shards is not None:
            # `Dataset.push_to_hub` expects an int (unlike DatasetDict which supports per-split dict).
            if isinstance(num_shards, dict):
                maybe = num_shards.get(str(split_name))
                if maybe is not None:
                    kwargs["num_shards"] = int(maybe)
            else:
                msg = f"`num_shards` must be a dict when pushing heterogeneous splits, got {type(num_shards).__name__}"
                raise TypeError(msg)
        if num_proc is not None:
            kwargs["num_proc"] = num_proc

        commits.append(ds.push_to_hub(**kwargs))

    return commits


DEFAULT_LOCAL_MAX_SHARD_SIZE = "500MB"
HUB_LAYOUT_AUX_FILES: tuple[str, ...] = (
    "build_manifest.json",
    "build_report.json",
    "text_provenance_inventory.json",
    "embedding_manifest.json",
)


def write_hub_layout_from_disk(
    dataset_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    max_shard_size: str | int | None = None,
    dataset_readme: str | os.PathLike[str] | None = None,
    copy_sidecars: bool = True,
) -> dict[str, list[str]]:
    """Write a local build in the Hub (Parquet-shard) layout, without uploading.

    Reads the ``DatasetDict.save_to_disk`` directory produced by ``primehkg-build`` /
    ``primehkg-features-build`` / ``primehkg-embed-build`` and writes, under
    ``out_dir``::

        <split>/<split>-00000-of-0000N.parquet   (one directory per split)
        build_manifest.json, build_report.json, ... (copied when present)
        sidecars/                                   (copied when present)
        README.md                                   (when ``dataset_readme`` is given)

    which is the layout of a Hugging Face snapshot of a dataset pushed with
    ``primehkg-hf-push``. The result can be passed as ``hub_repo_id=`` to
    ``primehkg.core.load_core_dataset`` (local directories are read in place).

    ``out_dir`` must not exist or must be empty, and must not be ``dataset_dir`` or
    lie inside it. Returns ``{split: [relative shard paths]}``.
    """
    import shutil

    from datasets.utils.py_utils import convert_file_size_to_int

    src = Path(dataset_dir).resolve()
    dst = Path(out_dir).resolve()
    if dst == src or src in dst.parents:
        msg = f"out_dir {str(dst)!r} must not be the dataset dir or inside it"
        raise ValueError(msg)
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        msg = f"out_dir {str(dst)!r} exists and is not an empty directory"
        raise FileExistsError(msg)

    dd = load_from_disk(str(src))
    if not isinstance(dd, DatasetDict):
        msg = f"Expected a DatasetDict at {str(src)!r}, got {type(dd).__name__}"
        raise TypeError(msg)

    max_bytes = convert_file_size_to_int(
        max_shard_size if max_shard_size is not None else DEFAULT_LOCAL_MAX_SHARD_SIZE
    )
    if max_bytes <= 0:
        msg = f"max_shard_size must be positive, got {max_shard_size!r}"
        raise ValueError(msg)

    dst.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {}
    for split_name, ds in dd.items():
        if not isinstance(ds, Dataset):
            msg = f"Expected split {split_name!r} to be a Dataset, got {type(ds).__name__}"
            raise TypeError(msg)
        name = str(split_name)
        nbytes = int(ds.data.nbytes)
        num_shards = max(1, -(-nbytes // max_bytes))
        num_shards = min(num_shards, max(1, len(ds)))
        split_dir = dst / name
        split_dir.mkdir(parents=True, exist_ok=True)
        rels: list[str] = []
        for index in range(num_shards):
            shard = ds.shard(num_shards=num_shards, index=index, contiguous=True)
            rel = f"{name}/{name}-{index:05d}-of-{num_shards:05d}.parquet"
            shard.to_parquet(str(dst / rel))
            rels.append(rel)
        written[name] = rels

    for aux_name in HUB_LAYOUT_AUX_FILES:
        aux = src / aux_name
        if aux.is_file():
            shutil.copy2(aux, dst / aux_name)

    sidecars = src / "sidecars"
    if copy_sidecars and sidecars.is_dir():
        shutil.copytree(sidecars, dst / "sidecars", symlinks=False)

    if dataset_readme is not None:
        readme = Path(dataset_readme)
        if not readme.is_file():
            msg = f"Dataset README path is not a file: {str(readme)!r}"
            raise FileNotFoundError(msg)
        shutil.copy2(readme, dst / "README.md")

    return written
