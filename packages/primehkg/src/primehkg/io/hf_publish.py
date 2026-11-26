from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import typer
from datasets import DatasetDict, load_from_disk
from typer.main import get_command

from primehkg.features.embedding_schema import EMBEDDING_CONFIG_NAME
from primehkg.io import hf_hub
from primehkg.io.cli_console import cli_error, cli_warning
from primehkg.io.hf_hub import (
    MultiSplitCreatePrNotSupportedError,
    push_hf_datasetdict_from_disk_to_hub,
    resolve_hf_token,
    upload_hf_dataset_readme,
    write_hub_layout_from_disk,
)

_AUX_UPLOAD_NAMES = (
    "build_manifest.json",
    "build_report.json",
    "text_provenance_inventory.json",
)


def _features_provenance_requires_inventory(dataset_dir: Path) -> bool:
    manifest_path = dataset_dir / "build_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if "provenance_registry_sha256" not in manifest:
        return False
    if (dataset_dir / "node_features_text").exists():
        return True
    try:
        dd = load_from_disk(str(dataset_dir))
    except Exception:
        return False
    return isinstance(dd, DatasetDict) and "node_features_text" in dd


@dataclass(frozen=True)
class HfPushCliArgs:
    """CLI options for ``primehkg-hf-push`` (Typer ``push`` command)."""

    dataset_dir: Path
    repo_id: str | None
    config_name: str
    private: bool
    public: bool
    revision: str | None
    commit_message: str | None
    commit_description: str | None
    create_pr: bool
    max_shard_size: str | None
    token: str | None
    dataset_readme: Path | None
    release_tag: str | None
    also_tag_latest: bool
    out_dir: Path | None = None


def run_local_layout(args: HfPushCliArgs) -> int:
    """``--out-dir`` mode: write the Hub layout locally; never contacts the Hub."""
    assert args.out_dir is not None
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        cli_error(f"dataset dir is not a directory: {dataset_dir}")
        return 2
    hub_only = [
        flag
        for flag, value in (
            ("--repo-id", args.repo_id),
            ("--revision", args.revision),
            ("--release-tag", args.release_tag),
            ("--token", args.token),
        )
        if value is not None
    ]
    if args.create_pr or args.also_tag_latest or args.private or args.public:
        hub_only.append("--create-pr/--also-tag-latest/--private/--public")
    if hub_only:
        cli_error(f"--out-dir writes locally; Hub-only options not allowed: {hub_only}")
        return 2
    if (
        _features_provenance_requires_inventory(dataset_dir)
        and not (dataset_dir / "text_provenance_inventory.json").is_file()
    ):
        cli_error(
            "features provenance build requires text_provenance_inventory.json "
            f"in {dataset_dir}"
        )
        return 2
    try:
        written = write_hub_layout_from_disk(
            dataset_dir,
            args.out_dir,
            max_shard_size=args.max_shard_size,
            dataset_readme=args.dataset_readme,
        )
    except (FileExistsError, FileNotFoundError, ValueError, TypeError) as exc:
        cli_error(str(exc))
        return 2
    summary = {split: len(paths) for split, paths in written.items()}
    sys.stdout.write(
        json.dumps({"out_dir": str(args.out_dir), "shards": summary}, sort_keys=True)
        + "\n"
    )
    return 0


def run_hf_push(args: HfPushCliArgs) -> int:
    if args.out_dir is not None:
        return run_local_layout(args)
    repo_id = args.repo_id
    if not repo_id:
        cli_error("--repo-id is required unless --out-dir is given")
        return 2
    if args.private and args.public:
        cli_error("choose at most one of --private / --public")
        return 2

    private: bool | None
    if args.private:
        private = True
    elif args.public:
        private = False
    else:
        private = None

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        cli_error(f"dataset dir is not a directory: {dataset_dir}")
        return 2

    if not args.private:
        try:
            on_disk = load_from_disk(str(dataset_dir))
        except (FileNotFoundError, OSError, ValueError):
            on_disk = None
        if isinstance(on_disk, DatasetDict) and EMBEDDING_CONFIG_NAME in on_disk:
            cli_error(f"{EMBEDDING_CONFIG_NAME} requires --private Hub push")
            return 2

    if resolve_hf_token(token=args.token) is None:
        cli_error(
            "missing HF token. Set `HF_TOKEN` (preferred) or `HUGGING_FACE_HUB_TOKEN`, "
            "or pass `--token`."
        )
        return 2

    readme_path: Path | None = None
    if args.dataset_readme is not None:
        readme_path = Path(args.dataset_readme)
        if not readme_path.is_file():
            cli_error(f"dataset readme path is not a file: {readme_path}")
            return 2

    if _features_provenance_requires_inventory(dataset_dir):
        inv = dataset_dir / "text_provenance_inventory.json"
        if not inv.is_file():
            cli_error(
                "features provenance build requires text_provenance_inventory.json "
                f"in {dataset_dir} (refusing Hub push)"
            )
            return 2

    try:
        push_hf_datasetdict_from_disk_to_hub(
            dataset_dir,
            repo_id,
            config_name=args.config_name,
            private=private,
            revision=args.revision,
            commit_message=args.commit_message,
            commit_description=args.commit_description,
            create_pr=bool(args.create_pr),
            max_shard_size=args.max_shard_size,
            token=args.token,
        )
    except MultiSplitCreatePrNotSupportedError as exc:
        cli_error(str(exc))
        return 2

    if readme_path is not None:
        upload_hf_dataset_readme(
            readme_path,
            repo_id,
            revision=args.revision,
            commit_message=args.commit_message,
            commit_description=args.commit_description,
            create_pr=bool(args.create_pr),
            token=args.token,
        )

    for aux_name in _AUX_UPLOAD_NAMES:
        aux_path = dataset_dir / aux_name
        if aux_path.is_file():
            hf_hub.upload_hf_dataset_file(
                aux_path,
                repo_id,
                aux_name,
                revision=args.revision,
                commit_message=args.commit_message,
                commit_description=args.commit_description,
                create_pr=bool(args.create_pr),
                token=args.token,
            )
        elif aux_name == "text_provenance_inventory.json":
            continue
        else:
            cli_warning(f"skipping upload of {aux_name} (not found in dataset dir)")

    if args.release_tag is not None:
        tag_msg = args.commit_message or f"primehkg_release {args.release_tag}"
        revision = args.revision or "main"
        hf_hub.create_hf_dataset_release_tag(
            repo_id,
            args.release_tag.strip(),
            revision=revision,
            tag_message=tag_msg,
            token=args.token,
        )
        if args.also_tag_latest:
            hf_hub.create_hf_dataset_release_tag(
                repo_id,
                "latest",
                revision=revision,
                tag_message=tag_msg,
                token=args.token,
            )
    return 0


app = typer.Typer(
    rich_markup_mode="rich",
    help=(
        "Push a local Hugging Face DatasetDict directory to the Hugging Face Hub, "
        "or write the same Hub layout to a local --out-dir."
    ),
)


def _argv_for_hf_push_cli(argv: list[str] | None) -> list[str]:
    a = list(argv) if argv is not None else list(sys.argv[1:])
    if a and a[0] == "push":
        return a[1:]
    return a


@app.command("push")
def push_command(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Directory from `datasets.DatasetDict.save_to_disk` (e.g. from write_hf_tables_to_disk).",
    ),
    repo_id: str | None = typer.Option(
        None,
        "--repo-id",
        help="Target Hub repo id, e.g. `org/primehkg-core` (required unless --out-dir).",
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help=(
            "Write the Hub layout (Parquet shards per split + build_manifest.json + "
            "sidecars/) to this local directory instead of uploading. No token, no "
            "network. The directory can be passed as hub_repo_id= to load_core_dataset."
        ),
    ),
    config_name: str = typer.Option(
        "default",
        "--config-name",
        help=(
            "HF Datasets config layout. Use 'default' for primehkg-core (one config per "
            "table: nodes, hyperedges, support). See docs/huggingface/hub-config-names.md."
        ),
    ),
    private: bool = typer.Option(
        False,
        "--private",
        help="Create/use a private Hub repo (if supported by your account/token).",
    ),
    public: bool = typer.Option(
        False,
        "--public",
        help="Mark the Hub repo as public (`private=False`).",
    ),
    revision: str | None = typer.Option(
        None, "--revision", help="Optional branch/revision name on the Hub repo."
    ),
    commit_message: str | None = typer.Option(
        None, "--commit-message", help="Optional Hub commit message."
    ),
    commit_description: str | None = typer.Option(
        None, "--commit-description", help="Optional longer Hub commit description."
    ),
    create_pr: bool = typer.Option(
        False,
        "--create-pr",
        help=(
            "Attempt to open a PR instead of committing directly. "
            "Not supported for multi-split PrimeHKG DatasetDict pushes."
        ),
    ),
    max_shard_size: str | None = typer.Option(
        None,
        "--max-shard-size",
        help="Optional shard size for each split's Dataset.push_to_hub (e.g. `500MB`).",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        help=(
            "Optional token override (discouraged). "
            "Prefer `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` in the environment."
        ),
    ),
    dataset_readme: Path | None = typer.Option(
        None,
        "--dataset-readme",
        help="Optional Markdown dataset card to upload as the Hub repo README.md.",
    ),
    release_tag: str | None = typer.Option(
        None,
        "--release-tag",
        help=(
            "After push, create/move a Hub git tag on the target revision "
            "(e.g. 0.1.2 aligned with primehkg_release). See docs/huggingface/hub-release-tags.md."
        ),
    ),
    also_tag_latest: bool = typer.Option(
        False,
        "--also-tag-latest",
        help="When --release-tag is set, also move tag `latest` to the same commit.",
    ),
) -> None:
    pargs = HfPushCliArgs(
        dataset_dir=dataset_dir,
        repo_id=repo_id,
        config_name=config_name,
        private=private,
        public=public,
        revision=revision,
        commit_message=commit_message,
        commit_description=commit_description,
        create_pr=create_pr,
        max_shard_size=max_shard_size,
        token=token,
        dataset_readme=dataset_readme,
        release_tag=release_tag,
        also_tag_latest=also_tag_latest,
        out_dir=out_dir,
    )
    raise typer.Exit(run_hf_push(pargs))


def main(argv: list[str] | None = None) -> int:
    cmd = get_command(app)
    args_list = _argv_for_hf_push_cli(argv)
    try:
        rc = cmd.main(
            args=args_list,
            prog_name="primehkg-hf-push",
            standalone_mode=False,
        )
    except SystemExit as e:
        if isinstance(e.code, int):
            return e.code
        return 1
    return int(rc) if rc is not None else 0


def entrypoint() -> None:
    """Console script entrypoint (returns process exit code via `SystemExit`)."""
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
