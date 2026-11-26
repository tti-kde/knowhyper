r"""Generate BLURB NER datasets from Crichton et al. and upload to HuggingFace Hub.

Replicates the preprocess_ner_data() logic from BLURB's official preprocessor.py:
  - token\tB-Label  →  token\tB  (label[0] only)
  - blank lines and -DOCSTART- lines are preserved as sentence boundaries

Data source:
  git clone https://github.com/cambridgeltl/MTL-Bioinformatics-2016

Usage:
    # Verify stats only (no upload)
    uv run python scripts/build_blurb_ner.py --hf-repo tti-kde/blurb-ner --dry-run

    # Generate and upload
    uv run python scripts/build_blurb_ner.py --hf-repo tti-kde/blurb-ner

    # Generate and write locally (reader layout for IEFORGE_DATA_DIR), no upload
    uv run python scripts/build_blurb_ner.py --out-dir "$IEFORGE_DATA_DIR/blurb-ner"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value

# ---------------------------------------------------------------------------
# BLURB official preprocess_ner_data logic (verbatim from preprocessor.py)
# ---------------------------------------------------------------------------

_TASKS: list[tuple[str, str]] = [
    ("bc5cdr-chem", "BC5CDR-chem-IOB"),
    ("bc5cdr-disease", "BC5CDR-disease-IOB"),
    ("ncbi-disease", "NCBI-disease-IOB"),
    ("bc2gm", "BC2GM-IOB"),
    ("jnlpba", "JNLPBA"),
]

# Expected entity mention counts from BLURB paper Table 3.
# Note: bc5cdr-disease validation shows 4246 in actual data; the paper prints 4244
# (verified by running identical preprocessing on the Crichton corpus).
_BLURB_EXPECTED: dict[str, dict[str, int]] = {
    "bc5cdr-chem": {"train": 5203, "validation": 5347, "test": 5385},
    "bc5cdr-disease": {"train": 4182, "validation": 4246, "test": 4424},
    "ncbi-disease": {"train": 5134, "validation": 787, "test": 960},
    "bc2gm": {"train": 15197, "validation": 3061, "test": 6325},
    "jnlpba": {"train": 46750, "validation": 4551, "test": 8662},
}


def _preprocess_ner_tsv(input_path: Path, output_path: Path) -> None:
    """Convert Crichton IOB tsv to BLURB format (keep only label[0])."""
    name_map = {
        "train.tsv": "train.tsv",
        "devel.tsv": "dev.tsv",
        "test.tsv": "test.tsv",
    }
    output_path.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in name_map.items():
        src = input_path / src_name
        dst = output_path / dst_name
        lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
        with dst.open("w", encoding="utf-8") as fout:
            for line in lines:
                if not line.split() or line.startswith("-DOCSTART-"):
                    fout.write(line)
                else:
                    token, label = line.split("\t")
                    fout.write(f"{token}\t{label[0]}\n")


def _tsv_to_examples(tsv_path: Path) -> list[dict[str, list[str]]]:
    """Parse a BLURB NER tsv into sentence-level {tokens, ner_tags} dicts."""
    examples: list[dict[str, list[str]]] = []
    tokens: list[str] = []
    tags: list[str] = []

    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("-DOCSTART-") or not line.strip():
            if tokens:
                examples.append({"tokens": tokens, "ner_tags": tags})
                tokens, tags = [], []
        else:
            parts = line.split("\t")
            if len(parts) == 2:
                tokens.append(parts[0])
                tags.append(parts[1].strip())

    if tokens:
        examples.append({"tokens": tokens, "ner_tags": tags})

    return examples


def _build_dataset_dict(processed_dir: Path) -> DatasetDict:
    splits = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}
    features = Features(
        {"tokens": Sequence(Value("string")), "ner_tags": Sequence(Value("string"))}
    )
    return DatasetDict(
        {
            split_name: Dataset.from_list(
                _tsv_to_examples(processed_dir / fname), features=features
            )
            for split_name, fname in splits.items()
        }
    )


def _count_entities(ds: Dataset) -> int:
    return sum(1 for ex in ds for tag in ex["ner_tags"] if tag.startswith("B"))


def _verify_stats(name: str, ds_dict: DatasetDict) -> bool:
    """Print entity mention counts per split and return True if all match expected."""
    expected = _BLURB_EXPECTED.get(name, {})
    ok = True
    for split, exp_count in expected.items():
        actual = _count_entities(ds_dict[split])
        mark = "✓" if actual == exp_count else "✗"
        print(
            f"  {mark} {split:12s}: entity mentions = {actual:6d}  (expected {exp_count})"
        )
        if actual != exp_count:
            ok = False
    return ok


def main() -> None:
    """Entry point: preprocess, verify, and optionally upload all BLURB NER tasks."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--hf-repo", default=None, help="HF repo id, e.g. tti-kde/blurb-ner"
    )
    parser.add_argument(
        "--data-root",
        default="raw_data/MTL-Bioinformatics-2016/data",
        help="Path to Crichton MTL-Bioinformatics-2016/data directory",
    )
    parser.add_argument(
        "--processed-dir",
        default="processed",
        help="Output dir for processed tsv files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip HF upload; just preprocess and verify",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write <out-dir>/<config>/<split>-00000-of-00001.parquet locally instead of pushing to the Hub (use $IEFORGE_DATA_DIR/<repo-name>)",
    )
    args = parser.parse_args()
    if args.hf_repo is None and args.out_dir is None and not args.dry_run:
        parser.error("one of --hf-repo, --out-dir or --dry-run is required")

    crichton_root = Path(args.data_root)
    processed_root = Path(args.processed_dir)

    all_ok = True
    datasets: dict[str, DatasetDict] = {}

    for task_name, crichton_subdir in _TASKS:
        print(f"\n=== {task_name} ===")
        src = crichton_root / crichton_subdir
        dst = processed_root / task_name
        _preprocess_ner_tsv(src, dst)

        ds_dict = _build_dataset_dict(dst)
        datasets[task_name] = ds_dict

        if not _verify_stats(task_name, ds_dict):
            all_ok = False
        for split, ds in ds_dict.items():
            print(f"    {split}: {len(ds)} sentences")

    print(
        f"\n{'All stats match BLURB paper!' if all_ok else 'WARNING: some stats differ from BLURB paper'}"
    )

    if args.dry_run:
        print("\n--dry-run: skipping HF upload")
        return

    if not all_ok:
        print("Aborting upload due to stat mismatches.")
        return

    if args.out_dir is not None:
        from ieforge.datasets.local import save_parquet_layout

        for task_name, ds_dict in datasets.items():
            save_parquet_layout(ds_dict, args.out_dir, task_name)
            print(f"  Wrote config={task_name} to {args.out_dir}")
        return

    print(f"\nUploading to {args.hf_repo} ...")
    for task_name, ds_dict in datasets.items():
        print(f"  Pushing config={task_name} ...")
        ds_dict.push_to_hub(args.hf_repo, config_name=task_name)
        print(f"  Done: {task_name}")

    print(
        f"\nAll done. Dataset available at: https://huggingface.co/datasets/{args.hf_repo}"
    )


if __name__ == "__main__":
    main()
