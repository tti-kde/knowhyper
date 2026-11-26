#!/usr/bin/env python3
"""Assemble the dataset-root prediction layout for #218 §0.2 steps 5-6.

Proposal generation writes five disjoint `oof_fold*/train.jsonl` parts plus
`full/{validation,test}.jsonl`. The knowledge arm reads
`predictions_root=<root>/<backbone>/<dataset>` and loads `{split}.jsonl` from THAT
directory, so the parts must be merged into one train file and the full-train exports
published beside it.

    uv run python scripts/finalize_span_ner_proposals.py \
        --predictions-root "$KNOWHYPER_HOME/artifacts/predictions/span_ner"

Merging is delegated to `ner_merge_predictions.merge_prediction_files`, which rejects
duplicate ids -- overlapping folds mean the OOF partition broke, i.e. leakage, so that
must fail rather than silently dedupe. `--expected-train-rows` additionally pins the
corpus size: a short merge would train the knowledge arm on a subset and still report
a normal score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ner_merge_predictions import merge_prediction_files

# BLURB train sizes, used as the default expected merge count per dataset.
TRAIN_ROWS: dict[str, int] = {
    "bc5cdr_chem": 4560,
    "bc2gm": 12574,
    "bc5cdr_disease": 4560,
    "jnlpba": 16807,
    "ncbi_disease": 5424,
}


@dataclass(frozen=True)
class FinalizeResult:
    """What one finalized cell contains."""

    train_rows: int
    train_sha: str
    validation_rows: int
    test_rows: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def finalize_cell(
    base: Path,
    *,
    num_folds: int = 5,
    expected_train_rows: int | None = None,
) -> FinalizeResult:
    """Merge one backbone x dataset cell's folds and publish all three splits."""
    parts = []
    for fold in range(num_folds):
        part = base / f"oof_fold{fold}" / "train.jsonl"
        if not part.is_file():
            msg = f"OOF part missing: {part}"
            raise FileNotFoundError(msg)
        parts.append(part)

    # Write to a temp name then replace, so a failed merge cannot leave a partial
    # train.jsonl that later reads as complete.
    merged_tmp = base / "train.jsonl.tmp"
    train_rows = merge_prediction_files(
        parts, merged_tmp, expected_count=expected_train_rows
    )
    merged_tmp.replace(base / "train.jsonl")

    published: dict[str, Path] = {"train": base / "train.jsonl"}
    for split in ("validation", "test"):
        source = base / "full" / f"{split}.jsonl"
        if not source.is_file():
            msg = f"full-train export missing: {source}"
            raise FileNotFoundError(msg)
        target = base / f"{split}.jsonl"
        shutil.copyfile(source, target)
        published[split] = target

    splits = {
        name: {"rows": _rows(path), "sha256": _sha256(path)}
        for name, path in published.items()
    }
    manifest = {
        "num_folds": num_folds,
        "splits": splits,
        "sources": {
            "train": [str(p.relative_to(base)) for p in parts],
            "validation": "full/validation.jsonl",
            "test": "full/test.jsonl",
        },
    }
    (base / "predictions.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return FinalizeResult(
        train_rows=train_rows,
        train_sha=splits["train"]["sha256"],
        validation_rows=splits["validation"]["rows"],
        test_rows=splits["test"]["rows"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions-root", required=True, type=Path)
    ap.add_argument("--num-folds", type=int, default=5)
    ap.add_argument(
        "--no-expected-rows",
        action="store_true",
        help="skip the corpus row-count check (not for campaign artifacts)",
    )
    args = ap.parse_args()

    failures = 0
    finalized = 0
    for backbone_dir in sorted(
        p for p in args.predictions_root.iterdir() if p.is_dir()
    ):
        for dataset_dir in sorted(p for p in backbone_dir.iterdir() if p.is_dir()):
            expected = (
                None if args.no_expected_rows else TRAIN_ROWS.get(dataset_dir.name)
            )
            try:
                result = finalize_cell(
                    dataset_dir,
                    num_folds=args.num_folds,
                    expected_train_rows=expected,
                )
            except (FileNotFoundError, ValueError) as error:
                print(
                    f"FAIL {backbone_dir.name}/{dataset_dir.name}: {error}",
                    file=sys.stderr,
                )
                failures += 1
                continue
            finalized += 1
            print(
                f"OK   {backbone_dir.name}/{dataset_dir.name}: "
                f"train {result.train_rows} (sha {result.train_sha[:16]}) "
                f"val {result.validation_rows} test {result.test_rows}"
            )

    print(f"\nfinalized {finalized} cells, {failures} failures")
    if finalized == 0:
        print("FATAL finalized 0 cells; wrong root?", file=sys.stderr)
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
