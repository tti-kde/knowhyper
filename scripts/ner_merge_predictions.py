"""Merge disjoint OOF NER prediction JSONL parts with strict validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def merge_prediction_files(
    inputs: list[Path], output: Path, *, expected_count: int | None = None
) -> int:
    """Merge prediction parts, rejecting malformed rows and duplicate IDs."""
    records: dict[str, dict[str, Any]] = {}
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                msg = f"Prediction row must be an object: {path}:{line_no}"
                raise TypeError(msg)
            prediction_id = str(payload.get("id", ""))
            if not prediction_id:
                msg = f"Missing prediction id: {path}:{line_no}"
                raise ValueError(msg)
            if prediction_id in records:
                msg = f"Duplicate prediction id {prediction_id!r}: {path}:{line_no}"
                raise ValueError(msg)
            records[prediction_id] = payload

    if expected_count is not None and len(records) != expected_count:
        msg = f"Merged {len(records)} predictions, expected {expected_count}"
        raise ValueError(msg)

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, int | str]:
        prediction_id = item[0]
        try:
            return (0, int(prediction_id))
        except ValueError:
            return (1, prediction_id)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for _, payload in sorted(records.items(), key=sort_key):
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return len(records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge OOF NER predictions")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    count = merge_prediction_files(
        args.inputs, args.output, expected_count=args.expected_count
    )
    print(f"Wrote {count} merged predictions to {args.output}")


if __name__ == "__main__":
    main()
