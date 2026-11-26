#!/usr/bin/env python
"""Plan the #220 span Knowledge-as-text campaign and emit node lines.

    # Phase 2 (n=3): 54 cells = 36 text + 18 matched base
    uv run python scripts/plan_span_ner_text.py --seeds 0-2 --cells-per-node 4

    # node lines for abci sweep -- note the grep: --emit-cells also prints a
    # human-readable summary, and feeding that into the file breaks the array
    uv run python scripts/plan_span_ner_text.py --seeds 0-2 --cells-per-node 4 \
        --emit-cells | grep '^CELLS=' | sed 's/^CELLS=//' > node_lines_phase2.txt

`base` is planned as a first-class arm: Phase 0.3 established that #218's
`shared_zero` carries ~11.8M fusion parameters in a 10x-LR group and is therefore
not a matched comparator. That is what takes Phase 3 from 120 to 180 cells.
"""

from __future__ import annotations

import argparse
from collections import Counter

from knowhyper.span_text_campaign import (
    ARMS,
    BACKBONES,
    DATASETS,
    SpanTextCampaignError,
    expand,
    pack,
)

# Measured on #218's comparable span cells; used only for a wall-clock estimate.
MINUTES_PER_CELL = 20


def _parse_seeds(tokens: list[str]) -> list[int]:
    """Accept `0-9`, `0,1,2` and bare integers, in the order given."""
    seeds: list[int] = []
    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part[1:]:
                start, _, end = part.partition("-")
                seeds.extend(range(int(start), int(end) + 1))
            else:
                seeds.append(int(part))
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbones", nargs="+", default=sorted(BACKBONES))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", default=["0-9"])
    parser.add_argument(
        "--cells-per-node",
        type=int,
        default=4,
        help=(
            "rt_HF has been observed delivering 4 GPUs rather than 8 on this "
            "reservation; size this from what the node actually reports"
        ),
    )
    parser.add_argument("--emit-cells", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        print("no seeds selected")
        return 2

    try:
        cells = expand(
            backbones=list(args.backbones),
            datasets=list(args.datasets),
            arms=list(args.arms),
            seeds=seeds,
        )
        nodes = pack(cells, cells_per_node=args.cells_per_node)
    except SpanTextCampaignError as exc:
        print(f"FATAL {exc}")
        return 2

    # Two live writers in one run directory lose both runs, so this is fatal rather
    # than a warning.
    run_dirs = [cell.run_dir for cell in cells]
    duplicates = [d for d, n in Counter(run_dirs).items() if n > 1]
    if duplicates:
        print(f"FATAL {len(duplicates)} duplicate run directories, e.g.:")
        for duplicate in duplicates[:5]:
            print(f"  {duplicate}")
        return 2

    print(f"cells: {len(cells)}")
    print(f"nodes: {len(nodes)} @ {args.cells_per_node} cells/node")
    # Cells on a node run concurrently, one per GPU, so a node costs about one
    # cell's wall-clock -- not the sum. Size walltime from a CONCURRENT
    # measurement, never a single-cell probe.
    print(f"est. wall-clock: ~{MINUTES_PER_CELL} min/node (cells run concurrently)")

    by_arm = Counter(cell.arm for cell in cells)
    by_backbone = Counter(cell.backbone for cell in cells)
    by_dataset = Counter(cell.dataset for cell in cells)
    print("by arm:      " + "  ".join(f"{k}={v}" for k, v in sorted(by_arm.items())))
    print(
        "by backbone: " + "  ".join(f"{k}={v}" for k, v in sorted(by_backbone.items()))
    )
    print(
        "by dataset:  " + "  ".join(f"{k}={v}" for k, v in sorted(by_dataset.items()))
    )

    if args.emit_cells:
        for node in nodes:
            specs = " ".join(f"{c.backbone}:{c.dataset}:{c.arm}:{c.seed}" for c in node)
            print(f"CELLS={specs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
