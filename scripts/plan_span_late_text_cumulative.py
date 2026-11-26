#!/usr/bin/env python3
"""Plan the one fixed #220 text + #231 ev_warm cumulative NER matrix.

The axes are deliberately not configurable: 2 backbones x 3 datasets x
real/shuffled graph evidence x 10 seeds = 120 cells. Allowing an arbitrary condition,
head or sweep option here would reopen the architecture search this experiment closes.
"""

from __future__ import annotations

import argparse

BACKBONES = ("pubmedbert", "biolinkbert")
DATASETS = ("bc5cdr_chem", "bc2gm", "bc5cdr_disease")
CONTROLS = ("real", "shuffled")
SEEDS = tuple(range(10))
CELLS_PER_NODE = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-cells", action="store_true")
    args = parser.parse_args()

    cells = [
        f"{backbone}:{dataset}:text_ev_warm:{control}:{seed}"
        for backbone in BACKBONES
        for dataset in DATASETS
        for seed in SEEDS
        for control in CONTROLS
    ]
    nodes = [
        cells[index : index + CELLS_PER_NODE]
        for index in range(0, len(cells), CELLS_PER_NODE)
    ]
    print(f"cells: {len(cells)}")
    print(f"nodes: {len(nodes)} @ {CELLS_PER_NODE} cells/node")
    print("matrix: 2 backbones x 3 datasets x 2 controls x 10 seeds")
    if args.emit_cells:
        for node in nodes:
            print(f"CELLS={' '.join(node)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
