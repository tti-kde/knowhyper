#!/usr/bin/env python3
"""Plan the fixed late-logit graph matrix for JNLPBA and NCBI-Disease.

Axes are intentionally closed: two backbones x two datasets x Structure only,
+attributes, +pretraining x real/shuffled x seeds 0--9 = 240 cells.  The zero
arm is the frozen base checkpoint and is not trained.  This is a missing-dataset
extension of the frozen #231 architecture, not a new search.
"""

from __future__ import annotations

import argparse

from knowhyper.span_late_campaign import expand, pack

BACKBONES = ["pubmedbert", "biolinkbert"]
DATASETS = ["jnlpba", "ncbi_disease"]
CONDITIONS = ["edge", "ev", "ev_warm"]
CONTROLS = ["real", "shuffled"]
SEEDS = list(range(10))
CELLS_PER_NODE = 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-cells", action="store_true")
    args = parser.parse_args()
    cells = expand(
        backbones=BACKBONES,
        datasets=DATASETS,
        conditions=CONDITIONS,
        controls=CONTROLS,
        seeds=SEEDS,
        shared_zero=False,
        encoder_root="required-at-runtime",
    )
    nodes = pack(cells, cells_per_node=CELLS_PER_NODE)
    print(f"cells: {len(cells)}")
    print(f"nodes: {len(nodes)} @ {CELLS_PER_NODE} cells/node")
    print("matrix: 2 backbones x 2 datasets x 3 conditions x 2 controls x 10 seeds")
    if args.emit_cells:
        for node in nodes:
            specs = [
                f"{cell.backbone}:{cell.dataset}:{cell.condition}:"
                f"{cell.control}:{cell.seed}"
                for cell in node
            ]
            print(f"CELLS={' '.join(specs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
