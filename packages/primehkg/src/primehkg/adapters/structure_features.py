from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from primehkg.adapters.pyg_hyper import PygHyperIncidence


def build_structure_feature_skeleton(
    incidence: dict[str, Any] | PygHyperIncidence,
) -> dict[str, Any]:
    """Build deterministic structure-only node features from incidence JSON (D-layer starter).

    Emits log1p hyperedge participation degree per node. No text or sequence modalities.
    """
    node_ids_obj = incidence.get("node_ids")
    incidence_pairs = incidence.get("incidence")
    if not isinstance(node_ids_obj, list) or not isinstance(incidence_pairs, list):
        msg = "incidence must contain node_ids list and incidence list"
        raise TypeError(msg)

    node_ids = [n for n in node_ids_obj if isinstance(n, str)]
    degree = [0] * len(node_ids)
    for pair in incidence_pairs:
        if not (isinstance(pair, list | tuple) and len(pair) == 2):
            continue
        ni, _hi = pair[0], pair[1]
        if isinstance(ni, int) and 0 <= ni < len(degree):
            degree[ni] += 1

    values = [float(math.log1p(d)) for d in degree]
    return {
        "version": 1,
        "kind": "structure_log1p_degree",
        "node_ids": node_ids,
        "feature_name": "log1p_hyperedge_degree",
        "values": values,
    }
