"""Build the type-matched substitution pool for the endpoint shuffled control.

The pool is derived from the SAME SubHG cache the arms read, so the substitutes
are entities that actually occur as endpoints in this dataset rather than
arbitrary graph nodes. It is built once at construction and is deterministic, so
the shuffled arm is reproducible across the train and eval passes of a cell.

Node ids carry their type: ``primekg:<type>:<local id>``. DDI's test split is
mostly ``drug`` but also contains ``exposure``, ``disease``,
``molecular_function``, ``biological_process``, ``pathway`` and
``effect/phenotype`` seeds, plus a small tail of ids that do not parse. That
spread is why the control is type-matched: a batch roll would substitute a
pathway vector for a drug and confound wrong-identity with wrong-type.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

#: Ids that do not match ``prefix:type:rest`` are pooled under this key so they
#: substitute only among themselves rather than being treated as one big type.
UNKNOWN_TYPE = "__unparsed__"


def node_type_of(node_id: str) -> str:
    """Return the type segment of a ``primekg:<type>:<id>`` node id."""
    parts = node_id.split(":")
    if len(parts) < 3:
        return UNKNOWN_TYPE
    return parts[1]


def build_endpoint_pool(
    *cache_dirs: str | Path,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return ``(pool_by_type, node_type_by_id)`` from cached SubHG splits.

    Only ``seed_node_ids`` are collected: the substitute for an endpoint must be
    something that occurs as an endpoint, not any node that happened to be
    retrieved. Pools are sorted so the mapping does not depend on filesystem
    order.

    Pass EVERY split the arm will see, not just train. The substitution looks a
    seed up by type, and a seed absent from the pool falls back to itself -- so a
    train-only pool would leave validation- and test-only entities unshuffled,
    silently weakening the control exactly on the split the headline number comes
    from. Types are derived from the id, so a missing seed still resolves to a
    type; pooling all splits is what guarantees the type has members to draw from.
    """
    paths = [Path(cache_dir) for cache_dir in cache_dirs]
    if not paths:
        msg = "build_endpoint_pool needs at least one cache directory"
        raise ValueError(msg)
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        msg = f"endpoint pool cache_dir does not exist: {', '.join(missing)}"
        raise FileNotFoundError(msg)

    by_type: dict[str, set[str]] = defaultdict(set)
    node_types: dict[str, str] = {}
    seen_any = False
    for cache_path in paths:
        for path in sorted(cache_path.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            seen_any = True
            for seed in payload.get("seed_node_ids") or ():
                type_key = node_type_of(seed)
                by_type[type_key].add(seed)
                node_types[seed] = type_key

    if not seen_any:
        msg = (
            f"endpoint pool found no readable cache entries under "
            f"{', '.join(str(path) for path in paths)}; an empty pool would "
            "silently disable the shuffle and make the control identical to the "
            "treatment"
        )
        raise ValueError(msg)

    return {key: sorted(values) for key, values in by_type.items()}, node_types


def resolve_pool_dir(dataset: str, fingerprint: str, split: str = "train") -> Path:
    """Locate the cached SubHG split the pool is drawn from.

    The fingerprint is passed in rather than written into a config: a literal
    here would silently go stale the moment any identity input moves, which is
    the failure `knowhyper.pins` exists to prevent. `$KNOWHYPER_HOME` is the
    documented central root, so the layout is derived, not configured.
    """
    import os

    home = os.environ.get("KNOWHYPER_HOME")
    if not home:
        msg = (
            "KNOWHYPER_HOME is unset, so the endpoint shuffle pool cannot be "
            "located; export it rather than defaulting to a relative path"
        )
        raise RuntimeError(msg)
    return Path(home) / "cache" / "subhg" / "v2" / fingerprint / dataset / split
