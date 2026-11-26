from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def merge_triplets_dedupe_kg_first(
    kg_triplets: list[dict[str, str]],
    more_triplets: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Append ``more_triplets`` after ``kg_triplets``, skipping edges that repeat ``(s, r, t)``.

    ``kg_triplets`` is preserved **row-for-row** (including repeated identical rows).
    Each appended triplet from ``more_triplets`` is dropped if its ``(s, r, t)``
    already appears anywhere in ``kg_triplets``.
    """
    seen: set[tuple[str, str, str]] = {(t["s"], t["r"], t["t"]) for t in kg_triplets}
    out: list[dict[str, str]] = [dict(t) for t in kg_triplets]
    for t in more_triplets:
        key = (t["s"], t["r"], t["t"])
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(t))
    return out


def load_primekg_edges_triplets(
    path: str | Path,
    index_to_node_id: dict[int, str],
    *,
    strict: bool = True,
) -> list[dict[str, str]]:
    """Load PrimeKG ``edges.csv`` as (s, r, t) triplets using a node_index join map.

    Expected columns (Dataverse): ``relation``, ``display_relation``, ``x_index``, ``y_index``.
    Uses ``relation`` as the triplet relation string ``r``.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            msg = "PrimeKG edges.csv is missing a header row."
            raise ValueError(msg)
        fieldnames_obj = reader.fieldnames
        if fieldnames_obj is None:
            msg = "PrimeKG edges.csv is missing a header row."
            raise ValueError(msg)
        fieldnames_list = list(fieldnames_obj)
        stripped = [c.strip() for c in fieldnames_list if c is not None]
        if len(set(stripped)) != len(stripped):
            counts = Counter(stripped)
            dupes = sorted([k for k, c in counts.items() if c > 1])
            msg = (
                "PrimeKG edges.csv header contains duplicate column names after stripping whitespace: "
                + ", ".join(dupes)
            )
            raise ValueError(msg)
        cols = {c.strip(): c for c in fieldnames_list if c is not None}
        required = {"relation", "x_index", "y_index"}
        missing = required - set(cols)
        if missing:
            msg = f"PrimeKG edges.csv missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        out: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            rel = (row.get(cols["relation"]) or "").strip()
            xi_raw = (row.get(cols["x_index"]) or "").strip()
            yi_raw = (row.get(cols["y_index"]) or "").strip()
            if not rel:
                if strict:
                    msg = f"empty relation at edges.csv row {row_number}"
                    raise ValueError(msg)
                continue
            try:
                xi = int(xi_raw)
                yi = int(yi_raw)
            except ValueError:
                if strict:
                    msg = f"invalid x_index/y_index at edges.csv row {row_number}"
                    raise ValueError(msg) from None
                continue
            s = index_to_node_id.get(xi)
            t = index_to_node_id.get(yi)
            if not s or not t:
                if strict:
                    msg = (
                        f"edges.csv row {row_number}: node_index not in join map "
                        f"(x_index={xi_raw!r}, y_index={yi_raw!r})"
                    )
                    raise ValueError(msg)
                continue
            out.append({"s": s, "r": rel, "t": t})
        return out
