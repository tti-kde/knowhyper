from __future__ import annotations

from typing import Any, TypedDict

from primehkg.builders.dedup import HyperedgeDedupStats, dedupe_hyperedges_by_id


class MergedHypergraphBundle(TypedDict):
    """Merged builder output with hyperedge dedup statistics."""

    hyperedges: list[dict[str, Any]]
    support: list[dict[str, Any]]
    hyperedge_dedup: HyperedgeDedupStats


def merge_hypergraph_segments(
    segments: list[tuple[str, dict[str, list[dict[str, Any]]]]],
) -> MergedHypergraphBundle:
    """Merge builder outputs into one ``hyperedges`` + ``support`` bundle.

    Each builder typically allocates its own ``support_id`` space (for example
    ``sup:1``, ``sup:2``, …). Combining segments without remapping can break the
    join key between ``hyperedges[*].support_id`` and ``support[*].support_id``.

    This function **renumbers** support rows to a single global namespace
    ``sup:1``, ``sup:2``, … in **caller segment order**, then **row order within**
    each segment's ``support`` list. Hyperedges are copied with ``support_id``
    rewritten using the per-segment map.

    ``segments`` is ``(builder_slug, {"hyperedges": [...], "support": [...]})``.
    ``builder_slug`` is informational (logging, future metadata); merge order is
    the list order you pass.

    See ``docs/builders/merging-builder-outputs.md``.
    """
    hyperedges_out: list[dict[str, Any]] = []
    support_out: list[dict[str, Any]] = []

    for _slug, part in segments:
        hes = part.get("hyperedges")
        sups = part.get("support")
        if not isinstance(hes, list) or not isinstance(sups, list):
            msg = "each segment must have list hyperedges and list support"
            raise TypeError(msg)

        id_map: dict[str, str] = {}
        for srow in sups:
            if not isinstance(srow, dict):
                msg = "support rows must be dicts"
                raise TypeError(msg)
            old = srow.get("support_id")
            if not isinstance(old, str) or not old:
                msg = f"support row missing support_id: {srow!r}"
                raise ValueError(msg)
            new_id = f"sup:{len(support_out) + 1}"
            if old in id_map:
                msg = f"duplicate support_id {old!r} within one segment"
                raise ValueError(msg)
            id_map[old] = new_id
            row = dict(srow)
            row["support_id"] = new_id
            if isinstance(row.get("sources"), list):
                row["sources"] = list(row["sources"])
            if isinstance(row.get("support_triplets"), list):
                row["support_triplets"] = list(row["support_triplets"])
            support_out.append(row)

        for he in hes:
            if not isinstance(he, dict):
                msg = "hyperedge rows must be dicts"
                raise TypeError(msg)
            sid = he.get("support_id")
            if not isinstance(sid, str) or not sid:
                msg = f"hyperedge missing support_id: {he!r}"
                raise ValueError(msg)
            if sid not in id_map:
                msg = (
                    f"hyperedge support_id {sid!r} not found in this segment's "
                    f"support table (keys mapped: {sorted(id_map)!r})"
                )
                raise ValueError(msg)
            he_out = dict(he)
            he_out["support_id"] = id_map[sid]
            hyperedges_out.append(he_out)

    hyperedges_out, dedup_stats, collapsed = dedupe_hyperedges_by_id(hyperedges_out)

    if collapsed:
        support_by_id = {s["support_id"]: s for s in support_out}
        dropped_ids: set[str] = set()
        for surviving_sid, dropped_sids in collapsed.items():
            survivor = support_by_id.get(surviving_sid)
            if survivor is None:
                continue
            for dropped_sid in dropped_sids:
                donor = support_by_id.get(dropped_sid)
                if donor is None:
                    continue
                for src in donor.get("sources", []):
                    if src not in survivor.setdefault("sources", []):
                        survivor["sources"].append(src)
                for trip in donor.get("support_triplets", []):
                    if trip not in survivor.setdefault("support_triplets", []):
                        survivor["support_triplets"].append(trip)
                dropped_ids.add(dropped_sid)
        if dropped_ids:
            support_out = [s for s in support_out if s["support_id"] not in dropped_ids]

    return {
        "hyperedges": hyperedges_out,
        "support": support_out,
        "hyperedge_dedup": dedup_stats,
    }
