from __future__ import annotations

from collections import defaultdict
from typing import Any

from primehkg.ids import he_id, normalize_node_id
from primehkg.policy.registry import load_relation_registry, normalize_relation


def build_protein_protein_bundle_hyperedges(
    triplets: list[dict[str, Any]],
    *,
    source_version: str = "unknown",
    max_partners_per_anchor: int = 8,
    min_partners_per_bundle: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    """Bundle multiple ``protein_protein`` neighbors around an anchor protein."""
    reg = load_relation_registry()
    neighbors: dict[str, set[str]] = defaultdict(set)
    pair_edges: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)

    valid: list[tuple[str, str, str]] = []
    for tr in triplets:
        s = tr.get("s")
        r = tr.get("r")
        t = tr.get("t")
        if not (isinstance(s, str) and isinstance(r, str) and isinstance(t, str)):
            continue
        s = normalize_node_id(s)
        r = r.strip()
        t = normalize_node_id(t)
        if not (s and r and t):
            continue
        valid.append((s, r, t))

    for s, r, t in sorted(valid, key=lambda x: (x[1], x[0], x[2])):
        if normalize_relation(reg, r) != "protein_protein":
            continue
        if not (
            s.startswith("primekg:protein:")
            and t.startswith("primekg:protein:")
            and s != t
        ):
            continue
        lo, hi = (s, t) if s < t else (t, s)
        pair_edges[(lo, hi)].add((s, r, t))
        neighbors[s].add(t)
        neighbors[t].add(s)

    hyperedges: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []

    for anchor in sorted(neighbors):
        partners_sorted = sorted(neighbors[anchor])
        if len(partners_sorted) < int(min_partners_per_bundle):
            continue
        keep = partners_sorted[: max(0, int(max_partners_per_anchor))]
        if len(keep) < int(min_partners_per_bundle):
            continue

        support_set: set[tuple[str, str, str]] = set()
        for partner in keep:
            lo, hi = (anchor, partner) if anchor < partner else (partner, anchor)
            support_set.update(pair_edges.get((lo, hi), set()))

        support_triplets = [
            {"s": s, "r": r, "t": t}
            for s, r, t in sorted(support_set, key=lambda x: (x[1], x[0], x[2]))
        ]

        rule_trace = {
            "builder": "protein_protein_bundle",
            "version": 1,
            "kind": "composed",
            "join_keys": ["protein_protein"],
            "generation_policy": {
                "max_partners_per_anchor": int(max_partners_per_anchor),
                "min_partners_per_bundle": int(min_partners_per_bundle),
                "tie_break": "partner_id_lex",
            },
        }

        node_ids = [anchor, *keep]
        roles = {anchor: "protein_anchor", **dict.fromkeys(keep, "protein")}
        hid = he_id(
            semantics="protein_protein_bundle",
            node_ids=node_ids,
            roles=roles,
            rule_trace=rule_trace,
        )
        support_id = f"sup:{hid}"

        supports.append(
            {
                "support_id": support_id,
                "sources": [{"name": "PrimeKG", "version": source_version}],
                "support_triplets": support_triplets,
                "rule_trace": rule_trace,
                "confidence": None,
            }
        )
        hyperedges.append(
            {
                "he_id": hid,
                "semantics": "protein_protein_bundle",
                "evidence_grade": "kg_composed",
                "polarity": "undirected",
                "node_ids": node_ids,
                "roles": roles,
                "support_id": support_id,
            }
        )

    return {"hyperedges": hyperedges, "support": supports}
