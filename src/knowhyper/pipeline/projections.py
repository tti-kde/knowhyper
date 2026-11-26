"""Deterministic structural controls for retrieved sub-hypergraphs."""

from __future__ import annotations

from itertools import combinations

from knowhyper.pipeline.types import HyperedgeRef, SubHypergraph


def binary_clique_projection(sub: SubHypergraph) -> SubHypergraph:
    """Return the canonical binary (2-section) projection of ``sub``.

    Hyperedges with at most two distinct participants are retained unchanged.
    Each higher-arity hyperedge is replaced by every unordered participant pair.
    The pair edges copy the parent semantics, polarity, score, score breakdown,
    and per-participant roles. Node order, seeds, and provenance are unchanged, so
    the projection can be applied after retrieval without changing cache identity.

    Repeated participant ids in a higher-arity edge are collapsed in first-seen
    order. This prevents malformed input from creating self-pairs or duplicate
    binary edges while keeping the transformation deterministic.
    """
    projected: list[HyperedgeRef] = []
    for hyperedge in sub.hyperedges:
        participants = tuple(dict.fromkeys(hyperedge.node_ids))
        if len(participants) <= 2:
            if participants == hyperedge.node_ids:
                projected.append(hyperedge)
            else:
                projected.append(
                    _copy_binary_edge(hyperedge, participants, suffix="deduplicated")
                )
            continue

        for left_index, right_index in combinations(range(len(participants)), 2):
            pair = (participants[left_index], participants[right_index])
            projected.append(
                _copy_binary_edge(
                    hyperedge,
                    pair,
                    suffix=f"pair:{left_index}:{right_index}",
                )
            )

    return SubHypergraph(
        seed_node_ids=sub.seed_node_ids,
        node_ids=sub.node_ids,
        hyperedges=tuple(projected),
        provenance=sub.provenance,
    )


def _copy_binary_edge(
    parent: HyperedgeRef, node_ids: tuple[str, ...], *, suffix: str
) -> HyperedgeRef:
    """Copy parent evidence onto one projected edge."""
    return HyperedgeRef(
        he_id=f"{parent.he_id}::binary:{suffix}",
        semantics=parent.semantics,
        node_ids=node_ids,
        roles={
            node_id: parent.roles[node_id]
            for node_id in node_ids
            if node_id in parent.roles
        },
        polarity=parent.polarity,
        score=parent.score,
        score_breakdown=parent.score_breakdown,
    )
