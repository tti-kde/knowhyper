"""Score-ranked hyperedge selection, shared by the R2b text arm and the graph arms.

R2b's serializer keeps the ``max_hyperedges`` highest-scoring hyperedges of the
retrieved sub-hypergraph; the fusion arms (``edge`` / ``EV``) encode all ``top_k``
of them.  Comparing those two directly therefore confounds the *representation*
question ("does typed structure survive being written as text?") with an
*evidence budget* difference of 3 versus 64 edges -- a null would be
unattributable between "text cannot carry structure" and "the text arm only saw
5% of the graph".

The GRAFT-style baseline (#189) needs a matched-evidence contrast, so the
selection policy is factored out of the serializer into :func:`select_top_ranked`
and applied to the *graph* arm as well through
:class:`TopRankedHyperedgeSelector`.  Both arms then read the same edge IDs by
construction rather than by two implementations agreeing; the parity is pinned by
``tests/pipeline/test_subhg_selection.py``.

Nothing here re-ranks, re-retrieves, or writes a cache.  Selection is a pure,
order-stable function of an already-retrieved :class:`SubHypergraph`, and it
returns a NEW frozen object -- the raw cached sub-HG must stay untouched because
the fusion arms read the same cache entry for their full-budget arms.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from knowhyper.pipeline.types import SubHypergraph

if TYPE_CHECKING:
    from collections.abc import Sequence

    from knowhyper.pipeline.types import HyperedgeRef

#: Bumped when the ranking key or the retention rule changes, so a policy edit
#: yields a different fingerprint instead of silently mixing two selections
#: within one comparison.
SELECTION_POLICY_VERSION = 1
SELECTION_POLICY = "top_ranked_score_he_id_v1"


def select_top_ranked(
    hyperedges: Sequence[HyperedgeRef], max_hyperedges: int
) -> tuple[HyperedgeRef, ...]:
    """Return the ``max_hyperedges`` highest-scoring edges, ordered by ``he_id``.

    The two-stage order is load-bearing and must match
    :meth:`SerializedSubHGSerializer.serialize` exactly:

    * **Select** by descending ``score``, breaking ties on ``he_id``.  Selecting
      by ``he_id`` alone would discard the retriever's ranking and keep an
      arbitrary 3 of 64 edges; leaving ties unbroken would make the choice depend
      on the retriever's incidental output order.
    * **Then order** the survivors by ``he_id``, so a re-run cannot reorder the
      rendered text or the encoded incidence.

    ``max_hyperedges`` larger than the input is not an error -- an instance whose
    retrieval returned fewer edges keeps all of them, which is also what the
    serializer does.
    """
    if max_hyperedges <= 0:
        msg = f"max_hyperedges must be positive, got {max_hyperedges}"
        raise ValueError(msg)
    ranked = sorted(hyperedges, key=lambda edge: (-float(edge.score), str(edge.he_id)))
    return tuple(sorted(ranked[:max_hyperedges], key=lambda edge: str(edge.he_id)))


def selected_edge_ids_hash(hyperedges: Sequence[HyperedgeRef]) -> str:
    """Digest the selected edge IDs, in order, for run provenance.

    Recorded per arm so the matched-evidence claim is auditable after the fact:
    the GRAFT-style text arm and the ``edge_top3`` graph arm must produce the same
    digest on the same instance.  IDs are the graph's own stable identifiers and
    carry no licensed text, so this is payload-free.
    """
    payload = json.dumps(
        [str(edge.he_id) for edge in hyperedges], separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class TopRankedHyperedgeSelector:
    """Restrict a retrieved sub-hypergraph to its top-ranked hyperedges.

    Wired into :class:`~knowhyper.ieforge.injector.KnowHyperInjector` as
    ``subhypergraph_transform``, which runs *after* cache retrieval and *before*
    entity-view augmentation.  That position matters in both directions:

    * After retrieval, so the raw ``top_k=64`` cache stays shared with the
      full-budget arms and needs no second precompute.
    * Before augmentation, so the entity view attaches attribute nodes to the
      3 retained edges' nodes only.  Running it afterwards would let attribute
      hyperedges compete for the 3 slots and the arm would no longer be
      structure-only.

    What the returned sub-hypergraph keeps:

    * the selected hyperedges, in ``he_id`` order,
    * every node participating in a selected hyperedge,
    * the head/tail **seed** nodes, even when no selected edge touches them --
      dropping an unreferenced seed would silently change which instances the
      downstream arms treat as linked, and the entity-view augmenter keys its
      attribute nodes off the seeds, and
    * ``provenance`` verbatim, so linked-mention records (which the serializer
      reads for target-pair roles) survive.  Two selection keys are added; no
      existing key is overwritten.
    """

    def __init__(self, max_hyperedges: int = 3) -> None:
        """Pin the edge budget and derive this policy's fingerprint."""
        if max_hyperedges <= 0:
            msg = (
                "TopRankedHyperedgeSelector needs a positive hyperedge budget, "
                f"got {max_hyperedges}"
            )
            raise ValueError(msg)
        self.max_hyperedges = int(max_hyperedges)
        identity = {
            "policy": SELECTION_POLICY,
            "policy_version": SELECTION_POLICY_VERSION,
            "max_hyperedges": self.max_hyperedges,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        self.fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def __call__(self, sub: SubHypergraph) -> SubHypergraph:
        """Return a new sub-hypergraph holding only the top-ranked hyperedges."""
        return self.select(sub)

    def select(self, sub: SubHypergraph) -> SubHypergraph:
        """Build the restricted sub-hypergraph without mutating ``sub``.

        An empty retrieval passes through with its (empty) edge tuple intact
        rather than raising: 46% of BioRED sub-HGs are empty, and an empty graph
        is inert downstream, not an error.
        """
        selected = select_top_ranked(sub.hyperedges, self.max_hyperedges)
        # Seeds first and in their original order, then newly reachable nodes in
        # first-appearance order, so node indices are a deterministic function of
        # the selection alone. `dict.fromkeys` de-duplicates while preserving that
        # order; a set would make the layout depend on hash randomization.
        retained = dict.fromkeys(sub.seed_node_ids)
        for edge in selected:
            for node_id in edge.node_ids:
                retained.setdefault(node_id, None)
        provenance: dict[str, Any] = dict(sub.provenance)
        provenance["subhg_selection_policy"] = SELECTION_POLICY
        provenance["subhg_selection_fingerprint"] = self.fingerprint
        provenance["subhg_selected_edge_ids_hash"] = selected_edge_ids_hash(selected)
        provenance["subhg_hyperedges_before_selection"] = len(sub.hyperedges)
        return SubHypergraph(
            seed_node_ids=sub.seed_node_ids,
            node_ids=tuple(retained),
            hyperedges=selected,
            provenance=provenance,
        )
