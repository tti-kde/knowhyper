"""#233 Phase A0: build the metadata `CandidateTextChannel` reads off the context.

The channel needs three things that only exist together inside the injector: the query's
candidate lattice, the retrieved sub-hypergraph's linked mentions, and the text the SERIALIZER
selected. A query builder could supply the first but has no access to the other two, so the
forwarding belongs here.

Three decisions, each preventing a specific wrong measurement:

* **Opt-in.** `khp_span_late_*` covers 400 recorded cells and must keep an unchanged context, so
  `enabled` defaults off at the call site and these keys are simply absent otherwise.
* **Node texts come from the serializer's SELECTED items**, not from every text the sub-HG's
  nodes could offer. T0 and T1 measured what the serializer selects with `max_targets=8`
  applied; forwarding the superset would hand the model more text than the audit counted, so the
  recorded coverage would not describe the arm.
* **Mentions are filtered to those whose node was actually selected.** An unlinked mention is not
  evidence (#171), and a linked mention whose node the serializer truncated away would make the
  channel raise on "no cached text" -- turning a legitimate truncation into a hard error.
"""

from __future__ import annotations

from typing import Any


class A0MetadataError(ValueError):
    """The A0 metadata cannot be built consistently."""


CANDIDATES_KEY = "a0_candidates"
MENTIONS_KEY = "a0_linked_mentions"
NODE_TEXTS_KEY = "a0_node_texts"


def build_a0_metadata(
    *,
    query: Any,
    sub: Any,
    serialized: Any,
    enabled: bool,
) -> dict[str, Any]:
    """Return the three `a0_*` keys, or `{}` when not enabled."""
    if not enabled:
        return {}

    metadata = getattr(query, "metadata", {}) or {}
    spans = list(metadata.get("entity_spans", []))
    labels = list(metadata.get("entity_types", []))
    if len(spans) != len(labels):
        msg = (
            f"entity_spans and entity_types must align: {len(spans)} vs {len(labels)}. "
            "A mismatch would label candidates with another candidate's type."
        )
        raise A0MetadataError(msg)

    text = str(getattr(query, "text", "") or "")
    candidates = [
        {
            "start": int(span[0]),
            "end": int(span[1]),
            "label": str(label),
            # The surface, taken from the source text so it matches what the lexical control
            # normalises -- not from `query.entities`, whose order is a separate contract.
            "text": text[int(span[0]) : int(span[1])],
        }
        for span, label in zip(spans, labels, strict=True)
    ]

    # What the SERIALIZER selected, keyed by node. `max_targets` has already been applied.
    node_texts: dict[str, list[str]] = {}
    for item in getattr(serialized, "items", ()):
        node_id = str(getattr(item, "node_id", "") or "")
        item_text = str(getattr(item, "text", "") or "")
        if not node_id or not item_text:
            continue
        node_texts.setdefault(node_id, []).append(item_text)

    provenance = getattr(sub, "provenance", {}) or {}
    mentions = [
        mention
        for mention in provenance.get("linked_mentions", [])
        # Both filters matter: no node at all is #171's unlinked state, and a node the
        # serializer did not select has no cached text, which the channel treats as a
        # cache/sub-HG disagreement rather than an absence.
        if mention.get("node_id") and str(mention["node_id"]) in node_texts
    ]

    return {
        CANDIDATES_KEY: candidates,
        MENTIONS_KEY: mentions,
        NODE_TEXTS_KEY: node_texts,
    }
