"""Canonical text serialization of a retrieved typed sub-hypergraph (R2b / C2).

R2a asks whether the *content* of retrieved KG text helps; it does not.  R2b asks
the separate question of whether the *typed structure* survives being written out
as text.  The two arms deliberately share only the serializer protocol, the
pair-aware collator, and token accounting -- selection policy and tests are
independent (roadmap 2026-07-19 comparison-methods, section 4.1-6).

Nothing here reads ``node_features_text``.  A hyperedge's semantics, member
roles and CTD polarity are structural facts already carried by
:class:`~knowhyper.pipeline.types.HyperedgeRef`, so this module needs no licensed
text payload and no license registry.  Entities are named by their stable
``node_id``; see ``_format`` for why display names are deliberately absent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from knowhyper.encoders.vocab import (
    POLARITY_VOCAB,
    REDAF_SEMANTICS_VOCAB,
    SEMANTICS_VOCAB,
    UNK,
)
from knowhyper.pipeline.subhg_selection import select_top_ranked

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ieforge.knowledge.core.schema import Query

    from knowhyper.pipeline.types import SubHypergraph

_POLICY_VERSION = 1
_TEMPLATE = "typed_subhg_v1"
_ROLE_NAMES = ("head", "tail")
_PROVENANCE_SCHEMA = "linked_mentions_role_v1"
# Replacement pools for the shuffled control, reusing the encoder's curated
# vocabularies so a corrupted value is always one the KG could really have
# produced -- a nonsense string would be distinguishable as noise rather than as
# wrong-but-plausible structure. `<unk>` is dropped: it is the encoder's
# out-of-vocab sentinel, never a serialized value.
#
# The pool must contain EVERY value live retrieval emits. A value missing from it
# can never be drawn as a replacement, so it appears only in the treated arm and
# each replacement standing in for it becomes a marker pointing back at one true
# value -- an INVERTIBLE relabelling a classifier can undo, rather than destruction
# of the type signal.
#
# `SEMANTICS_VOCAB` alone is not enough: it was curated 2026-06-16 from primehkg's
# builders (11 values) and omits most of what retrieval actually returns, so a
# hand-added pair would still have aborted ~14% of ChemProt and ~51% of BioRED
# instances via the fail-closed guard below. `REDAF_SEMANTICS_VOCAB` is the
# frequency-ordered list built from the real ChemProt cache (40 values), so the
# union covers both. `HyperedgeRef.polarity` likewise defaults to the literal
# `"unknown"`, which `POLARITY_VOCAB` deliberately folds into `<unk>`.
# `_assert_in_vocabulary` still fails closed on anything outside the pool.
#
# The pool is a fallback, not the primary source. Membership here does not prove a
# value is PRODUCIBLE: `pathway_membership` occurs in zero primehkg files and zero
# of 4.48M sampled hyperedges, and 13 further entries never appear in any cache. A
# replacement the treated arm can never emit is a marker that identifies the
# control outright -- the mirror image of the invertible-relabelling hazard above.
# So `_value_pools` draws from the values observed in the instance's OWN edges
# first and only falls back to the pool when an instance offers no alternative.
# Values that exist in no primehkg builder and in no cache: drawing one would mark
# the control. `pathway_membership` is KnowHyper-local (3 references, all our own).
_UNPRODUCIBLE_SEMANTICS = frozenset({"pathway_membership"})
_SEMANTICS_VOCABULARY: tuple[str, ...] = tuple(
    dict.fromkeys(
        value
        for value in (*SEMANTICS_VOCAB, *REDAF_SEMANTICS_VOCAB)
        if value != UNK and value not in _UNPRODUCIBLE_SEMANTICS
    )
)
_POLARITY_VOCABULARY: tuple[str, ...] = (
    *(value for value in POLARITY_VOCAB if value != UNK),
    "unknown",
)
# One hyperedge per serialized item, and `_truncate_knowledge` splits the token
# budget round-robin ACROSS items -- so the item count, not the total, decides
# whether a single item survives intact. At R2a's N=64 against the shared
# 128-token payload each block would get 2 tokens and render as "[ hg", cutting
# the semantics value off before it is ever reached and making the treated and
# shuffled arms byte-identical after truncation. R2a is not comparable here: it
# emits at most 2 items (head/tail definitions), so it never divides the budget
# more than two ways.
#
# A 2-member block is ~29-35 PubMedBERT tokens, but cost grows ~8-10 tokens per
# additional member (a 4-member block is ~56, a 20-member one ~176), and arity is
# unbounded: ChemProt has an arity>2 edge in 43.9% of instances (max 410),
# BioRED 13.5%, DDI 2.7% (docs/reports/2026-07-19-binary-projection-feasibility.md).
# So 3 blocks fit 128 only when they are narrow. The collator therefore keeps whole
# blocks and DROPS the ones that do not fit (see `_truncate_whole_segments`) rather
# than cutting them mid-field, and reports `knowledge_items_dropped`.
# tests/ieforge/test_serialized_subhg_collation.py pins this at the real budget.
_DEFAULT_MAX_HYPEREDGES = 3
# Members rendered per hyperedge before eliding the rest as "+N more". Without a
# cap a 410-member edge renders thousands of tokens, and `polarity=` -- rendered
# last, and the one typed signal this project has positive evidence for -- is the
# first field lost to truncation. The cap bounds a block to ~65 tokens even with
# long node ids; it cannot GUARANTEE the payload budget on its own, which is why
# the collator drops whole over-budget blocks instead of trimming them.
_DEFAULT_MAX_MEMBERS = 4


@dataclass(frozen=True)
class SerializedHyperedge:
    """One hyperedge rendered as canonical text.

    ``role`` / ``mention`` / ``text`` mirror the field names
    :class:`~knowhyper.ieforge.text_collator.PairAwareTextKnowledgeCollator`
    already consumes, so the R2b arm reuses the tokenizer/budget path unchanged.
    ``role`` is the *target-pair* role this hyperedge is attached to (or ``seed``),
    not the member role inside the hyperedge -- those live in ``text``.
    """

    role: str
    he_id: str
    semantics: str
    polarity: str
    node_ids: tuple[str, ...]
    member_roles: tuple[tuple[str, str], ...]
    text: str
    #: The block as the TREATED arm renders it. The collator keeps or drops whole
    #: blocks by this string's length, not by ``text``, so the treated and shuffled
    #: arms always retain the same number of blocks: corrupted values differ in
    #: length from the originals, and budgeting on ``text`` would let one arm carry
    #: more structure than the other. Equal to ``text`` in the treated arm.
    budget_text: str = ""
    mention: str = ""


@dataclass(frozen=True)
class SerializedSubHG:
    """Canonical text plus per-hyperedge entries for budget-aware tokenization."""

    text: str
    items: tuple[SerializedHyperedge, ...]
    fingerprint: str


class SerializedSubHGSerializer:
    """Render a retrieved sub-hypergraph as deterministic typed text.

    Selection policy, independent from R2a's definition policy:

    * every hyperedge of the retrieved sub-hypergraph is eligible (R2a restricts
      itself to the linked target seeds),
    * the ``max_hyperedges`` **highest-scoring** hyperedges are kept.  The
      retriever returns ``top_k`` score-ranked edges and the fusion arms encode
      that ranking, so selecting any other subset would mean R2b and the fusion
      arms measure different structure -- a null would then be unattributable
      between "structure does not transfer as text" and "we serialized the wrong
      3 of 64 edges".  ``he_id`` breaks score ties, so the choice stays
      deterministic without discarding relevance, and
    * the kept edges are then rendered in ``he_id`` order so a re-run cannot
      reorder the text.

    ``shuffle_values`` produces the *shuffled control* arm: the frame, the field
    names, the hyperedge count and the connectivity are preserved, while every
    ``semantics`` and ``polarity`` value is replaced by a different member of the
    curated vocabulary.  ``R2b - shuffled`` therefore isolates the *typed content*
    from the effect of the ``[HG]`` frame itself.  R2a's sham arm showed the frame
    alone moves the score as much as real text does, so without this control an
    R2b null could not distinguish "structure does not transfer" from "the model
    only ever saw the frame".

    Two known limits of the control, both deliberate:

    * Node ids and member role names are **not** corrupted, so connectivity still
      leaks into the control.  It isolates the hyperedge *type* signal, not the
      whole structure; the ``off_short`` contrast bounds the rest.
    * Replacement values come from a shared vocabulary, so a corrupted string may
      tokenize to a different length than the original.  The token budget is
      therefore matched only in expectation, not per instance -- exact per-instance
      matching would require a within-instance permutation, which degenerates to
      the identity on the many instances whose hyperedges share one value (see
      ``_value_pools``).  The collator's ``knowledge_token_count`` diagnostic must
      be reported for both arms so any residual length gap stays auditable.
    """

    #: Metadata key the injector publishes items under, kept distinct from R2a's
    #: ``retrieved_definitions`` so a misconfigured arm yields empty knowledge
    #: rather than silently consuming the other arm's payload.
    items_key = "serialized_subhg"

    def __init__(
        self,
        *,
        max_hyperedges: int = _DEFAULT_MAX_HYPEREDGES,
        max_members: int = _DEFAULT_MAX_MEMBERS,
        include_polarity: bool = True,
        shuffle_values: bool = False,
        shuffle_seed: int = 0,
        semantics_vocabulary: Sequence[str] = _SEMANTICS_VOCABULARY,
        polarity_vocabulary: Sequence[str] = _POLARITY_VOCABULARY,
        required_provenance_schema: str | None = _PROVENANCE_SCHEMA,
    ) -> None:
        if max_hyperedges <= 0:
            msg = "SerializedSubHGSerializer needs a positive hyperedge budget"
            raise ValueError(msg)
        if max_members <= 0:
            msg = "SerializedSubHGSerializer needs a positive member budget"
            raise ValueError(msg)
        if shuffle_values and len(set(semantics_vocabulary)) < 2:
            msg = (
                "the shuffled control needs at least two distinct semantics values, "
                "otherwise it would reproduce the treated arm verbatim"
            )
            raise ValueError(msg)
        if shuffle_values and include_polarity and len(set(polarity_vocabulary)) < 2:
            msg = (
                "the shuffled control needs at least two distinct polarity values "
                "while polarity is rendered"
            )
            raise ValueError(msg)
        self._max_hyperedges = int(max_hyperedges)
        self._max_members = int(max_members)
        self._include_polarity = bool(include_polarity)
        self._shuffle_values = bool(shuffle_values)
        self._shuffle_seed = int(shuffle_seed)
        self._semantics_vocabulary = tuple(dict.fromkeys(semantics_vocabulary))
        self._polarity_vocabulary = tuple(dict.fromkeys(polarity_vocabulary))
        self._required_provenance_schema = required_provenance_schema
        identity = {
            "policy": "typed_subhg_hyperedges",
            "policy_version": _POLICY_VERSION,
            "template": _TEMPLATE,
            "max_hyperedges": self._max_hyperedges,
            "max_members": self._max_members,
            "include_polarity": self._include_polarity,
            "shuffle_values": self._shuffle_values,
            "shuffle_seed": self._shuffle_seed,
            # Folded in so a vocabulary edit re-precomputes the control's text
            # cache instead of silently mixing two corruption schemes.
            "semantics_vocabulary": self._semantics_vocabulary,
            "polarity_vocabulary": self._polarity_vocabulary,
            "provenance_schema": required_provenance_schema,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        self.fingerprint = hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def serialize(self, query: Query, sub: SubHypergraph) -> SerializedSubHG:
        """Serialize the retrieved hyperedges in stable ``he_id`` order.

        ``query`` is part of the serializer protocol shared with R2a but is
        deliberately unused here: target-pair roles are read from retrieval
        provenance so a cache hit and a fresh retrieval serialize identically.
        """
        del query
        self._assert_provenance(sub)
        roles = self._roles_by_node(sub)
        # Keep the highest-scoring edges (he_id breaks ties), then render in he_id
        # order. Selecting by he_id alone would throw away the retriever's ranking
        # and serialize an arbitrary ~3 of top_k=64 edges.
        #
        # Delegated to the shared selector rather than kept inline: the GRAFT-style
        # baseline (#189) needs a GRAPH arm over the SAME 3 edges, and the whole
        # point of that contrast is that both arms see identical evidence. Two
        # copies of this two-stage sort would let the arms drift apart silently --
        # the delta would then measure a selection difference and read as a
        # representation difference.
        selected = select_top_ranked(sub.hyperedges, self._max_hyperedges)
        semantics_values, polarity_values = self._value_pools(selected)

        items: list[SerializedHyperedge] = []
        for index, edge in enumerate(selected):
            node_ids = tuple(str(node_id) for node_id in edge.node_ids)
            member_roles = tuple(
                (str(node_id), str(edge.roles.get(node_id, ""))) for node_id in node_ids
            )
            semantics = semantics_values[index]
            polarity = polarity_values[index]
            items.append(
                SerializedHyperedge(
                    role=self._pair_role(node_ids, roles),
                    he_id=str(edge.he_id),
                    semantics=semantics,
                    polarity=polarity,
                    node_ids=node_ids,
                    member_roles=member_roles,
                    text=self._format(
                        semantics=semantics,
                        polarity=polarity,
                        member_roles=member_roles,
                        pair_roles=roles,
                    ),
                    # Always the TREATED rendering, so both arms are budgeted on
                    # the same length and retain the same number of blocks.
                    budget_text=self._format(
                        semantics=str(edge.semantics),
                        polarity=str(edge.polarity) if self._include_polarity else "",
                        member_roles=member_roles,
                        pair_roles=roles,
                    ),
                )
            )
        text = " ".join(item.text for item in items)
        return SerializedSubHG(
            text=text, items=tuple(items), fingerprint=self.fingerprint
        )

    def _assert_provenance(self, sub: SubHypergraph) -> None:
        if self._required_provenance_schema is None:
            return
        actual = sub.provenance.get("linked_mentions_schema")
        if actual != self._required_provenance_schema:
            msg = (
                "serialized sub-hypergraph text requires linked-mention provenance "
                f"schema {self._required_provenance_schema!r}, got {actual!r}"
            )
            raise ValueError(msg)

    @staticmethod
    def _roles_by_node(sub: SubHypergraph) -> Mapping[str, str]:
        """Map linked node id -> target-pair role, preferring the head record.

        Roles come from retrieval provenance rather than the query so a cached
        sub-hypergraph serializes identically to a freshly retrieved one.
        """
        roles: dict[str, str] = {}
        for record in sub.provenance.get("linked_mentions", []):
            node_id = str(record.get("node_id", ""))
            pair_role = str(record.get("pair_role", ""))
            if not node_id or pair_role not in _ROLE_NAMES:
                continue
            current = roles.get(node_id)
            if current is None or _ROLE_NAMES.index(pair_role) < _ROLE_NAMES.index(
                current
            ):
                roles[node_id] = pair_role
        return roles

    @staticmethod
    def _pair_role(node_ids: Sequence[str], roles: Mapping[str, str]) -> str:
        """Label a hyperedge by which target endpoints it touches."""
        touched = sorted(
            {roles[node_id] for node_id in node_ids if node_id in roles},
            key=_ROLE_NAMES.index,
        )
        if not touched:
            return "seed"
        return "pair" if len(touched) == len(_ROLE_NAMES) else touched[0]

    def _value_pools(
        self, edges: Sequence[Any]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the per-edge semantics/polarity values to render.

        For the primary arm these are the edges' own values.

        For the shuffled control each value is replaced by a *different* member of
        the corpus-wide vocabulary (``semantics_vocabulary`` /
        ``polarity_vocabulary``), chosen deterministically from the edge's own
        ``he_id``.  Two properties matter and a within-instance permutation
        provides neither:

        * **The control can never degenerate into the treated arm.** Permuting
          within an instance is the identity whenever that instance's hyperedges
          all share one value -- the common case, since DDI retrieval is dominated
          by ``drug_drug`` and polarity is mostly ``unknown``.  A single-hyperedge
          instance is likewise unpermutable.  Such instances would silently be
          scored as controls while carrying the *real* structure, biasing
          ``R2b - shuffled`` toward zero, i.e. manufacturing the null this arm
          exists to test for.
        * **Corruption is uniform across instances.** With a permutation, how
          badly an instance is corrupted depends on how diverse its own edges are,
          so the treatment intensity would correlate with sub-hypergraph
          composition.

        Keyed off ``he_id`` rather than an RNG so the output stays canonical:
        no dependence on Python's salted ``hash`` or on global random state, and a
        cache hit reproduces a fresh serialization byte for byte.
        """
        semantics = tuple(str(edge.semantics) for edge in edges)
        polarity = tuple(
            str(edge.polarity) if self._include_polarity else "" for edge in edges
        )
        if not self._shuffle_values:
            return semantics, polarity
        self._assert_in_vocabulary(semantics, polarity)
        # Prefer values this corpus demonstrably produces: anything the treated arm
        # cannot emit would identify the control.
        semantics_pool = self._observed_pool(semantics, self._semantics_vocabulary)
        polarity_pool = self._observed_pool(polarity, self._polarity_vocabulary)
        shuffled_semantics = tuple(
            self._replacement(str(edge.he_id), value, semantics_pool)
            for edge, value in zip(edges, semantics, strict=True)
        )
        shuffled_polarity = tuple(
            value
            if not value
            else self._replacement(str(edge.he_id), value, polarity_pool)
            for edge, value in zip(edges, polarity, strict=True)
        )
        return shuffled_semantics, shuffled_polarity

    @staticmethod
    def _observed_pool(
        observed: Sequence[str], fallback: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Prefer the instance's own distinct values; fall back to the vocabulary.

        A value drawn from the configured pool but never producible by retrieval
        (`pathway_membership` is in no primehkg builder and no cache) would make the
        control trivially identifiable. Using the instance's own values keeps every
        replacement one the treated arm demonstrably emits. Single-valued instances
        offer no alternative, so those fall back to the pool -- which is why the
        pool must still cover everything retrieval emits.
        """
        distinct = tuple(dict.fromkeys(value for value in observed if value))
        return distinct if len(distinct) >= 2 else fallback

    def _assert_in_vocabulary(
        self, semantics: Sequence[str], polarity: Sequence[str]
    ) -> None:
        """Fail closed when a real value is missing from a replacement pool.

        A value outside the pool can never be *drawn* as a replacement, so it only
        ever appears in the treated arm. Every replacement that stands in for it
        then becomes a marker pointing back at one true value, and the corruption
        degrades into a relabelling a classifier can invert -- the control would
        silently stop destroying the type signal it exists to destroy.

        This is data-dependent, so ``__init__``'s size checks cannot catch it: a
        new primehkg builder emitting an uncurated ``semantics`` must stop the run
        rather than quietly weaken the control.
        """
        unknown_semantics = sorted(set(semantics) - set(self._semantics_vocabulary))
        if unknown_semantics:
            msg = (
                "the shuffled control cannot corrupt semantics values missing from "
                f"its vocabulary: {unknown_semantics}; add them to "
                "semantics_vocabulary (this changes the serializer fingerprint)"
            )
            raise ValueError(msg)
        unknown_polarity = sorted(
            {value for value in polarity if value} - set(self._polarity_vocabulary)
        )
        if unknown_polarity:
            msg = (
                "the shuffled control cannot corrupt polarity values missing from "
                f"its vocabulary: {unknown_polarity}; add them to "
                "polarity_vocabulary (this changes the serializer fingerprint)"
            )
            raise ValueError(msg)

    def _replacement(self, he_id: str, value: str, vocabulary: tuple[str, ...]) -> str:
        """Pick a vocabulary entry that is deterministic in ``he_id`` but != value.

        Returns ``value`` unchanged only when the vocabulary offers no alternative,
        which ``__init__`` rejects, so in practice the value always changes.
        """
        alternatives = tuple(entry for entry in vocabulary if entry != value)
        if not alternatives:
            return value
        digest = hashlib.sha256(f"{self._shuffle_seed}\0{he_id}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % len(alternatives)
        return alternatives[index]

    def _format(
        self,
        *,
        semantics: str,
        polarity: str,
        member_roles: Sequence[tuple[str, str]],
        pair_roles: Mapping[str, str],
    ) -> str:
        """Render one hyperedge as ``[HG] ... [/HG]``.

        Target-pair endpoints are written as their role token rather than an
        identifier.  Two reasons: DDI masks both drug surfaces in the source
        sentence and the collator's ``redact`` policy rewrites any target surface
        found in knowledge text, so emitting an endpoint name there would either
        leak the masked identity or be silently replaced -- and a display name is
        not available anyway (``HyperedgeRef`` carries only ``node_ids``, and
        retrieved neighbours never appear in the source text).  Neighbours keep
        their stable ``node_id``, which is what makes the connectivity readable.

        At most ``max_members`` members are rendered, with the remainder summarised
        as ``+N more``.  Hyperedges are unbounded in arity (``max_hyperedge_size``
        is unset in every retriever profile), and the collator divides the payload
        budget evenly across blocks and then cuts each one from the right.  A
        4-member ``pathway_membership`` edge already overruns its share, and since
        ``polarity`` is rendered last it is the first field lost -- silently, in
        BOTH arms, which would delete the one typed signal this project has
        positive evidence for (CTD polarity) while leaving the arms looking intact.
        Capping members keeps every *field* present in every retained block; the
        elision is visible in the text rather than being an invisible truncation.
        """
        fields = [f"semantics={semantics}"]
        shown = list(member_roles[: self._max_members])
        for node_id, member_role in shown:
            label = pair_roles.get(node_id)
            name = f"[{label.upper()}_ENTITY]" if label else node_id
            fields.append(f"{member_role or 'member'}={name}" if member_role else name)
        elided = len(member_roles) - len(shown)
        if elided > 0:
            fields.append(f"+{elided} more")
        if self._include_polarity:
            fields.append(f"polarity={polarity}")
        return "[HG] " + "; ".join(fields) + " [/HG]"
