"""Exact chemical external-id resolver backed by ``chemical_xref_v1``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from primehkg.linking.protocols import LinkCandidate, LinkMatchKind
from primehkg.linking.resolvers import _candidate

if TYPE_CHECKING:
    from collections.abc import Collection

    from primehkg.linking.chemical_xref_index import ChemicalXrefIndex

CHEMICAL_EXTERNAL_NAMESPACES = frozenset({"mesh", "pubchem", "drugbank", "chebi"})


class ChemicalExternalIdResolver:
    """Resolve MeSH / PubChem / DrugBank / ChEBI via ``ChemicalXrefIndex``.

    Hard ``allowed_node_types`` filtering is applied here (unlike soft
    ``node_type_hint`` re-rank in ``LinkPipeline``). When the sidecar index is
    missing, resolve is a no-op and ``last_resolve_audit`` reports unloaded.
    """

    def __init__(self, index: ChemicalXrefIndex | None) -> None:
        self._index = index
        self.last_resolve_audit: dict[str, Any] = {
            "chemical_xref_sidecar_loaded": index is not None,
        }

    def resolve(
        self,
        query: str,
        *,
        namespace: str | None = None,
        node_type_hint: str | Collection[str] | None = None,
        allowed_node_types: Collection[str] | None = None,
        top_k: int = 5,
    ) -> list[LinkCandidate]:
        del node_type_hint
        audit: dict[str, Any] = {
            "chemical_xref_sidecar_loaded": self._index is not None,
        }
        if self._index is not None:
            meta = self._index.meta
            audit["chemical_xref_schema_version"] = meta.get("schema_version")
            audit["chemical_xref_fingerprint"] = meta.get("inputs_fingerprint")
        self.last_resolve_audit = audit

        if self._index is None:
            return []
        if namespace is None:
            return []
        ns = namespace.strip().lower()
        if ns not in CHEMICAL_EXTERNAL_NAMESPACES:
            return []
        q = (query or "").strip()
        if not q:
            return []

        before = self._index.lookup(q, namespace=ns, live_only=True)
        audit["candidate_count_before_type_filter"] = len(before)
        allowed = (
            {t.strip() for t in allowed_node_types}
            if allowed_node_types is not None
            else None
        )
        after = (
            before if allowed is None else [c for c in before if c.node_type in allowed]
        )
        audit["candidate_count_after_type_filter"] = len(after)
        audit["mapping_method"] = "chemical_xref_v1"
        audit["mapping_path_count"] = sum(1 for c in after if c.mapping_path)
        if any("ambiguous_external_id" in c.audit_flags for c in after):
            audit["ambiguous_external_id"] = True
        if not after:
            # Distinguish mapping-absent vs type-filtered-away vs no-live.
            non_live = self._index.lookup(q, namespace=ns, live_only=False)
            if not non_live:
                audit["mapping_absent"] = True
            elif not before:
                audit["no_live_terminal_node"] = True
            elif allowed is not None and before:
                audit["type_filter_nil"] = True

        self.last_resolve_audit = audit

        out: list[LinkCandidate] = []
        for cand in after[: max(0, int(top_k))]:
            flags = list(cand.audit_flags)
            flags.append("chemical_xref_v1")
            if cand.mapping_relation:
                flags.append(f"mapping_relation:{cand.mapping_relation}")
            out.append(
                _candidate(
                    node_id=cand.node_id,
                    score=0.95,
                    match_kind=LinkMatchKind.EXACT_XREF,
                    matched_surface=q,
                    matched_field=f"chemical_xref.{ns}",
                    score_breakdown={"base": 0.95},
                    audit_flags=flags,
                )
            )
        return out
