"""HyperGraphRAG-style structured context formatting (Phase 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from primehkg.features.he_text_sources import render_he_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from primehkg.core.dataset import CoreDataset
    from primehkg.retrieval.khop import Subhypergraph
    from primehkg.retrieval.protocols import ContextOutput
    from primehkg.retrieval.retrieve import RetrievalResult


def _node_label_map(dataset: CoreDataset) -> dict[str, str]:
    return {
        str(row["node_id"]): str(row.get("name") or row["node_id"])
        for row in dataset.nodes
    }


def _support_index(dataset: CoreDataset) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for row in dataset.support:
        sid = row.get("support_id")
        if isinstance(sid, str):
            out[sid] = row
    return out


def _collect_node_ids(
    *,
    seed_node_ids: Sequence[str],
    results: Sequence[RetrievalResult],
    subhypergraph: Subhypergraph | None,
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for nid in seed_node_ids:
        key = str(nid)
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    if subhypergraph is not None:
        for nid in subhypergraph.get("node_ids", []):
            key = str(nid)
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    for row in results:
        he_raw = row.get("hyperedge") or {}
        he = he_raw if isinstance(he_raw, dict) else {}
        node_ids_raw = he.get("node_ids", [])
        if not isinstance(node_ids_raw, list):
            continue
        for nid in node_ids_raw:
            key = str(nid)
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


class DefaultContextFormatter:
    """Format ranked retrieval output into entities / relationships / sources."""

    def __init__(self, *, template_id: str = "natural_sentence_v1") -> None:
        """Select HE text template registry used for relationship descriptions."""
        self._template_id = template_id

    def format(
        self,
        *,
        results: list[RetrievalResult],
        seed_node_ids: Sequence[str],
        subhypergraph: Subhypergraph | None,
        dataset: CoreDataset,
    ) -> ContextOutput:
        """Build structured context from ranked hyperedges and optional subhypergraph."""
        node_labels = _node_label_map(dataset)
        seed_set = {str(nid) for nid in seed_node_ids}
        node_meta: dict[str, tuple[str, str]] = {}
        for row in dataset.nodes:
            nid = str(row["node_id"])
            node_meta[nid] = (
                str(row.get("name") or nid),
                str(row.get("node_type") or ""),
            )

        entities: list[dict[str, str]] = []
        for nid in _collect_node_ids(
            seed_node_ids=seed_node_ids,
            results=results,
            subhypergraph=subhypergraph,
        ):
            name, node_type = node_meta.get(nid, (node_labels.get(nid, nid), ""))
            entities.append(
                {
                    "node_id": nid,
                    "name": name,
                    "node_type": node_type,
                    "is_seed": "true" if nid in seed_set else "false",
                }
            )

        relationships: list[dict[str, str]] = []
        support_ids: list[str] = []
        for row in results:
            he = dict(row.get("hyperedge") or {})
            he_id = str(row.get("he_id") or he.get("he_id") or "")
            text_source = render_he_text(
                he,
                node_labels=node_labels,
                template_id=self._template_id,
            )
            relationships.append(
                {
                    "he_id": he_id,
                    "semantics": str(he.get("semantics") or ""),
                    "evidence_grade": str(he.get("evidence_grade") or ""),
                    "polarity": str(he.get("polarity") or ""),
                    "score": str(row.get("score", 0.0)),
                    "description": text_source.text,
                    "node_ids": ",".join(str(n) for n in he.get("node_ids", [])),
                }
            )
            sid = he.get("support_id")
            if isinstance(sid, str) and sid not in support_ids:
                support_ids.append(sid)

        support_by_id = _support_index(dataset)
        sources: list[dict[str, str]] = []
        for support_id in support_ids:
            support_row = support_by_id.get(support_id)
            if support_row is None:
                sources.append(
                    {
                        "support_id": support_id,
                        "source_name": "",
                        "source_version": "",
                    }
                )
                continue
            raw_sources = support_row.get("sources") or []
            if not raw_sources:
                sources.append(
                    {
                        "support_id": support_id,
                        "source_name": "",
                        "source_version": "",
                    }
                )
                continue
            if not isinstance(raw_sources, list):
                continue
            for src in raw_sources:
                if not isinstance(src, dict):
                    continue
                src_map = cast("dict[str, Any]", src)
                sources.append(
                    {
                        "support_id": support_id,
                        "source_name": str(src_map.get("name") or ""),
                        "source_version": str(src_map.get("version") or ""),
                    }
                )

        return {
            "entities": entities,
            "relationships": relationships,
            "sources": sources,
        }
