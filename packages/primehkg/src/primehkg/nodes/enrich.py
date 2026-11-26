from __future__ import annotations

from typing import TYPE_CHECKING, Any

from primehkg.nodes.feature_refs import merge_feature_refs
from primehkg.nodes.merge import merge_xrefs
from primehkg.nodes.synonyms import is_genuine_synonym

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def _iter_dicts(values: object) -> Iterable[dict[str, Any]]:
    if isinstance(values, list | tuple):
        for v in values:
            if isinstance(v, dict):
                yield v


def _norm_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


def _dedupe_sorted_strs(values: Iterable[str]) -> list[str]:
    return sorted({v for v in values if isinstance(v, str) and v})


def _choose_display_name(
    candidates: list[dict[str, Any]],
    priority: list[str],
    fallback: str,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Choose a deterministic display name from candidates."""
    normed: list[dict[str, Any]] = []
    for row in candidates:
        name = _norm_str(row.get("name"))
        if name is None:
            continue
        normed.append(
            {
                "name": name,
                "source": _norm_str(row.get("source")) or "unknown",
                "source_version": _norm_str(row.get("source_version")) or "unknown",
            }
        )

    normed_sorted = sorted(
        normed,
        key=lambda r: (
            str(r.get("source") or ""),
            str(r.get("source_version") or ""),
            str(r.get("name") or ""),
        ),
    )

    if not normed_sorted:
        return fallback, [], {"method": "fallback", "priority": list(priority)}

    by_source: dict[str, list[str]] = {}
    for r in normed_sorted:
        src = str(r.get("source") or "unknown")
        by_source.setdefault(src, []).append(str(r["name"]))

    for src in priority:
        if src in by_source:
            chosen = sorted(set(by_source[src]))[0]
            return (
                chosen,
                normed_sorted,
                {
                    "method": "priority_source",
                    "priority": list(priority),
                    "chosen_source": src,
                },
            )

    chosen_any = sorted({str(r["name"]) for r in normed_sorted})[0]
    return (
        chosen_any,
        normed_sorted,
        {"method": "lexicographic", "priority": list(priority)},
    )


def _infer_update_source(update: Mapping[str, Any]) -> tuple[str, str]:
    # Prefer explicit source hints, otherwise derive from name provenance if unambiguous.
    source = _norm_str(update.get("source") or update.get("incoming_source"))
    version = _norm_str(
        update.get("source_version") or update.get("incoming_source_version")
    )
    if source and version:
        return source, version

    sources: set[str] = set()
    versions: set[str] = set()
    for row in _iter_dicts(update.get("names")):
        s = _norm_str(row.get("source"))
        v = _norm_str(row.get("source_version"))
        if s:
            sources.add(s)
        if v:
            versions.add(v)

    if len(sources) == 1:
        source = next(iter(sources))
    if len(versions) == 1:
        version = next(iter(versions))

    return source or "unknown", version or "unknown"


def _merge_source_meta(base: dict[str, Any], patch: object) -> dict[str, Any]:
    if not isinstance(patch, dict):
        return base
    out = dict(base)
    for k, v in patch.items():
        if not isinstance(k, str) or not k.strip():
            continue
        kk = k.strip()
        existing = out.get(kk)
        if isinstance(existing, dict) and isinstance(v, dict):
            out[kk] = _merge_source_meta(existing, v)
        elif isinstance(existing, list) and isinstance(v, list):
            out[kk] = [*existing, *v]
        else:
            out[kk] = v
    return out


def enrich_nodes_v1(
    *,
    base_nodes: list[dict[str, Any]],
    node_updates: dict[str, dict[str, Any]],
    display_name_priority_by_type: dict[str, list[str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for node in base_nodes:
        if not isinstance(node, dict):
            continue

        nid = node.get("node_id")
        if not isinstance(nid, str):
            out.append(dict(node))
            continue

        upd = node_updates.get(nid) or {}
        node_type = str(node.get("node_type") or "unknown")

        source_meta: dict[str, Any] = dict(node.get("source_meta") or {})
        source_meta = _merge_source_meta(source_meta, upd.get("source_meta_patch"))
        xrefs_in: dict[str, list[str]] = dict(node.get("xrefs") or {})

        incoming_source, incoming_source_version = _infer_update_source(upd)
        evidence = _norm_str(upd.get("evidence"))

        if isinstance(upd.get("xrefs"), dict):
            xrefs_out, xref_prov = merge_xrefs(
                base=xrefs_in,
                incoming=upd["xrefs"],
                incoming_source=incoming_source,
                incoming_source_version=incoming_source_version,
                evidence=evidence,
                provenance=list(source_meta.get("xref_provenance") or []),
            )
            source_meta["xref_provenance"] = xref_prov
        else:
            xrefs_out = xrefs_in

        name_candidates = list(_iter_dicts(source_meta.get("name_provenance")))
        name_candidates.extend(list(_iter_dicts(upd.get("names"))))
        priority = list(display_name_priority_by_type.get(node_type) or [])
        fallback_name = str(node.get("name") or nid)
        chosen_name, normed_name_prov, rule = _choose_display_name(
            name_candidates, priority, fallback_name
        )
        if normed_name_prov:
            source_meta["name_provenance"] = normed_name_prov
            source_meta["display_name_rule"] = rule

        syn_values: list[str] = []
        syn_base = node.get("synonyms")
        if isinstance(syn_base, list):
            syn_values.extend([s for s in syn_base if isinstance(s, str)])

        for row in _iter_dicts(upd.get("synonyms")):
            name = _norm_str(row.get("name"))
            if name is not None:
                syn_values.append(name)

        syn_values = [
            s
            for s in syn_values
            if is_genuine_synonym(s, name=chosen_name, node_id=nid)
        ]
        syn_norm = _dedupe_sorted_strs(syn_values)

        feature_refs = merge_feature_refs(
            dict(node.get("feature_refs") or {}),
            upd.get("feature_refs")
            if isinstance(upd.get("feature_refs"), dict)
            else None,
        )

        out.append(
            {
                **dict(node),
                "name": chosen_name,
                "synonyms": syn_norm,
                "xrefs": xrefs_out,
                "source_meta": source_meta,
                "feature_refs": feature_refs,
            }
        )

    return out
