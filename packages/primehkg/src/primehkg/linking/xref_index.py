"""Build xref / label lookup indexes from enriched ``nodes`` rows."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from primehkg.nodes.normalize import normalize_namespace, normalize_xref_id

if TYPE_CHECKING:
    from datasets import Dataset

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[.,;]+")


def normalize_label(text: str) -> str:
    """Deterministic label normalization for exact / fuzzy linking."""
    lowered = (text or "").strip().lower()
    no_punct = _PUNCT_RE.sub("", lowered)
    return _WS_RE.sub(" ", no_punct).strip()


def _coerce_json(value: Any) -> Any:
    """Return ``value``, JSON-decoding it first if it is a string.

    Parquet-backed ``Dataset`` rows serialize nested columns (``xrefs``,
    ``synonyms``) to JSON strings, whereas in-memory fixtures pass them as
    native ``dict``/``list``. Both must index identically, so decode a string
    payload before the caller's ``isinstance`` check. Non-JSON strings and any
    other type are returned unchanged (the caller's type guard then rejects
    them), so this never raises.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return value


@dataclass
class XrefIndex:
    """In-memory xref and label indexes over PrimeHKG node rows."""

    xref_to_nodes: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    label_to_nodes: dict[str, list[str]] = field(default_factory=dict)
    node_ids: set[str] = field(default_factory=set)

    @classmethod
    def from_nodes(cls, nodes: Dataset) -> XrefIndex:
        idx = cls()
        for row in nodes:
            node_id = row.get("node_id")
            if not isinstance(node_id, str) or not node_id.strip():
                continue
            nid = node_id.strip()
            idx.node_ids.add(nid)

            name = row.get("name")
            if isinstance(name, str) and name.strip():
                idx._add_label(normalize_label(name), nid)

            syns = _coerce_json(row.get("synonyms"))
            if isinstance(syns, list):
                for syn in syns:
                    if isinstance(syn, str) and syn.strip():
                        idx._add_label(normalize_label(syn), nid)

            xrefs = _coerce_json(row.get("xrefs"))
            if isinstance(xrefs, dict):
                for raw_ns, ids in xrefs.items():
                    if not isinstance(raw_ns, str):
                        continue
                    try:
                        ns = normalize_namespace(raw_ns)
                    except ValueError:
                        continue
                    if not isinstance(ids, list):
                        continue
                    for raw_id in ids:
                        if not isinstance(raw_id, str) or not raw_id.strip():
                            continue
                        try:
                            xid = normalize_xref_id(ns, raw_id)
                        except ValueError:
                            continue
                        idx._add_xref(ns, xid, nid)
        return idx

    def _add_xref(self, ns: str, xid: str, node_id: str) -> None:
        key = (ns, xid)
        cur = self.xref_to_nodes.get(key, [])
        if node_id not in cur:
            cur = sorted([*cur, node_id])
            self.xref_to_nodes[key] = cur
        if ns == "mondo" and xid.startswith("MONDO:"):
            local = xid[6:]
            if local.isdigit():
                alt_key = (ns, local.lstrip("0") or "0")
                cur_alt = self.xref_to_nodes.get(alt_key, [])
                if node_id not in cur_alt:
                    self.xref_to_nodes[alt_key] = sorted([*cur_alt, node_id])
        if ns == "mesh" and xid.upper().startswith("MESH:"):
            local = xid.split(":", 1)[1]
            alt_key = (ns, local)
            if alt_key != key:
                cur_alt = self.xref_to_nodes.get(alt_key, [])
                if node_id not in cur_alt:
                    self.xref_to_nodes[alt_key] = sorted([*cur_alt, node_id])
        if ns == "drugbank":
            alt_key = (ns, xid.lower())
            if alt_key != key:
                cur_alt = self.xref_to_nodes.get(alt_key, [])
                if node_id not in cur_alt:
                    self.xref_to_nodes[alt_key] = sorted([*cur_alt, node_id])

    def _add_label(self, label: str, node_id: str) -> None:
        # Purely-numeric labels (e.g. a bare ClinVar VariationID like "100" or a PrimeKG
        # numeric id used as a fallback name) are ambiguous across node_types and redundant
        # with exact_node_id resolution — keep them out of the label index to avoid thousands
        # of colliding exact-label buckets. See docs/adr/2026-06-13-variant-node-core-scope.md.
        if not label or label.isdigit():
            return
        cur = self.label_to_nodes.get(label, [])
        if node_id not in cur:
            cur = sorted([*cur, node_id])
            self.label_to_nodes[label] = cur

    def lookup_node_id(self, raw: str) -> list[str]:
        nid = raw.strip()
        if nid in self.node_ids:
            return [nid]
        return []

    def lookup_xref(self, namespace: str, raw_id: str) -> list[str]:
        ns = normalize_namespace(namespace)
        xid = normalize_xref_id(ns, raw_id)
        return list(self.xref_to_nodes.get((ns, xid), []))

    def lookup_label(self, raw: str) -> list[str]:
        return list(self.label_to_nodes.get(normalize_label(raw), []))

    def all_labels(self) -> list[tuple[str, list[str]]]:
        return sorted(self.label_to_nodes.items(), key=lambda kv: kv[0])
