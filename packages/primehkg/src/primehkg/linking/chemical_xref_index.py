"""In-memory index over ``chemical_xref_v1`` mapping rows."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from primehkg.io.chemical_xref_sidecar import load_chemical_xref_sidecar
from primehkg.linking.pipeline import node_type_of

_RELATION_RANK = {
    "direct_xref": 0,
    "composed_bridge": 1,
    "unichem_pubchem_to_drugbank": 2,
    "unichem_pubchem_to_chebi": 3,
    "ctd_mesh_to_pubchem": 4,
}


def normalize_chemical_external_id(external_id: str, *, namespace: str) -> str:
    """Strip common prefixes; return bare id for the given namespace."""
    s = external_id.strip()
    ns = namespace.strip().lower()
    upper = s.upper()
    if ns == "mesh":
        return s.removeprefix("MESH:").removeprefix("mesh:")
    if ns == "pubchem":
        return s.removeprefix("CID:").removeprefix("cid:").lstrip("0") or "0"
    if ns == "drugbank":
        if upper.startswith("DRUGBANK:"):
            return s.split(":", 1)[1].strip()
        return s
    if ns == "chebi":
        return s.removeprefix("CHEBI:").removeprefix("chebi:")
    return s


@dataclass(frozen=True)
class ChemicalXrefCandidate:
    node_id: str
    node_type: str
    score: float | None
    mapping_path: tuple[str, ...]
    mapping_relation: str
    sources: tuple[tuple[str, str], ...]
    ambiguity_count: int
    audit_flags: tuple[str, ...]


def _path_tuple(path: str) -> tuple[str, ...]:
    parts = [p for p in str(path).split("->") if p]
    return tuple(parts)


def _candidate_sort_key(c: ChemicalXrefCandidate) -> tuple[Any, ...]:
    return (
        _RELATION_RANK.get(c.mapping_relation, 99),
        0 if c.ambiguity_count == 1 else 1,
        0 if c.node_type == "drug" else 1 if c.node_type == "chebi" else 2,
        c.node_id,
        c.mapping_path,
    )


class ChemicalXrefIndex:
    """Lookup index over chemical_xref_v1 rows."""

    def __init__(
        self, rows: list[dict[str, Any]], *, meta: dict[str, Any] | None = None
    ) -> None:
        self._meta = meta or {}
        self._by_ext: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._by_node: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            ns = str(row["external_namespace"]).strip().lower()
            eid = str(row["external_id"]).strip()
            nid = str(row["node_id"]).strip()
            self._by_ext.setdefault((ns, eid), []).append(row)
            if nid:
                self._by_node.setdefault(nid, []).append(row)

    @classmethod
    def from_sidecar_dir(
        cls,
        sidecar_dir: str | Path,
        *,
        expected_fingerprint: str | None = None,
    ) -> ChemicalXrefIndex:
        meta, rows = load_chemical_xref_sidecar(
            sidecar_dir, expected_fingerprint=expected_fingerprint
        )
        return cls(rows, meta=meta)

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    def lookup(
        self,
        external_id: str,
        *,
        namespace: str,
        allowed_node_types: Collection[str] | None = None,
        live_only: bool = True,
    ) -> list[ChemicalXrefCandidate]:
        ns = namespace.strip().lower()
        eid = normalize_chemical_external_id(external_id, namespace=ns)
        rows = self._by_ext.get((ns, eid), [])
        allowed = (
            {t.strip() for t in allowed_node_types}
            if allowed_node_types is not None
            else None
        )
        best_by_node: dict[str, ChemicalXrefCandidate] = {}
        for row in rows:
            if live_only and not bool(row.get("is_live_node")):
                continue
            nid = str(row["node_id"])
            ntype = str(row.get("node_type") or node_type_of(nid) or "")
            if allowed is not None and ntype not in allowed:
                continue
            flags: list[str] = []
            amb = int(row.get("ambiguity_count") or 0)
            if amb > 1:
                flags.append("ambiguous_external_id")
            if not bool(row.get("is_live_node")):
                flags.append("no_live_terminal_node")
            cand = ChemicalXrefCandidate(
                node_id=nid,
                node_type=ntype,
                score=None,
                mapping_path=_path_tuple(str(row.get("mapping_path") or "")),
                mapping_relation=str(row.get("mapping_relation") or ""),
                sources=(
                    (
                        str(row.get("mapping_source") or ""),
                        str(row.get("mapping_source_version") or ""),
                    ),
                ),
                ambiguity_count=amb,
                audit_flags=tuple(flags),
            )
            prev = best_by_node.get(nid)
            if prev is None or _candidate_sort_key(cand) < _candidate_sort_key(prev):
                best_by_node[nid] = cand
        return sorted(best_by_node.values(), key=_candidate_sort_key)

    def iter_external_ids(self, namespace: str) -> list[str]:
        ns = namespace.strip().lower()
        return sorted({eid for (n, eid) in self._by_ext if n == ns})

    def reverse_lookup(self, node_id: str) -> list[ChemicalXrefCandidate]:
        rows = self._by_node.get(str(node_id).strip(), [])
        out: list[ChemicalXrefCandidate] = []
        for row in rows:
            nid = str(row["node_id"])
            ntype = str(row.get("node_type") or node_type_of(nid) or "")
            amb = int(row.get("ambiguity_count") or 0)
            flags: list[str] = []
            if amb > 1:
                flags.append("ambiguous_external_id")
            out.append(
                ChemicalXrefCandidate(
                    node_id=nid,
                    node_type=ntype,
                    score=None,
                    mapping_path=_path_tuple(str(row.get("mapping_path") or "")),
                    mapping_relation=str(row.get("mapping_relation") or ""),
                    sources=(
                        (
                            str(row.get("mapping_source") or ""),
                            str(row.get("mapping_source_version") or ""),
                        ),
                    ),
                    ambiguity_count=amb,
                    audit_flags=tuple(flags),
                )
            )
        out.sort(
            key=lambda c: (
                c.mapping_path[0] if c.mapping_path else "",
                c.mapping_relation,
                c.node_id,
            )
        )
        return out

    def select_preferred(
        self, candidates: list[ChemicalXrefCandidate]
    ) -> ChemicalXrefCandidate | None:
        if not candidates:
            return None
        return sorted(candidates, key=_candidate_sort_key)[0]
