from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from primehkg.nodes.normalize import normalize_xref_id

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_SYNONYM_EXACT = re.compile(r'^synonym: "(.*)" EXACT\b(.*)$')


def _coerce_json(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return value


def read_mondo_synonym_updates(
    path: str | Path, *, source_version: str | None = None
) -> dict[str, dict[str, object]]:
    """Parse MONDO OBO EXACT synonyms (excluding DEPRECATED) keyed by MONDO id."""
    p = Path(path)
    version = source_version if source_version is not None else p.name
    by_mondo: dict[str, set[str]] = defaultdict(set)
    current: str | None = None
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("["):
                current = None
            elif s.startswith("id: MONDO:"):
                current = s.removeprefix("id: ").strip()
            elif current and s.startswith("synonym:"):
                m = _SYNONYM_EXACT.match(s)
                if m and "DEPRECATED" not in m.group(2):
                    name = m.group(1).replace('\\"', '"').replace("\\n", "\n")
                    if name:
                        by_mondo[current].add(name)
    out: dict[str, dict[str, object]] = {}
    for mondo_id in sorted(by_mondo):
        synonyms = [
            {"name": name, "source": "MONDO", "source_version": version}
            for name in sorted(by_mondo[mondo_id])
        ]
        out[mondo_id] = {
            "synonyms": synonyms,
            "source": "MONDO",
            "source_version": version,
        }
    return out


def bind_mondo_synonym_updates_to_nodes(
    mondo_updates: Mapping[str, dict[str, object]],
    nodes: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Bind MONDO-keyed synonym updates onto disease nodes via ``xrefs.mondo``."""
    out: dict[str, dict[str, object]] = {}
    for node in nodes:
        node_id_raw = node.get("node_id")
        if not isinstance(node_id_raw, str) or not node_id_raw.strip():
            continue
        node_id = node_id_raw.strip()

        xrefs = _coerce_json(node.get("xrefs"))
        if not isinstance(xrefs, dict):
            continue
        mondo_raw = xrefs.get("mondo")
        if not isinstance(mondo_raw, list):
            continue

        by_name: dict[str, dict[str, Any]] = {}
        source_version: str | None = None
        for raw_id in mondo_raw:
            if not isinstance(raw_id, str) or not raw_id.strip():
                continue
            try:
                mondo_id = normalize_xref_id("mondo", raw_id)
            except ValueError:
                continue
            upd = mondo_updates.get(mondo_id)
            if upd is None:
                continue
            if source_version is None:
                sv = upd.get("source_version")
                if isinstance(sv, str):
                    source_version = sv
            syns = upd.get("synonyms")
            if not isinstance(syns, list):
                continue
            for syn in syns:
                if not isinstance(syn, dict):
                    continue
                syn_d = cast("dict[str, object]", syn)
                name = syn_d.get("name")
                if isinstance(name, str) and name and name not in by_name:
                    by_name[name] = cast("dict[str, Any]", syn)

        if not by_name:
            continue
        out[node_id] = {
            "synonyms": [by_name[k] for k in sorted(by_name)],
            "source": "MONDO",
            "source_version": source_version or "",
        }
    return out
