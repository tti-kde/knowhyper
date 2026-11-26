from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any, Literal, cast

import yaml

if TYPE_CHECKING:
    from pathlib import Path

Status = Literal["active", "defer", "deprecated_alias"]
DeferReason = Literal[
    "ambiguous_semantics",
    "explosion_risk",
    "licensing",
    "needs_external_ontology",
    "duplicate_of",
]


@dataclass(frozen=True, slots=True)
class RelationEntry:
    """One PrimeKG relation string policy entry in the registry."""

    relation: str
    status: Status
    notes: str
    defer_reason: DeferReason | None = None
    maps_to: str | None = None


def load_relation_registry_bytes() -> bytes:
    return (
        resources.files("primehkg.policy")
        .joinpath("relation_registry.yaml")
        .read_bytes()
    )


def load_relation_registry(path: Path | None = None) -> dict[str, RelationEntry]:
    raw = load_relation_registry_bytes() if path is None else path.read_bytes()
    obj: Any = yaml.safe_load(raw)
    if not isinstance(obj, dict):
        msg = "relation registry: top-level must be a mapping"
        raise TypeError(msg)
    if int(obj.get("version", -1)) < 1:
        msg = "relation registry: version must be >= 1"
        raise ValueError(msg)
    rels = obj.get("relations")
    if not isinstance(rels, list):
        msg = "relation registry: relations must be a list"
        raise TypeError(msg)

    out: dict[str, RelationEntry] = {}
    for row in rels:
        if not isinstance(row, dict):
            msg = "relation registry: each relation entry must be a mapping"
            raise TypeError(msg)
        rel = str(row.get("relation", "")).strip()
        status = str(row.get("status", "")).strip()
        notes = str(row.get("notes", "")).strip()
        if not rel or not status or not notes:
            msg = f"relation registry: invalid entry (relation={rel!r}, status={status!r})"
            raise ValueError(msg)
        if status not in {"active", "defer", "deprecated_alias"}:
            msg = f"relation registry: invalid status for {rel!r}: {status!r}"
            raise ValueError(msg)
        st: Status = cast("Status", status)

        defer_reason = row.get("defer_reason")
        dr: DeferReason | None
        if defer_reason is None:
            dr = None
        else:
            drs = str(defer_reason).strip()
            if drs not in {
                "ambiguous_semantics",
                "explosion_risk",
                "licensing",
                "needs_external_ontology",
                "duplicate_of",
            }:
                msg = f"relation registry: invalid defer_reason for {rel!r}: {drs!r}"
                raise ValueError(msg)
            dr = cast("DeferReason", drs)

        maps_to = row.get("maps_to")
        mt = str(maps_to).strip() if maps_to is not None else None
        if mt == "":
            mt = None

        if status == "defer" and dr is None:
            msg = f"relation registry: defer entries must set defer_reason ({rel})"
            raise ValueError(msg)
        if status == "deprecated_alias" and not mt:
            msg = f"relation registry: deprecated_alias must set maps_to ({rel})"
            raise ValueError(msg)

        if rel in out:
            msg = f"relation registry: duplicate relation key: {rel!r}"
            raise ValueError(msg)
        out[rel] = RelationEntry(
            relation=rel, status=st, notes=notes, defer_reason=dr, maps_to=mt
        )

    for rel, ent in sorted(out.items()):
        if ent.status != "deprecated_alias":
            continue
        seen: set[str] = {rel}
        cur = ent.maps_to
        while cur is not None:
            if cur in seen:
                msg = f"relation registry: alias cycle involving {rel!r}"
                raise ValueError(msg)
            seen.add(cur)
            nxt = out.get(cur)
            if nxt is None:
                msg = f"relation registry: alias maps_to missing target: {rel!r} -> {cur!r}"
                raise ValueError(msg)
            if nxt.status != "deprecated_alias":
                break
            cur = nxt.maps_to

    return out


# Relations consumed only by the phenotype builder as canonical size-2 atomics.
PHENOTYPE_BUILDER_RELATIONS: frozenset[str] = frozenset(
    {"drug_effect", "disease_phenotype_positive"}
)


def normalize_relation(registry: dict[str, RelationEntry], relation: str) -> str:
    r = relation.strip()
    cur = r
    for _ in range(32):
        ent = registry.get(cur)
        if ent is None:
            return cur
        if ent.status == "deprecated_alias" and ent.maps_to:
            cur = ent.maps_to.strip()
            continue
        return cur
    msg = f"relation registry: too many alias hops for {r!r}"
    raise ValueError(msg)
