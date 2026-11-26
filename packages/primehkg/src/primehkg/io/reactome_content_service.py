from __future__ import annotations

from collections.abc import Callable
from typing import Any

from primehkg.io.curated_node_ids import (
    curated_chebi_node_id,
    normalize_uniprot_accession,
)

FetchJson = Callable[[str], Any]  # stId/dbId -> ContentService JSON (dict|list)


def _node_id_for_reference(database: object, identifier: object) -> str | None:
    """Map a referenceEntity (database, identifier) to a node_id, or None.

    ``chebi`` -> ``chebi:<id>``; ``uniprot`` -> bare AC. Other databases -> None.
    """
    if not isinstance(database, str) or not isinstance(identifier, str):
        return None
    ident = identifier.strip()
    if not ident:
        return None
    db = database.strip().lower()
    if "chebi" in db:
        return curated_chebi_node_id(ident)
    if "uniprot" in db:
        return normalize_uniprot_accession(ident)
    return None


def _resolve_entity(
    st_id: str,
    *,
    fetch_json: FetchJson,
    entity_cache: dict[str, list[str]],
    _visiting: frozenset[str] = frozenset(),
) -> list[str]:
    """Recursively resolve a PhysicalEntity stId to leaf node_ids (ChEBI/UniProt).

    Memoized in ``entity_cache``. Cycle-safe via ``_visiting``. Unresolvable -> [].
    """
    if st_id in entity_cache:
        return entity_cache[st_id]
    if st_id in _visiting:
        return []
    try:
        obj = fetch_json(st_id)
    except Exception:  # network/JSON robustness
        return []
    if not isinstance(obj, dict):
        return []

    schema = obj.get("schemaClass")
    is_composite = schema in {"Complex", "DefinedSet", "CandidateSet", "EntitySet"}
    result: list[str] = []
    if schema in {"SimpleEntity", "EntityWithAccessionedSequence"}:
        ref = obj.get("referenceEntity")
        if isinstance(ref, dict):
            node_id = _node_id_for_reference(
                ref.get("databaseName"), ref.get("identifier")
            )
            if node_id is not None:
                result = [node_id]
    elif schema in {"Complex", "DefinedSet", "CandidateSet", "EntitySet"}:
        child_keys = ("hasComponent", "hasMember", "hasCandidate")
        next_visiting = _visiting | {st_id}
        seen: set[str] = set()
        for key in child_keys:
            members = obj.get(key)
            if not isinstance(members, list):
                continue
            for member in members:
                if not isinstance(member, dict):
                    continue
                child_st = member.get("stId")
                if not isinstance(child_st, str) or not child_st:
                    continue
                for node_id in _resolve_entity(
                    child_st,
                    fetch_json=fetch_json,
                    entity_cache=entity_cache,
                    _visiting=next_visiting,
                ):
                    if node_id not in seen:
                        seen.add(node_id)
                        result.append(node_id)

    if not (is_composite and _visiting):
        entity_cache[st_id] = result
    return result


def _entity_st_id(obj: Any) -> str | None:
    if isinstance(obj, dict):
        st = obj.get("stId")
        if isinstance(st, str) and st:
            return st
    return None


def _roles_from_reaction(
    reaction: dict[str, Any], *, fetch_json: FetchJson
) -> list[tuple[str, str]]:
    """Extract (role, entity_stId) pairs from a /data/query reaction object.

    catalystActivity / regulatedBy items carry a dbId; a secondary fetch resolves the
    physical entity. A failing secondary fetch drops only that participant.
    """
    pairs: list[tuple[str, str]] = []

    pairs.extend(
        ("input", st)
        for st in (_entity_st_id(x) for x in reaction.get("input") or [])
        if st is not None
    )
    pairs.extend(
        ("output", st)
        for st in (_entity_st_id(x) for x in reaction.get("output") or [])
        if st is not None
    )

    for item in reaction.get("catalystActivity") or []:
        db_id = item.get("dbId") if isinstance(item, dict) else None
        if db_id is None:
            continue
        try:
            detail = fetch_json(str(db_id))
        except Exception:  # secondary fetch failure drops only this participant
            continue
        if not isinstance(detail, dict):
            continue
        st = _entity_st_id(detail.get("physicalEntity"))
        if st is not None:
            pairs.append(("catalyst", st))

    for item in reaction.get("regulatedBy") or []:
        db_id = item.get("dbId") if isinstance(item, dict) else None
        if db_id is None:
            continue
        try:
            detail = fetch_json(str(db_id))
        except Exception:  # secondary fetch failure drops only this participant
            continue
        if not isinstance(detail, dict):
            continue
        schema = detail.get("schemaClass")
        if schema == "PositiveRegulation":
            role = "positive_regulator"
        elif schema == "NegativeRegulation":
            role = "negative_regulator"
        else:
            continue
        st = _entity_st_id(detail.get("regulator"))
        if st is not None:
            pairs.append((role, st))

    return pairs


def resolve_reaction_participants(
    reaction_id: str,
    *,
    fetch_json: FetchJson,
    entity_cache: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """Resolve a reaction stId to sorted, unique (role, node_id) rows. [] on failure."""
    try:
        reaction = fetch_json(reaction_id)
    except Exception:  # reaction fetch failure -> empty
        return []
    if not isinstance(reaction, dict):
        return []

    rows: set[tuple[str, str]] = set()
    for role, entity_st in _roles_from_reaction(reaction, fetch_json=fetch_json):
        for node_id in _resolve_entity(
            entity_st, fetch_json=fetch_json, entity_cache=entity_cache
        ):
            rows.add((role, node_id))
    return sorted(rows)
