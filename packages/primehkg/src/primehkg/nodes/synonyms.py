from __future__ import annotations


def normalize_synonym_key(value: object) -> str:
    """Lowercase and remove all whitespace (Issue #463 genuine-synonym key)."""
    if not isinstance(value, str):
        return ""
    return "".join(value.lower().split())


def local_id_from_node_id(node_id: str) -> str:
    return node_id.rsplit(":", 1)[-1]


def is_genuine_synonym(surface: object, *, name: object, node_id: str) -> bool:
    if not isinstance(surface, str) or not surface.strip():
        return False
    key = normalize_synonym_key(surface)
    if not key:
        return False
    if key == normalize_synonym_key(name):
        return False
    return key != normalize_synonym_key(local_id_from_node_id(node_id))
