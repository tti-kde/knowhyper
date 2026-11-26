from __future__ import annotations

import hashlib
import json
import re
from typing import Any


class _MessageValueError(ValueError):
    _MSG = "Invalid value."

    def __str__(self) -> str:
        return self._MSG


class _SemanticsError(_MessageValueError):
    pass


class _SemanticsWhitespaceError(_SemanticsError):
    _MSG = "Invalid semantics: leading/trailing whitespace is not allowed."


class _SemanticsColonError(_SemanticsError):
    _MSG = "Invalid semantics: ':' is not allowed."


class _SemanticsCharsetError(_SemanticsError):
    _MSG = "Invalid semantics: must match regex ^[a-z0-9_]+$."


class _RoleKeyCollisionError(_MessageValueError):
    _MSG = "Role key collision after normalization."


class _NodeIdEmptyError(_MessageValueError):
    _MSG = "Invalid node_ids: empty node_id after normalization."


class _NodeIdDuplicateError(_MessageValueError):
    _MSG = "Invalid node_ids: duplicates after normalization."


class _RolesNodeMismatchError(_MessageValueError):
    _MSG = "Invalid roles: roles keys must match node_ids after normalization."


class _RoleValueEmptyError(_MessageValueError):
    _MSG = "Invalid roles: role values must be non-empty after normalization."


def normalize_node_id(node_id: str) -> str:
    return node_id.strip()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def he_id(
    *,
    semantics: str,
    node_ids: list[str],
    roles: dict[str, str],
    rule_trace: dict[str, Any],
) -> str:
    if semantics != semantics.strip():
        raise _SemanticsWhitespaceError
    if ":" in semantics:
        raise _SemanticsColonError
    if not re.fullmatch(r"^[a-z0-9_]+$", semantics):
        raise _SemanticsCharsetError

    norm_nodes = [normalize_node_id(n) for n in node_ids]
    if any(not n for n in norm_nodes):
        raise _NodeIdEmptyError
    if len(set(norm_nodes)) != len(norm_nodes):
        raise _NodeIdDuplicateError
    norm_nodes_sorted = sorted(norm_nodes)

    roles_normalized: dict[str, str] = {}
    for raw_k, raw_v in roles.items():
        k = normalize_node_id(raw_k)
        if k in roles_normalized:
            raise _RoleKeyCollisionError
        v = raw_v.strip()
        if not v:
            raise _RoleValueEmptyError
        roles_normalized[k] = v

    if set(roles_normalized.keys()) != set(norm_nodes):
        raise _RolesNodeMismatchError

    norm_roles = {k: roles_normalized[k] for k in sorted(roles_normalized)}
    payload = {
        "semantics": semantics,
        "node_ids": norm_nodes_sorted,
        "roles": norm_roles,
        "rule_trace": rule_trace,
    }
    # Use 128 bits (32 hex chars) to reduce collision risk at scale.
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:32]
    return f"he:{semantics}:{digest}"
