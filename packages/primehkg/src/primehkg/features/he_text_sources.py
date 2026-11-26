from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import yaml


@dataclass(frozen=True)
class HeTextSource:
    """One textualized hyperedge."""

    he_id: str
    template_id: str
    text: str
    semantics: str
    node_count: int


def load_he_text_templates(template_id: str) -> dict[str, str]:
    """Load per-semantics templates from the packaged YAML registry."""
    policy_dir = files("primehkg.policy")
    yaml_path = policy_dir / "he_text_templates.yaml"
    content = yaml_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    templates = data.get("templates", {})
    if template_id not in templates:
        available = sorted(templates.keys())
        msg = f"template_id {template_id!r} not found; available: {available}"
        raise ValueError(msg)
    return dict(templates[template_id])


def _node_role_pairs(node_ids: list[str], roles_raw: Any) -> list[tuple[str, str]]:
    """Pair node_ids with role names (production schema: dict[node_id, role])."""
    if isinstance(roles_raw, dict):
        return [
            (nid, str(roles_raw.get(nid, f"node_{i + 1}")))
            for i, nid in enumerate(node_ids)
        ]
    if isinstance(roles_raw, list):
        return [
            (nid, str(role)) for nid, role in zip(node_ids, roles_raw, strict=False)
        ]
    return [(nid, f"node_{i + 1}") for i, nid in enumerate(node_ids)]


def render_he_text(
    he: dict[str, Any],
    *,
    node_labels: dict[str, str],
    template_id: str,
    templates: dict[str, str] | None = None,
) -> HeTextSource:
    """Render a single hyperedge into natural language text."""
    if templates is None:
        templates = load_he_text_templates(template_id)

    semantics = str(he.get("semantics", ""))
    node_ids: list[str] = list(he.get("node_ids", []))
    pairs = _node_role_pairs(node_ids, he.get("roles", {}))

    template_str = templates.get(semantics, templates.get("_fallback", ""))

    slot_values: dict[str, str] = {}
    join_parts: list[str] = []
    for i, (nid, role) in enumerate(pairs):
        label = node_labels.get(nid, nid)
        slot_values[role] = label
        slot_values[f"node_{i + 1}"] = label
        slot_values[f"role_{i + 1}"] = role
        join_parts.append(f"{label} ({role})")

    slot_values["nodes_join"] = ", ".join(join_parts)
    slot_values["semantics"] = semantics

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return slot_values.get(key, f"<{key}>")

    text = re.sub(r"\{(\w+)\}", _replace, template_str)
    text = re.sub(r",\s*<[^>]+>", "", text)

    return HeTextSource(
        he_id=str(he.get("he_id", "")),
        template_id=template_id,
        text=text.strip(),
        semantics=semantics,
        node_count=len(node_ids),
    )
