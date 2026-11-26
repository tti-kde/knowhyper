from __future__ import annotations

from typing import Any

from primehkg.features.he_text_sources import load_he_text_templates, render_he_text


def build_he_text_rows(
    *,
    hyperedges: list[dict[str, Any]],
    node_labels: dict[str, str],
    template_id: str,
) -> list[dict[str, Any]]:
    """Textualize all hyperedges using the named template."""
    templates = load_he_text_templates(template_id)
    rows: list[dict[str, Any]] = []
    for he in hyperedges:
        src = render_he_text(
            he,
            node_labels=node_labels,
            template_id=template_id,
            templates=templates,
        )
        rows.append(
            {
                "he_id": src.he_id,
                "template_id": src.template_id,
                "text": src.text,
                "semantics": src.semantics,
                "node_count": src.node_count,
            }
        )
    return rows
