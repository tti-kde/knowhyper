from __future__ import annotations

import hashlib
import json
import pathlib
from typing import TYPE_CHECKING, Any

from primehkg.io.build_manifest import stable_json_dumps

if TYPE_CHECKING:
    from primehkg.features.embedding_text_sources import TextSource


def compute_text_snapshot_hash(sources: list[TextSource]) -> str:
    """SHA-256 over stable JSON lines sorted by ``(node_id, source_modality, text)``."""
    sorted_sources = sorted(
        sources, key=lambda s: (s.node_id, s.source_modality, s.text)
    )
    lines = [
        stable_json_dumps(
            {
                "node_id": s.node_id,
                "source_modality": s.source_modality,
                "text": s.text,
            }
        )
        for s in sorted_sources
    ]
    payload = "\n".join(lines)
    if lines:
        payload += "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_embed_manifest(path: str | pathlib.Path, *, manifest: dict[str, Any]) -> None:
    """Write ``embed_manifest.json`` under ``path`` (directory or file path)."""
    p = pathlib.Path(path)
    out_file = p if p.suffix == ".json" else p / "embed_manifest.json"
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    out_file.write_text(text, encoding="utf-8")
