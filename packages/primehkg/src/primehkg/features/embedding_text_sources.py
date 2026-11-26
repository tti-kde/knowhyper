from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextSource:
    """One embeddable text attribute for a canonical node."""

    node_id: str
    source_modality: str
    text: str
    source: str
    source_version: str


def _load_core_node_ids(core_dir: str | Path) -> frozenset[str]:
    from datasets import DatasetDict, load_from_disk

    loaded = load_from_disk(str(core_dir))
    if not isinstance(loaded, DatasetDict):
        msg = f"expected DatasetDict at {core_dir!r}, got {type(loaded).__name__}"
        raise TypeError(msg)
    nodes = loaded["nodes"]
    ids: set[str] = set()
    for row in nodes:
        nid = row.get("node_id") if isinstance(row, dict) else None
        if isinstance(nid, str) and nid:
            ids.add(nid)
    return frozenset(ids)


def _core_source_version(core_dir: str | Path) -> str:
    manifest_path = Path(core_dir) / "build_manifest.json"
    if not manifest_path.is_file():
        return "unknown"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    fp = data.get("inputs_fingerprint") if isinstance(data, dict) else None
    if isinstance(fp, str) and fp:
        return fp
    return "unknown"


def _non_empty_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def collect_text_sources(
    *,
    core_dir: Path | str,
    text_dataset_dir: Path | str | None,
) -> list[TextSource]:
    """Collect embeddable text from core nodes and optional features text split."""
    from datasets import DatasetDict, load_from_disk

    core_path = Path(core_dir)
    loaded = load_from_disk(str(core_path))
    if not isinstance(loaded, DatasetDict):
        msg = f"expected DatasetDict at {core_path!r}, got {type(loaded).__name__}"
        raise TypeError(msg)

    allowed_node_ids = _load_core_node_ids(core_path)
    core_version = _core_source_version(core_path)
    sources: list[TextSource] = []

    for row in loaded["nodes"]:
        if not isinstance(row, dict):
            continue
        node_id = row.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        name = _non_empty_text(row.get("name"))
        if name is None:
            continue
        sources.append(
            TextSource(
                node_id=node_id,
                source_modality="display_name",
                text=name,
                source="primehkg-core",
                source_version=core_version,
            )
        )

    if text_dataset_dir is not None:
        text_loaded = load_from_disk(str(text_dataset_dir))
        if not isinstance(text_loaded, DatasetDict):
            msg = (
                f"expected DatasetDict at {text_dataset_dir!r}, "
                f"got {type(text_loaded).__name__}"
            )
            raise TypeError(msg)
        text_split = text_loaded["node_features_text"]
        for row in text_split:
            if not isinstance(row, dict):
                continue
            node_id = row.get("node_id")
            if not isinstance(node_id, str) or node_id not in allowed_node_ids:
                continue
            modality = row.get("modality")
            if not isinstance(modality, str) or not modality:
                continue
            text = _non_empty_text(row.get("text"))
            if text is None:
                continue
            source = row.get("source")
            source_version = row.get("source_version")
            if not isinstance(source, str) or not source:
                continue
            if not isinstance(source_version, str) or not source_version:
                continue
            sources.append(
                TextSource(
                    node_id=node_id,
                    source_modality=modality,
                    text=text,
                    source=source,
                    source_version=source_version,
                )
            )

    sources.sort(key=lambda s: (s.node_id, s.source_modality))
    return sources
