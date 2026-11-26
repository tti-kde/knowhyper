from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

from datasets import DatasetDict, load_from_disk

from primehkg.io.build_manifest import stable_json_dumps


@dataclass(frozen=True)
class EmbeddingInput:
    """One normalized source payload to encode into an embedding row."""

    record_id: str
    target: Literal["node", "hyperedge"]
    input_kind: str
    source_modality: str
    text: str
    input_hash: str
    source: str
    source_version: str
    license_tag: str = ""


def normalize_embedding_input(value: object) -> str | None:
    """Return NFKC-normalized, stripped text, or None for empty/non-text values."""
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFKC", value).strip()
    if not text:
        return None
    return text


def normalized_input_hash(value: str) -> str:
    """SHA-256 hex digest over the normalized embedding input text."""
    text = normalize_embedding_input(value) or ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_node_inputs(
    *,
    core_dir: str | Path,
    features_dir: str | Path | None,
    input_kinds: frozenset[str] | None = None,
) -> list[EmbeddingInput]:
    """Collect node embedding inputs from core text and optional feature splits."""
    core = _load_dataset_dict(core_dir)
    allowed_node_ids = _core_node_ids(core)
    core_source_version = _manifest_inputs_fingerprint(core_dir)
    rows: list[EmbeddingInput] = []

    if _input_kind_allowed("text", input_kinds):
        for row in core["nodes"]:
            if not isinstance(row, dict):
                continue
            node_id = _non_empty_str(row.get("node_id"))
            text = normalize_embedding_input(row.get("name"))
            if node_id is None or text is None:
                continue
            rows.append(
                _embedding_input(
                    record_id=node_id,
                    target="node",
                    input_kind="text",
                    source_modality="display_name",
                    text=text,
                    source="primehkg-core",
                    source_version=core_source_version,
                )
            )

    if features_dir is not None:
        features = _load_dataset_dict(features_dir)
        rows.extend(
            _collect_node_text_feature_inputs(
                features=features,
                allowed_node_ids=allowed_node_ids,
                input_kinds=input_kinds,
            )
        )
        rows.extend(
            _collect_node_sequence_inputs(
                features=features,
                allowed_node_ids=allowed_node_ids,
                input_kinds=input_kinds,
            )
        )
        rows.extend(
            _collect_node_property_vector_inputs(
                core=core,
                features=features,
                input_kinds=input_kinds,
            )
        )

    rows.sort(
        key=lambda row: (
            row.record_id,
            row.input_kind,
            row.source_modality,
            row.input_hash,
        )
    )
    return rows


def collect_he_text_inputs(*, he_text_dir: str | Path) -> list[EmbeddingInput]:
    """Collect hyperedge text embedding inputs from the he_features_text split."""
    loaded = _load_dataset_dict(he_text_dir)
    source_version = _manifest_inputs_fingerprint(he_text_dir)
    rows: list[EmbeddingInput] = []
    for row in loaded["he_features_text"]:
        if not isinstance(row, dict):
            continue
        he_id = _non_empty_str(row.get("he_id"))
        template_id = _non_empty_str(row.get("template_id"))
        text = normalize_embedding_input(row.get("text"))
        if he_id is None or template_id is None or text is None:
            continue
        rows.append(
            _embedding_input(
                record_id=he_id,
                target="hyperedge",
                input_kind="hyperedge_text",
                source_modality=template_id,
                text=text,
                source="primehkg-he-text-build",
                source_version=source_version,
            )
        )

    rows.sort(
        key=lambda row: (
            row.record_id,
            row.input_kind,
            row.source_modality,
            row.input_hash,
        )
    )
    return rows


def iter_input_chunks(
    *,
    targets: frozenset[str],
    core_dir: str | Path,
    features_dir: str | Path | None,
    he_text_dir: str | Path | None,
    node_input_kinds: frozenset[str] | None,
    chunk_size: int,
) -> Iterator[list[EmbeddingInput]]:
    """Yield EmbeddingInput chunks of <= chunk_size, streaming node then HE inputs.

    Reuses collect_node_inputs / collect_he_text_inputs so per-input identity
    (input_hash, source_modality, ...) is identical to the non-chunked path; only the
    delivery is chunked. The build re-sorts all rows at the end, so chunk order does not
    affect the output.
    """
    if chunk_size < 1:
        msg = f"chunk_size must be >= 1, got {chunk_size}"
        raise ValueError(msg)

    def _emit(items: list[EmbeddingInput]) -> Iterator[list[EmbeddingInput]]:
        for start in range(0, len(items), chunk_size):
            yield items[start : start + chunk_size]

    if "node" in targets:
        node_items = collect_node_inputs(
            core_dir=core_dir, features_dir=features_dir, input_kinds=node_input_kinds
        )
        yield from _emit(node_items)
    if "hyperedge" in targets:
        if he_text_dir is None:
            msg = "he_text_dir is required when targets includes hyperedge"
            raise ValueError(msg)
        he_items = collect_he_text_inputs(he_text_dir=he_text_dir)
        yield from _emit(he_items)


def input_snapshot_hash5(items: Iterable[tuple[str, str, str, str, str]]) -> str:
    """SHA-256 over stable JSON lines identifying normalized inputs, from pre-reduced 5-tuples.

    Each tuple is (record_id, target, input_kind, source_modality, input_hash).
    """
    lines = [
        stable_json_dumps(
            {
                "record_id": record_id,
                "target": target,
                "input_kind": input_kind,
                "source_modality": source_modality,
                "input_hash": input_hash,
            }
        )
        for record_id, target, input_kind, source_modality, input_hash in sorted(items)
    ]
    payload = "\n".join(lines)
    if lines:
        payload += "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def input_snapshot_hash(inputs: list[EmbeddingInput]) -> str:
    """SHA-256 over stable JSON lines identifying the normalized inputs."""
    return input_snapshot_hash5(
        (row.record_id, row.target, row.input_kind, row.source_modality, row.input_hash)
        for row in inputs
    )


def _load_dataset_dict(path: str | Path) -> DatasetDict:
    loaded = load_from_disk(str(path))
    if not isinstance(loaded, DatasetDict):
        msg = f"expected DatasetDict at {path!r}, got {type(loaded).__name__}"
        raise TypeError(msg)
    return loaded


def _manifest_inputs_fingerprint(path: str | Path) -> str:
    manifest_path = Path(path) / "build_manifest.json"
    if not manifest_path.is_file():
        return "unknown"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = data.get("inputs_fingerprint") if isinstance(data, dict) else None
    if isinstance(fingerprint, str) and fingerprint:
        return fingerprint
    return "unknown"


def _core_node_ids(core: DatasetDict) -> frozenset[str]:
    ids: set[str] = set()
    for row in core["nodes"]:
        if not isinstance(row, dict):
            continue
        node_id = _non_empty_str(row.get("node_id"))
        if node_id is not None:
            ids.add(node_id)
    return frozenset(ids)


def _collect_node_text_feature_inputs(
    *,
    features: DatasetDict,
    allowed_node_ids: frozenset[str],
    input_kinds: frozenset[str] | None,
) -> list[EmbeddingInput]:
    if not _input_kind_allowed("text", input_kinds):
        return []
    if "node_features_text" not in features:
        return []

    rows: list[EmbeddingInput] = []
    for row in features["node_features_text"]:
        if not isinstance(row, dict):
            continue
        node_id = _non_empty_str(row.get("node_id"))
        if node_id is None or node_id not in allowed_node_ids:
            continue
        modality = _non_empty_str(row.get("modality"))
        text = normalize_embedding_input(row.get("text"))
        source = _non_empty_str(row.get("source"))
        source_version = _non_empty_str(row.get("source_version"))
        if modality is None or text is None or source is None or source_version is None:
            continue
        rows.append(
            _embedding_input(
                record_id=node_id,
                target="node",
                input_kind="text",
                source_modality=modality,
                text=text,
                source=source,
                source_version=source_version,
                license_tag=_optional_str(row.get("license_tag")),
            )
        )
    return rows


def _collect_node_sequence_inputs(
    *,
    features: DatasetDict,
    allowed_node_ids: frozenset[str],
    input_kinds: frozenset[str] | None,
) -> list[EmbeddingInput]:
    if "node_features_sequence" not in features:
        return []

    input_kind_by_sequence_type = {
        "protein": "protein_sequence",
        "nucleotide": "nucleotide_sequence",
        "smiles": "smiles",
    }
    rows: list[EmbeddingInput] = []
    for row in features["node_features_sequence"]:
        if not isinstance(row, dict):
            continue
        node_id = _non_empty_str(row.get("node_id"))
        if node_id is None or node_id not in allowed_node_ids:
            continue
        sequence_type = _non_empty_str(row.get("sequence_type"))
        if sequence_type is None:
            continue
        input_kind = input_kind_by_sequence_type.get(sequence_type)
        if input_kind is None or not _input_kind_allowed(input_kind, input_kinds):
            continue
        text = normalize_embedding_input(row.get("sequence"))
        source = _non_empty_str(row.get("source"))
        source_version = _non_empty_str(row.get("source_version"))
        if text is None or source is None or source_version is None:
            continue
        rows.append(
            _embedding_input(
                record_id=node_id,
                target="node",
                input_kind=input_kind,
                source_modality=sequence_type,
                text=text,
                source=source,
                source_version=source_version,
                license_tag=_optional_str(row.get("license_tag")),
            )
        )
    return rows


def _collect_node_property_vector_inputs(
    *,
    core: DatasetDict,
    features: DatasetDict,
    input_kinds: frozenset[str] | None,
) -> list[EmbeddingInput]:
    if not _input_kind_allowed("property_vector", input_kinds):
        return []
    if "node_features_property" not in features:
        return []

    from primehkg.features.property_vector import (
        build_property_vectors,
        property_vector_input_text,
    )
    from primehkg.policy.node_property_registry import load_node_property_registry

    registry = load_node_property_registry()
    registry_order = tuple(sorted(registry.by_id.keys()))
    property_vectors = build_property_vectors(
        nodes=[dict(row) for row in core["nodes"]],
        property_rows=[dict(row) for row in features["node_features_property"]],
        registry_order=registry_order,
    )

    rows: list[EmbeddingInput] = []
    for vector in property_vectors:
        text = property_vector_input_text(
            property_ids=vector.property_ids,
            values=vector.values,
            missing_mask=vector.missing_mask,
        )
        rows.append(
            _embedding_input(
                record_id=vector.node_id,
                target="node",
                input_kind="property_vector",
                source_modality="property_numeric_v1",
                text=text,
                source=vector.source,
                source_version=vector.source_version,
            )
        )
    return rows


def _embedding_input(
    *,
    record_id: str,
    target: Literal["node", "hyperedge"],
    input_kind: str,
    source_modality: str,
    text: str,
    source: str,
    source_version: str,
    license_tag: str = "",
) -> EmbeddingInput:
    return EmbeddingInput(
        record_id=record_id,
        target=target,
        input_kind=input_kind,
        source_modality=source_modality,
        text=text,
        input_hash=normalized_input_hash(text),
        source=source,
        source_version=source_version,
        license_tag=license_tag,
    )


def _input_kind_allowed(input_kind: str, input_kinds: frozenset[str] | None) -> bool:
    return input_kinds is None or input_kind in input_kinds


def _non_empty_str(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value


def _optional_str(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value
