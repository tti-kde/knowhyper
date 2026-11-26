"""Raw-evidence text serialization for controlled knowledge baselines.

This module deliberately does not depend on ``KnowledgeRepr`` or an encoder.
Text baselines must be generated from licensed raw evidence, never by attempting
to invert an embedding produced for the neural knowledge path.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pyarrow.parquet as pq

from knowhyper.provenance.values import is_concrete_provenance_value

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ieforge.knowledge.core.schema import Query

    from knowhyper.pipeline.types import SubHypergraph
    from knowhyper.provenance.text_source_licenses import SourceLicenseRegistry

_TEXT_TABLE = "node_features_text"
_POLICY_VERSION = 5
_ROLE_NAMES = ("head", "tail")
_PROVENANCE_SCHEMA = "linked_mentions_role_v1"
_TEXT_PROVENANCE_COLUMNS = (
    "source_field",
    "upstream_source",
    "upstream_release",
    "retrieved_at",
    "source_url",
    "source_artifact_sha256",
    "license_id",
    "license_url",
    "usage_scope",
    "redistribution_scope",
    "attribution_text",
    "provenance_chain",
    "provenance_status",
    "provenance_registry_digest",
)
_IndexCacheKey = tuple[str, tuple[str, ...], str, str, str, str]
_INDEX_CACHE: dict[_IndexCacheKey, DefinitionEvidenceIndex] = {}


@dataclass(frozen=True)
class DefinitionEvidence:
    """One raw textual evidence row from ``primehkg-features``."""

    node_id: str
    modality: str
    text: str
    content_hash: str
    language: str = "en"
    source: str = ""
    source_version: str = ""
    license_tag: str | None = None
    source_field: str = ""
    upstream_source: str = ""
    upstream_release: str = ""
    retrieved_at: str = ""
    source_url: str = ""
    source_artifact_sha256: str = ""
    license_id: str = ""
    license_url: str = ""
    usage_scope: str = ""
    redistribution_scope: str = ""
    attribution_text: str = ""
    provenance_chain: str = ""
    provenance_status: str = ""
    provenance_registry_digest: str = ""


@dataclass(frozen=True)
class SerializedDefinition:
    """Selected definition annotated with its target-pair role."""

    role: str
    node_id: str
    mention: str
    modality: str
    text: str
    content_hash: str
    language: str = "en"
    source: str = ""
    source_version: str = ""
    license_tag: str | None = None
    license_id: str = ""
    license_url: str = ""
    usage_scope: str = ""
    redistribution_scope: str = ""
    payload_class: str = ""
    source_field: str = ""
    source_release: str = ""
    upstream_source: str = ""
    upstream_release: str = ""
    reviewed_at: str = ""
    registry_digest: str = ""
    primehkg_provenance_registry_digest: str = ""
    source_artifact_sha256: str = ""
    retrieved_at: str = ""
    source_url: str = ""
    attribution_text: str = ""
    provenance_chain: str = ""
    provenance_status: str = ""
    primehkg_license_id: str = ""
    primehkg_license_url: str = ""
    primehkg_usage_scope: str = ""
    primehkg_redistribution_scope: str = ""


@dataclass(frozen=True)
class SerializedDefinitions:
    """Human-readable text plus structured entries for budget-aware tokenization."""

    text: str
    items: tuple[SerializedDefinition, ...]
    fingerprint: str


class DefinitionEvidenceIndex:
    """In-memory raw ``description``/``definition`` lookup by entity node id."""

    def __init__(
        self,
        features_dir: str | Path,
        *,
        modalities: Sequence[str] = ("description", "definition"),
        expected_commit: str | None = None,
        expected_inputs_fingerprint: str | None = None,
        expected_text_provenance_schema: str | None = None,
        expected_provenance_registry_sha256: str | None = None,
    ) -> None:
        root = Path(features_dir).resolve()
        manifest = self._validate_source(
            root,
            expected_commit=expected_commit,
            expected_inputs_fingerprint=expected_inputs_fingerprint,
            expected_text_provenance_schema=expected_text_provenance_schema,
            expected_provenance_registry_sha256=(expected_provenance_registry_sha256),
        )
        self.provenance_schema = manifest.get("text_provenance_schema")
        self.provenance_registry_sha256 = manifest.get("provenance_registry_sha256")
        wanted = frozenset(modalities)
        records: list[DefinitionEvidence] = []
        files = sorted(Path(root, _TEXT_TABLE).glob("*.parquet"))
        if not files:
            msg = f"no {_TEXT_TABLE} parquet under {features_dir}"
            raise FileNotFoundError(msg)
        base_columns = [
            "node_id",
            "modality",
            "text",
            "content_hash",
            "language",
            "source",
            "source_version",
            "license_tag",
        ]
        physical_row_count = 0
        provenance_status_counts: Counter[str] = Counter()
        for path in files:
            available = frozenset(pq.read_schema(path).names)
            missing_base = [name for name in base_columns if name not in available]
            if missing_base:
                msg = (
                    f"raw text table {path} is missing required columns: {missing_base}"
                )
                raise ValueError(msg)
            if expected_text_provenance_schema is not None:
                missing_provenance = [
                    name for name in _TEXT_PROVENANCE_COLUMNS if name not in available
                ]
                if missing_provenance:
                    msg = (
                        f"raw text table {path} does not implement "
                        f"{expected_text_provenance_schema}: {missing_provenance}"
                    )
                    raise ValueError(msg)
            columns = base_columns + [
                name for name in _TEXT_PROVENANCE_COLUMNS if name in available
            ]
            table = pq.read_table(path, columns=columns)
            physical_row_count += table.num_rows
            for row in table.to_pylist():
                self._validate_provenance_row(row, manifest)
                status = row.get("provenance_status")
                if status is not None:
                    provenance_status_counts[str(status)] += 1
                modality = str(row["modality"])
                text = str(row["text"] or "").strip()
                if modality not in wanted or not text:
                    continue
                records.append(
                    DefinitionEvidence(
                        node_id=str(row["node_id"]),
                        modality=modality,
                        text=text,
                        content_hash=str(row["content_hash"]),
                        language=str(row["language"] or ""),
                        source=str(row["source"] or ""),
                        source_version=str(row["source_version"] or ""),
                        license_tag=(
                            str(row["license_tag"])
                            if row["license_tag"] is not None
                            else None
                        ),
                        **{
                            name: str(row.get(name) or "")
                            for name in _TEXT_PROVENANCE_COLUMNS
                        },
                    )
                )
        self._validate_row_counts(
            manifest,
            physical_row_count,
            provenance_status_counts=provenance_status_counts,
        )
        self._by_node = self._group(records)

    @staticmethod
    def _validate_source(
        root: Path,
        *,
        expected_commit: str | None,
        expected_inputs_fingerprint: str | None,
        expected_text_provenance_schema: str | None,
        expected_provenance_registry_sha256: str | None,
    ) -> dict[str, Any]:
        if expected_commit is not None and expected_commit not in root.parts:
            msg = f"raw feature snapshot is not pinned to commit {expected_commit}: {root}"
            raise ValueError(msg)
        manifest_path = root / "build_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError as exc:
            if not any(
                (
                    expected_inputs_fingerprint,
                    expected_text_provenance_schema,
                    expected_provenance_registry_sha256,
                )
            ):
                return {}
            msg = f"cannot verify raw feature manifest: {manifest_path}"
            raise ValueError(msg) from exc
        except json.JSONDecodeError as exc:
            msg = f"cannot verify raw feature manifest: {manifest_path}"
            raise ValueError(msg) from exc
        if not isinstance(manifest, dict):
            msg = f"raw feature manifest must be a JSON object: {manifest_path}"
            raise TypeError(msg)
        actual = manifest.get("inputs_fingerprint")
        if (
            expected_inputs_fingerprint is not None
            and actual != expected_inputs_fingerprint
        ):
            msg = (
                "raw feature inputs fingerprint mismatch: "
                f"expected {expected_inputs_fingerprint}, got {actual}"
            )
            raise ValueError(msg)
        actual_schema = manifest.get("text_provenance_schema")
        if (
            expected_text_provenance_schema is not None
            and actual_schema != expected_text_provenance_schema
        ):
            msg = (
                "raw text provenance schema mismatch: "
                f"expected {expected_text_provenance_schema}, got {actual_schema}"
            )
            raise ValueError(msg)
        actual_registry = manifest.get("provenance_registry_sha256")
        if (
            expected_provenance_registry_sha256 is not None
            and actual_registry != expected_provenance_registry_sha256
        ):
            msg = (
                "raw text provenance registry mismatch: "
                f"expected {expected_provenance_registry_sha256}, got "
                f"{actual_registry}"
            )
            raise ValueError(msg)
        return manifest

    @staticmethod
    def _validate_provenance_row(row: dict[str, Any], manifest: dict[str, Any]) -> None:
        registry_digest = manifest.get("provenance_registry_sha256")
        row_digest = row.get("provenance_registry_digest")
        if registry_digest is not None and row_digest != registry_digest:
            msg = (
                "raw text row provenance registry digest mismatch: "
                f"manifest={registry_digest}, row={row_digest}"
            )
            raise ValueError(msg)
        source_inventory = manifest.get("text_provenance_sources")
        if not isinstance(source_inventory, list):
            if row.get("provenance_status") == "complete":
                msg = (
                    "complete raw text row requires manifest "
                    "text_provenance_sources inventory"
                )
                raise ValueError(msg)
            return
        source = str(row.get("source") or "").casefold()
        source_field = str(row.get("source_field") or "")
        matches = [
            item
            for item in source_inventory
            if isinstance(item, dict)
            and str(item.get("source") or "").casefold() == source
            and str(item.get("source_field") or "") == source_field
        ]
        if len(matches) > 1:
            msg = f"duplicate manifest provenance source entry: {source}/{source_field}"
            raise ValueError(msg)
        if not matches:
            if row.get("provenance_status") == "complete":
                msg = (
                    "complete raw text row has no matching manifest provenance "
                    f"source entry: {source}/{source_field}"
                )
                raise ValueError(msg)
            return
        expected_artifact = matches[0].get("source_artifact_sha256")
        actual_artifact = row.get("source_artifact_sha256")
        if expected_artifact is not None and actual_artifact != expected_artifact:
            msg = (
                "raw text source artifact digest mismatch for "
                f"{source}/{source_field}: manifest={expected_artifact}, "
                f"row={actual_artifact}"
            )
            raise ValueError(msg)

    @staticmethod
    def _validate_row_counts(
        manifest: dict[str, Any],
        actual: int,
        *,
        provenance_status_counts: Counter[str],
    ) -> None:
        outputs = manifest.get("outputs")
        expected = outputs.get(_TEXT_TABLE) if isinstance(outputs, dict) else None
        if expected is not None and expected != actual:
            msg = f"raw text row count mismatch: manifest={expected}, parquet={actual}"
            raise ValueError(msg)
        completeness = manifest.get("provenance_completeness")
        if not isinstance(completeness, dict):
            return
        if completeness.get("rows_total") != actual:
            msg = (
                "raw text provenance completeness row count mismatch: "
                f"manifest={completeness.get('rows_total')}, parquet={actual}"
            )
            raise ValueError(msg)
        subtotal = sum(
            int(completeness.get(name, 0))
            for name in ("complete", "partial", "unknown")
        )
        if subtotal != actual:
            msg = (
                "raw text provenance completeness counts do not sum to rows_total: "
                f"{subtotal} != {actual}"
            )
            raise ValueError(msg)
        expected_counts = {
            name: int(completeness.get(name, 0))
            for name in ("complete", "partial", "unknown")
        }
        invalid_statuses = set(provenance_status_counts) - set(expected_counts)
        if invalid_statuses:
            msg = f"raw text rows have invalid provenance_status: {invalid_statuses}"
            raise ValueError(msg)
        actual_counts = {
            name: provenance_status_counts.get(name, 0) for name in expected_counts
        }
        if actual_counts != expected_counts:
            msg = (
                "raw text provenance status counts mismatch: "
                f"manifest={expected_counts}, parquet={actual_counts}"
            )
            raise ValueError(msg)

    @classmethod
    def from_records(
        cls, records: Iterable[DefinitionEvidence]
    ) -> DefinitionEvidenceIndex:
        """Build a small index without parquet, primarily for fixtures/tests."""
        obj = cls.__new__(cls)
        object.__setattr__(obj, "provenance_schema", None)
        object.__setattr__(obj, "provenance_registry_sha256", None)
        object.__setattr__(obj, "_by_node", cls._group(records))
        return obj

    @staticmethod
    def _group(
        records: Iterable[DefinitionEvidence],
    ) -> dict[str, tuple[DefinitionEvidence, ...]]:
        grouped: dict[str, dict[tuple[str, ...], DefinitionEvidence]] = {}
        for record in records:
            # The raw table has legitimate repeated (node, modality) rows. Exact
            # content duplicates must not depend on parquet row order.
            # Content equality does not transfer provenance or licensing rights.
            # Preserve identical text independently for every source decision so
            # the license filter, rather than lexicographic source order, chooses
            # the eligible row.
            key = (
                record.modality,
                record.content_hash,
                record.source,
                record.source_version,
                record.license_tag or "",
                record.source_field,
                record.upstream_source,
                record.upstream_release,
                record.source_artifact_sha256,
                record.license_id,
                record.usage_scope,
                record.redistribution_scope,
                record.provenance_status,
                record.provenance_registry_digest,
            )
            current = grouped.setdefault(record.node_id, {}).get(key)
            if current is None or (
                record.source,
                record.text,
                record.language,
                record.source_version,
                record.license_tag or "",
            ) < (
                current.source,
                current.text,
                current.language,
                current.source_version,
                current.license_tag or "",
            ):
                grouped[record.node_id][key] = record
        return {
            node_id: tuple(
                sorted(
                    values.values(),
                    key=lambda row: (
                        row.modality,
                        row.source,
                        row.source_version,
                        row.content_hash,
                        row.source_field,
                        row.upstream_release,
                        row.provenance_registry_digest,
                        row.text,
                    ),
                )
            )
            for node_id, values in sorted(grouped.items())
        }

    def candidates(self, node_id: str) -> tuple[DefinitionEvidence, ...]:
        """Return all raw candidates for a node; selection belongs to the policy."""
        return self._by_node.get(node_id, ())


class NodeNameIndex:
    """``node_id -> name`` taken from the KG core node table.

    Names are a first-class attribute of the node itself and are present for
    100% of nodes (137,587/137,587 in v0.12.3, including all 7,957 drug nodes).
    They are NOT reliably available from ``node_features_text``: that table is a
    features sidecar whose ``short_label`` rows cover ontology entities but not
    PrimeKG drug nodes, which is why a name-bearing arm must read the core node
    table instead of filtering the features table.
    """

    def __init__(self, core: Any) -> None:
        self._by_node: dict[str, str] = {}
        for row in core.nodes:
            node_id = str(row["node_id"])
            name = str(row.get("name") or "").strip()
            if name:
                self._by_node[node_id] = name

    def __len__(self) -> int:
        """Return how many nodes carry a name."""
        return len(self._by_node)

    def name_for(self, node_id: str) -> str | None:
        """Return the node's display name, or ``None`` when the node is unknown."""
        return self._by_node.get(node_id)


def load_definition_evidence_index(
    features_dir: str | Path,
    *,
    modalities: Sequence[str] = ("description", "definition"),
    expected_commit: str | None = None,
    expected_inputs_fingerprint: str | None = None,
    expected_text_provenance_schema: str | None = None,
    expected_provenance_registry_sha256: str | None = None,
) -> DefinitionEvidenceIndex:
    """Load one immutable raw index per process and share it across split collators.

    ieforge constructs separate train/validation/test knowledge collators. Without
    this factory each would independently materialize the same raw parquet table.
    DataLoader fork workers then share the already-built immutable pages copy-on-write.
    """
    key = (
        str(Path(features_dir).resolve()),
        tuple(str(item) for item in modalities),
        str(expected_commit),
        str(expected_inputs_fingerprint),
        str(expected_text_provenance_schema),
        str(expected_provenance_registry_sha256),
    )
    cached = _INDEX_CACHE.get(key)
    if cached is None:
        cached = DefinitionEvidenceIndex(
            features_dir,
            modalities=modalities,
            expected_commit=expected_commit,
            expected_inputs_fingerprint=expected_inputs_fingerprint,
            expected_text_provenance_schema=expected_text_provenance_schema,
            expected_provenance_registry_sha256=(expected_provenance_registry_sha256),
        )
        _INDEX_CACHE[key] = cached
    return cached


class RetrievedDefinitionsSerializer:
    """Select exactly one definition-like text for each target seed entity.

    Selection is independent from typed-sub-HG serialization (R2b): only linked
    seed entities are eligible, ``description`` wins over ``definition``, and
    ties are broken by stable evidence metadata rather than dataset row order.
    """

    #: Metadata key the injector publishes items under. Declared explicitly rather
    #: than relying on a default so a new arm cannot silently inherit this one's
    #: key and train with no knowledge. Deliberately NOT part of ``identity``
    #: below, so declaring it leaves every completed R2a text cache valid.
    items_key = "retrieved_definitions"

    def __init__(
        self,
        evidence_index: DefinitionEvidenceIndex,
        *,
        modalities: Sequence[str] = ("description", "definition"),
        node_name_index: NodeNameIndex | None = None,
        license_registry: SourceLicenseRegistry | None = None,
        allowed_payload_classes: Sequence[str] = (),
        audience: str = "legacy",
        raw_text_cache_mode: str = "restricted",
        cache_root: str | Path | None = None,
        source_fingerprint: str = "unversioned",
        source_revision: str = "unversioned",
        required_provenance_schema: str | None = _PROVENANCE_SCHEMA,
        max_targets: int | None = None,
    ) -> None:
        if not modalities:
            msg = "RetrievedDefinitionsSerializer needs at least one modality"
            raise ValueError(msg)
        if audience not in {"legacy", "publishable", "internal"}:
            msg = f"unsupported text-evidence audience: {audience!r}"
            raise ValueError(msg)
        if raw_text_cache_mode not in {"restricted", "disabled"}:
            msg = f"unsupported raw text cache mode: {raw_text_cache_mode!r}"
            raise ValueError(msg)
        if audience != "legacy" and license_registry is None:
            msg = "license-aware text selection requires a source license registry"
            raise ValueError(msg)
        if license_registry is not None:
            license_registry.validate_for_audience(audience)
        if audience == "publishable":
            if evidence_index.provenance_schema is None:
                msg = (
                    "publishable text selection requires a versioned PrimeHKG "
                    "text provenance schema; use audience=legacy for legacy snapshots"
                )
                raise ValueError(msg)
            if evidence_index.provenance_registry_sha256 is None:
                msg = (
                    "publishable text selection requires a pinned PrimeHKG "
                    "provenance registry digest"
                )
                raise ValueError(msg)
        self._index = evidence_index
        self._modalities = tuple(modalities)
        self._node_name_index = node_name_index
        self._priority = {name: index for index, name in enumerate(self._modalities)}
        self._license_registry = license_registry
        self._allowed_payload_classes = frozenset(allowed_payload_classes)
        self._audience = audience
        self._raw_text_cache_mode = raw_text_cache_mode
        self._requires_complete_upstream_provenance = audience == "publishable"
        self._cache_root = (
            Path(cache_root)
            if cache_root is not None and raw_text_cache_mode != "disabled"
            else None
        )
        self._required_provenance_schema = required_provenance_schema
        if max_targets is not None and max_targets < 1:
            msg = f"max_targets must be >= 1, got {max_targets}"
            raise ValueError(msg)
        # Upper bound on serialized items. RE has exactly two targets (head/tail) and
        # never needs this; NER's generic target branch can return the WHOLE seed set,
        # which is unbounded from the collator's point of view. The collator spends
        # ~4 wrapper tokens per item before any content, so an uncapped instance blew
        # the budget: "max_knowledge_tokens is too small to preserve definition
        # boundary tags: need 382, got 256" -- 95 items of pure tags. `injector.top_k`
        # does NOT bound this; it caps hyperedges, not targets.
        self._max_targets = max_targets
        identity = {
            "policy": "targeted_seed_definitions",
            "policy_version": _POLICY_VERSION,
            "modalities": self._modalities,
            "license_registry_sha256": (
                license_registry.digest if license_registry is not None else None
            ),
            "allowed_payload_classes": sorted(self._allowed_payload_classes),
            "audience": audience,
            "raw_text_cache_mode": raw_text_cache_mode,
            "source_fingerprint": source_fingerprint,
            "source_revision": source_revision,
            # This is the digest of PrimeHKG's upstream provenance registry. It
            # is intentionally distinct from license_registry_sha256, which is
            # KnowHyper's downstream selection policy digest.
            "primehkg_provenance_registry_sha256": (
                evidence_index.provenance_registry_sha256
            ),
            "primehkg_text_provenance_schema": evidence_index.provenance_schema,
            "template": "pair_roles_v1",
            "provenance_schema": required_provenance_schema,
        }
        # Added only when a narrowing is actually requested, so every arm built
        # before the name fallback existed keeps its exact fingerprint — and
        # hence its text cache. A name-bearing arm sets it and gets its own cache.
        if self._node_name_index is not None:
            identity["node_name_fallback"] = "core_nodes_v1"
        # Same rule: added ONLY when a cap is requested, so every existing arm keeps
        # its exact fingerprint and its text cache. A capped arm gets its own.
        if self._max_targets is not None:
            identity["max_targets"] = int(self._max_targets)
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        self.fingerprint = hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def serialize(self, query: Query, sub: SubHypergraph) -> SerializedDefinitions:
        """Serialize definitions for the linked target pair in stable role order."""
        targets = self._targets(query, sub)
        # Truncate AFTER _targets so the ordering it establishes (role order for RE,
        # linked-mention order then sorted seeds for NER) decides which survive.
        if self._max_targets is not None:
            targets = targets[: self._max_targets]
        target_signature = [list(target) for target in targets]
        cache_path = self._cache_path(query, sub.seed_node_ids, target_signature)
        if cache_path is not None:
            cached = self._load(cache_path, sub.seed_node_ids, target_signature)
            if cached is not None:
                return cached

        selected: list[SerializedDefinition] = []
        for role, node_id, mention in targets:
            selected_evidence = self._select(node_id)
            if selected_evidence is None:
                continue
            evidence, license_info = selected_evidence
            selected.append(
                SerializedDefinition(
                    role=role,
                    node_id=node_id,
                    mention=mention,
                    modality=evidence.modality,
                    text=evidence.text,
                    content_hash=evidence.content_hash,
                    language=evidence.language,
                    source=evidence.source,
                    source_version=evidence.source_version,
                    license_tag=evidence.license_tag,
                    license_id=(license_info.license_id if license_info else ""),
                    license_url=(license_info.license_url if license_info else ""),
                    usage_scope=(
                        license_info.usage_scope.value if license_info else ""
                    ),
                    redistribution_scope=(
                        license_info.redistribution_scope if license_info else ""
                    ),
                    payload_class=(license_info.payload_class if license_info else ""),
                    source_field=(
                        evidence.source_field
                        or (license_info.source_field if license_info else "")
                    ),
                    source_release=(
                        license_info.source_release if license_info else ""
                    ),
                    upstream_source=(
                        evidence.upstream_source
                        or (license_info.upstream_source if license_info else "")
                    ),
                    upstream_release=(
                        evidence.upstream_release
                        or (license_info.upstream_release if license_info else "")
                    ),
                    reviewed_at=(license_info.reviewed_at if license_info else ""),
                    registry_digest=(
                        license_info.registry_digest if license_info else ""
                    ),
                    primehkg_provenance_registry_digest=(
                        evidence.provenance_registry_digest
                    ),
                    source_artifact_sha256=evidence.source_artifact_sha256,
                    retrieved_at=evidence.retrieved_at,
                    source_url=evidence.source_url,
                    attribution_text=evidence.attribution_text,
                    provenance_chain=evidence.provenance_chain,
                    provenance_status=evidence.provenance_status,
                    primehkg_license_id=evidence.license_id,
                    primehkg_license_url=evidence.license_url,
                    primehkg_usage_scope=evidence.usage_scope,
                    primehkg_redistribution_scope=(evidence.redistribution_scope),
                )
            )
        result = SerializedDefinitions(
            text=" ".join(self._format(item) for item in selected),
            items=tuple(selected),
            fingerprint=self.fingerprint,
        )
        if cache_path is not None:
            self._store(cache_path, sub.seed_node_ids, target_signature, result)
        return result

    def _select(self, node_id: str) -> tuple[DefinitionEvidence, Any | None] | None:
        if self._license_registry is None:
            evidence = self._legacy_select(node_id)
            return (evidence, None) if evidence is not None else None
        eligible: list[tuple[DefinitionEvidence, Any | None]] = []
        for row in self._index.candidates(node_id):
            if row.modality not in self._priority:
                continue
            if self._requires_complete_upstream_provenance and (
                row.provenance_status != "complete"
                or not is_concrete_provenance_value(row.upstream_release)
                or not is_concrete_provenance_value(row.retrieved_at)
                or row.usage_scope != "allow"
                or not row.license_id
                or (row.license_tag is not None and row.license_tag != row.license_id)
                or (
                    row.redistribution_scope == "attribution_required"
                    and not row.attribution_text.strip()
                )
            ):
                continue
            license_info = None
            if self._license_registry is not None:
                license_info = self._license_registry.decision_for(
                    source=row.source,
                    modality=row.source_field or row.modality,
                    license_tag=row.license_id or row.license_tag,
                )
                if license_info is None:
                    continue
                if row.provenance_status == "complete" and (
                    row.license_id != license_info.license_id
                    or row.source_version != license_info.source_release
                    or row.redistribution_scope != license_info.redistribution_scope
                    or row.upstream_source != license_info.upstream_source
                    or row.upstream_release != license_info.upstream_release
                ):
                    continue
                if (
                    self._allowed_payload_classes
                    and license_info.payload_class not in self._allowed_payload_classes
                ):
                    continue
            eligible.append((row, license_info))
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda item: (
                item[0].language != "en",
                self._priority[item[0].modality],
                item[1].payload_class if item[1] is not None else "",
                item[0].source,
                item[0].source_version,
                item[0].license_tag or "",
                item[0].content_hash,
                item[0].text,
            ),
        )

    def _name_evidence(self, node_id: str) -> DefinitionEvidence | None:
        """Synthesize evidence carrying the node's KG name (C1n fallback only)."""
        if self._node_name_index is None:
            return None
        name = self._node_name_index.name_for(node_id)
        if not name:
            return None
        return DefinitionEvidence(
            node_id=node_id,
            modality="name",
            text=name,
            content_hash=hashlib.sha256(name.encode()).hexdigest(),
            language="en",
            source="primehkg-core",
            source_field="name",
        )

    def _legacy_select(self, node_id: str) -> DefinitionEvidence | None:
        """Preserve the exact pre-v2 selector for explicit legacy comparisons.

        When ``node_name_index`` is set (the C1n arm), a node with no definition
        falls back to its KG name instead of contributing nothing. A real
        definition always wins: the fallback only runs when nothing is eligible.

        The name comes from the core node table, not from ``node_features_text``.
        The features table's ``short_label`` rows cover ontology entities but NOT
        PrimeKG drug nodes, so filtering it yields zero names for DDI; the core
        table has a name for 100% of nodes, drugs included.
        """
        eligible = [
            row
            for row in self._index.candidates(node_id)
            if row.modality in self._priority
        ]
        if not eligible:
            return self._name_evidence(node_id)
        return min(
            eligible,
            key=lambda row: (
                self._priority[row.modality],
                row.language != "en",
                row.source,
                row.source_version,
                row.license_tag or "",
                row.content_hash,
                row.text,
            ),
        )

    def _targets(self, query: Query, sub: SubHypergraph) -> list[tuple[str, str, str]]:
        actual_schema = sub.provenance.get("linked_mentions_schema")
        if (
            self._required_provenance_schema is not None
            and actual_schema != self._required_provenance_schema
        ):
            msg = (
                "retrieved definitions require linked-mention provenance schema "
                f"{self._required_provenance_schema!r}, got {actual_schema!r}"
            )
            raise ValueError(msg)
        seeds = set(sub.seed_node_ids)
        linked = sub.provenance.get("linked_mentions", [])
        if self._required_provenance_schema == _PROVENANCE_SCHEMA:
            return self._targets_from_role_provenance(linked, seeds)
        raw_groups = query.metadata.get("target_entity_groups")
        if raw_groups is not None:
            groups = {
                str(role): {str(text) for text in mentions}
                for role, mentions in dict(raw_groups).items()
            }
            return self._targets_from_groups(linked, seeds, groups)

        if linked and len(query.entities) == 2 and len(set(query.entities)) == 2:
            groups = {
                role: {str(query.entities[index])}
                for index, role in enumerate(_ROLE_NAMES)
            }
            return self._targets_from_groups(linked, seeds, groups)

        if linked and self._required_provenance_schema is not None:
            msg = (
                "pair roles cannot be inferred safely; configure a pair-aware query "
                "builder that sets metadata.target_entity_groups"
            )
            raise ValueError(msg)

        targets: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for index, record in enumerate(linked):
            node_id = str(record.get("node_id", ""))
            if not node_id or node_id not in seeds or node_id in seen:
                continue
            role = _ROLE_NAMES[index] if index < len(_ROLE_NAMES) else f"seed_{index}"
            targets.append((role, node_id, str(record.get("text", node_id))))
            seen.add(node_id)
        if targets:
            return targets
        return [
            (f"seed_{index}", node_id, node_id)
            for index, node_id in enumerate(sorted(seeds))
        ]

    @staticmethod
    def _targets_from_role_provenance(
        linked: Any, seeds: set[str]
    ) -> list[tuple[str, str, str]]:
        targets: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for record in sorted(
            linked,
            key=lambda record: (
                _ROLE_NAMES.index(str(record.get("pair_role")))
                if str(record.get("pair_role")) in _ROLE_NAMES
                else len(_ROLE_NAMES),
                -float(record.get("score", 0.0)),
                str(record.get("node_id", "")),
            ),
        ):
            role = record.get("pair_role")
            node_id = str(record.get("node_id", ""))
            if role not in _ROLE_NAMES:
                msg = f"role-aware provenance has invalid pair_role: {role!r}"
                raise ValueError(msg)
            role_node = (str(role), node_id)
            if not node_id or node_id not in seeds or role_node in seen:
                continue
            targets.append((str(role), node_id, str(record.get("text", node_id))))
            seen.add(role_node)
        return targets

    @staticmethod
    def _targets_from_groups(
        linked: Any,
        seeds: set[str],
        groups: dict[str, set[str]],
    ) -> list[tuple[str, str, str]]:
        matched: dict[str, list[dict[str, Any]]] = {role: [] for role in _ROLE_NAMES}
        for record in linked:
            node_id = str(record.get("node_id", ""))
            mention = str(record.get("text", ""))
            if not node_id or node_id not in seeds:
                continue
            roles = [role for role in _ROLE_NAMES if mention in groups.get(role, set())]
            if len(roles) > 1:
                msg = f"linked mention {mention!r} is ambiguous between target roles"
                raise ValueError(msg)
            if roles:
                matched[roles[0]].append(dict(record))

        targets: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for role in _ROLE_NAMES:
            records = sorted(
                matched[role],
                key=lambda record: (
                    -float(record.get("score", 0.0)),
                    str(record.get("node_id", "")),
                    str(record.get("text", "")),
                ),
            )
            for record in records:
                node_id = str(record["node_id"])
                if node_id in seen:
                    continue
                targets.append((role, node_id, str(record["text"])))
                seen.add(node_id)
        return targets

    @staticmethod
    def _format(item: SerializedDefinition) -> str:
        label = item.role.upper()
        return f"[{label}_DEFINITION] {item.mention}: {item.text} [/{label}_DEFINITION]"

    def _cache_path(
        self,
        query: Query,
        seed_node_ids: tuple[str, ...],
        target_signature: list[list[str]],
    ) -> Path | None:
        if self._cache_root is None:
            return None
        metadata = query.metadata
        if not all(name in metadata for name in ("dataset", "split", "id")):
            return None
        dataset = self._safe_cache_component(metadata["dataset"], field="dataset")
        split = self._safe_cache_component(metadata["split"], field="split")
        instance = "\0".join((dataset, split, str(metadata["id"])))
        instance_digest = hashlib.sha256(instance.encode()).hexdigest()
        target_identity = json.dumps(
            {
                "seed_node_ids": list(seed_node_ids),
                "target_signature": target_signature,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        target_digest = hashlib.sha256(target_identity.encode()).hexdigest()
        path = (
            self._cache_root
            / self.fingerprint
            / dataset
            / split
            / f"{instance_digest}.{target_digest}.json"
        )
        self._assert_cache_containment(path)
        return path

    @staticmethod
    def _safe_cache_component(value: object, *, field: str) -> str:
        component = str(value)
        if (
            not component
            or component in {".", ".."}
            or Path(component).is_absolute()
            or len(Path(component).parts) != 1
            or "/" in component
            or "\\" in component
            or "\0" in component
        ):
            msg = (
                f"cache metadata {field!r} must be a non-empty single path "
                f"component, got {component!r}"
            )
            raise ValueError(msg)
        return component

    def _assert_cache_containment(self, path: Path) -> None:
        if self._cache_root is None:
            return
        root = self._cache_root.resolve(strict=False)
        candidate = path.resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            msg = f"raw text cache path resolves outside cache root: {path}"
            raise ValueError(msg) from exc

    def _load(
        self,
        path: Path,
        seed_node_ids: tuple[str, ...],
        target_signature: list[list[str]],
    ) -> SerializedDefinitions | None:
        try:
            if self._raw_text_cache_mode == "restricted":
                parent_fd = self._open_restricted_cache_parent(path, create=False)
                if parent_fd is None:
                    return None
                try:
                    self._assert_cache_containment(path)
                    payload = self._read_restricted_cache_file(
                        path,
                        parent_fd=parent_fd,
                    )
                finally:
                    os.close(parent_fd)
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["fingerprint"] != self.fingerprint:
                return None
            if payload["seed_node_ids"] != list(seed_node_ids):
                return None
            if payload["target_signature"] != target_signature:
                return None
            items = tuple(SerializedDefinition(**item) for item in payload["items"])
            return SerializedDefinitions(
                text=str(payload["text"]),
                items=items,
                fingerprint=self.fingerprint,
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None

    def _store(
        self,
        path: Path,
        seed_node_ids: tuple[str, ...],
        target_signature: list[list[str]],
        result: SerializedDefinitions,
    ) -> None:
        payload: dict[str, Any] = {
            "fingerprint": self.fingerprint,
            "seed_node_ids": list(seed_node_ids),
            "target_signature": target_signature,
            "text": result.text,
            "items": [asdict(item) for item in result.items],
        }
        if self._raw_text_cache_mode == "restricted":
            if self._cache_root is None:
                msg = "restricted raw text cache requires a cache root"
                raise ValueError(msg)
            self._cache_root.mkdir(parents=True, exist_ok=True)
            parent_fd = self._open_restricted_cache_parent(path, create=True)
            if parent_fd is None:
                msg = (
                    f"could not create restricted raw text cache parent: {path.parent}"
                )
                raise OSError(msg)
            try:
                self._assert_cache_containment(path)
                self._write_restricted_cache_file(
                    path,
                    payload,
                    parent_fd=parent_fd,
                )
            finally:
                os.close(parent_fd)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _open_restricted_cache_parent(
        self,
        path: Path,
        *,
        create: bool,
    ) -> int | None:
        """Return an fd for the restricted parent without following symlinks.

        Threat model: ``cache_root`` is a caller-controlled, potentially shared
        trust boundary and is neither chmodded nor rejected when it is itself a
        symlink. Every serializer-owned component below it is untrusted: path
        traversal, symlink substitution, and non-directory ancestors fail
        closed. ``dir_fd`` traversal with ``O_NOFOLLOW`` narrows TOCTOU windows
        by retaining each verified parent while opening the next component.
        The returned split-directory descriptor remains open across file I/O,
        so a concurrent path substitution cannot redirect the read or replace.
        """
        if self._cache_root is None:
            return None
        relative_parent = path.parent.relative_to(self._cache_root)
        flags = os.O_RDONLY | os.O_DIRECTORY
        root_fd = os.open(self._cache_root, flags)
        parent_fd = root_fd
        try:
            for component in relative_parent.parts:
                if create:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=parent_fd)
                    except FileExistsError:
                        pass
                try:
                    child_fd = os.open(
                        component,
                        flags | os.O_NOFOLLOW,
                        dir_fd=parent_fd,
                    )
                except FileNotFoundError:
                    if create:
                        raise
                    return None
                except OSError as exc:
                    msg = (
                        "raw text cache ancestor must be a real directory, "
                        f"not a symlink or special file: {component!r}"
                    )
                    raise ValueError(msg) from exc
                if parent_fd != root_fd:
                    os.close(parent_fd)
                parent_fd = child_fd
                os.fchmod(parent_fd, 0o700)
            if parent_fd == root_fd:
                return os.dup(root_fd)
            result_fd = parent_fd
            parent_fd = root_fd
            return result_fd
        finally:
            if parent_fd != root_fd:
                os.close(parent_fd)
            os.close(root_fd)

    @staticmethod
    def _reject_unsafe_cache_file(path: Path, *, parent_fd: int) -> None:
        try:
            mode = os.stat(
                path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            ).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            msg = (
                "raw text cache entry must be a regular file; symlinks and "
                f"special files are rejected: {path}"
            )
            raise ValueError(msg)

    def _read_restricted_cache_file(
        self,
        path: Path,
        *,
        parent_fd: int,
    ) -> dict[str, Any]:
        self._reject_unsafe_cache_file(path, parent_fd=parent_fd)
        try:
            fd = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            mode = os.stat(
                path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            ).st_mode
            if stat.S_ISLNK(mode):
                msg = f"raw text cache entry must not be a symlink: {path}"
                raise ValueError(msg) from exc
            raise
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                msg = f"raw text cache entry must be a regular file: {path}"
                raise ValueError(msg)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, encoding="utf-8") as stream:
                fd = -1
                return json.load(stream)
        finally:
            if fd >= 0:
                os.close(fd)

    def _write_restricted_cache_file(
        self,
        path: Path,
        payload: dict[str, Any],
        *,
        parent_fd: int,
    ) -> None:
        self._reject_unsafe_cache_file(path, parent_fd=parent_fd)
        tmp_name = f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
        fd = -1
        try:
            fd = os.open(
                tmp_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                fd = -1
                json.dump(payload, stream, sort_keys=True)
            self._reject_unsafe_cache_file(path, parent_fd=parent_fd)
            os.replace(
                tmp_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(tmp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
