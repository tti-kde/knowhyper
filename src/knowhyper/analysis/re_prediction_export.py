"""Payload-free per-instance prediction artifacts for relation extraction."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
LEAKAGE_OVERLAP_TYPES = frozenset(
    {
        "model_input_exact",
        "model_input_entity_masked_exact",
        "shared_16gram",
    }
)
PAIR_DISJOINT_OVERLAP_TYPES = frozenset({"pair_unordered"})
_IDENTITY_PATTERN = re.compile(r"^[A-Za-z0-9_.:+/|-]{1,128}$")
_INSTANCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:#/+|*-]{1,256}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_POPULATIONS = ("official", "leakage_cleared", "pair_disjoint")
_METRIC_FIELDS = frozenset({"precision", "recall", "f1", "tp", "fp", "fn", "n"})


@dataclass(frozen=True)
class PredictionRecord:
    """The only fields permitted in a prediction JSONL row."""

    dataset: str
    split: str
    instance_id: str
    gold_label_id: int
    gold_label: str
    predicted_label_id: int
    predicted_label: str
    predicted_score: float
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate fields used by downstream pairing and metric code."""
        if self.schema_version != SCHEMA_VERSION:
            msg = f"unsupported prediction schema version: {self.schema_version}"
            raise ValueError(msg)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (self.gold_label_id, self.predicted_label_id)
        ):
            msg = "gold and predicted label IDs must be non-negative integers"
            raise ValueError(msg)
        if not _IDENTITY_PATTERN.fullmatch(
            self.dataset
        ) or not _IDENTITY_PATTERN.fullmatch(self.split):
            msg = "dataset and split must be payload-free identifiers"
            raise ValueError(msg)
        if not _INSTANCE_ID_PATTERN.fullmatch(self.instance_id):
            msg = "instance_id must be a payload-free identifier"
            raise ValueError(msg)
        if not _IDENTITY_PATTERN.fullmatch(
            self.gold_label
        ) or not _IDENTITY_PATTERN.fullmatch(self.predicted_label):
            msg = "gold and predicted labels must be payload-free identifiers"
            raise ValueError(msg)
        if not math.isfinite(self.predicted_score) or not (
            0.0 <= self.predicted_score <= 1.0
        ):
            msg = "predicted_score must be finite and in [0, 1]"
            raise ValueError(msg)


@dataclass(frozen=True)
class ExclusionMasks:
    """Test instance IDs excluded by each preregistered sensitivity analysis."""

    leakage: frozenset[str] = frozenset()
    pair_disjoint: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PredictionBundleManifest:
    """Allowlisted provenance and metric metadata for one prediction artifact."""

    dataset: str
    split: str
    experiment: str
    condition: str
    seed: int
    checkpoint_sha256: str
    resolved_config_sha256: str
    extra_overrides_sha256: str
    negative_label: str
    num_workers: int
    batch_size: int
    overlap_instances_sha256: str | None
    applied_exclusion_counts: dict[str, int]
    metrics: dict[str, dict[str, float | int]]
    ieforge_official_metrics: dict[str, float]

    def __post_init__(self) -> None:
        """Reject arbitrary text and malformed identity hashes."""
        for name, value in (
            ("dataset", self.dataset),
            ("split", self.split),
            ("experiment", self.experiment),
            ("condition", self.condition),
            ("negative_label", self.negative_label),
        ):
            if not _IDENTITY_PATTERN.fullmatch(value):
                msg = f"invalid manifest identity field {name}: {value!r}"
                raise ValueError(msg)
        for name, digest in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("resolved_config_sha256", self.resolved_config_sha256),
            ("extra_overrides_sha256", self.extra_overrides_sha256),
        ):
            if not _SHA256_PATTERN.fullmatch(digest):
                msg = f"invalid manifest SHA-256 field {name}"
                raise ValueError(msg)
        if self.overlap_instances_sha256 is not None and not _SHA256_PATTERN.fullmatch(
            self.overlap_instances_sha256
        ):
            msg = "invalid manifest SHA-256 field overlap_instances_sha256"
            raise ValueError(msg)
        if self.num_workers < 4:
            msg = "manifest num_workers must be >=4"
            raise ValueError(msg)
        if self.batch_size < 1:
            msg = "manifest batch_size must be positive"
            raise ValueError(msg)
        if set(self.applied_exclusion_counts) != {
            "leakage_cleared",
            "pair_disjoint",
        } or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in self.applied_exclusion_counts.values()
        ):
            msg = "manifest applied_exclusion_counts has invalid fields or values"
            raise ValueError(msg)
        if set(self.metrics) != set(_POPULATIONS):
            msg = "manifest metrics must contain all preregistered populations"
            raise ValueError(msg)
        for population, values in self.metrics.items():
            if set(values) != _METRIC_FIELDS:
                msg = f"manifest metrics has invalid payload for {population}"
                raise ValueError(msg)
            if any(
                not isinstance(values[field], int)
                or isinstance(values[field], bool)
                or values[field] < 0
                for field in ("tp", "fp", "fn", "n")
            ) or any(
                not isinstance(values[field], int | float)
                or isinstance(values[field], bool)
                or not math.isfinite(float(values[field]))
                or not 0.0 <= float(values[field]) <= 1.0
                for field in ("precision", "recall", "f1")
            ):
                msg = f"manifest metrics has invalid values for {population}"
                raise ValueError(msg)
        if set(self.ieforge_official_metrics) != {
            "precision",
            "recall",
            "f1",
        } or not all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
            for value in self.ieforge_official_metrics.values()
        ):
            msg = "manifest ieforge_official_metrics has invalid fields or values"
            raise ValueError(msg)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with a stable canonical encoding."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_overlap_masks(path: Path) -> ExclusionMasks:
    """Load test-side preregistration masks from a leakage-audit JSONL.

    Only train/validation-to-test rows are accepted. Duplicate IDs and duplicate
    overlap groups collapse into a set by construction.
    """
    leakage: set[str] = set()
    pair_disjoint: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            overlap_type = str(row.get("overlap_type", ""))
            target = (
                leakage
                if overlap_type in LEAKAGE_OVERLAP_TYPES
                else pair_disjoint
                if overlap_type in PAIR_DISJOINT_OVERLAP_TYPES
                else None
            )
            if target is None:
                continue
            ids = _test_ids_from_overlap_row(row, line_number=line_number)
            target.update(ids)
    return ExclusionMasks(
        leakage=frozenset(leakage),
        pair_disjoint=frozenset(pair_disjoint),
    )


def _test_ids_from_overlap_row(
    row: Mapping[str, Any], *, line_number: int
) -> Sequence[str]:
    left_split = str(row.get("left_split", ""))
    right_split = str(row.get("right_split", ""))
    if left_split == "test" and right_split in {"train", "validation"}:
        raw_ids = row.get("left_instance_ids")
    elif right_split == "test" and left_split in {"train", "validation"}:
        raw_ids = row.get("right_instance_ids")
    else:
        return ()
    if not isinstance(raw_ids, list) or not all(
        isinstance(item, str) and item for item in raw_ids
    ):
        msg = f"invalid test instance IDs at overlap JSONL line {line_number}"
        raise ValueError(msg)
    return raw_ids


def validate_prediction_population(
    left: Sequence[PredictionRecord],
    right: Sequence[PredictionRecord],
) -> None:
    """Fail closed unless paired conditions have identical ordered populations."""
    if len(left) != len(right):
        msg = f"paired prediction length mismatch: {len(left)} != {len(right)}"
        raise ValueError(msg)
    left_keys = [_population_key(record) for record in left]
    right_keys = [_population_key(record) for record in right]
    if left_keys != right_keys:
        mismatch = next(
            (
                index
                for index, (left_key, right_key) in enumerate(
                    zip(left_keys, right_keys, strict=True)
                )
                if left_key != right_key
            ),
            None,
        )
        msg = f"paired prediction population/order mismatch at row {mismatch}"
        raise ValueError(msg)


def validate_bundle_manifests(
    left: PredictionBundleManifest,
    right: PredictionBundleManifest,
) -> None:
    """Require paired bundles to share all population-defining provenance."""
    for field in (
        "dataset",
        "split",
        "seed",
        "negative_label",
        "overlap_instances_sha256",
        "applied_exclusion_counts",
    ):
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value != right_value:
            msg = (
                f"paired bundle manifest mismatch for {field}: "
                f"{left_value!r} != {right_value!r}"
            )
            raise ValueError(msg)


def _population_key(record: PredictionRecord) -> tuple[str, str, str, int, str]:
    return (
        record.dataset,
        record.split,
        record.instance_id,
        record.gold_label_id,
        record.gold_label,
    )


def validate_unique_instance_ids(records: Sequence[PredictionRecord]) -> None:
    """Reject ambiguous artifacts rather than silently overwriting duplicate IDs."""
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = (record.dataset, record.split, record.instance_id)
        if key in seen:
            msg = f"duplicate prediction instance ID: {key!r}"
            raise ValueError(msg)
        seen.add(key)


def validate_masks_for_population(
    records: Sequence[PredictionRecord], masks: ExclusionMasks
) -> None:
    """Reject masks containing IDs outside the current prediction population."""
    population = {record.instance_id for record in records}
    unknown_leakage = masks.leakage - population
    unknown_pairs = masks.pair_disjoint - population
    if unknown_leakage or unknown_pairs:
        msg = (
            "overlap mask contains IDs outside prediction population: "
            f"leakage={len(unknown_leakage)} pair_disjoint={len(unknown_pairs)}"
        )
        raise ValueError(msg)


def compute_micro_f1(
    records: Iterable[PredictionRecord], *, negative_label: str
) -> dict[str, float | int]:
    """Match ieforge's exact-label micro F1 with the negative class excluded."""
    tp = fp = fn = count = 0
    for record in records:
        count += 1
        gold_positive = record.gold_label != negative_label
        pred_positive = record.predicted_label != negative_label
        exact = record.predicted_label == record.gold_label
        if pred_positive and gold_positive and exact:
            tp += 1
        elif pred_positive:
            fp += 1
        if gold_positive and not exact:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n": count,
    }


def compute_sensitivity_metrics(
    records: Sequence[PredictionRecord],
    *,
    negative_label: str,
    masks: ExclusionMasks,
) -> dict[str, dict[str, float | int]]:
    """Compute official and preregistered sensitivity populations."""
    validate_unique_instance_ids(records)
    validate_masks_for_population(records, masks)
    return {
        "official": compute_micro_f1(records, negative_label=negative_label),
        "leakage_cleared": compute_micro_f1(
            (record for record in records if record.instance_id not in masks.leakage),
            negative_label=negative_label,
        ),
        "pair_disjoint": compute_micro_f1(
            (
                record
                for record in records
                if record.instance_id not in masks.pair_disjoint
            ),
            negative_label=negative_label,
        ),
    }


def compute_paired_sensitivity_metrics(
    left: Sequence[PredictionRecord],
    right: Sequence[PredictionRecord],
    *,
    negative_label: str,
    masks: ExclusionMasks,
) -> dict[str, Any]:
    """Compute matched metrics and deltas after a fail-closed population check."""
    validate_prediction_population(left, right)
    left_metrics = compute_sensitivity_metrics(
        left, negative_label=negative_label, masks=masks
    )
    right_metrics = compute_sensitivity_metrics(
        right, negative_label=negative_label, masks=masks
    )
    return {
        "left": left_metrics,
        "right": right_metrics,
        "delta_f1": {
            population: float(right_metrics[population]["f1"])
            - float(left_metrics[population]["f1"])
            for population in _POPULATIONS
        },
    }


def write_prediction_bundle(
    output_dir: Path,
    records: Sequence[PredictionRecord],
    manifest: PredictionBundleManifest,
) -> None:
    """Atomically write a restrictive, payload-free prediction bundle."""
    validate_unique_instance_ids(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    predictions_path = output_dir / "predictions.jsonl"
    manifest_path = output_dir / "manifest.json"
    _atomic_write(
        predictions_path,
        "".join(
            json.dumps(asdict(record), sort_keys=True) + "\n" for record in records
        ),
    )
    _atomic_write(
        manifest_path,
        json.dumps(
            {
                **asdict(manifest),
                "schema_version": SCHEMA_VERSION,
                "prediction_rows": len(records),
                "raw_text_written_to_artifact": False,
                "prediction_jsonl_sha256": sha256_file(predictions_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def read_predictions(path: Path) -> list[PredictionRecord]:
    """Read and validate a prediction JSONL produced by this module."""
    allowed = set(PredictionRecord.__dataclass_fields__)
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            unexpected = set(row) - allowed
            if unexpected:
                msg = (
                    f"unexpected prediction fields at line {line_number}: "
                    f"{sorted(unexpected)}"
                )
                raise ValueError(msg)
            records.append(PredictionRecord(**row))
    validate_unique_instance_ids(records)
    return records


def read_prediction_bundle(
    output_dir: Path,
) -> tuple[list[PredictionRecord], PredictionBundleManifest]:
    """Read a bundle and verify its allowlisted manifest and content digest."""
    predictions_path = output_dir / "predictions.jsonl"
    manifest_path = output_dir / "manifest.json"
    records = read_predictions(predictions_path)
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_fields = set(PredictionBundleManifest.__dataclass_fields__)
    generated_fields = {
        "schema_version",
        "prediction_rows",
        "raw_text_written_to_artifact",
        "prediction_jsonl_sha256",
    }
    unexpected = set(raw_manifest) - manifest_fields - generated_fields
    missing = manifest_fields - set(raw_manifest)
    if unexpected or missing:
        msg = (
            "invalid prediction manifest fields: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
        raise ValueError(msg)
    if raw_manifest.get("schema_version") != SCHEMA_VERSION:
        msg = "prediction manifest schema version mismatch"
        raise ValueError(msg)
    if raw_manifest.get("raw_text_written_to_artifact") is not False:
        msg = "prediction manifest does not assert payload-free output"
        raise ValueError(msg)
    if raw_manifest.get("prediction_rows") != len(records):
        msg = "prediction manifest row count mismatch"
        raise ValueError(msg)
    if raw_manifest.get("prediction_jsonl_sha256") != sha256_file(predictions_path):
        msg = "prediction JSONL digest mismatch"
        raise ValueError(msg)
    manifest = PredictionBundleManifest(
        **{field: raw_manifest[field] for field in manifest_fields}
    )
    if records and (
        any(record.dataset != manifest.dataset for record in records)
        or any(record.split != manifest.split for record in records)
    ):
        msg = "prediction rows do not match manifest dataset/split"
        raise ValueError(msg)
    observed_labels = {
        label
        for record in records
        for label in (record.gold_label, record.predicted_label)
    }
    if records and manifest.negative_label not in observed_labels:
        msg = "manifest negative label is absent from prediction label space"
        raise ValueError(msg)
    recomputed_official = compute_micro_f1(
        records, negative_label=manifest.negative_label
    )
    _require_metric_match(
        manifest.metrics["official"],
        recomputed_official,
        context="manifest official metric",
    )
    _require_metric_match(
        manifest.ieforge_official_metrics,
        recomputed_official,
        context="ieforge official metric",
        fields=("precision", "recall", "f1"),
    )
    official_n = int(recomputed_official["n"])
    for population, count_key in (
        ("leakage_cleared", "leakage_cleared"),
        ("pair_disjoint", "pair_disjoint"),
    ):
        sensitivity_n = int(manifest.metrics[population]["n"])
        applied_count = manifest.applied_exclusion_counts[count_key]
        if applied_count != official_n - sensitivity_n:
            msg = f"manifest exclusion count mismatch for {population}"
            raise ValueError(msg)
        _require_metric_self_consistency(
            manifest.metrics[population], context=population
        )
    return records, manifest


def _require_metric_match(
    actual: Mapping[str, float | int],
    expected: Mapping[str, float | int],
    *,
    context: str,
    fields: Sequence[str] = ("precision", "recall", "f1", "tp", "fp", "fn", "n"),
) -> None:
    for field in fields:
        left = actual[field]
        right = expected[field]
        if isinstance(left, float) or isinstance(right, float):
            matches = math.isclose(
                float(left), float(right), abs_tol=1e-12, rel_tol=0.0
            )
        else:
            matches = left == right
        if not matches:
            msg = f"{context} mismatch for {field}: {left!r} != {right!r}"
            raise ValueError(msg)


def _require_metric_self_consistency(
    metric: Mapping[str, float | int], *, context: str
) -> None:
    tp = int(metric["tp"])
    fp = int(metric["fp"])
    fn = int(metric["fn"])
    n = int(metric["n"])
    if min(tp, fp, fn, n) < 0 or max(tp, fp, fn) > n:
        msg = f"{context} has invalid metric counts"
        raise ValueError(msg)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    _require_metric_match(
        metric,
        {"precision": precision, "recall": recall, "f1": f1},
        context=context,
        fields=("precision", "recall", "f1"),
    )


def write_json_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one restrictive JSON artifact with atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        temporary_path.chmod(0o600)
        temporary_path.replace(path)
        path.chmod(0o600)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
