from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class NodeTypeCoveragePolicy:
    """Coverage policy for one PrimeKG node type."""

    node_type: str
    phase: int
    primary_group: tuple[str, ...]
    stretch_metrics: tuple[str, ...] = ()
    pass_threshold: float = 0.95


@dataclass(frozen=True, slots=True)
class ModalityCoveragePolicy:
    """Packaged modality coverage matrix for goal A KPI."""

    version: int
    pass_threshold: float
    by_node_type: dict[str, NodeTypeCoveragePolicy]


def load_modality_coverage_bytes() -> bytes:
    return (
        resources.files("primehkg.policy")
        .joinpath("modality_coverage.yaml")
        .read_bytes()
    )


def modality_coverage_fingerprint() -> str:
    return hashlib.sha256(load_modality_coverage_bytes()).hexdigest()


def _coerce_str_tuple(value: Any, *, field: str, node_type: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        msg = f"modality coverage: {node_type!r} {field} must be a list"
        raise TypeError(msg)
    out: list[str] = []
    for item in value:
        s = str(item).strip()
        if not s:
            msg = f"modality coverage: {node_type!r} {field} entries must be non-empty"
            raise ValueError(msg)
        out.append(s)
    return tuple(out)


def load_modality_coverage(path: Path | None = None) -> ModalityCoveragePolicy:
    raw = load_modality_coverage_bytes() if path is None else path.read_bytes()
    obj: Any = yaml.safe_load(raw)
    if not isinstance(obj, dict):
        msg = "modality coverage: top-level must be a mapping"
        raise TypeError(msg)

    version = obj.get("version")
    if not isinstance(version, int) or version < 1:
        msg = "modality coverage: version must be an integer >= 1"
        raise ValueError(msg)

    global_threshold = obj.get("pass_threshold")
    if not isinstance(global_threshold, (int, float)):
        msg = "modality coverage: pass_threshold must be numeric"
        raise TypeError(msg)
    global_threshold_f = float(global_threshold)
    if not 0.0 < global_threshold_f <= 1.0:
        msg = "modality coverage: pass_threshold must be in (0, 1]"
        raise ValueError(msg)

    rows = obj.get("node_types")
    if not isinstance(rows, list):
        msg = "modality coverage: node_types must be a list"
        raise TypeError(msg)

    by_node_type: dict[str, NodeTypeCoveragePolicy] = {}
    for row in rows:
        if not isinstance(row, dict):
            msg = "modality coverage: each node_types entry must be a mapping"
            raise TypeError(msg)

        node_type = str(row.get("node_type", "")).strip()
        if not node_type:
            msg = "modality coverage: node_type must be non-empty"
            raise ValueError(msg)

        phase = row.get("phase")
        if not isinstance(phase, int) or phase < 1:
            msg = f"modality coverage: {node_type!r} phase must be an integer >= 1"
            raise ValueError(msg)

        primary_group = _coerce_str_tuple(
            row.get("primary_group"), field="primary_group", node_type=node_type
        )
        if not primary_group:
            msg = f"modality coverage: {node_type!r} primary_group must be non-empty"
            raise ValueError(msg)

        stretch_raw = row.get("stretch_metrics", [])
        stretch_metrics = _coerce_str_tuple(
            stretch_raw, field="stretch_metrics", node_type=node_type
        )

        row_threshold = row.get("pass_threshold")
        if row_threshold is None:
            pass_threshold = global_threshold_f
        else:
            if not isinstance(row_threshold, (int, float)):
                msg = f"modality coverage: {node_type!r} pass_threshold must be numeric"
                raise TypeError(msg)
            pass_threshold = float(row_threshold)
            if not 0.0 < pass_threshold <= 1.0:
                msg = (
                    f"modality coverage: {node_type!r} pass_threshold must be in (0, 1]"
                )
                raise ValueError(msg)

        if node_type in by_node_type:
            msg = f"modality coverage: duplicate node_type key: {node_type!r}"
            raise ValueError(msg)

        by_node_type[node_type] = NodeTypeCoveragePolicy(
            node_type=node_type,
            phase=phase,
            primary_group=primary_group,
            stretch_metrics=stretch_metrics,
            pass_threshold=pass_threshold,
        )

    if not by_node_type:
        msg = "modality coverage: node_types must contain at least one entry"
        raise ValueError(msg)

    return ModalityCoveragePolicy(
        version=version,
        pass_threshold=global_threshold_f,
        by_node_type=by_node_type,
    )


def p1_node_types(policy: ModalityCoveragePolicy | None = None) -> tuple[str, ...]:
    cov = policy if policy is not None else load_modality_coverage()
    return tuple(sorted(nt for nt, ent in cov.by_node_type.items() if ent.phase == 1))


def node_types_for_profile(
    profile: str,
    policy: ModalityCoveragePolicy | None = None,
) -> tuple[str, ...]:
    cov = policy if policy is not None else load_modality_coverage()
    key = profile.strip().lower()
    if key == "full-v1":
        phases = {1, 2, 3}
    elif key == "full-v1-p1":
        phases = {1}
    else:
        phases = {1}
    return tuple(
        sorted(nt for nt, ent in cov.by_node_type.items() if ent.phase in phases)
    )
