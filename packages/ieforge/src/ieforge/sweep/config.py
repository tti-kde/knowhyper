"""Sweep configuration schema, loading, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from omegaconf import OmegaConf

from ieforge import CONFIGS_DIR

_SWEEP_DIR = CONFIGS_DIR / "sweep"
_VALID_METHODS = frozenset({"bayes", "grid", "random"})


@dataclass(frozen=True)
class SweepConfig:
    """Validated sweep definition loaded from configs/sweep/*.yaml."""

    name: str
    method: str
    base_overrides: list[str]
    parameters: dict[str, dict]
    early_terminate: dict | None = None


def parse_sweep_config(raw: dict[str, Any]) -> SweepConfig:
    """Validate a raw sweep mapping and return a SweepConfig.

    Raises ValueError if method is unknown, or parameters / base_overrides empty.
    """
    method = raw.get("method")
    if method not in _VALID_METHODS:
        msg = f"sweep method must be one of {sorted(_VALID_METHODS)}, got {method!r}"
        raise ValueError(msg)

    parameters = raw.get("parameters") or {}
    if not isinstance(parameters, dict) or not parameters:
        msg = "sweep parameters must be a non-empty mapping"
        raise ValueError(msg)

    base_overrides = raw.get("base_overrides") or []
    if not isinstance(base_overrides, list) or not base_overrides:
        msg = "sweep base_overrides must be a non-empty list (need experiment/dataset)"
        raise ValueError(msg)

    return SweepConfig(
        name=str(raw.get("name", "")),
        method=method,
        base_overrides=list(base_overrides),
        parameters=dict(parameters),
        early_terminate=raw.get("early_terminate"),
    )


def load_sweep_config(name: str) -> SweepConfig:
    """Load and validate configs/sweep/<name>.yaml into a SweepConfig."""
    path = _SWEEP_DIR / f"{name}.yaml"
    if not path.is_file():
        msg = f"sweep config not found: {path}"
        raise FileNotFoundError(msg)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(raw, dict)
    return parse_sweep_config(cast("dict[str, Any]", raw))
