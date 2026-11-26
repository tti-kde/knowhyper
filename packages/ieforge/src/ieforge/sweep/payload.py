"""Build a wandb sweep API payload from a SweepConfig."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ieforge import CONFIGS_DIR
from ieforge.sweep.translate import parameters_to_wandb

if TYPE_CHECKING:
    from ieforge.sweep.config import SweepConfig

_CONFIGS_DIR = str(CONFIGS_DIR)
_MODE_TO_GOAL = {"max": "maximize", "min": "minimize"}

# ``${args_no_hyphens}`` is a wandb macro the agent expands into the sampled
# hyperparameters as `key=value` tokens, which Hydra consumes as overrides.
_WANDB_ARGS_MACRO = "${args_no_hyphens}"


def _trial_command(sweep_name: str) -> list[str]:
    """Build the per-trial command the wandb agent runs for ``sweep_name``.

    The interpreter is pinned to the absolute path of the Python running
    ``sweep create`` (``sys.executable``) rather than wandb's ``${interpreter}``
    macro, which hardcodes a bare ``python`` that need not be on the agent's PATH
    (e.g. uv-managed venvs expose only ``python3``/the venv binary, so a bare
    ``python`` exits 127 and every trial dies before training).
    """
    return [
        sys.executable,
        "-m",
        "ieforge.cli.main",
        "sweep",
        "run",
        f"sweep={sweep_name}",
        _WANDB_ARGS_MACRO,
    ]


def resolve_monitor(base_overrides: list[str]) -> tuple[str, str]:
    """Resolve the optimization target from the experiment's metric config.

    Composes the train config with ``base_overrides`` and reads
    ``cfg.metric.monitor`` / ``cfg.metric.monitor_mode``.
    """
    with initialize_config_dir(config_dir=_CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="train", overrides=list(base_overrides))
    monitor = OmegaConf.select(cfg, "metric.monitor")
    mode = OmegaConf.select(cfg, "metric.monitor_mode")
    if monitor is None or mode is None:
        msg = (
            "could not resolve metric.monitor/monitor_mode from base_overrides "
            f"{base_overrides!r}; check the experiment/metric config"
        )
        raise ValueError(msg)
    goal = _MODE_TO_GOAL.get(str(mode).strip().lower())
    if goal is None:
        msg = f"unsupported monitor_mode {mode!r}; expected 'max' or 'min'"
        raise ValueError(msg)
    return str(monitor), goal


def build_sweep_payload(sweep: SweepConfig) -> dict:
    """Construct the dict passed to wandb.sweep()."""
    name, goal = resolve_monitor(sweep.base_overrides)
    payload: dict = {
        "name": sweep.name,
        # wandb's classic agent reads command["program"] unconditionally even
        # though our command template does not reference ${program}; it must be
        # present or the agent raises KeyError before launching any trial.
        "program": "ieforge",
        "method": sweep.method,
        "metric": {"name": name, "goal": goal},
        "parameters": parameters_to_wandb(sweep.parameters),
        "command": _trial_command(sweep.name),
    }
    if sweep.early_terminate is not None:
        payload["early_terminate"] = sweep.early_terminate
    return payload
