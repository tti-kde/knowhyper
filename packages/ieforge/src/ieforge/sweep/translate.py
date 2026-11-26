"""Pure conversions between wandb.config and Hydra override strings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def _render_value(value: Any) -> str:
    """Render a Python value as a Hydra override RHS string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        inner = ",".join(_render_value(v) for v in value)
        return f"[{inner}]"
    return str(value)


def wandb_config_to_overrides(
    wandb_config: Mapping[str, Any],
    allowed_keys: Iterable[str],
) -> list[str]:
    """Convert a wandb.config mapping to Hydra override strings.

    Only keys present in ``allowed_keys`` are emitted; any extra keys wandb adds
    (such as ``_wandb``) are dropped so they never reach Hydra.
    """
    allowed = set(allowed_keys)
    overrides: list[str] = []
    for key, value in wandb_config.items():
        if key not in allowed:
            continue
        overrides.append(f"{key}={_render_value(value)}")
    return overrides


def parameters_to_wandb(parameters: dict[str, dict]) -> dict[str, dict]:
    """Return wandb sweep ``parameters`` for our sweep parameters.

    Our sweep config already uses wandb's parameter spec verbatim, so this is an
    identity passthrough; it exists to keep the create-side conversion explicit
    and symmetric with wandb_config_to_overrides on the run side.
    """
    return parameters
