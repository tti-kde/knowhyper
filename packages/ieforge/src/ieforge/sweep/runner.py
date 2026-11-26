"""wandb sweep orchestration: create sweeps and run trial agents."""

from __future__ import annotations

import logging
import os
import subprocess
import sys

from ieforge.sweep.config import load_sweep_config
from ieforge.sweep.payload import build_sweep_payload

log = logging.getLogger(__name__)


def _require_wandb():
    """Import wandb lazily and fail with an actionable message if unavailable."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - exercised via message only
        msg = "wandb is required for sweeps; install it and run `wandb login`."
        raise RuntimeError(msg) from exc
    return wandb


def _resolve_entity(wandb, entity: str | None) -> str | None:  # noqa: ANN001
    """Fall back to the logged-in user's default wandb entity when unset.

    ``wandb.agent`` cannot resolve a bare sweep id without an entity (it raises
    "entityName required for project query"), and ``wandb.sweep`` is clearer when
    the entity is explicit, so resolve the account default here when the caller
    did not pass ``--entity``.
    """
    if entity is not None:
        return entity
    try:
        return wandb.Api().default_entity
    except Exception:  # default entity lookup is best-effort
        return None


def create_sweep(name: str, *, project: str, entity: str | None) -> str:
    """Create a wandb sweep from configs/sweep/<name>.yaml and return its id."""
    wandb = _require_wandb()
    sweep = load_sweep_config(name)
    payload = build_sweep_payload(sweep)
    return wandb.sweep(payload, project=project, entity=_resolve_entity(wandb, entity))


def run_agent(
    sweep_id: str,
    *,
    count: int | None,
    project: str,
    entity: str | None,
) -> None:
    """Start a wandb agent that runs trials for the given sweep id.

    Trials are launched by the wandb ``command`` embedded in the sweep payload
    (`ieforge sweep run sweep=<name>`), so no ``function`` is passed here.
    """
    wandb = _require_wandb()
    wandb.agent(
        sweep_id,
        count=count,
        project=project,
        entity=_resolve_entity(wandb, entity),
    )


def _invoke_train(overrides: list[str]) -> None:
    """Invoke the Hydra train entrypoint as a subprocess with the given overrides."""
    cmd = [
        sys.executable,
        "-m",
        "ieforge.cli._hydra_entry",
        "--config-name=train",
        *overrides,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        msg = f"trial training exited with code {result.returncode}"
        raise RuntimeError(msg)


def run_trial_from_args(args: list[str]) -> None:
    """Run one trial from raw `key=value` args passed by the wandb agent.

    The wandb command template passes `sweep=<name>` plus the sampled
    hyperparameters as `key=value` tokens. We extract the sweep name, keep only
    sampled keys declared in that sweep's parameters, force the training run to
    log to wandb, isolate its output directory, and train.

    The training subprocess (not this process) owns the wandb run: it inherits
    the agent's ``WANDB_*`` environment (sweep id, the agent-assigned run id), so
    its ``WandbLogger`` attaches to the run the sweep controller tracks and logs
    the optimization metric there. We therefore do NOT call ``wandb.init()`` here
    -- doing so would create a separate run in this process, disjoint from the
    subprocess where ``trainer.fit`` actually logs the metric.
    """
    name = None
    sampled_pairs: dict[str, str] = {}
    # The wandb agent passes sampled values as already-stringified `key=value`
    # tokens (via the command's ${args_no_hyphens}), not a wandb.config mapping,
    # so we parse them directly here rather than via translate.wandb_config_to_overrides.
    for token in args:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key == "sweep":
            name = value
        else:
            sampled_pairs[key] = value
    if not name:
        msg = "sweep run requires a `sweep=<name>` argument from the agent command"
        raise ValueError(msg)

    sweep = load_sweep_config(name)
    allowed = set(sweep.parameters.keys())
    sampled = [f"{k}={v}" for k, v in sampled_pairs.items() if k in allowed]

    dropped = sorted(set(sampled_pairs) - allowed)
    if dropped:
        # A sampled key that is not a declared sweep parameter would otherwise be
        # silently discarded, leaving the trial to train on defaults while still
        # reporting a metric -- corrupting the search. Surface it loudly.
        log.warning(
            "sweep trial dropped sampled keys not declared in parameters: %s",
            dropped,
        )

    # wandb's agent sets WANDB_RUN_ID for the trial; fall back to a stable label.
    run_id = os.environ.get("WANDB_RUN_ID") or "trial"
    output_override = (
        f"output_dir=outputs/sweep/{name}/{run_id}/"
        "${task.name}/${dataset.name}/${model.name}"
    )
    overrides = [
        *sweep.base_overrides,
        "logger=wandb",
        *sampled,
        output_override,
    ]
    _invoke_train(overrides)
