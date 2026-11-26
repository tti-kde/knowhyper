"""Fail-closed runtime audit for #233's candidate-owned A0 arms.

Shape-correct plumbing has twice produced a plausible validation F1 while the treatment was
dead.  This callback records facts that cannot be inferred from F1: the registered treatment
parameters are present in the optimizer exactly once, all anchors are frozen, real training
batches contain evidence and interventions, the logit delta is non-zero, and treatment weights
actually move after the warmup horizon.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

FILENAME = "a0_trainability.json"


class A0TrainabilityCallback(pl.Callback):
    """Audit treatment trainability and make a dead route fail the job."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        min_train_steps: int = 150,
        fail_on_dead: bool = True,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.min_train_steps = int(min_train_steps)
        self.fail_on_dead = bool(fail_on_dead)
        self._before: dict[str, torch.Tensor] = {}
        self._target_parameters: dict[str, torch.nn.Parameter] = {}
        self._optimizer_counts: dict[int, int] = {}
        self._frozen_anchor_optimizer_memberships = 0
        self._hook: Any = None
        self._diagnostics = {
            "train_batches": 0,
            "masked_candidates": 0,
            "with_evidence": 0,
            "intervened": 0,
            "nonzero_delta": 0,
        }
        self._written = False

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        del trainer, stage
        model = getattr(pl_module, "model", None)
        if model is not None and self._hook is None:
            self._hook = model.register_forward_hook(self._on_forward)

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        model = getattr(pl_module, "model", None)
        if model is None:
            msg = "A0 trainability audit could not find pl_module.model"
            raise RuntimeError(msg)
        source = str(getattr(model, "evidence_source", ""))
        if source == "text":
            treatment = getattr(model, "text_encoder", None)
            prefix = "text_encoder"
        elif source == "lexical":
            treatment = getattr(model, "_lexical_projection", None)
            prefix = "lexical_projection"
        else:
            msg = f"A0 trainability audit does not accept source={source!r}"
            raise RuntimeError(msg)
        if treatment is None:
            msg = f"A0 {source} treatment is absent before optimizer construction"
            raise RuntimeError(msg)

        self._target_parameters = {
            f"{prefix}.{name}": parameter
            for name, parameter in treatment.named_parameters()
        }
        if not self._target_parameters:
            msg = f"A0 {source} treatment has no parameters"
            raise RuntimeError(msg)

        optimizer_ids = [
            id(parameter)
            for optimizer in trainer.optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self._optimizer_counts = {
            parameter_id: optimizer_ids.count(parameter_id)
            for parameter_id in set(optimizer_ids)
        }
        bad_treatment = [
            name
            for name, parameter in self._target_parameters.items()
            if not parameter.requires_grad
            or self._optimizer_counts.get(id(parameter), 0) != 1
        ]
        if bad_treatment:
            msg = (
                "A0 treatment parameters must be trainable and appear exactly once in the "
                f"optimizer; bad={bad_treatment[:8]}"
            )
            raise RuntimeError(msg)

        head = getattr(model, "signed_head", None)
        if head is None:
            msg = "A0 signed residual head is absent"
            raise RuntimeError(msg)
        bad_head = [
            name
            for name, parameter in head.named_parameters()
            if not parameter.requires_grad
            or self._optimizer_counts.get(id(parameter), 0) != 1
        ]
        if bad_head:
            msg = (
                "A0 signed-head parameters must appear exactly once in the optimizer; "
                f"bad={bad_head[:8]}"
            )
            raise RuntimeError(msg)

        anchors = [getattr(model, "inner", None), *getattr(model, "_oof_anchors", ())]
        trainable_anchor_parameters = [
            f"anchor{anchor_index}.{name}"
            for anchor_index, anchor in enumerate(anchors)
            if anchor is not None
            for name, parameter in anchor.named_parameters()
            if parameter.requires_grad
        ]
        if trainable_anchor_parameters:
            msg = f"A0 anchor is trainable: {trainable_anchor_parameters[:8]}"
            raise RuntimeError(msg)
        # ieforge's ordinary single-LR optimizer enumerates the registered full anchor even
        # when requires_grad=False. AdamW skips parameters whose grad is None, so this is inert
        # and creates no optimizer state. The five OOF anchors are unregistered and absent.
        # Record membership for transparency; freezing is the load-bearing condition.
        self._frozen_anchor_optimizer_memberships = sum(
            self._optimizer_counts.get(id(parameter), 0)
            for anchor in anchors
            if anchor is not None
            for parameter in anchor.parameters()
        )

        # CPU float clones allow an exact per-tensor max delta at the end. The largest arm is
        # one extra BERT (~440 MB), small relative to an rt_HF node's host memory and not saved
        # into Lightning callback state.
        self._before = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in self._target_parameters.items()
        }

    def _on_forward(self, module: torch.nn.Module, args: Any, output: Any) -> None:
        del args
        if not module.training:
            return
        diagnostics = getattr(output, "knowledge_diagnostics", None)
        if not isinstance(diagnostics, dict):
            return
        evidence = diagnostics.get("evidence_valid")
        intervened = diagnostics.get("intervened")
        delta = diagnostics.get("logit_delta")
        if evidence is None or intervened is None or delta is None:
            return
        keep = getattr(output, "span_mask", None)
        if keep is None:
            keep = torch.ones_like(evidence, dtype=torch.bool)
        else:
            keep = keep.reshape(evidence.shape).bool()
        evidence = evidence.bool() & keep
        intervened = intervened.bool() & keep
        nonzero = delta.detach().abs().amax(dim=-1).reshape(evidence.shape) > 0
        self._diagnostics["train_batches"] += 1
        self._diagnostics["masked_candidates"] += int(keep.sum())
        self._diagnostics["with_evidence"] += int(evidence.sum())
        self._diagnostics["intervened"] += int(intervened.sum())
        self._diagnostics["nonzero_delta"] += int((nonzero & keep).sum())

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._write_and_enforce(global_step=int(trainer.global_step))

    def teardown(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        del pl_module, stage
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        # `on_train_end` is the normal path. Teardown still writes a record if a custom
        # Lightning loop skips that hook, but does not turn an unrelated exception into a
        # second exception.
        if self._before and not self._written:
            self._write_and_enforce(global_step=int(trainer.global_step), enforce=False)

    def _write_and_enforce(self, *, global_step: int, enforce: bool = True) -> None:
        if self._written:
            return
        deltas: dict[str, float] = {}
        for name, parameter in self._target_parameters.items():
            now = parameter.detach().float().cpu()
            deltas[name] = float((now - self._before[name]).abs().max())
        moved = sorted(name for name, delta in deltas.items() if delta > 0.0)
        max_delta = max(deltas.values(), default=0.0)
        failures: list[str] = []
        if global_step < self.min_train_steps:
            failures.append(f"global_step={global_step} < {self.min_train_steps}")
        if not moved:
            failures.append("no treatment tensor moved")
        failures.extend(
            f"{key}=0"
            for key in ("with_evidence", "intervened", "nonzero_delta")
            if self._diagnostics[key] == 0
        )

        record = {
            "global_step": global_step,
            "min_train_steps": self.min_train_steps,
            "treatment_tensors": len(self._target_parameters),
            "moved_tensors": len(moved),
            "unchanged_tensors": len(self._target_parameters) - len(moved),
            "max_abs_delta": max_delta,
            "frozen_anchor_optimizer_memberships": (
                self._frozen_anchor_optimizer_memberships
            ),
            "moved_names": moved,
            "diagnostics": dict(self._diagnostics),
            "failures": failures,
            "pass": not failures,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / FILENAME).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        self._written = True
        if enforce and failures and self.fail_on_dead:
            msg = f"A0 runtime trainability gate failed: {failures}"
            raise RuntimeError(msg)
