"""Write the #218 Phase 2 span diagnostics beside the run, in-job.

`$KNOWHYPER_HOME/runs/**/checkpoints` is cleaned, so a diagnostics probe submitted
after a campaign finishes finds no checkpoint and skips every cell -- the failure
recorded in the `abci-run-checkpoints-are-not-durable` note. The record therefore has
to be produced by the training job itself.

Implemented as a `register_forward_hook` on the model rather than
`on_validation_batch_end`, because ieforge's `IELightningModule.validation_step`
returns `None`: Lightning hands the callback an empty `outputs` and the model output is
unreachable there. A forward hook sees every output whichever step produced it.

A model with no knowledge branch (baseline, text arm) never sets
`knowledge_diagnostics`, and this callback then writes NOTHING rather than an empty
record -- an all-zero file would read as "measured, dead branch" for a cell that has no
branch at all.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from knowhyper.ieforge.span_diagnostics import SpanDiagnosticsAccumulator

logger = logging.getLogger(__name__)

FILENAME = "knowledge_diagnostics.json"


class SpanDiagnosticsCallback(pl.Callback):
    """Accumulate per-batch span diagnostics and write one JSON per cell."""

    def __init__(self, output_dir: str | Path) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self._accumulator = SpanDiagnosticsAccumulator()
        self._handle: Any = None
        self._seen = False

    # ------------------------------------------------------------------ hooks

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Register the forward hook once.

        Lightning calls setup for fit AND validate, so this must be idempotent:
        two hooks on one model would count every batch twice and halve every mean.
        """
        del trainer, stage
        if self._handle is not None:
            return
        model = getattr(pl_module, "model", None)
        if model is None:
            return
        self._handle = model.register_forward_hook(self._on_forward)

    def teardown(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Remove the hook and write the record."""
        del trainer, pl_module, stage
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._write()

    # ---------------------------------------------------------------- internals

    def _on_forward(self, _module: Any, args: Any, output: Any) -> None:
        diagnostics = getattr(output, "knowledge_diagnostics", None)
        if diagnostics is None:
            return
        # The public candidate mask lives on the batch, which is the forward's first
        # positional argument. Without it the reduction averages over padding
        # candidates -- mostly zeros at max_span_width 12 -- and a healthy branch
        # reports a near-zero mean residual, i.e. looks dead.
        span_mask = None
        if args:
            batch = args[0]
            if isinstance(batch, dict):
                span_mask = batch.get("span_mask")
        try:
            self._accumulator.update(diagnostics, span_mask=span_mask)
        except ValueError:
            # Mixed controls in one cell is a harness bug, not a reason to kill a
            # training run that is otherwise fine. Record it and stop accumulating.
            logger.exception("span diagnostics disabled for this cell")
            if self._handle is not None:
                self._handle.remove()
                self._handle = None
            return
        self._seen = True

    def _write(self) -> None:
        if not self._seen:
            # No knowledge branch on this model (baseline / text arm). Writing an
            # all-zero record would read as a measured dead branch.
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / FILENAME
        record = self._accumulator.as_dict()
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        verdicts = record.get("verdicts", {})
        tripped = sorted(name for name, hit in verdicts.items() if hit)
        if tripped:
            # Phase 2 stops on these. Surface them in driver.log rather than leaving
            # them only in a JSON nobody reads until aggregation.
            logger.warning("span diagnostics verdicts tripped: %s", ", ".join(tripped))
        logger.info("wrote span knowledge diagnostics to %s", path)
