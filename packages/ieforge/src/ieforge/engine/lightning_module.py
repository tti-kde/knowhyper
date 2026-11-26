from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from lightning_fabric.utilities.apply_func import move_data_to_device
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import LRScheduler

from ieforge.data.schema import BioREDirectPairPrediction, BioREDPairPrediction
from ieforge.losses.base import IELoss
from ieforge.metrics.base import IEMetric
from ieforge.models.base import IEModel
from ieforge.tasks.base import IETask
from ieforge.tasks.biored import BioREDPairTask
from ieforge.tasks.bioredirect import BioREDirectPairTask
from ieforge.tasks.doc_cls import DocClassificationTask
from ieforge.tasks.pico import PICOTask
from ieforge.tasks.qa import QATask
from ieforge.tasks.re import RETask
from ieforge.tasks.sentence_similarity import SentenceSimilarityTask


class IELightningModule(pl.LightningModule):
    """Lightning module that wires together model, task, loss, and metrics."""

    def __init__(
        self,
        model: IEModel,
        task: IETask,
        loss_fn: IELoss,
        metrics: IEMetric,
        optimizer_cfg: Any,
        scheduler_cfg: Any = None,
        accumulate_grad_batches: int = 1,
    ) -> None:
        """Store all components, optimizer/scheduler configs, and accumulation factor.

        ``accumulate_grad_batches`` lets the module reproduce ``Trainer``-style
        gradient accumulation manually: under manual optimization (which is the
        only mode this module supports) Lightning rejects the trainer-level
        ``accumulate_grad_batches`` flag, so we apply the same effect here by
        scaling each micro-batch loss by 1/N and only running
        ``optimizer.step`` / ``scheduler.step`` / ``optimizer.zero_grad`` every
        Nth call to ``training_step``.
        """
        super().__init__()
        self.automatic_optimization = False
        self.model = model
        self.task = task
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        if accumulate_grad_batches < 1:
            msg = "accumulate_grad_batches must be a positive integer"
            raise ValueError(msg)
        self.accumulate_grad_batches = accumulate_grad_batches
        self._micro_batch_counter = 0

    def transfer_batch_to_device(
        self,
        batch: dict[str, Any],
        device: torch.device,
        dataloader_idx: int,
    ) -> dict[str, Any]:
        """Move tensor-like batch data to device while preserving metadata fields."""
        del dataloader_idx
        movable = {
            key: value
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        metadata = {
            key: value
            for key, value in batch.items()
            if not isinstance(value, torch.Tensor)
        }
        return {**move_data_to_device(movable, device), **metadata}

    def _re_chunk_size(self) -> int:
        if not isinstance(self.task, RETask):
            return 0
        return self.task.train_pair_chunk_size

    def _re_val_chunk_size(self) -> int:
        if not isinstance(self.task, RETask):
            return 0
        return self.task.val_pair_chunk_size

    def _chunked_re_forward(self, batch: dict[str, Any], chunk_size: int) -> list[Any]:
        """Run chunked forward passes without gradients; return concatenated decoded preds."""
        total_examples = int(batch["input_ids"].shape[0])
        all_outputs = []
        with torch.no_grad():
            for start in range(0, total_examples, chunk_size):
                end = min(start + chunk_size, total_examples)
                sub_batch = self._slice_re_batch(batch, start, end)
                all_outputs.append(self.model(sub_batch))
        return all_outputs

    def _slice_re_batch(
        self, batch: dict[str, Any], start: int, end: int
    ) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, (torch.Tensor, list)):
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    def _chunked_re_forward_backward(
        self,
        batch: dict[str, Any],
        chunk_size: int,
        loss_scale: float = 1.0,
    ) -> torch.Tensor:
        """Run chunked forward+backward passes and return a detached weighted-mean loss.

        Each chunk's loss is scaled by its example fraction and backpropagated immediately,
        freeing the chunk's activations before the next forward pass.  ``loss_scale``
        multiplies that fraction by an outer factor so callers can fold in a
        gradient-accumulation 1/N term without changing the per-chunk normalization.
        The returned scalar is detached: it is used only for logging, not for further
        differentiation, and reflects the unscaled (pre-accumulation) mean loss so
        progress bars stay comparable across accumulation factors.
        """
        total_examples = int(batch["input_ids"].shape[0])
        logged_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        for start in range(0, total_examples, chunk_size):
            end = min(start + chunk_size, total_examples)
            sub_batch = self._slice_re_batch(batch, start, end)
            outputs = self.model(sub_batch)
            chunk_loss = self.loss_fn(outputs, sub_batch)
            weight = (end - start) / total_examples
            scaled_chunk_loss = chunk_loss * weight * loss_scale
            if self._trainer is not None:
                self.manual_backward(scaled_chunk_loss)
            else:
                scaled_chunk_loss.backward()
            logged_loss = (
                logged_loss + chunk_loss.detach().to(dtype=torch.float32) * weight
            )
        return logged_loss

    @staticmethod
    def _re_batch_is_empty(batch: dict[str, Any]) -> bool:
        """Return True when the RE collator produced a zero-pair batch.

        RECollator emits a 0-row batch when every example in it has no allowed
        entity pair; such a batch must not be forwarded to the model.
        """
        input_ids = batch.get("input_ids")
        return isinstance(input_ids, torch.Tensor) and input_ids.shape[0] == 0

    def training_step(self, batch: dict[str, Any], _batch_idx: int) -> torch.Tensor:
        """Run a forward pass, accumulate gradients, and step the optimizer.

        Under manual optimization, this method owns the full update cycle:
        zero_grad -> forward -> backward -> optimizer.step -> scheduler.step.

        When ``accumulate_grad_batches > 1`` the cycle is unrolled across N
        consecutive training_step calls: ``zero_grad`` runs on the first call
        of each window, each micro-batch's loss is scaled by 1/N before
        backward, and ``opt.step`` / ``scheduler.step`` only run on the Nth
        call. When called outside a Trainer (e.g. in unit tests), the
        optimizer/scheduler calls are skipped so that direct invocations still
        work.
        """
        if self._re_batch_is_empty(batch):
            return torch.zeros((), device=self.device, dtype=torch.float32)

        has_trainer = self._trainer is not None
        accum = self.accumulate_grad_batches
        is_first_in_window = self._micro_batch_counter == 0

        opt = None
        if has_trainer:
            opt = self.optimizers()
            if not isinstance(opt, LightningOptimizer):
                msg = "IELightningModule expects exactly one optimizer"
                raise RuntimeError(msg)
            if is_first_in_window:
                opt.zero_grad()

        loss_scale = 1.0 / accum

        chunk_size = self._re_chunk_size()
        input_ids = batch.get("input_ids")
        if (
            chunk_size <= 0
            or not isinstance(input_ids, torch.Tensor)
            or input_ids.shape[0] <= chunk_size
        ):
            outputs = self.model(batch)
            loss = self.loss_fn(outputs, batch)
            scaled_loss = loss * loss_scale
            if has_trainer:
                self.manual_backward(scaled_loss)
            else:
                scaled_loss.backward()
            logged_loss = loss.detach()
        else:
            logged_loss = self._chunked_re_forward_backward(
                batch, chunk_size, loss_scale=loss_scale
            )

        self._micro_batch_counter += 1
        is_step_boundary = self._micro_batch_counter >= accum
        if has_trainer and is_step_boundary:
            assert opt is not None
            opt.step()
            scheduler = self.lr_schedulers()
            if scheduler is not None:
                if not isinstance(scheduler, LRScheduler):
                    msg = "IELightningModule expects a single LR scheduler"
                    raise RuntimeError(msg)
                scheduler.step()
        if is_step_boundary:
            self._micro_batch_counter = 0

        self.log("train/loss", logged_loss, on_step=True, on_epoch=True, prog_bar=True)
        return logged_loss

    def _metric_inputs(
        self, batch: dict[str, Any], preds: list[Any]
    ) -> tuple[list[Any], list[Any], list[str]]:
        """Return references and texts for the current task family."""
        texts = batch.get("text", [""] * len(preds))
        if isinstance(self.task, BioREDirectPairTask):
            required = ("pmids", "id1s", "id2s", "id1_types", "id2_types", "label_text")
            missing = [k for k in required if k not in batch]
            if missing:
                msg = (
                    "BioREDirectPairTask batch missing metadata for metric "
                    "evaluation: " + ", ".join(missing)
                )
                raise ValueError(msg)
            n = len(preds)
            novelty_text = batch.get("novelty_text", [None] * n)
            direction_text = batch.get("direction_text", ["None"] * n)
            gold = [
                BioREDirectPairPrediction(
                    pmid=pmid,
                    id1=id1,
                    id2=id2,
                    id1_type=id1_type,
                    id2_type=id2_type,
                    gold_label=label_text,
                    pred_label=label_text,
                    novelty=novelty,
                    direction=direction,
                    gold_direction=direction,
                )
                for pmid, id1, id2, id1_type, id2_type, label_text, novelty, direction in zip(
                    batch["pmids"],
                    batch["id1s"],
                    batch["id2s"],
                    batch["id1_types"],
                    batch["id2_types"],
                    batch["label_text"],
                    novelty_text,
                    direction_text,
                    strict=True,
                )
            ]
            return preds, gold, texts
        if isinstance(self.task, BioREDPairTask):
            required = (
                "pmids",
                "id1s",
                "id2s",
                "id1_types",
                "id2_types",
                "label_text",
            )
            missing = [key for key in required if key not in batch]
            if missing:
                msg = (
                    "BioREDPairTask batch missing metadata for metric evaluation: "
                    + ", ".join(missing)
                )
                raise ValueError(msg)
            novelty_text = batch.get("novelty_text", [None] * len(preds))
            gold = [
                BioREDPairPrediction(
                    pmid=pmid,
                    id1=id1,
                    id2=id2,
                    id1_type=id1_type,
                    id2_type=id2_type,
                    gold_label=label_text,
                    pred_label=label_text,
                    novelty=novelty,
                )
                for pmid, id1, id2, id1_type, id2_type, label_text, novelty in zip(
                    batch["pmids"],
                    batch["id1s"],
                    batch["id2s"],
                    batch["id1_types"],
                    batch["id2_types"],
                    batch["label_text"],
                    novelty_text,
                    strict=True,
                )
            ]
            return preds, gold, texts
        if isinstance(
            self.task,
            QATask | DocClassificationTask | PICOTask | SentenceSimilarityTask,
        ):
            if "label_text" not in batch:
                msg = f"{type(self.task).__name__} batch missing label_text for metric evaluation"
                raise ValueError(msg)
            return preds, batch["label_text"], texts
        refs = (
            batch["entities"]
            if "entities" in batch
            else batch.get("relations", [[] for _ in preds])
        )
        return preds, refs, texts

    def _eval_forward(self, batch: dict[str, Any]) -> Any:
        """Run forward pass for evaluation, chunked if val_pair_chunk_size is set."""
        chunk_size = self._re_val_chunk_size()
        input_ids = batch.get("input_ids")
        if (
            chunk_size <= 0
            or not isinstance(input_ids, torch.Tensor)
            or input_ids.shape[0] <= chunk_size
        ):
            return self.model(batch)
        chunk_outputs = self._chunked_re_forward(batch, chunk_size)
        combined_logits = torch.cat([o.logits for o in chunk_outputs], dim=0)
        return type("_Outputs", (), {"logits": combined_logits})()

    def validation_step(self, batch: dict[str, Any], _batch_idx: int) -> None:
        """Run inference and accumulate metric state for validation."""
        if self._re_batch_is_empty(batch):
            return
        outputs = self._eval_forward(batch)
        preds = self.task.decode(outputs, batch)
        preds, refs, texts = self._metric_inputs(batch, preds)
        self.metrics.update(preds, refs, texts)

    def test_step(self, batch: dict[str, Any], _batch_idx: int) -> None:
        """Run inference and accumulate metric state for testing."""
        if self._re_batch_is_empty(batch):
            return
        outputs = self._eval_forward(batch)
        preds = self.task.decode(outputs, batch)
        preds, refs, texts = self._metric_inputs(batch, preds)
        self.metrics.update(preds, refs, texts)

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics and reset accumulators."""
        scores = self.metrics.compute()
        self.log_dict({f"val/{k}": v for k, v in scores.items()}, prog_bar=True)
        self.metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Log test metrics and reset accumulators."""
        scores = self.metrics.compute()
        self.log_dict({f"test/{k}": v for k, v in scores.items()})
        self.metrics.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Build and return the optimizer, and optionally a learning rate scheduler."""
        optimizer = instantiate(self.optimizer_cfg, params=self.parameters())
        if self.scheduler_cfg is None:
            return {"optimizer": optimizer}
        scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Scheduler is stepped manually in training_step; this metadata is ignored under manual optimization.
                "frequency": 1,
            },
        }
