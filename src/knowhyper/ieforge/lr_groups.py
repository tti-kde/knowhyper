"""Differential learning rate for the knowledge-fusion branch.

Motivation (NER report 2026-07-02 §5.2 + DDI arm-A/B mechanistic null):
ieforge's ``IELightningModule.configure_optimizers`` builds a single optimizer
over ``self.parameters()`` at one LR (2e-5, tuned for the pretrained BERT trunk).
The freshly-initialized knowledge branch (cross-attention, AllSet encoder, gate,
fusion) is residual-added to that strong trunk and, at the trunk's low LR, barely
trains — its weights stay near Xavier init and every learnable fusion control
(gate/temperature/coverage) collapses to its initialization. Giving the knowledge
modules a much higher LR than the backbone is the standard fix for a fresh module
bolted onto a pretrained backbone.

ieforge hardcodes ``IELightningModule`` in ``_hydra_entry`` (not swappable via
config), so we inject through the one configurable seam — the trainer
``callbacks`` list. :class:`KnowledgeLRCallback` replaces the module's
``configure_optimizers`` during ``setup(stage="fit")`` (which Lightning runs
before it builds optimizers) with a param-group-aware version that reproduces
ieforge's optimizer/scheduler construction but splits params into
backbone (base LR) and knowledge (base LR x ``knowledge_lr_mult``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
from hydra.utils import instantiate

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch

# Backbone = the inner HF classifier. Everything else under the model (cross_attn,
# encoder, gate, gate_mlp, modality_fusion, coverage_mlp) is the knowledge branch.
BACKBONE_PREFIX = "model.inner."


def split_knowledge_params(
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
    backbone_prefix: str = BACKBONE_PREFIX,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Partition trainable params into (backbone, knowledge) by name prefix.

    Backbone = params whose name starts with ``backbone_prefix`` (the pretrained
    inner BERT). Knowledge = every other trainable param (the fusion branch).
    Frozen params (``requires_grad=False``) are dropped from both lists.
    """
    backbone: list[torch.nn.Parameter] = []
    knowledge: list[torch.nn.Parameter] = []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        (backbone if name.startswith(backbone_prefix) else knowledge).append(p)
    return backbone, knowledge


def build_param_groups(
    pl_module: pl.LightningModule,
    knowledge_lr_mult: float,
    backbone_prefix: str = BACKBONE_PREFIX,
) -> list[dict[str, Any]]:
    """Two AdamW param groups: backbone at base LR, knowledge at base LR x mult.

    Reads the base LR from the module's ``optimizer_cfg`` (the same dict ieforge
    passes to the optimizer). The backbone group omits ``lr`` so it inherits the
    optimizer default; the knowledge group sets ``lr = base_lr * mult``. Empty
    groups are dropped so a knowledge-free model still yields a valid optimizer.
    """
    opt_cfg: Any = pl_module.optimizer_cfg  # ieforge duck-typed attr
    base_lr = float(opt_cfg["lr"])
    backbone, knowledge = split_knowledge_params(
        pl_module.named_parameters(), backbone_prefix
    )
    groups: list[dict[str, Any]] = []
    if backbone:
        groups.append({"params": backbone})  # inherits base_lr
    if knowledge:
        groups.append({"params": knowledge, "lr": base_lr * knowledge_lr_mult})
    return groups


class KnowledgeLRCallback(pl.Callback):
    """Give the knowledge branch a higher LR than the pretrained backbone.

    Added to the trainer ``callbacks`` list. During ``setup(stage="fit")`` it
    replaces ``pl_module.configure_optimizers`` with a param-group-aware version
    (backbone @ base LR, knowledge @ base LR x ``knowledge_lr_mult``) that
    otherwise reproduces ieforge's optimizer + scheduler wiring exactly.
    ``knowledge_lr_mult=1.0`` reproduces the single-LR baseline.
    """

    def __init__(
        self,
        knowledge_lr_mult: float = 1.0,
        backbone_prefix: str = BACKBONE_PREFIX,
    ) -> None:
        """Store the LR multiplier and backbone name prefix."""
        super().__init__()
        if knowledge_lr_mult <= 0:
            msg = f"knowledge_lr_mult must be > 0, got {knowledge_lr_mult}"
            raise ValueError(msg)
        self.knowledge_lr_mult = float(knowledge_lr_mult)
        self.backbone_prefix = backbone_prefix

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Swap in the param-group optimizer builder before Lightning uses it."""
        del trainer
        if stage != "fit":
            return
        mult = self.knowledge_lr_mult
        prefix = self.backbone_prefix

        def configure_optimizers() -> Any:
            opt_cfg: Any = pl_module.optimizer_cfg  # ieforge duck-typed attrs
            sch_cfg: Any = pl_module.scheduler_cfg
            groups = build_param_groups(pl_module, mult, prefix)
            # Build via a partial so the plain-python param-group list is passed
            # straight to the optimizer. instantiate(..., params=groups) would
            # OmegaConf-convert the group dicts and AdamW would then reject them
            # ("one of the params is a DictConfig").
            optimizer_factory = instantiate(opt_cfg, _partial_=True)
            optimizer = optimizer_factory(groups)
            if sch_cfg is None:
                return {"optimizer": optimizer}
            scheduler = instantiate(sch_cfg, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    # interval/frequency are dead metadata under ieforge's
                    # automatic_optimization=False: it steps the scheduler
                    # manually in training_step. Kept to match ieforge's shape.
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        # ieforge hardcodes IELightningModule (not swappable via config), so we
        # override the instance method here — setattr avoids the method-assign
        # type error while Lightning still calls the instance attribute.
        setattr(pl_module, "configure_optimizers", configure_optimizers)  # noqa: B010
