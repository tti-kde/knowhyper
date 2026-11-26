"""Hydra entrypoints for train and eval commands."""

import logging
import math
from typing import Any, cast

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ieforge import CONFIGS_DIR
from ieforge.data.datamodule import IEDataModule, IEEvalDataModule
from ieforge.engine.lightning_module import IELightningModule
from ieforge.knowledge.core.config import strip_names
from ieforge.metrics.biored import write_biored_predictions_jsonl
from ieforge.tasks.biored import BioREDPairTask
from ieforge.tasks.doc_cls import DocClassificationTask
from ieforge.tasks.ner import NERTask
from ieforge.tasks.pico import PICOTask
from ieforge.tasks.qa import QATask
from ieforge.tasks.sentence_similarity import SentenceSimilarityTask
from ieforge.tasks.span_ner import SpanNERTask

log = logging.getLogger(__name__)

_CONFIGS_DIR = str(CONFIGS_DIR)

# Keys in the dataset config that belong to IEDataModule, not the dataset constructor.
_DATAMODULE_KEYS = frozenset({"batch_size", "num_workers", "name", "task_family"})
_RE_TASK_SCHEMAS_USING_TASK_LABEL_SPACE = frozenset(
    {"biored_entity_pair", "biored_entity_pair_plus_type_plus_novelty"}
)
_RE_METRIC_TARGETS_USING_NEGATIVE_LABEL = frozenset(
    {
        "ieforge.metrics.relation_f1.RelationF1",
        "ieforge.metrics.biored_pair_f1.BioREDPairF1",
        "ieforge.metrics.biored_pair_type_novelty_f1.BioREDPairTypeNoveltyF1",
    }
)

# dataset task_family -> the set of task.name values that legitimately consume it.
_DATASET_TASK_FAMILY_COMPAT: dict[str, frozenset[str]] = {
    "ner": frozenset({"ner", "span_ner"}),
    "re": frozenset({"re"}),
    "biored": frozenset({"biored_pair", "bioredirect_pair"}),
    "qa": frozenset({"qa"}),
    "doc_cls": frozenset({"doc_cls"}),
    "pico": frozenset({"pico"}),
    "sentence_similarity": frozenset({"sentence_similarity"}),
}


def _validate_task_dataset_compatibility(cfg: DictConfig) -> None:
    """Fail fast when the configured task does not match the dataset's family.

    Reads ``dataset.task_family``; when it is absent (e.g. a custom dataset), this
    is a no-op so existing setups keep working.
    """
    dataset_family = OmegaConf.select(cfg, "dataset.task_family", default=None)
    if dataset_family is None:
        return
    task_name = OmegaConf.select(cfg, "task.name", default=None)
    allowed = _DATASET_TASK_FAMILY_COMPAT.get(
        dataset_family, frozenset({dataset_family})
    )
    if task_name not in allowed:
        dataset_name = OmegaConf.select(cfg, "dataset.name", default="<unknown>")
        msg = (
            f"Dataset {dataset_name!r} has task_family={dataset_family!r}, but the "
            f"configured task is {task_name!r}. Select a matching experiment, "
            f"e.g. `experiment=re dataset=chemprot` for RE datasets."
        )
        raise ValueError(msg)


def _dataset_instantiation_cfg(dataset_cfg: DictConfig) -> DictConfig:
    """Return a copy of a dataset config with IEDataModule-only keys removed."""
    raw = cast("dict", OmegaConf.to_container(dataset_cfg, resolve=True))
    filtered = {k: v for k, v in raw.items() if k not in _DATAMODULE_KEYS}
    return OmegaConf.create(filtered)


# Keys in the model config that are metadata, not model constructor args.
_MODEL_NON_CTOR_KEYS = frozenset({"tokenizer_name"})


def _model_instantiation_cfg(model_cfg: DictConfig) -> DictConfig:
    """Return a copy of a model config with non-constructor keys removed."""
    raw = cast("dict", OmegaConf.to_container(model_cfg, resolve=True))
    filtered = {k: v for k, v in raw.items() if k not in _MODEL_NON_CTOR_KEYS}
    return OmegaConf.create(filtered)


def _build_datamodule(
    cfg: DictConfig,
    collator: Any,
    splits: list[str],
    train_collator: Any = None,
    test_collator: Any = None,
) -> IEDataModule | IEEvalDataModule:
    """Construct an IEDataModule by instantiating dataset splits from config.

    If ``cfg.train_dataset`` is present, it is used for the train split (with its
    own ``batch_size``), while ``cfg.dataset`` is used for validation and test.
    This allows pair-level train batching alongside document-level eval batching.
    """
    requested = set(splits)
    train_dataset_cfg = OmegaConf.select(cfg, "train_dataset", default=None)

    if requested == {"test"}:
        test_cfg = cast(
            "DictConfig",
            OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True)),
        )
        test_cfg.split = "test"
        test_dataset = _build_dataset(test_cfg, "test", root_cfg=cfg)
        return IEEvalDataModule(
            test_dataset=test_dataset,
            collator=collator,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
        )

    required_splits = ("train", "validation", "test")
    missing = [split for split in required_splits if split not in requested]
    if missing:
        msg = f"Missing required dataset split: {', '.join(missing)}"
        raise ValueError(msg)

    datasets = {}
    for split in required_splits:
        if split == "train" and train_dataset_cfg is not None:
            src_cfg = cast(
                "DictConfig",
                OmegaConf.create(
                    OmegaConf.to_container(train_dataset_cfg, resolve=True)
                ),
            )
        else:
            src_cfg = cast(
                "DictConfig",
                OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True)),
            )
        src_cfg.split = split
        datasets[split] = _build_dataset(src_cfg, split, root_cfg=cfg)

    train_batch_size = (
        train_dataset_cfg.batch_size
        if train_dataset_cfg is not None
        else cfg.dataset.batch_size
    )
    return IEDataModule(
        train_dataset=datasets["train"],
        val_dataset=datasets["validation"],
        test_dataset=datasets["test"],
        collator=collator,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        train_collator=train_collator,
        train_batch_size=train_batch_size,
        test_collator=test_collator,
    )


def _compute_training_steps(cfg: DictConfig, num_examples: int) -> int:
    """Compute total training steps from dataset size, batch size, and epochs."""
    train_dataset_cfg = OmegaConf.select(cfg, "train_dataset", default=None)
    batch_size = (
        train_dataset_cfg.batch_size
        if train_dataset_cfg is not None
        else cfg.dataset.batch_size
    )
    steps_per_epoch = math.ceil(num_examples / batch_size)
    accumulate = cfg.trainer.get("accumulate_grad_batches", 1)
    return math.ceil(steps_per_epoch / accumulate) * cfg.trainer.max_epochs


def _build_dataset(
    dataset_cfg: DictConfig,
    split: str,
    *,
    root_cfg: DictConfig | None = None,
) -> Any:
    """Construct a dataset split from its config, optionally wrapped by augmentor."""
    base = instantiate(_dataset_instantiation_cfg(dataset_cfg), split=split)
    if root_cfg is None or split != "train":
        return base
    aug_cfg = OmegaConf.select(root_cfg, "augmentor", default=None)
    if aug_cfg is None:
        return base
    from ieforge.data.augmentation import AugmentedDataset

    aug_clean = strip_names(OmegaConf.to_container(aug_cfg, resolve=True))
    return AugmentedDataset(base, instantiate(OmegaConf.create(aug_clean)))


def _build_task(cfg: DictConfig, dataset: Any) -> Any:
    """Construct the configured task with dataset-derived labels."""
    target = cfg.task.get("_target_")
    if target == "ieforge.tasks.ner.NERTask":
        return NERTask(
            label_list=dataset.label_list,
            label_scheme=cfg.task.label_scheme,
            max_length=cfg.task.max_length,
            name=cfg.task.name,
        )
    if target == "ieforge.tasks.span_ner.SpanNERTask":
        return SpanNERTask(
            label_list=dataset.label_list,
            max_length=cfg.task.max_length,
            max_span_width=cfg.task.max_span_width,
            threshold=cfg.task.get("threshold", 0.5),
            decoder=cfg.task.get("decoder", "greedy"),
            name=cfg.task.name,
        )
    if target == "ieforge.tasks.re.RETask":
        from ieforge.tasks.re import RETask

        schema = cfg.task.get("schema", "default")
        if schema in _RE_TASK_SCHEMAS_USING_TASK_LABEL_SPACE:
            label_list = list(cfg.task.label_list)
            negative_label = cfg.task.negative_label
        else:
            label_list = dataset.label_list
            negative_label = dataset.negative_label

        return RETask(
            label_list=label_list,
            entity_representation=cfg.task.get("entity_representation", "dummify"),
            negative_label=negative_label,
            max_length=cfg.task.get("max_length", 512),
            context_strategy=cfg.task.get("context_strategy", "full_text"),
            window_sentences=cfg.task.get("window_sentences", 0),
            schema=schema,
            train_pair_chunk_size=cfg.task.get("train_pair_chunk_size", 0),
            val_pair_chunk_size=cfg.task.get("val_pair_chunk_size", 0),
            train_negative_downsample_ratio=cfg.task.get(
                "train_negative_downsample_ratio", 0
            ),
            train_exclude_negative=cfg.task.get("train_exclude_negative", False),
            name=cfg.task.get("name", "re"),
        )
    if target == "ieforge.tasks.biored.BioREDPairTask":
        cfg_label_list = OmegaConf.select(cfg.task, "label_list", default=None)
        label_list = (
            list(cfg_label_list)
            if cfg_label_list is not None
            else list(dataset.label_list)
        )
        return BioREDPairTask(
            label_list=label_list,
            negative_label=cfg.task.get("negative_label", dataset.negative_label),
            max_length=cfg.task.max_length,
            name=cfg.task.name,
        )
    if target == "ieforge.tasks.qa.QATask":
        return QATask(
            label_list=dataset.label_list,
            max_length=cfg.task.max_length,
            name=cfg.task.name,
        )
    if target == "ieforge.tasks.doc_cls.DocClassificationTask":
        return DocClassificationTask(
            label_list=list(dataset.label_list),
            threshold=cfg.task.threshold,
            max_length=cfg.task.max_length,
            name=cfg.task.name,
        )
    if target == "ieforge.tasks.pico.PICOTask":
        return PICOTask(
            label_list=list(dataset.label_list),
            max_length=cfg.task.max_length,
            name=cfg.task.name,
        )
    if target == "ieforge.tasks.sentence_similarity.SentenceSimilarityTask":
        return SentenceSimilarityTask(
            label_list=list(dataset.label_list),
            max_length=cfg.task.max_length,
            name=cfg.task.name,
        )
    if target == "ieforge.tasks.bioredirect.BioREDirectPairTask":
        from ieforge.tasks.bioredirect import BioREDirectPairTask

        direction_labels = list(cfg.task.get("direction_labels", []))
        return BioREDirectPairTask(
            relation_labels=list(cfg.task.relation_labels),
            novelty_labels=list(cfg.task.novelty_labels),
            direction_labels=direction_labels or None,
            negative_label=cfg.task.get("negative_label", "None"),
            max_length=cfg.task.max_length,
            name=cfg.task.name,
        )
    msg = f"Unsupported task target: {target}"
    raise ValueError(msg)


def _build_collator(cfg: DictConfig, task: Any) -> tuple[Any, int]:
    """Construct the collator for the configured task.

    Returns (collator, num_added_tokens). num_added_tokens is > 0 only when
    the task is RETask with a typed_marker entity_representation, in which case
    entity type marker strings are registered as additional_special_tokens.
    """
    from ieforge.tasks.re import RETask

    raw_tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
    if not isinstance(raw_tokenizer, PreTrainedTokenizerBase):
        msg = (
            f"AutoTokenizer.from_pretrained({cfg.model.tokenizer_name!r}) "
            f"returned {type(raw_tokenizer).__name__}, expected PreTrainedTokenizerBase"
        )
        raise TypeError(msg)
    tokenizer = cast("PreTrainedTokenizerBase", raw_tokenizer)

    num_added_tokens = 0
    if isinstance(task, RETask) and task.entity_representation in {
        "typed_marker",
        "typed_marker_all_mentions",
    }:
        special = task.special_tokens()
        if special:
            num_added_tokens = tokenizer.add_special_tokens(
                {"additional_special_tokens": special}
            )
            if num_added_tokens != len(special):
                log.warning(
                    "Requested %d special tokens but tokenizer registered %d; "
                    "some markers may already exist in the base vocabulary",
                    len(special),
                    num_added_tokens,
                )

    if isinstance(task, BioREDPairTask):
        special = task.special_tokens()
        if special:
            num_added_tokens = tokenizer.add_special_tokens(
                {"additional_special_tokens": special}
            )
            if num_added_tokens != len(special):
                log.warning(
                    "Requested %d BioRED marker tokens but tokenizer registered %d; "
                    "some markers may already exist in the base vocabulary",
                    len(special),
                    num_added_tokens,
                )
        from ieforge.data.collators import BioREDPairCollator

        return BioREDPairCollator(task=task, tokenizer=tokenizer), num_added_tokens

    from ieforge.tasks.bioredirect import BioREDirectPairTask as _BioREDirectPairTask

    if isinstance(task, _BioREDirectPairTask):
        special = task.special_tokens()
        if special:
            num_added_tokens = tokenizer.add_special_tokens(
                {"additional_special_tokens": special}
            )
        from ieforge.data.collators import BioREDirectCollator

        soft_prompt_len = int(getattr(task, "soft_prompt_len", 0))
        return (
            BioREDirectCollator(
                task=task, tokenizer=tokenizer, soft_prompt_len=soft_prompt_len
            ),
            num_added_tokens,
        )
    if isinstance(task, SpanNERTask):
        from ieforge.data.collators import SpanNERCollator

        return SpanNERCollator(task=task, tokenizer=tokenizer), num_added_tokens
    if isinstance(task, NERTask):
        from ieforge.data.collators import NERCollator

        return NERCollator(task=task, tokenizer=tokenizer), num_added_tokens
    if isinstance(task, QATask):
        from ieforge.data.collators import QACollator

        return QACollator(task=task, tokenizer=tokenizer), num_added_tokens
    if isinstance(task, DocClassificationTask):
        from ieforge.data.collators import DocClassificationCollator

        return (
            DocClassificationCollator(task=task, tokenizer=tokenizer),
            num_added_tokens,
        )
    if isinstance(task, PICOTask):
        from ieforge.data.collators import PICOCollator

        return PICOCollator(task=task, tokenizer=tokenizer), num_added_tokens
    if isinstance(task, SentenceSimilarityTask):
        from ieforge.data.collators import SentenceSimilarityCollator

        return (
            SentenceSimilarityCollator(task=task, tokenizer=tokenizer),
            num_added_tokens,
        )

    from ieforge.data.collators import RECollator

    if isinstance(task, RETask):
        return RECollator(task=task, tokenizer=tokenizer), num_added_tokens
    msg = f"Unsupported task type: {type(task)!r}"
    raise ValueError(msg)


def _build_train_collator(task: Any, eval_collator: Any) -> Any:
    """Return a separate train-time collator for RE; otherwise reuse the eval one.

    For ``RETask`` we construct a new collator with ``is_training=True`` so that
    ``RETask.encode`` can apply train-time negative downsampling. All other tasks
    reuse the same collator instance for train and eval.
    """
    from ieforge.data.collators import RECollator
    from ieforge.tasks.re import RETask

    if isinstance(task, RETask) and isinstance(eval_collator, RECollator):
        return RECollator(
            task=task,
            tokenizer=eval_collator.tokenizer,
            is_training=True,
        )
    return eval_collator


_KNOWLEDGE_COLLATOR_CONTROL_KEYS = frozenset(
    {"enabled", "train_enabled", "eval_enabled"}
)


def _wrap_knowledge_collator(cfg: DictConfig, collator: Any, *, split_role: str) -> Any:
    """Optionally wrap a collator with the configured knowledge collator."""
    collator_cfg = OmegaConf.select(cfg, "knowledge.collator", default=None)
    if collator_cfg is None or not bool(collator_cfg.get("enabled", False)):
        return collator

    if split_role == "train":
        split_enabled = bool(collator_cfg.get("train_enabled", True))
    elif split_role in {"eval", "validation", "test"}:
        split_enabled = bool(collator_cfg.get("eval_enabled", False))
    else:
        msg = f"Unsupported collator split_role: {split_role!r}"
        raise ValueError(msg)
    if not split_enabled:
        return collator

    raw = OmegaConf.to_container(collator_cfg, resolve=True)
    assert isinstance(raw, dict)
    instantiation_cfg = strip_names(
        {
            key: value
            for key, value in raw.items()
            if key not in _KNOWLEDGE_COLLATOR_CONTROL_KEYS
        }
    )
    if "_target_" not in instantiation_cfg:
        msg = "knowledge.collator must include a _target_ when enabled"
        raise ValueError(msg)
    dataset_name = OmegaConf.select(
        cfg, "dataset.name", default=None
    ) or OmegaConf.select(cfg, "dataset.task", default=None)
    instance_metadata = (
        {"dataset": dataset_name, "split": split_role}
        if dataset_name is not None
        else None
    )
    return instantiate(
        OmegaConf.create(instantiation_cfg),
        base_collator=collator,
        instance_metadata=instance_metadata,
    )


_METRIC_CONFIG_KEYS = frozenset({"name", "monitor", "monitor_mode", "monitor_label"})


def _metric_instantiation_cfg(cfg: DictConfig) -> DictConfig:
    """Return a copy of cfg.metric without trainer-only metadata fields."""
    raw = cast("dict", OmegaConf.to_container(cfg.metric, resolve=True))
    filtered = {k: v for k, v in raw.items() if k not in _METRIC_CONFIG_KEYS}
    return OmegaConf.create(filtered)


def _build_metric(cfg: DictConfig, dataset: Any) -> Any:
    """Construct the configured metric with task-specific dataset metadata."""
    metric_cfg = _metric_instantiation_cfg(cfg)
    target = metric_cfg.get("_target_")

    if cfg.task.get("_target_") == "ieforge.tasks.re.RETask":
        schema = cfg.task.get("schema", "default")
        if target in _RE_METRIC_TARGETS_USING_NEGATIVE_LABEL:
            negative_label = dataset.negative_label
            if schema in _RE_TASK_SCHEMAS_USING_TASK_LABEL_SPACE:
                negative_label = cfg.task.negative_label
            return instantiate(metric_cfg, negative_label=negative_label)

    if target == "ieforge.metrics.pico_macro_f1.PICOMacroF1":
        return instantiate(metric_cfg, label_list=list(dataset.label_list))
    return instantiate(metric_cfg)


@hydra.main(version_base=None, config_path=_CONFIGS_DIR, config_name="train")
def train(cfg: DictConfig) -> None:
    """Train an IE model using the resolved Hydra config."""
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    _validate_task_dataset_compatibility(cfg)
    if "seed" in cfg:
        seed_everything(cfg.seed, workers=True)

    train_dataset_cfg = OmegaConf.select(cfg, "train_dataset", default=None)
    train_dataset = _build_dataset(
        train_dataset_cfg if train_dataset_cfg is not None else cfg.dataset,
        "train",
        root_cfg=cfg,
    )
    task = _build_task(cfg, train_dataset)
    base_collator, num_added_tokens = _build_collator(cfg, task)
    train_collator = _build_train_collator(task, base_collator)
    train_collator = _wrap_knowledge_collator(cfg, train_collator, split_role="train")
    val_collator = _wrap_knowledge_collator(cfg, base_collator, split_role="validation")
    test_collator = _wrap_knowledge_collator(cfg, base_collator, split_role="test")
    datamodule = _build_datamodule(
        cfg,
        val_collator,
        ["train", "validation", "test"],
        train_collator=train_collator,
        test_collator=test_collator,
    )
    # train() always requests all three splits, so _build_datamodule always
    # returns the full IEDataModule (never the test-only IEEvalDataModule).
    assert isinstance(datamodule, IEDataModule)
    model = instantiate(
        _model_instantiation_cfg(cfg.model),
        label_list=task.label_list,
        extra_token_count=num_added_tokens,
    )
    loss_fn = instantiate(cfg.loss)
    metrics = _build_metric(cfg, train_dataset)

    optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
    assert isinstance(optimizer_cfg, dict)

    scheduler_cfg: dict | None = None
    if "scheduler" in cfg:
        num_steps = _compute_training_steps(cfg, len(datamodule.train_dataset))
        raw = OmegaConf.to_container(cfg.scheduler, resolve=True)
        assert isinstance(raw, dict)
        raw["num_training_steps"] = num_steps
        scheduler_cfg = raw
        log.info(
            "Scheduler: %d warmup / %d total steps",
            scheduler_cfg["num_warmup_steps"],
            num_steps,
        )

    accumulate_grad_batches = int(cfg.trainer.get("accumulate_grad_batches", 1))
    lightning_module = IELightningModule(
        model=model,
        task=task,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    logger = (
        instantiate(cfg.logger) if "logger" in cfg and cfg.logger is not None else True
    )
    # IELightningModule applies accumulate_grad_batches manually because it uses
    # manual optimization, which Lightning's Trainer.accumulate_grad_batches does
    # not support. Force the trainer's own value back to 1 via instantiate
    # overrides so Lightning does not raise a MisconfigurationException; the real
    # gradient accumulation is already wired into IELightningModule above.
    trainer_overrides: dict[str, Any] = {
        "logger": logger,
        "accumulate_grad_batches": 1,
    }
    trainer = instantiate(cfg.trainer, **trainer_overrides)
    trainer.fit(lightning_module, datamodule=datamodule)
    if not bool(getattr(trainer, "fast_dev_run", False)):
        trainer.test(lightning_module, datamodule=datamodule, ckpt_path="best")


@hydra.main(version_base=None, config_path=_CONFIGS_DIR, config_name="eval")
def eval_cmd(cfg: DictConfig) -> None:
    """Evaluate a trained IE model using the resolved Hydra config."""
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    _validate_task_dataset_compatibility(cfg)

    test_dataset = _build_dataset(cfg.dataset, "test", root_cfg=cfg)
    task = _build_task(cfg, test_dataset)
    collator, num_added_tokens = _build_collator(cfg, task)
    collator = _wrap_knowledge_collator(cfg, collator, split_role="test")
    datamodule = _build_datamodule(cfg, collator, ["test"])
    model = instantiate(
        _model_instantiation_cfg(cfg.model),
        label_list=task.label_list,
        extra_token_count=num_added_tokens,
    )
    metrics = _build_metric(cfg, test_dataset)

    loss_fn = instantiate(cfg.loss)

    raw_opt = (
        OmegaConf.to_container(cfg.optimizer, resolve=True)
        if "optimizer" in cfg
        else {
            "_target_": "torch.optim.AdamW",
            "lr": 2e-5,
        }
    )
    assert isinstance(raw_opt, dict)
    optimizer_cfg = raw_opt
    lightning_module = IELightningModule(
        model=model,
        task=task,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer_cfg=optimizer_cfg,
    )

    logger = (
        instantiate(cfg.logger) if "logger" in cfg and cfg.logger is not None else True
    )
    trainer: pl.Trainer = instantiate(cfg.trainer, logger=logger)
    trainer.test(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    prediction_output = OmegaConf.select(cfg, "prediction_output", default=None)
    if prediction_output is not None and hasattr(metrics, "last_predictions"):
        write_biored_predictions_jsonl(metrics.last_predictions, prediction_output)


if __name__ == "__main__":
    import sys

    argv = " ".join(sys.argv)
    if "--config-name=train_re" in argv:
        train()
    elif "--config-name=eval_re" in argv or "--config-name=eval" in argv:
        eval_cmd()
    else:
        train()
