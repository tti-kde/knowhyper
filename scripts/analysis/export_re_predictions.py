#!/usr/bin/env python3
"""Export payload-free RE predictions and preregistered sensitivity metrics.

Example:
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run python \
    scripts/analysis/export_re_predictions.py \
    --experiment ddi_retrieved_definitions --condition r2a --seed 42 \
    --checkpoint /path/to/best.ckpt --num-workers 8 \
    --overlap-instances /path/to/split_overlap_instances.jsonl \
    --output-dir /path/to/predictions/r2a_seed42 \
    trainer=difflr model.gate_init=1.0
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from knowhyper.analysis.re_prediction_export import (
    ExclusionMasks,
    PredictionBundleManifest,
    PredictionRecord,
    compute_sensitivity_metrics,
    load_overlap_masks,
    sha256_file,
    sha256_json,
    write_prediction_bundle,
)


class IdentityPreservingCollator:
    """Attach stable dataset IDs after the ordinary ieforge collator runs."""

    def __init__(self, collator: Any) -> None:
        self.collator = collator

    def __call__(self, examples: list[Any]) -> dict[str, Any]:
        batch = self.collator(examples)
        if "example_ids" in batch:
            ids = [str(item) for item in batch["example_ids"]]
        else:
            ids = [str(example.id) for example in examples]
        labels = batch.get("labels")
        row_count = (
            int(labels.shape[0]) if isinstance(labels, torch.Tensor) else len(ids)
        )
        if len(ids) != row_count:
            msg = (
                "cannot align prediction IDs with classifier rows: "
                f"ids={len(ids)} rows={row_count}"
            )
            raise ValueError(msg)
        batch["_prediction_instance_ids"] = ids
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overlap-instances", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("extra", nargs="*", help="Additional Hydra overrides after --")
    args = parser.parse_args()
    if args.num_workers < 4:
        parser.error("--num-workers must be >=4")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def compose_components(
    args: argparse.Namespace,
) -> tuple[Any, Any, DataLoader[Any], int]:
    from ieforge import CONFIGS_DIR as IEFORGE_CONFIGS_DIR
    from ieforge.cli._hydra_entry import (
        _build_collator,
        _build_dataset,
        _build_task,
        _wrap_knowledge_collator,
    )

    GlobalHydra.instance().clear()
    configs_dir = Path(__file__).parents[2] / "src" / "knowhyper" / "configs"
    overrides = [
        f"experiment={args.experiment}",
        f"dataset.split={args.split}",
        f"dataset.num_workers={args.num_workers}",
        f"dataset.batch_size={args.batch_size}",
        f"seed={args.seed}",
        f"hydra.searchpath=[file://{configs_dir}]",
        *args.extra,
    ]
    with initialize_config_dir(config_dir=str(IEFORGE_CONFIGS_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=overrides)
    dataset = _build_dataset(cfg.dataset, args.split, root_cfg=cfg)
    task = _build_task(cfg, dataset)
    collator, num_added_tokens = _build_collator(cfg, task)
    collator = _wrap_knowledge_collator(cfg, collator, split_role=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=IdentityPreservingCollator(collator),
        num_workers=args.num_workers,
    )
    return cfg, task, loader, num_added_tokens


def load_lightning_module(
    cfg: Any, task: Any, num_added_tokens: int, checkpoint: Path, device: str
) -> Any:
    from hydra.utils import instantiate
    from ieforge.cli._hydra_entry import (
        _build_metric,
        _model_instantiation_cfg,
    )
    from ieforge.engine.lightning_module import IELightningModule

    model = instantiate(
        _model_instantiation_cfg(cfg.model),
        label_list=task.label_list,
        extra_token_count=num_added_tokens,
    )
    dataset_for_metric = type(
        "_MetricDataset",
        (),
        {"negative_label": task.negative_label, "label_list": task.label_list},
    )()
    metric = _build_metric(cfg, dataset_for_metric)
    module = IELightningModule(
        model=model,
        task=task,
        loss_fn=instantiate(cfg.loss),
        metrics=metric,
        optimizer_cfg={"_target_": "torch.optim.AdamW", "lr": 2e-5},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        msg = "checkpoint must contain a mapping"
        raise TypeError(msg)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        msg = "checkpoint must contain a non-empty Lightning state_dict"
        raise ValueError(msg)
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        msg = "checkpoint state_dict must map string keys to tensors"
        raise TypeError(msg)
    module.load_state_dict(state_dict, strict=True)
    module.eval()
    module.to(device)
    return module


def extract_records(
    module: Any,
    loader: DataLoader[Any],
    *,
    dataset: str,
    split: str,
    device: str,
) -> tuple[list[PredictionRecord], dict[str, float]]:
    records: list[PredictionRecord] = []
    with torch.no_grad():
        for batch in loader:
            if module._re_batch_is_empty(batch):
                continue
            batch_device = module.transfer_batch_to_device(
                batch, torch.device(device), dataloader_idx=0
            )
            outputs = module._eval_forward(batch_device)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probabilities = torch.softmax(logits.float(), dim=-1)
            scores, predicted_ids = probabilities.max(dim=-1)
            gold_ids = batch_device["labels"].tolist()
            instance_ids = batch_device["_prediction_instance_ids"]
            for instance_id, gold_id, predicted_id, score in zip(
                instance_ids,
                gold_ids,
                predicted_ids.tolist(),
                scores.tolist(),
                strict=True,
            ):
                records.append(
                    PredictionRecord(
                        dataset=dataset,
                        split=split,
                        instance_id=str(instance_id),
                        gold_label_id=int(gold_id),
                        gold_label=str(module.task.label_list[gold_id]),
                        predicted_label_id=int(predicted_id),
                        predicted_label=str(module.task.label_list[predicted_id]),
                        predicted_score=float(score),
                    )
                )
            decoded = module.task.decode(outputs, batch_device)
            predictions, references, texts = module._metric_inputs(
                batch_device, decoded
            )
            module.metrics.update(predictions, references, texts)
    official = module.metrics.compute()
    return records, {str(key): float(value) for key, value in official.items()}


def main() -> None:
    args = parse_args()
    cfg, task, loader, num_added_tokens = compose_components(args)
    module = load_lightning_module(
        cfg, task, num_added_tokens, args.checkpoint, args.device
    )
    dataset = str(cfg.dataset.name)
    records, ieforge_metrics = extract_records(
        module,
        loader,
        dataset=dataset,
        split=args.split,
        device=args.device,
    )
    masks = (
        load_overlap_masks(args.overlap_instances)
        if args.overlap_instances
        else ExclusionMasks()
    )
    metrics = compute_sensitivity_metrics(
        records,
        negative_label=str(task.negative_label),
        masks=masks,
    )
    for key in ("precision", "recall", "f1"):
        if abs(float(metrics["official"][key]) - ieforge_metrics[key]) > 1e-12:
            msg = (
                f"export metric does not match ieforge {key}: "
                f"{metrics['official'][key]} != {ieforge_metrics[key]}"
            )
            raise ValueError(msg)

    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    write_prediction_bundle(
        args.output_dir,
        records,
        PredictionBundleManifest(
            dataset=dataset,
            split=args.split,
            experiment=args.experiment,
            condition=args.condition,
            seed=args.seed,
            checkpoint_sha256=sha256_file(args.checkpoint),
            resolved_config_sha256=sha256_json(resolved_config),
            extra_overrides_sha256=sha256_json(list(args.extra)),
            negative_label=str(task.negative_label),
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            overlap_instances_sha256=(
                sha256_file(args.overlap_instances)
                if args.overlap_instances is not None
                else None
            ),
            applied_exclusion_counts={
                "leakage_cleared": len(masks.leakage),
                "pair_disjoint": len(masks.pair_disjoint),
            },
            metrics=metrics,
            ieforge_official_metrics=ieforge_metrics,
        ),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
