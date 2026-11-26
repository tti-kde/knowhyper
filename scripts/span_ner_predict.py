#!/usr/bin/env python
"""Export ieforge span-NER entities and candidate lattices as prediction JSONL.

Train predictions must be out-of-fold. Validation/test predictions must come
from the corresponding full-train checkpoint. The output is accepted by
``NERPredictionQueryBuilder(mention_source="candidates")``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, cast

import torch
from ieforge import CONFIGS_DIR as IEFORGE_CONFIGS_DIR
from torch.utils.data import DataLoader

from knowhyper.ieforge.span_predictions import span_candidates_from_logits
from knowhyper.ieforge.span_proposal_provenance import (
    ProposalProvenance,
    assert_checkpoint_matches_backbone,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _PredictionCollator:
    def __init__(self, base_collator: Any) -> None:
        self.base_collator = base_collator

    def __call__(self, examples: list[Any]) -> dict[str, Any]:
        batch = self.base_collator(examples)
        batch["id"] = [str(example.id) for example in examples]
        return batch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Span NER prediction export")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--searchpath", required=True)
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "ieforge/knowhyper model config for the first pass, e.g. "
            "pubmedbert_span_classifier or biolinkbert_span_classifier. Candidate "
            "JSONL is backbone-specific: it must match the checkpoint and must not "
            "be reused across backbones."
        ),
    )
    parser.add_argument(
        "--experiment",
        default="span_ner",
        help="knowledge-free span experiment used to compose the first pass",
    )
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--candidate-top-k", type=int, default=16)
    parser.add_argument("--oof-num-folds", type=int)
    parser.add_argument("--oof-fold", type=int)
    parser.add_argument("--fold-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    oof_requested = args.oof_num_folds is not None or args.oof_fold is not None
    if oof_requested and (args.oof_num_folds is None or args.oof_fold is None):
        msg = "--oof-num-folds and --oof-fold must be supplied together"
        raise ValueError(msg)
    requested_splits = [split.strip() for split in args.splits.split(",")]
    if oof_requested and requested_splits != ["train"]:
        msg = "OOF prediction requires --splits train"
        raise ValueError(msg)

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    overrides = [
        f"experiment={args.experiment}",
        f"dataset={args.dataset}",
        f"model={args.model}",
        f"hydra.searchpath=[file://{args.searchpath}]",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(IEFORGE_CONFIGS_DIR)):
        cfg = compose(config_name="train", overrides=overrides)

    datamodule_keys = frozenset({"batch_size", "num_workers", "name", "task_family"})
    model_meta_keys = frozenset({"tokenizer_name"})

    def dataset_config(raw_cfg: Any) -> Any:
        raw = OmegaConf.to_container(raw_cfg, resolve=True)
        if not isinstance(raw, dict):
            msg = "dataset config must resolve to a mapping"
            raise TypeError(msg)
        return OmegaConf.create(
            {key: value for key, value in raw.items() if key not in datamodule_keys}
        )

    train_dataset = instantiate(dataset_config(cfg.dataset), split="train")
    from ieforge.tasks.span_ner import SpanNERTask

    task = SpanNERTask(
        label_list=train_dataset.label_list,
        max_length=cfg.task.max_length,
        max_span_width=cfg.task.max_span_width,
        threshold=cfg.task.threshold,
        decoder=cfg.task.decoder,
        name=cfg.task.name,
    )

    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    tokenizer = cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(cfg.model.tokenizer_name),
    )
    raw_model = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(raw_model, dict):
        msg = "model config must resolve to a mapping"
        raise TypeError(msg)
    model_cfg = OmegaConf.create(
        {key: value for key, value in raw_model.items() if key not in model_meta_keys}
    )
    model = instantiate(model_cfg, label_list=task.label_list)

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Fail closed before loading: a checkpoint from another backbone loads cleanly
    # (the span head shapes agree across BERT-base encoders) and writes a plausible
    # but wrong candidate lattice.
    provenance = ProposalProvenance(
        model_config=args.model,
        tokenizer_name=str(cfg.model.tokenizer_name),
        encoder_name=str(cfg.model.model_name),
        vocab_size=int(model.encoder.get_input_embeddings().weight.shape[0]),
    )
    assert_checkpoint_matches_backbone(state_dict, provenance=provenance)

    prefix = "model."
    model.load_state_dict(
        {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    from ieforge.data.collators import SpanNERCollator

    collator = _PredictionCollator(SpanNERCollator(task=task, tokenizer=tokenizer))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, Any]] = {}

    for split in requested_splits:
        if oof_requested:
            from knowhyper.ieforge.ner_oof import NERFoldDataset

            dataset = NERFoldDataset(
                task=str(cfg.dataset.task),
                split="train",
                num_folds=args.oof_num_folds,
                fold=args.oof_fold,
                fold_seed=args.fold_seed,
                role="holdout",
            )
        else:
            dataset = instantiate(dataset_config(cfg.dataset), split=split)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collator,
            shuffle=False,
            num_workers=args.num_workers,
        )

        path = output_dir / f"{split}.jsonl"
        count = 0
        with torch.no_grad(), path.open("w") as writer:
            for batch in loader:
                device_batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                outputs = model(device_batch)
                decoded = task.decode(outputs, device_batch)
                for row, (example_id, text, entities) in enumerate(
                    zip(batch["id"], batch["text"], decoded, strict=True)
                ):
                    candidates = span_candidates_from_logits(
                        outputs.logits[row],
                        text=text,
                        word_char_spans=batch["word_char_spans"][row],
                        entity_types=task.entity_types,
                        decoded_entities=entities,
                        top_k=args.candidate_top_k,
                    )
                    record = {
                        "id": str(example_id),
                        "text": text,
                        "entities": [
                            {
                                "start": entity.start,
                                "end": entity.end,
                                "text": entity.text,
                                "label": entity.label,
                            }
                            for entity in entities
                        ],
                        "candidates": [candidate.to_dict() for candidate in candidates],
                    }
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        digest = _sha256(path)
        written[split] = {"rows": count, "sha256": digest}
        logger.info("Wrote %d records to %s (sha256 %s)", count, path, digest[:16])

    # The SubHG namespace is scoped by these digests, so the manifest is not
    # optional bookkeeping: without it a cache cannot be tied back to the exact
    # proposal distribution that produced it.
    manifest = {
        "dataset": str(cfg.dataset.task),
        "experiment": args.experiment,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "candidate_top_k": args.candidate_top_k,
        "splits": written,
        "oof": (
            {
                "num_folds": args.oof_num_folds,
                "fold": args.oof_fold,
                "fold_seed": args.fold_seed,
            }
            if oof_requested
            else None
        ),
        **provenance.to_dict(),
    }
    manifest_path = output_dir / "predictions.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote provenance manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
