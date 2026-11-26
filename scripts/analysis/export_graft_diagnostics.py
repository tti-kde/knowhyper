#!/usr/bin/env python3
"""Export payload-free GRAFT-style gate diagnostics and resource cost (#189).

The end-task delta alone cannot distinguish the two ways this arm can produce a
null:

* the gate LEARNED to ignore evidence (interesting: parallel text encoding does
  not transmit typed structure), versus
* the gate COLLAPSED to a constant at initialization and never read evidence at
  all (uninteresting: the arm never ran the experiment).

Those look identical in aggregate F1 and differ obviously in the per-instance gate
distribution, so the gate is exported per instance rather than summarised. The
same sidecar carries ``source_predicted_label`` beside ``final_predicted_label``,
which is what makes "the evidence branch changed this instance's decision"
countable instead of inferred.

Resource cost is recorded here too, because the issue's comparison is explicitly
NOT parameter-matched: this arm carries a second full encoder. Disclosing the cost
is the alternative to pretending it away.

Payload-free: instance ids, labels, scores, gate values and counts only. No raw
node ids, no serialized evidence text, no licensed payload.

Example:
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run python \
    scripts/analysis/export_graft_diagnostics.py \
    --experiment ddi_graft_style_real --condition graft_real --seed 42 \
    --checkpoint /path/to/best.ckpt --num-workers 8 \
    --output-dir "$KNOWHYPER_HOME/analysis/re/graft_real_seed42" \
    trainer=nolog
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from omegaconf import OmegaConf

from knowhyper.analysis.graft_diagnostics import (
    GraftDiagnosticRecord,
    ResourceReport,
    gate_summary,
    parameter_report,
    write_graft_diagnostics,
)


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=3,
        help="batches excluded from the throughput measurement (cuDNN autotune, "
        "allocator warm-up); they are still scored",
    )
    parser.add_argument("extra", nargs="*", help="Additional Hydra overrides after --")
    args = parser.parse_args()
    if args.num_workers < 4:
        parser.error("--num-workers must be >=4")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def extract_records(
    module: Any,
    loader: Any,
    *,
    dataset: str,
    split: str,
    device: str,
    warmup_batches: int,
) -> tuple[list[GraftDiagnosticRecord], ResourceReport]:
    """Score the split, collecting per-instance gate diagnostics and timing.

    Throughput excludes the first ``warmup_batches`` batches: the first forward
    pass pays cuDNN autotune and allocator growth, which on a short split would
    otherwise dominate and make the two arms' costs incomparable.
    """
    records: list[GraftDiagnosticRecord] = []
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timed_examples = 0
    timed_seconds = 0.0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if module._re_batch_is_empty(batch):
                continue
            batch_device = module.transfer_batch_to_device(
                batch, torch.device(device), dataloader_idx=0
            )
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = module._eval_forward(batch_device)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            final_probs = torch.softmax(outputs.logits.float(), dim=-1)
            final_scores, final_ids = final_probs.max(dim=-1)
            # `source_logits` is the branch a `base`-arm model would have produced,
            # so comparing the two label columns counts decision flips directly.
            source_probs = torch.softmax(outputs.source_logits.float(), dim=-1)
            source_scores, source_ids = source_probs.max(dim=-1)
            gate = outputs.knowledge_gate.float().flatten()

            rows = int(outputs.logits.shape[0])
            if index >= warmup_batches:
                timed_examples += rows
                timed_seconds += elapsed

            present = batch_device.get("knowledge_evidence_present")
            token_count = batch_device.get("knowledge_evidence_token_count")
            retained = batch_device.get("knowledge_evidence_retained_blocks")
            labels = module.task.label_list
            for row, instance_id in enumerate(batch_device["_prediction_instance_ids"]):
                records.append(
                    GraftDiagnosticRecord(
                        dataset=dataset,
                        split=split,
                        instance_id=str(instance_id),
                        source_predicted_label=str(labels[int(source_ids[row])]),
                        source_predicted_score=float(source_scores[row]),
                        final_predicted_label=str(labels[int(final_ids[row])]),
                        final_predicted_score=float(final_scores[row]),
                        knowledge_gate=float(gate[row]),
                        evidence_present=(
                            bool(present[row]) if present is not None else False
                        ),
                        evidence_token_count=(
                            int(token_count[row]) if token_count is not None else 0
                        ),
                        retained_block_count=(
                            int(retained[row]) if retained is not None else 0
                        ),
                    )
                )

    peak_bytes = None
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_bytes = int(torch.cuda.max_memory_allocated())
    return records, ResourceReport(
        parameters=parameter_report(module.model),
        max_memory_allocated_bytes=peak_bytes,
        examples_per_second=(
            timed_examples / timed_seconds if timed_seconds > 0 else None
        ),
        timed_examples=timed_examples,
        timed_seconds=timed_seconds,
        warmup_batches=warmup_batches,
    )


def main() -> None:
    args = parse_args()
    # Reuse the prediction exporter's composition and checkpoint loading verbatim:
    # a second copy could drift and silently score a different dataloader order,
    # which would misalign every instance id in the sidecar.
    from export_re_predictions import compose_components, load_lightning_module

    cfg, task, loader, num_added_tokens = compose_components(args)
    module = load_lightning_module(
        cfg, task, num_added_tokens, args.checkpoint, args.device
    )
    if not hasattr(module.model, "evidence_encoder"):
        msg = (
            f"{type(module.model).__name__} has no evidence_encoder; this exporter "
            "requires a ParallelTextGatedKnowledgeModel arm"
        )
        raise TypeError(msg)

    dataset = str(cfg.dataset.name)
    records, resources = extract_records(
        module,
        loader,
        dataset=dataset,
        split=args.split,
        device=args.device,
        warmup_batches=args.warmup_batches,
    )
    summary = gate_summary(records)
    # The selector budget: on a GRAFT text arm selection happens INSIDE the serializer
    # (subhg_serialization delegates to select_top_ranked), so there is no
    # `subhypergraph_transform` node -- only the graph arms have one. Reading that path
    # alone recorded `selector: null` in every manifest, i.e. the one field that proves
    # the same-evidence contrast held was silently absent from the artifact whose whole
    # purpose is making a null readable. Fall back to the serializer, then refuse a
    # None rather than writing it.
    selector_budget = OmegaConf.select(
        cfg, "knowledge.pipeline.injector.subhypergraph_transform.max_hyperedges"
    )
    if selector_budget is None:
        selector_budget = OmegaConf.select(
            cfg, "knowledge.pipeline.injector.text_serializer.max_hyperedges"
        )
    # The two payloads select DIFFERENT units, so the budget key differs: the
    # serialized arms cap hyperedges (`max_hyperedges`), the definitions arms cap
    # target entities (`max_targets`). Demanding max_hyperedges universally made every
    # text-arm diagnostics export die on this guard. `max_targets` may legitimately be
    # unset on ChemProt/DDI, which have exactly two targets, so record the unit
    # alongside the number rather than raising on an absent cap.
    budget_unit = "hyperedges"
    if selector_budget is None:
        selector_budget = OmegaConf.select(
            cfg, "knowledge.pipeline.injector.text_serializer.max_targets"
        )
        budget_unit = "targets"
    payload = OmegaConf.select(cfg, "knowledge.collator.payload")
    if selector_budget is None and payload != "definitions":
        # Only the definitions payload is allowed an absent cap. A serialized arm with
        # no selection budget means the same-evidence field would be null, which is the
        # one thing this sidecar exists to make auditable.
        msg = (
            "no selection budget resolved for a serialized-payload arm; the manifest "
            "would record null for the field that proves same-evidence parity"
        )
        raise ValueError(msg)
    fingerprints = {
        "selector": selector_budget,
        "selector_unit": budget_unit,
        "serializer_max_hyperedges": OmegaConf.select(
            cfg, "knowledge.pipeline.injector.text_serializer.max_hyperedges"
        ),
        "evidence_mode": OmegaConf.select(cfg, "knowledge.collator.evidence_mode"),
        "max_evidence_tokens": OmegaConf.select(
            cfg, "knowledge.collator.max_evidence_tokens"
        ),
        "retriever": OmegaConf.select(
            cfg, "knowledge.pipeline.injector.retriever._target_"
        ),
    }
    write_graft_diagnostics(
        args.output_dir,
        records,
        manifest={
            "dataset": dataset,
            "split": args.split,
            "experiment": args.experiment,
            "condition": args.condition,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "batch_size": args.batch_size,
            "config": fingerprints,
            "gate_summary": summary,
            "resources": resources.as_dict(),
        },
    )
    print(
        json.dumps(
            {"gate_summary": summary, "resources": resources.as_dict()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
