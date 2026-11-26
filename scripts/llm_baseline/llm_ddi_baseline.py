#!/usr/bin/env python3
"""Run the protocol-matched instruction-tuned LLM baseline on DDI (#314, Agent B).

    uv run python scripts/llm_baseline/llm_ddi_baseline.py \\
        --arm KG --split test \\
        --cache-split-dir $KNOWHYPER_HOME/cache/subhg/v2/45caf65e8a07ea90/ddi/test \\
        --output-root $KNOWHYPER_HOME/runs/llm_baseline_2026_09 --condition qwen35_9b_KG

Writes ``<output-root>/ddi/<condition>/seed_0/{export/predictions.jsonl,
raw_outputs.jsonl, manifest.json, cell.json}`` -- the same layout the matrix
cells use, so ``scripts/bootstrap/re_paired_bootstrap.py`` can pair the arms.

The prompt, demonstrations, decoding and invalid-output policy are fixed in
``knowhyper.analysis.llm_re_baseline``; this script only executes them and
records provenance (model revision, prompt fingerprint, token counts, latency,
peak memory, invalid rate).  Nothing here is selected on the split being
scored.  Run ``--split validation`` first to confirm the parser and the
evidence plumbing; ``--split test`` once.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ieforge.knowledge.core.schema import Query

from knowhyper.analysis.llm_re_baseline import (
    DDI_LABELS,
    DDI_NEGATIVE,
    PromptSpec,
    annotate_pair_roles,
    cache_entry_path,
    label_id_order,
    parse_label,
    prediction_record,
    recorded_instance_id,
    render_messages,
    select_demonstrations,
)
from knowhyper.analysis.re_prediction_export import compute_micro_f1
from knowhyper.cache.store import subhypergraph_from_dict
from knowhyper.pipeline.subhg_serialization import (
    SerializedSubHGSerializer,
)

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--arm", choices=("noKG", "KG"), required=True)
    ap.add_argument("--split", choices=("validation", "test"), required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument(
        "--cache-split-dir", type=Path, help="SubHG cache split dir (KG arm)"
    )
    ap.add_argument("--max-hyperedges", type=int, default=64)
    ap.add_argument(
        "--zero-shot",
        action="store_true",
        help="no demonstrations (camera-ready zero-shot arm); default keeps the "
        "recorded one-example-per-label prompt",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N rows")
    ap.add_argument(
        "--no-sort-by-length",
        dest="sort_by_length",
        action="store_false",
        help="disable length-bucketed batching (wall-clock only; greedy outputs are unchanged)",
    )
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32")
    )
    args = ap.parse_args(argv)
    if args.arm == "KG" and args.cache_split_dir is None:
        ap.error("--cache-split-dir is required for the KG arm")
    return args


def load_split(split: str) -> tuple[list[Any], Any]:
    """Return ``(examples, dataset)``; the dataset carries ``label_list``."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from ieforge import CONFIGS_DIR as IEFORGE_CONFIGS_DIR
    from ieforge.cli._hydra_entry import _build_dataset

    GlobalHydra.instance().clear()
    configs_dir = Path(__file__).resolve().parents[3] / "src" / "knowhyper" / "configs"
    with initialize_config_dir(config_dir=str(IEFORGE_CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=ddi_baseline",
                f"dataset.split={split}",
                f"hydra.searchpath=[file://{configs_dir}]",
            ],
        )
    dataset = _build_dataset(cfg.dataset, split, root_cfg=cfg)
    return [dataset[i] for i in range(len(dataset))], dataset


def load_examples(split: str) -> list[Any]:
    """Examples only, for callers that do not need the label order."""
    return load_split(split)[0]


def _has_both_endpoints_edge(sub: Any) -> bool:
    """Does any RETRIEVED hyperedge contain both endpoint nodes?

    Such an edge means the graph asserts that this very pair interacts -- most
    of the DDI decision.  It is the same evidence the fusion arm consumes, so it
    is not leakage, but its frequency has to be reported alongside any KG gain
    rather than left for a reviewer to find.

    Computed from ``sub.hyperedges`` and the annotated endpoint NODE IDS, not
    from the serialized string (#314 review F3).  The renderer shows at most
    ``max_members`` members per edge and elides the rest as ``+N more``, so a
    text scan silently missed every edge whose tail sat past that cap and the
    reported fraction was a lower bound rather than a measurement.  The rendered
    view is budgeted output; the evidence is the hyperedge.
    """
    roles = {
        str(record.get("node_id")): record.get("pair_role")
        for record in sub.provenance.get("linked_mentions", [])
        if record.get("node_id") is not None
    }
    heads = {node for node, role in roles.items() if role == "head"}
    tails = {node for node, role in roles.items() if role == "tail"}
    if not heads or not tails:
        return False
    return any(
        (set(edge.node_ids) & heads) and (set(edge.node_ids) & tails)
        for edge in sub.hyperedges
    )


def _has_both_endpoints_rendered(text: str) -> bool:
    """The OLD text-based test, kept so the two numbers can be compared.

    Retained only to report ``both_endpoints_rendered_fraction`` beside the
    corrected figure, making the correction visible in the record instead of a
    number silently changing between revisions.
    """
    return any(
        "[HEAD_ENTITY]" in block and "[TAIL_ENTITY]" in block
        for block in text.split("[HG]")
    )


def evidence_for(
    example: Any, cache_split_dir: Path, serializer: SerializedSubHGSerializer
) -> tuple[str, str, Any]:
    """Return ``(evidence_text, status, annotated_sub)``.

    The annotated sub-hypergraph is returned alongside the rendered text because
    the both-endpoints diagnostic must be computed from the hyperedges, which
    the rendered text only partially shows (#314 review F3). ``None`` where
    there is nothing to inspect.
    """
    path = cache_entry_path(cache_split_dir, str(example.id))
    if path is None:
        return "", "missing", None
    sub = subhypergraph_from_dict(json.loads(path.read_text(encoding="utf-8")))
    if not sub.hyperedges:
        return "", "empty", None
    head = example.entities[example.head_idx]
    tail = example.entities[example.tail_idx]
    annotated = annotate_pair_roles(sub, head.mention or [], tail.mention or [])
    return (
        serializer.serialize(Query(text=str(example.text)), annotated).text,
        "ok",
        annotated,
    )


def load_model(
    model_id: str, revision: str, device: str, dtype_name: str
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoTokenizer

    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, padding_side="left"
    )
    if tokenizer is None:
        msg = f"no tokenizer for {model_id}@{revision}"
        raise RuntimeError(msg)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore[union-attr]
    # Pick the Auto class from the checkpoint's own declared architecture rather
    # than guessing.  Qwen3.5-9B declares ``Qwen3_5ForConditionalGeneration``: it
    # is a multimodal checkpoint, so ``AutoModelForCausalLM`` (which maps
    # ``qwen3_5`` to the text-only ``Qwen3_5ForCausalLM``) cannot even be
    # constructed from the top-level config -- verified offline against the
    # pinned revision, where ForConditionalGeneration matches all 760 module
    # keys and CausalLM raises on the missing ``vocab_size``.  We only ever feed
    # it text; the vision tower is loaded and unused.
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
    )

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    architectures = list(getattr(config, "architectures", None) or [])
    if any(a.endswith("ForConditionalGeneration") for a in architectures):
        loaders = [AutoModelForImageTextToText, AutoModelForCausalLM]
    else:
        loaders = [AutoModelForCausalLM, AutoModelForImageTextToText]
    model = None
    errors: list[str] = []
    for loader in loaders:
        try:
            model = loader.from_pretrained(model_id, revision=revision, dtype=dtype)
            break
        except Exception as exc:  # reported below if every loader fails
            errors.append(f"{loader.__name__}: {type(exc).__name__}: {exc}")
    if model is None:
        msg = (
            f"could not load {model_id}@{revision} (architectures {architectures!r}); "
            + " | ".join(errors)
        )
        raise RuntimeError(msg)
    model.to(device)
    model.eval()
    return tokenizer, model


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def generate_batch(
    tokenizer: Any, model: Any, prompts: list[str], *, max_new_tokens: int, device: str
) -> tuple[list[str], list[int]]:
    import torch

    # truncation is explicitly off: a prompt that does not fit must fail, not be
    # silently shortened.  main() asserts every prompt fits before generating.
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=False
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[:, encoded["input_ids"].shape[1] :]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    return texts, [int(n) for n in prompt_lengths]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    train = load_examples("train")
    demos = () if args.zero_shot else select_demonstrations(train)
    del train
    spec = PromptSpec(demonstrations=demos, with_evidence=args.arm == "KG")
    examples, dataset = load_split(args.split)
    # Export ids follow ieforge's own label_list, not the prompt order, so this
    # predictions.jsonl encodes labels exactly as every recorded RE export does.
    export_labels = label_id_order(dataset)
    if set(export_labels) != set(DDI_LABELS):
        msg = (
            f"dataset label_list {export_labels!r} does not cover the prompt "
            f"labels {DDI_LABELS!r}"
        )
        raise ValueError(msg)
    if args.limit is not None:
        examples = examples[: args.limit]
    serializer = (
        SerializedSubHGSerializer(max_hyperedges=args.max_hyperedges)
        if args.arm == "KG"
        else None
    )

    tokenizer, model = load_model(args.model, args.revision, args.device, args.dtype)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    prompts: list[str] = []
    evidence_status: dict[str, int] = {"ok": 0, "empty": 0, "missing": 0, "n/a": 0}
    evidence_chars: list[int] = []
    both_endpoints = 0
    both_endpoints_rendered = 0
    for example in examples:
        evidence = None
        if serializer is not None and args.cache_split_dir is not None:
            evidence, status, annotated = evidence_for(
                example, args.cache_split_dir, serializer
            )
            evidence_status[status] += 1
            evidence_chars.append(len(evidence))
            if annotated is not None:
                both_endpoints += int(_has_both_endpoints_edge(annotated))
            both_endpoints_rendered += int(_has_both_endpoints_rendered(evidence))
        else:
            evidence_status["n/a"] += 1
        prompts.append(
            render_prompt(
                tokenizer, render_messages(spec, str(example.text), evidence or None)
            )
        )

    # Fail before generating if any prompt exceeds the model's context.  With
    # truncation off an overflow would otherwise surface as a CUDA error or as
    # silently mangled input; the KG arm sits near the 64-edge serializer cap on
    # essentially every instance, so the margin is worth asserting explicitly.
    prompt_lengths = [
        len(ids) for ids in tokenizer(prompts, truncation=False).input_ids
    ]
    max_context = int(
        getattr(model.config, "max_position_embeddings", 0)
        or getattr(
            getattr(model.config, "text_config", None), "max_position_embeddings", 0
        )
        or getattr(tokenizer, "model_max_length", 0)
    )
    budget = max_context - args.max_new_tokens
    if max_context > 0 and max(prompt_lengths) >= budget:
        msg = (
            f"longest prompt is {max(prompt_lengths)} tokens but the context budget is "
            f"{budget} ({max_context} - {args.max_new_tokens} new tokens); "
            "refusing to run with truncation disabled"
        )
        raise ValueError(msg)

    # A KG arm pointed at the wrong cache directory would otherwise run to
    # completion as a silent second copy of the noKG arm.  Empty sub-HGs are
    # legitimate (DDI has them); a *missing* file is not.
    if serializer is not None and evidence_status["missing"]:
        msg = (
            f"{evidence_status['missing']}/{len(examples)} instances have no entry in "
            f"{args.cache_split_dir}; refusing to score a KG arm against the wrong cache"
        )
        raise ValueError(msg)

    # Batch prompts of similar length together so a batch is not dominated by
    # left padding to its longest member.  Greedy decoding is per-sequence, so
    # this changes only wall-clock, not the outputs -- and the results are put
    # back in dataset order below before anything is scored.
    order = (
        sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        if args.sort_by_length
        else list(range(len(prompts)))
    )
    started = time.perf_counter()
    outputs_by_index: dict[int, str] = {}
    tokens_by_index: dict[int, int] = {}
    for start in range(0, len(order), args.batch_size):
        idx = order[start : start + args.batch_size]
        texts, lengths = generate_batch(
            tokenizer,
            model,
            [prompts[i] for i in idx],
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        for i, text, length in zip(idx, texts, lengths, strict=True):
            outputs_by_index[i] = text
            tokens_by_index[i] = length
        done = min(start + args.batch_size, len(order))
        if done % (args.batch_size * 20) == 0 or done == len(order):
            print(f"  {done}/{len(order)}", flush=True)
    elapsed = time.perf_counter() - started
    outputs = [outputs_by_index[i] for i in range(len(prompts))]
    prompt_tokens = [tokens_by_index[i] for i in range(len(prompts))]

    records = []
    raw_rows = []
    statuses: dict[str, int] = {"json": 0, "substring": 0, "invalid": 0}
    for example, output in zip(examples, outputs, strict=True):
        label, status = parse_label(output, DDI_LABELS)
        statuses[status] += 1
        predicted = label if label is not None else DDI_NEGATIVE
        instance_id = recorded_instance_id(str(example.id))
        records.append(
            prediction_record(
                instance_id=instance_id,
                gold_label=str(example.label),
                predicted_label=predicted,
                labels=export_labels,
                valid=label is not None,
                split=args.split,
            )
        )
        raw_rows.append({"instance_id": instance_id, "output": output, "parse": status})
    ids = [r.instance_id for r in records]
    if len(set(ids)) != len(ids):
        msg = "instance ids are not unique after stripping the pair suffix"
        raise ValueError(msg)
    metrics = compute_micro_f1(records, negative_label=DDI_NEGATIVE)

    cell = args.output_root / "ddi" / args.condition / "seed_0"
    export = cell / "export"
    export.mkdir(parents=True, exist_ok=True)
    with (export / "predictions.jsonl").open("w") as fh:
        for record in records:
            fh.write(json.dumps(dataclasses.asdict(record), sort_keys=True) + "\n")
    with (cell / "raw_outputs.jsonl").open("w") as fh:
        for row in raw_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    peak = (
        torch.cuda.max_memory_allocated() / 2**30
        if args.device.startswith("cuda")
        else None
    )
    manifest = {
        "arm": args.arm,
        "split": args.split,
        "condition": args.condition,
        "model": args.model,
        "revision": args.revision,
        "dtype": args.dtype,
        "loader_class": type(model).__name__,
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": False,
            "batch_size": args.batch_size,
            "sort_by_length": args.sort_by_length,
            "sort_by_length_note": (
                "prompts are batched in ascending prompt-length order to cut left "
                "padding, then restored to dataset order; greedy decoding is "
                "per-sequence so the predictions do not depend on batching"
            ),
        },
        "seeds": {
            "n_runs": 1,
            "note": (
                "the LLM baseline is inference-only: there are no training seeds, "
                "so seed-level statistics over this arm are not meaningful and only "
                "the instance-level (paired bootstrap) CI should be reported"
            ),
        },
        "prompt_version": spec.version,
        "prompt_fingerprint": spec.fingerprint(),
        "system_prompt_sha256": hashlib.sha256(spec.system_prompt.encode()).hexdigest(),
        "demonstrations": [dataclasses.asdict(d) for d in demos],
        "demonstration_rule": (
            "none (zero-shot)"
            if args.zero_shot
            else "first train example per label in dataset order, <= 320 chars"
        ),
        "evidence": {
            "cache_split_dir": str(args.cache_split_dir)
            if args.cache_split_dir
            else None,
            "serializer_fingerprint": serializer.fingerprint if serializer else None,
            "max_hyperedges": args.max_hyperedges if serializer else None,
            "status_counts": evidence_status,
            "chars_mean": statistics.fmean(evidence_chars) if evidence_chars else None,
            "chars_max": max(evidence_chars) if evidence_chars else None,
            "both_endpoints_edge_fraction": (
                both_endpoints / len(examples) if serializer is not None else None
            ),
            "both_endpoints_rendered_fraction": (
                both_endpoints_rendered / len(examples)
                if serializer is not None
                else None
            ),
            "both_endpoints_edge_note": (
                "both_endpoints_edge_fraction: fraction of instances whose RETRIEVED "
                "evidence contains at least one hyperedge whose member set holds both "
                "the head and the tail drug node, i.e. the graph already asserts that "
                "this pair interacts. Same evidence the fusion arm consumes; report "
                "any KG gain against this number. "
                "both_endpoints_rendered_fraction is the superseded text-based count "
                "(#314 review F3): the serializer renders at most max_members members "
                "per edge and elides the rest as '+N more', so scanning the rendered "
                "string missed every edge whose tail sat past that cap. It is a lower "
                "bound, kept only so the correction is visible."
            )
            if serializer is not None
            else None,
        },
        "input_length": {
            "prompt_tokens_mean": statistics.fmean(prompt_lengths),
            "prompt_tokens_p50": statistics.median(prompt_lengths),
            "prompt_tokens_max": max(prompt_lengths),
            "model_max_context": max_context,
            "arms_not_length_matched_note": (
                "the two arms are NOT matched on input length: noKG prompts are a few "
                "hundred tokens, KG prompts a few thousand, because the KG arm appends "
                "up to max_hyperedges serialized hyperedges. Cost, latency and memory "
                "therefore differ by roughly an order of magnitude between arms."
            ),
            "max_hyperedges_note": (
                "max_hyperedges matches the fusion arm's retrieval top_k (64); the "
                "recorded GRAFT-style text comparator uses 3, so the two text arms do "
                "NOT share a serialization budget"
            ),
        },
        "n": len(records),
        "limit": args.limit,
        "parse_status_counts": statuses,
        "invalid_rate": statuses["invalid"] / len(records) if records else None,
        "metrics_official": metrics,
        "prompt_tokens": {
            "mean": statistics.fmean(prompt_tokens),
            "max": max(prompt_tokens),
            "total": sum(prompt_tokens),
        },
        "latency_seconds_total": elapsed,
        "latency_seconds_per_instance": elapsed / len(records) if records else None,
        "peak_gpu_memory_gib": peak,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(0)
        if args.device.startswith("cuda")
        else None,
    }
    (cell / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (cell / "cell.json").write_text(
        json.dumps(
            {
                "dataset": "ddi",
                "method": args.condition,
                "seed": 0,
                "split": args.split,
                "test_f1": metrics["f1"] if args.split == "test" else None,
                f"{args.split}_f1": metrics["f1"],
                "status": "completed",
                "export_dir": str(export),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "metrics_official",
                    "parse_status_counts",
                    "invalid_rate",
                    "latency_seconds_total",
                    "peak_gpu_memory_gib",
                )
            },
            indent=2,
        )
    )
    print(f"wrote {cell}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
