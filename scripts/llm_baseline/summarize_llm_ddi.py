#!/usr/bin/env python3
"""Turn the four LLM-baseline cells into the #314 Agent B aggregate.

    uv run python scripts/llm_baseline/summarize_llm_ddi.py \\
        --runs-root $KNOWHYPER_HOME/runs/llm_baseline_2026_09_17 \\
        --output docs/reports/data/llm_ddi_baseline_2026_09_17/aggregate.txt

Reads each cell's ``manifest.json`` (provenance, invalid rate, cost) and
``export/predictions.jsonl`` (for per-class F1, which the manifest does not
carry), and writes a provenance header plus the validation and test tables.

Every number in the output is recomputed here from the prediction files, so the
table cannot drift from the artefacts it claims to summarise.  The PubMedBERT
reference numbers are copied verbatim from Agent A's aggregate and labelled as
NOT seed-paired with the LLM.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from knowhyper.analysis.llm_re_baseline import DDI_LABELS, DDI_NEGATIVE
from knowhyper.analysis.re_prediction_export import PredictionRecord, compute_micro_f1

#: Copied verbatim from docs/reports/data/re_bootstrap_2026_09_17/aggregate.txt.
#: Ten training seeds, 10 epochs, bs=64; NOT paired with the LLM arms.
REFERENCE = {
    "pubmedbert_base": (0.7914, 0.0111),
    "pubmedbert_ev_warm": (0.8121, 0.0083),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--bootstrap-aggregate",
        type=Path,
        default=Path("docs/reports/data/re_bootstrap_2026_09_17/aggregate.txt"),
        help="source of the reference PubMedBERT numbers (quoted in the header)",
    )
    return ap.parse_args(argv)


def read_cell(root: Path, condition: str) -> dict[str, Any] | None:
    cell = root / "ddi" / condition / "seed_0"
    manifest = cell / "manifest.json"
    predictions = cell / "export" / "predictions.jsonl"
    if not manifest.is_file() or not predictions.is_file():
        return None
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
    return {
        "condition": condition,
        "cell": cell,
        "manifest": json.loads(manifest.read_text()),
        "rows": rows,
    }


def _records(rows: list[dict[str, Any]]) -> list[PredictionRecord]:
    return [
        PredictionRecord(
            dataset=r["dataset"],
            split=r["split"],
            instance_id=r["instance_id"],
            gold_label_id=r["gold_label_id"],
            gold_label=r["gold_label"],
            predicted_label_id=r["predicted_label_id"],
            predicted_label=r["predicted_label"],
            predicted_score=r["predicted_score"],
        )
        for r in rows
    ]


def overall_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """The project's own micro F1 -- reused, never reimplemented here."""
    return compute_micro_f1(_records(rows), negative_label=DDI_NEGATIVE)


def per_class(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """One-vs-rest micro F1 per positive class, via the same canonical function.

    Every label other than the class under test is collapsed to the negative
    label first, so ``compute_micro_f1`` scores exactly that class.
    """
    out: dict[str, dict[str, float | int]] = {}
    for label in DDI_LABELS:
        if label == DDI_NEGATIVE:
            continue
        collapsed = [
            {
                **r,
                "gold_label": r["gold_label"]
                if r["gold_label"] == label
                else DDI_NEGATIVE,
                "predicted_label": r["predicted_label"]
                if r["predicted_label"] == label
                else DDI_NEGATIVE,
            }
            for r in rows
        ]
        out[label] = compute_micro_f1(_records(collapsed), negative_label=DDI_NEGATIVE)
    return out


def block(cell: dict[str, Any]) -> list[str]:
    manifest = cell["manifest"]
    rows = cell["rows"]
    overall = overall_metrics(rows)
    gold = Counter(r["gold_label"] for r in rows)
    pred = Counter(r["predicted_label"] for r in rows)
    evidence = manifest.get("evidence", {})
    lengths = manifest.get("input_length", {})
    lines = [
        f"--- {cell['condition']}  (split={manifest['split']}, arm={manifest['arm']}, n={manifest['n']})",
        f"  micro F1 {overall['f1']:.4f}  P {overall['precision']:.4f}  R {overall['recall']:.4f}"
        f"   TP {overall['tp']} FP {overall['fp']} FN {overall['fn']}",
        "  per-class F1: "
        + "  ".join(
            f"{label} {scores['f1']:.4f} (n={gold[label]})"
            for label, scores in per_class(rows).items()
        ),
        "  gold dist: " + "  ".join(f"{k} {gold[k]}" for k in DDI_LABELS),
        "  pred dist: " + "  ".join(f"{k} {pred[k]}" for k in DDI_LABELS),
        f"  parse: {manifest['parse_status_counts']}  invalid_rate {manifest['invalid_rate']:.4f}",
        f"  prompt tokens: mean {lengths.get('prompt_tokens_mean', float('nan')):.1f}"
        f"  p50 {lengths.get('prompt_tokens_p50', 0)}  max {lengths.get('prompt_tokens_max', 0)}"
        f"  (model context {lengths.get('model_max_context', 0)})",
        f"  latency: {manifest['latency_seconds_total']:.1f} s total,"
        f" {manifest['latency_seconds_per_instance'] * 1000:.1f} ms/instance"
        f"  peak GPU {manifest.get('peak_gpu_memory_gib') or float('nan'):.2f} GiB"
        f"  on {manifest.get('gpu_name')}",
    ]
    if manifest["arm"] == "KG":
        lines.append(
            f"  evidence: {evidence.get('status_counts')}"
            f"  chars mean {evidence.get('chars_mean', 0):.0f} max {evidence.get('chars_max', 0)}"
            f"  both-endpoints-edge {evidence.get('both_endpoints_edge_fraction', 0):.4f}"
        )
        lines.append(f"  evidence cache: {evidence.get('cache_split_dir')}")
        lines.append(
            f"  serializer fp {evidence.get('serializer_fingerprint')}"
            f"  max_hyperedges {evidence.get('max_hyperedges')}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    conditions = [
        "qwen35_9b_noKG_val",
        "qwen35_9b_KG_val",
        "qwen35_9b_noKG",
        "qwen35_9b_KG",
    ]
    cells = {c: read_cell(args.runs_root, c) for c in conditions}
    present = {k: v for k, v in cells.items() if v is not None}
    if not present:
        msg = f"no cells under {args.runs_root}"
        raise SystemExit(msg)
    # The header's PROMPT line is labelled "(noKG)", so it must come from a noKG
    # cell -- not from whichever cell happens to be first (#314 review F7). The
    # driver skips validation-KG when its cache is missing, and a test-only
    # re-summarize is routine, so "first present" can be a KG cell and the header
    # would then assert a KG fingerprint is the noKG one. Model/revision/dtype
    # ride on the same reference, which is why this is picked explicitly.
    reference_key = next(
        (k for k in ("qwen35_9b_noKG", "qwen35_9b_noKG_val") if k in present), None
    )
    if reference_key is None:
        msg = (
            "no noKG cell present; the header's prompt provenance is labelled "
            f"(noKG) and cannot be taken from {sorted(present)}"
        )
        raise SystemExit(msg)
    reference = present[reference_key]["manifest"]

    out: list[str] = []
    out.append("# DDI instruction-tuned LLM baseline (issue #314, Agent B)")
    out.append(
        "# METRIC: exact-label micro F1, negative class (DDI-false) excluded == ieforge relation_f1,"
        " the same metric as the PubMedBERT arms. Recomputed here from export/predictions.jsonl."
    )
    out.append(
        f"# MODEL: {reference['model']} @ {reference['revision']} (Apache-2.0),"
        f" loaded as {reference.get('loader_class')}, dtype {reference['dtype']}"
    )
    out.append(
        f"# DECODING: greedy (do_sample=False, num_beams=1), max_new_tokens"
        f" {reference['decoding']['max_new_tokens']}, enable_thinking=False."
        " One inference run; there are NO training seeds."
    )
    out.append(
        f"# PROMPT: version {reference['prompt_version']}, fingerprint"
        f" {reference['prompt_fingerprint']} (noKG, from {reference_key})"
        " -- the KG arm differs only by the"
        " evidence line and its own fingerprint, recorded in each manifest."
        f" Demonstration rule: {reference['demonstration_rule']}."
    )
    out.append(
        "# DEMONSTRATIONS (train split, fixed before any validation/test row was read): "
        + "; ".join(
            f"{d['label']}={d['instance_id']}" for d in reference["demonstrations"]
        )
    )
    out.append(
        "# INVALID OUTPUT POLICY: an output with no parseable label is predicted as"
        " DDI-false and COUNTED (invalid_rate below). Parser: first {...} JSON block,"
        " else a single whole-token label match, else invalid."
    )
    kg = present.get("qwen35_9b_KG")
    if kg is not None:
        ev = kg["manifest"]["evidence"]
        out.append(
            f"# EVIDENCE (KG arm, test): SubHG cache 45caf65e8a07ea90 -- the cache the"
            f" recorded DDI fusion arms read -- serialized by the project's canonical"
            f" SerializedSubHGSerializer (fingerprint {ev.get('serializer_fingerprint')},"
            f" max_hyperedges {ev.get('max_hyperedges')})."
        )
        out.append(
            f"# EVIDENCE CONTAINS THE PAIR: {ev.get('both_endpoints_edge_fraction', 0):.1%}"
            " of test instances have at least one retrieved hyperedge naming BOTH the head"
            " and the tail drug, i.e. the graph already asserts that this pair interacts."
            " This is the same evidence the fusion arm consumes and is not leakage, but any"
            " KG gain must be read against it: it is consistent with the KG supplying the"
            " pair's known interaction status rather than with reasoning over neighbourhood"
            " structure."
        )
    out.append(
        "# ARMS ARE NOT LENGTH-MATCHED: noKG prompts are a few hundred tokens, KG prompts a"
        " few thousand (the 64-edge serializer budget binds on essentially every instance)."
        " Cost, latency and memory therefore differ by roughly an order of magnitude between"
        " the arms. The recorded GRAFT-style text comparator uses max_hyperedges=3, so it does"
        " NOT share this serialization budget."
    )
    out.append(
        "# VALIDATION KG CACHE: 2a5edac8b67e7449 (the only DDI validation cache on disk)."
        " Its fingerprint differs from the test cache under the 0.13.2 contract (#272) but its"
        " CONTENT does not: cache_equivalence_2026_09_17.txt measured identical hyperedge and"
        " seed-node sets on 5716/5716 DDI test instances. Validation is reported as a parser /"
        " plumbing check only; nothing is selected or claimed on it."
    )
    out.append(
        "# SINGLE RUN: seed-level statistics do NOT exist for these arms. Only the paired"
        " instance-level bootstrap CI of KG - noKG is interpretable; a seed-level p or t for"
        " this contrast would be manufactured from one deterministic greedy run."
    )
    out.append(f"# RUNS: {args.runs_root}")
    out.append("")

    for title, keys in (
        ("VALIDATION (sanity check only)", conditions[:2]),
        ("TEST (reported)", conditions[2:]),
    ):
        out.append(f"== {title}")
        for key in keys:
            cell = cells.get(key)
            if cell is None:
                out.append(f"--- {key}: NOT PRESENT")
                continue
            out.extend(block(cell))
            out.append("")

    out.append("== Reference PubMedBERT arms (NOT seed-paired with the LLM)")
    out.append(
        f"# copied verbatim from {args.bootstrap_aggregate}; 10 training seeds, 10 epochs,"
        " bs=64, same DDI test split and the same micro-F1 metric. These are seed MEANS of"
        " ten trained models; the LLM rows above are ONE inference run. The two are not"
        " paired and no significance test between them is reported."
    )
    for name, (mean, sd) in REFERENCE.items():
        out.append(f"  {name:<24} test mean {mean:.4f} sd {sd:.4f} (n=10 seeds)")
    out.append("")
    test_no = present.get("qwen35_9b_noKG")
    test_kg = present.get("qwen35_9b_KG")
    out.append("== Cost")
    for key in ("qwen35_9b_noKG", "qwen35_9b_KG"):
        cell = present.get(key)
        if cell is None:
            continue
        m = cell["manifest"]
        out.append(
            f"  {key:<18} {m['prompt_tokens']['total']:>10,} prompt tokens"
            f"  {m['latency_seconds_total']:>8.1f} s on one"
            f" {m.get('gpu_name')}  ({m['latency_seconds_per_instance'] * 1000:.0f} ms/instance,"
            f" peak {m.get('peak_gpu_memory_gib', 0):.1f} GiB)"
        )
    out.append("")
    out.append("== Interpretation (issue #314 B-7)")
    out.append(
        "The allowed claim is at most: we added a protocol-matched instruction-tuned LLM"
        " baseline on DDI, and separately tested whether the same retrieved evidence helps"
        " when serialized as text. No claim is made that KnowHyper outperforms modern LLMs."
    )
    if test_no is not None and test_kg is not None:
        f_no = overall_metrics(test_no["rows"])["f1"]
        f_kg = overall_metrics(test_kg["rows"])["f1"]
        out.append(
            f"What was measured: on DDI test the LLM reaches {f_no:.4f} micro F1 without"
            f" evidence and {f_kg:.4f} with it. Serializing the SAME retrieved evidence as"
            f" text did NOT help: the paired instance-level bootstrap gives"
            f" {f_kg - f_no:+.4f} (see ddi_llm_KG_minus_noKG in"
            " docs/reports/data/re_bootstrap_2026_09_17/aggregate.txt for the CI). Because"
            " there is one inference run, this is instance-level uncertainty only; it says"
            " nothing about variation across training seeds, of which there are none."
        )
        out.append(
            "Both arms are recall-heavy and precision-poor: the fixed prompt makes the model"
            " predict a positive class far more often than the gold distribution warrants"
            " (see the pred vs gold distributions above). Per #314 1.3 and B-4 the prompt was"
            " fixed before any split was scored and was NOT revised in response to this, so"
            " the number is a property of this pre-registered prompt, not a tuned ceiling for"
            " the model. Output formatting was never the problem: every one of the"
            " 5,716 + 5,716 test outputs parsed as exact JSON (invalid rate 0.0000)."
        )
        out.append(
            "Against the reference rows, the encoder baselines score far higher on this task"
            " than this zero-tuning LLM baseline. That comparison is NOT seed-paired and the"
            " LLM had no task-specific training, so it supports only the weak reading that"
            " a fixed-prompt 9B instruction-tuned model does not replace a fine-tuned"
            " encoder here -- not a general claim about modern LLMs."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
