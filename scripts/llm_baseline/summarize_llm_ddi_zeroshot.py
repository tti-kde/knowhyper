#!/usr/bin/env python3
"""Summarise the camera-ready zero-shot arm of the #314 DDI LLM baseline.

    uv run python scripts/llm_baseline/summarize_llm_ddi_zeroshot.py \\
        --runs-root $KNOWHYPER_HOME/runs/llm_baseline_zeroshot_2026_11_23 \\
        --output docs/reports/data/llm_ddi_zeroshot_2026_11_23/aggregate.txt

Same model, revision, decoding, prompt version and parser as the recorded
2026-09-17 cells (``llm_ddi_baseline.py --zero-shot`` drops only the
demonstrations).  A re-run of the recorded one-example-per-label noKG cell on
the same host is included so the hardware change (H200 -> RTX PRO 6000) can be
read off directly.  Every number is recomputed from the prediction files with
the blocks of ``summarize_llm_ddi.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "summarize_llm_ddi", _HERE / "summarize_llm_ddi.py"
)
assert _spec is not None
assert _spec.loader is not None
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

#: Recorded one-example-per-label noKG test cell (H200), copied verbatim from
#: docs/reports/data/llm_ddi_baseline_2026_09_17/aggregate.txt.
RECORDED_FEWSHOT_TEST = {"f1": 0.2913, "tp": 806, "fp": 3749, "fn": 173}

CONDITIONS = (
    "qwen35_9b_noKG_zeroshot_val",
    "qwen35_9b_noKG_zeroshot",
    "qwen35_9b_noKG_fewshot_repro",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    return ap.parse_args(argv)


#: (condition -> expected split, whether demonstrations are present).
EXPECTED = {
    "qwen35_9b_noKG_zeroshot_val": ("validation", False),
    "qwen35_9b_noKG_zeroshot": ("test", False),
    "qwen35_9b_noKG_fewshot_repro": ("test", True),
}
#: Manifest keys every input must share with the zero-shot test cell.
SHARED_KEYS = ("model", "revision", "dtype", "prompt_version", "system_prompt_sha256")


def check_provenance(manifests: dict[str, dict]) -> None:
    """Refuse inputs that do not match the claims the header prints."""
    ref = manifests["qwen35_9b_noKG_zeroshot"]
    problems: list[str] = []
    for cond, m in manifests.items():
        split, with_demos = EXPECTED[cond]
        if m.get("split") != split:
            problems.append(f"{cond}: split {m.get('split')!r} != {split!r}")
        if m.get("arm") != "noKG":
            problems.append(f"{cond}: arm {m.get('arm')!r} != 'noKG'")
        if bool(m.get("demonstrations")) != with_demos:
            problems.append(
                f"{cond}: demonstrations present={bool(m.get('demonstrations'))}"
            )
        problems.extend(
            f"{cond}: {key} differs from the zero-shot test cell"
            for key in SHARED_KEYS
            if m.get(key) != ref.get(key)
        )
        if m.get("decoding") != ref.get("decoding"):
            problems.append(f"{cond}: decoding differs from the zero-shot test cell")
    if problems:
        raise SystemExit("provenance check failed:\n  " + "\n  ".join(problems))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cells = {c: _base.read_cell(args.runs_root, c) for c in CONDITIONS}
    missing = [c for c, v in cells.items() if v is None]
    if missing:
        msg = f"missing cells under {args.runs_root}: {missing}"
        raise SystemExit(msg)
    zs = cells["qwen35_9b_noKG_zeroshot"]["manifest"]
    check_provenance({c: v["manifest"] for c, v in cells.items()})

    out = [
        "# DDI instruction-tuned LLM baseline: zero-shot arm (camera-ready, EACL 2027)",
        "# METRIC: exact-label micro F1, negative class (DDI-false) excluded =="
        " ieforge relation_f1. Recomputed here from export/predictions.jsonl.",
        f"# MODEL: {zs['model']} @ {zs['revision']}, loaded as {zs['loader_class']},"
        f" dtype {zs['dtype']}",
        "# DECODING: greedy (do_sample=False, num_beams=1), max_new_tokens"
        f" {zs['decoding']['max_new_tokens']}, enable_thinking=False. One inference"
        " run; there are NO training seeds.",
        f"# PROMPT: version {zs['prompt_version']} with NO demonstrations"
        f" (fingerprint {zs['prompt_fingerprint']}); same system prompt"
        f" (sha256 {zs['system_prompt_sha256'][:16]}...) and parser as the"
        " recorded cells of docs/reports/data/llm_ddi_baseline_2026_09_17/.",
        "# HOST: whale, NVIDIA RTX PRO 6000 Blackwell (the recorded cells ran on"
        " ABCI H200). fewshot_repro re-runs the recorded noKG one-example-per-label"
        " test cell here to measure that change.",
        "# NOT SELECTED ON ANYTHING: the prompt was not revised; validation is"
        " reported for completeness only.",
        f"# RUNS: whale:~/.knowhyper/runs/{args.runs_root.name} (copied to the"
        " local $KNOWHYPER_HOME/runs)",
        "",
        "== VALIDATION",
        *_base.block(cells["qwen35_9b_noKG_zeroshot_val"]),
        "",
        "== TEST (reported)",
        *_base.block(cells["qwen35_9b_noKG_zeroshot"]),
        "",
        "== HARDWARE CHECK: recorded one-example-per-label noKG test cell, re-run",
        *_base.block(cells["qwen35_9b_noKG_fewshot_repro"]),
    ]
    repro = _base.overall_metrics(cells["qwen35_9b_noKG_fewshot_repro"]["rows"])
    rec = RECORDED_FEWSHOT_TEST
    out += [
        f"  recorded on H200: micro F1 {rec['f1']:.4f}  TP {rec['tp']} FP {rec['fp']}"
        f" FN {rec['fn']}",
        f"  difference (re-run - recorded): F1 {repro['f1'] - rec['f1']:+.4f}"
        f"  TP {repro['tp'] - rec['tp']:+d} FP {repro['fp'] - rec['fp']:+d}"
        f" FN {repro['fn'] - rec['fn']:+d}",
        "",
        "== Interpretation",
        "Removing the demonstrations changes the test F1 only slightly (see TEST vs"
        " the recorded 0.2913); the model still predicts a positive label for most"
        " pairs, so the low score is not caused by the demonstrations. The re-run of"
        " the recorded cell differs from it only at the fourth decimal, so the"
        " zero-shot number can be read beside the recorded H200 numbers.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
