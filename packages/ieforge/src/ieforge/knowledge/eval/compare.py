from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from ieforge.knowledge.eval.schema import (
    KnowledgeEvalError,
    load_report_json,
)


def compare_reports(baseline: Path, candidate: Path) -> dict[str, Any]:
    """Compare shared numeric summary metrics from two knowledge eval reports."""
    baseline_report = load_report_json(baseline)
    candidate_report = load_report_json(candidate)

    baseline_metrics = baseline_report.summary.metrics
    candidate_metrics = candidate_report.summary.metrics
    shared_metric_names = set(baseline_metrics) & set(candidate_metrics)

    metrics: dict[str, dict[str, float]] = {}
    for name in sorted(shared_metric_names):
        baseline_value = baseline_metrics[name]
        candidate_value = candidate_metrics[name]
        if baseline_value is None or candidate_value is None:
            continue
        baseline_number = _require_numeric_metric(
            baseline_value,
            f"{baseline.name}: summary.metrics.{name}",
        )
        candidate_number = _require_numeric_metric(
            candidate_value,
            f"{candidate.name}: summary.metrics.{name}",
        )
        metrics[name] = {
            "baseline": baseline_number,
            "candidate": candidate_number,
            "delta": candidate_number - baseline_number,
        }

    baseline_example_ids = {example.id for example in baseline_report.examples}
    candidate_example_ids = {example.id for example in candidate_report.examples}

    return {
        "baseline": baseline.name,
        "candidate": candidate.name,
        "baseline_num_examples": baseline_report.summary.num_examples,
        "candidate_num_examples": candidate_report.summary.num_examples,
        "shared_num_examples": len(baseline_example_ids & candidate_example_ids),
        "metrics": metrics,
    }


def write_comparison_json(comparison: dict[str, Any], output: Path) -> None:
    """Write a report comparison as deterministic JSON."""
    try:
        contents = (
            json.dumps(comparison, allow_nan=False, indent=2, sort_keys=True) + "\n"
        )
    except ValueError as exc:
        msg = f"invalid comparison JSON: {exc}"
        raise KnowledgeEvalError(msg) from exc
    output.write_text(contents, encoding="utf-8")


def _require_numeric_metric(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{field_name} must be numeric or null"
        raise KnowledgeEvalError(msg)
    return float(value)
