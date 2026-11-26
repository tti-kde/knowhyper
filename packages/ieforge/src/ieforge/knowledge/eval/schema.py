from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path


class KnowledgeEvalError(ValueError):
    """Raised for invalid knowledge eval inputs or reports."""


@dataclass(frozen=True)
class KnowledgeEvalExample:
    """Single knowledge evaluation input example loaded from JSONL."""

    id: str
    query: str
    expected_fact_ids: list[str] = field(default_factory=list)
    expected_text: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeEvalArtifacts:
    """Artifact presence flags recorded for one evaluated example."""

    has_text_context: bool
    has_prompt_embeds: bool
    has_prompt_attention_mask: bool
    graph_feature_keys: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class KnowledgeEvalExampleResult:
    """Evaluation output for one knowledge evaluation example."""

    id: str
    query: str
    expected_fact_ids: list[str]
    retrieved_fact_ids: list[str]
    metrics: dict[str, float | None]
    artifacts: KnowledgeEvalArtifacts
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeEvalSummary:
    """Aggregate counts and metrics for a knowledge evaluation report."""

    num_examples: int
    metrics: dict[str, float | None]
    num_scored_examples: dict[str, int]


@dataclass(frozen=True)
class KnowledgeEvalReport:
    """Serializable knowledge evaluation report."""

    summary: KnowledgeEvalSummary
    examples: list[KnowledgeEvalExampleResult]


_MISSING = object()


def load_eval_examples(path: Path) -> list[KnowledgeEvalExample]:
    """Load and validate knowledge evaluation examples from a JSONL file."""
    examples: list[KnowledgeEvalExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"line {line_no}: invalid JSON: {exc.msg}"
                raise KnowledgeEvalError(msg) from exc
            examples.append(_example_from_jsonl_payload(payload, line_no))
    return examples


def report_to_dict(report: KnowledgeEvalReport) -> dict[str, Any]:
    """Convert a knowledge evaluation report to a JSON-compatible dictionary."""
    return {
        "summary": {
            "num_examples": report.summary.num_examples,
            "metrics": dict(report.summary.metrics),
            "num_scored_examples": dict(report.summary.num_scored_examples),
        },
        "examples": [
            {
                "id": example.id,
                "query": example.query,
                "expected_fact_ids": list(example.expected_fact_ids),
                "retrieved_fact_ids": list(example.retrieved_fact_ids),
                "metrics": dict(example.metrics),
                "artifacts": {
                    "has_text_context": example.artifacts.has_text_context,
                    "has_prompt_embeds": example.artifacts.has_prompt_embeds,
                    "has_prompt_attention_mask": (
                        example.artifacts.has_prompt_attention_mask
                    ),
                    "graph_feature_keys": list(example.artifacts.graph_feature_keys),
                },
                "metadata": dict(example.metadata),
            }
            for example in report.examples
        ],
    }


def report_from_dict(payload: dict[str, Any]) -> KnowledgeEvalReport:
    """Build a knowledge evaluation report from a JSON-compatible dictionary."""
    report_payload = _require_mapping(payload, "report")
    summary_payload = _require_mapping(
        _required_field(report_payload, "summary", "report"), "report.summary"
    )
    examples_payload = _require_list(
        _required_field(report_payload, "examples", "report"), "report.examples"
    )

    return KnowledgeEvalReport(
        summary=_summary_from_dict(summary_payload),
        examples=[
            _example_result_from_dict(
                _require_mapping(example, f"report.examples[{index}]"),
                f"report.examples[{index}]",
            )
            for index, example in enumerate(examples_payload)
        ],
    )


def write_report_json(report: KnowledgeEvalReport, path: Path) -> None:
    """Write a knowledge evaluation report to an indented JSON file."""
    try:
        contents = (
            json.dumps(
                report_to_dict(report),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except ValueError as exc:
        msg = f"invalid report JSON: {exc}"
        raise KnowledgeEvalError(msg) from exc
    path.write_text(contents, encoding="utf-8")


def load_report_json(path: Path) -> KnowledgeEvalReport:
    """Load a knowledge evaluation report from a JSON file."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        msg = f"invalid report JSON: {exc.msg}"
        raise KnowledgeEvalError(msg) from exc
    return report_from_dict(_require_mapping(payload, "report"))


def _example_from_jsonl_payload(
    payload: object,
    line_no: int,
) -> KnowledgeEvalExample:
    if not isinstance(payload, dict):
        msg = f"line {line_no}: row must be a JSON object"
        raise KnowledgeEvalError(msg)

    payload_dict = cast("dict[str, Any]", payload)
    id_value = _required_string(payload_dict, "id", line_no)
    query = _required_string(payload_dict, "query", line_no)
    expected_fact_ids = _optional_string_list(
        payload_dict, "expected_fact_ids", line_no
    )
    expected_text = _optional_string_list(payload_dict, "expected_text", line_no)
    metadata = payload_dict.get("metadata", {})
    if not isinstance(metadata, dict):
        msg = f"line {line_no}: metadata must be an object"
        raise KnowledgeEvalError(msg)

    return KnowledgeEvalExample(
        id=id_value,
        query=query,
        expected_fact_ids=expected_fact_ids,
        expected_text=expected_text,
        metadata=dict(metadata),
    )


def _required_string(payload: dict[str, Any], field_name: str, line_no: int) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        msg = f"line {line_no}: {field_name} must be a string"
        raise KnowledgeEvalError(msg)
    return value


def _optional_string_list(
    payload: dict[str, Any],
    field_name: str,
    line_no: int,
) -> list[str]:
    value = payload.get(field_name, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        msg = f"line {line_no}: {field_name} must be a list of strings"
        raise KnowledgeEvalError(msg)
    return cast("list[str]", list(value))


def _summary_from_dict(payload: dict[str, Any]) -> KnowledgeEvalSummary:
    return KnowledgeEvalSummary(
        num_examples=_require_int(
            _required_field(payload, "num_examples", "report.summary"),
            "report.summary.num_examples",
        ),
        metrics=_require_metrics(
            _required_field(payload, "metrics", "report.summary"),
            "report.summary.metrics",
        ),
        num_scored_examples=_require_int_mapping(
            _required_field(payload, "num_scored_examples", "report.summary"),
            "report.summary.num_scored_examples",
        ),
    )


def _example_result_from_dict(
    payload: dict[str, Any],
    context: str,
) -> KnowledgeEvalExampleResult:
    artifacts_context = f"{context}.artifacts"
    artifacts_payload = _require_mapping(
        _required_field(payload, "artifacts", context),
        artifacts_context,
    )
    metadata = _optional_mapping(payload, "metadata", context)
    return KnowledgeEvalExampleResult(
        id=_require_string(_required_field(payload, "id", context), f"{context}.id"),
        query=_require_string(
            _required_field(payload, "query", context),
            f"{context}.query",
        ),
        expected_fact_ids=_require_string_list(
            _required_field(payload, "expected_fact_ids", context),
            f"{context}.expected_fact_ids",
        ),
        retrieved_fact_ids=_require_string_list(
            _required_field(payload, "retrieved_fact_ids", context),
            f"{context}.retrieved_fact_ids",
        ),
        metrics=_require_metrics(
            _required_field(payload, "metrics", context),
            f"{context}.metrics",
        ),
        artifacts=KnowledgeEvalArtifacts(
            has_text_context=_require_bool(
                _required_field(
                    artifacts_payload, "has_text_context", artifacts_context
                ),
                f"{artifacts_context}.has_text_context",
            ),
            has_prompt_embeds=_require_bool(
                _required_field(
                    artifacts_payload, "has_prompt_embeds", artifacts_context
                ),
                f"{artifacts_context}.has_prompt_embeds",
            ),
            has_prompt_attention_mask=_require_bool(
                _required_field(
                    artifacts_payload,
                    "has_prompt_attention_mask",
                    artifacts_context,
                ),
                f"{artifacts_context}.has_prompt_attention_mask",
            ),
            graph_feature_keys=_require_string_list(
                _required_field(
                    artifacts_payload, "graph_feature_keys", artifacts_context
                ),
                f"{artifacts_context}.graph_feature_keys",
            ),
        ),
        metadata=metadata,
    )


def _required_field(
    payload: dict[str, Any],
    field_name: str,
    context: str,
) -> object:
    value = payload.get(field_name, _MISSING)
    if value is _MISSING:
        msg = f"{context}.{field_name} is required"
        raise KnowledgeEvalError(msg)
    return value


def _require_mapping(payload: object, field_name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        msg = f"{field_name} must be an object"
        raise KnowledgeEvalError(msg)
    return cast("dict[str, Any]", payload)


def _optional_mapping(
    payload: dict[str, Any],
    field_name: str,
    context: str,
) -> dict[str, Any]:
    value = payload.get(field_name, {})
    if not isinstance(value, dict):
        msg = f"{context}.{field_name} must be an object"
        raise KnowledgeEvalError(msg)
    return dict(value)


def _require_list(payload: object, field_name: str) -> list[Any]:
    if not isinstance(payload, list):
        msg = f"{field_name} must be a list"
        raise KnowledgeEvalError(msg)
    return payload


def _require_string(payload: object, field_name: str) -> str:
    if not isinstance(payload, str):
        msg = f"{field_name} must be a string"
        raise KnowledgeEvalError(msg)
    return payload


def _require_string_list(payload: object, field_name: str) -> list[str]:
    if not isinstance(payload, list) or not all(
        isinstance(item, str) for item in payload
    ):
        msg = f"{field_name} must be a list of strings"
        raise KnowledgeEvalError(msg)
    return cast("list[str]", list(payload))


def _require_bool(payload: object, field_name: str) -> bool:
    if not isinstance(payload, bool):
        msg = f"{field_name} must be a bool"
        raise KnowledgeEvalError(msg)
    return payload


def _require_int(payload: object, field_name: str) -> int:
    if isinstance(payload, bool) or not isinstance(payload, int):
        msg = f"{field_name} must be an integer"
        raise KnowledgeEvalError(msg)
    return payload


def _reject_json_constant(value: str) -> None:
    msg = f"invalid report JSON constant: {value}"
    raise KnowledgeEvalError(msg)


def _require_finite_metric(payload: float, field_name: str) -> float:
    try:
        value = float(payload)
    except OverflowError as exc:
        msg = f"{field_name} must be finite"
        raise KnowledgeEvalError(msg) from exc
    if not math.isfinite(value):
        msg = f"{field_name} must be finite"
        raise KnowledgeEvalError(msg)
    return value


def _require_metrics(payload: object, field_name: str) -> dict[str, float | None]:
    mapping = _require_mapping(payload, field_name)
    metrics: dict[str, float | None] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            msg = f"{field_name} keys must be strings"
            raise KnowledgeEvalError(msg)
        if value is None:
            metrics[key] = None
        elif isinstance(value, bool) or not isinstance(value, int | float):
            msg = f"{field_name}.{key} must be numeric or null"
            raise KnowledgeEvalError(msg)
        else:
            metric_value = _require_finite_metric(value, f"{field_name}.{key}")
            metrics[key] = metric_value
    return metrics


def _require_int_mapping(payload: object, field_name: str) -> dict[str, int]:
    mapping = _require_mapping(payload, field_name)
    counts: dict[str, int] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            msg = f"{field_name} keys must be strings"
            raise KnowledgeEvalError(msg)
        counts[key] = _require_int(value, f"{field_name}.{key}")
    return counts
