"""Payload-free records and summaries for the GRAFT-style arm (#189).

Separated from ``scripts/analysis/export_graft_diagnostics.py`` so the record
schema, the gate summary and the parameter accounting are unit-testable without a
GPU or a checkpoint. The script owns the dataloader and the forward pass; this
module owns what gets written.

Nothing here may carry evidence content. Fields are instance ids, labels, scores,
gate values and counts -- the same class of aggregate the R2a/R2b arms already
publish. Raw node ids and serialized ``[HG]`` text are deliberately absent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

#: Written into the manifest so a later reader can tell which schema produced a
#: sidecar rather than inferring it from which columns happen to be present.
SCHEMA = "graft_style_diagnostics_v1"


@dataclass(frozen=True)
class GraftDiagnosticRecord:
    """One instance's source-only vs gated decision, plus its evidence shape.

    ``source_predicted_label`` is what THIS model's source branch alone predicted, in
    the same forward pass. Holding both columns makes "the evidence branch changed this
    decision" a count rather than an inference across two runs with different seeds.

    It is NOT the `base` arm's decision, and must not be described as one: the source
    branch was trained JOINTLY with the evidence branch and the shared classifier, so
    its weights are not an independently-trained baseline's. The `g=0` guarantee says
    the two agree when evidence is ABSENT, which is a statement about this model, not a
    claim that the branch reproduces `base`. Use the `base` arm itself for that.
    """

    dataset: str
    split: str
    instance_id: str
    source_predicted_label: str
    source_predicted_score: float
    final_predicted_label: str
    final_predicted_score: float
    knowledge_gate: float
    evidence_present: bool
    evidence_token_count: int
    retained_block_count: int

    @property
    def decision_changed(self) -> bool:
        return self.source_predicted_label != self.final_predicted_label


@dataclass(frozen=True)
class ResourceReport:
    """Parameter, memory and throughput cost of the arm.

    The issue's comparison is explicitly NOT parameter-matched -- this arm carries a
    second full encoder -- so the cost is recorded and disclosed rather than
    minimised. ``examples_per_second`` excludes warm-up batches; without that the
    first batch's cuDNN autotune dominates a short split and the arms stop being
    comparable on speed.
    """

    parameters: dict[str, int]
    max_memory_allocated_bytes: int | None
    examples_per_second: float | None
    timed_examples: int
    timed_seconds: float
    warmup_batches: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameters": dict(self.parameters),
            "max_memory_allocated_bytes": self.max_memory_allocated_bytes,
            "examples_per_second": self.examples_per_second,
            "timed_examples": self.timed_examples,
            "timed_seconds": self.timed_seconds,
            "warmup_batches": self.warmup_batches,
        }


def parameter_report(model: Any) -> dict[str, int]:
    """Count total/trainable parameters and the three GRAFT-specific components.

    Component counts are reported separately because "this arm has more
    parameters" is not the interesting disclosure -- WHERE they are is. Nearly all
    of the excess is the second encoder, and the fusion machinery (cross-attention
    + gate) is small; a reader can only check that from a breakdown.
    """
    report = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    components = {
        "source_encoder": getattr(getattr(model, "inner", None), "model", None),
        "evidence_encoder": getattr(model, "evidence_encoder", None),
        "cross_attention": getattr(model, "cross_attn", None),
        "fusion_gate": getattr(model, "gate_mlp", None),
    }
    for label, module in components.items():
        report[label] = (
            sum(p.numel() for p in module.parameters()) if module is not None else 0
        )
    return report


def gate_summary(records: Sequence[GraftDiagnosticRecord]) -> dict[str, Any]:
    """Aggregate the gate distribution and the decision-flip rate.

    ``gate_sd`` over PRESENT-evidence rows is the collapse detector: a gate that
    never learned to read evidence has sd ~0 while still producing a plausible
    mean, and that distinguishes "the model ignored evidence it could see" from
    "the gate was never trained" -- two readings of the same null.
    """
    if not records:
        return {
            "count": 0,
            "evidence_present_count": 0,
            "gate_mean": None,
            "gate_sd": None,
            "gate_min": None,
            "gate_max": None,
            "gate_mean_present": None,
            "gate_sd_present": None,
            "decision_changed_count": 0,
            "decision_changed_rate": None,
        }
    gates = [record.knowledge_gate for record in records]
    present = [record.knowledge_gate for record in records if record.evidence_present]
    changed = sum(1 for record in records if record.decision_changed)
    return {
        "count": len(records),
        "evidence_present_count": len(present),
        "gate_mean": _mean(gates),
        "gate_sd": _sd(gates),
        "gate_min": min(gates),
        "gate_max": max(gates),
        "gate_mean_present": _mean(present),
        "gate_sd_present": _sd(present),
        "decision_changed_count": changed,
        "decision_changed_rate": changed / len(records),
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sd(values: Sequence[float]) -> float | None:
    """Sample standard deviation; None for fewer than two values."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def write_graft_diagnostics(
    output_dir: Path,
    records: Sequence[GraftDiagnosticRecord],
    manifest: dict[str, Any],
) -> None:
    """Write the sidecar and its manifest, records sorted by instance id.

    Sorted so two arms' sidecars can be joined by line order as well as by id, and
    so a diff between re-runs is empty rather than a reordering.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: record.instance_id)
    with (output_dir / "graft_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    payload = {"schema": SCHEMA, "record_count": len(ordered), **manifest}
    (output_dir / "graft_diagnostics_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
