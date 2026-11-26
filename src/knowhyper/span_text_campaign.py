"""Cell plan for the #220 span-level Knowledge-as-text campaign (Table 6).

Deliberately separate from `knowhyper.span_campaign`: that module's axis is
condition x control (`edge`/`ev`/`ev_warm` x `real`/`shuffled`/`zero`) for the
graph-encoder route. #220's axis is a three-arm text route, and #220 names #218
the canonical producer of proposals and caches -- shared INPUTS, separate plans.

| paper (Table 6)     | arm         | config family                |
|---------------------|-------------|------------------------------|
| Text-only backbone  | `base`      | `span_ner_base*`             |
| Knowledge as text   | `text_real` | `span_ner_retrieved_text*`   |
| sham control        | `text_sham` | `span_ner_text_sham*`        |

The primary contrast is `text_real - text_sham` (content), NOT `text_real - base`
(total effect). The sham holds mention surface, segment, boundary tags and payload
token count fixed and masks only the semantic body, so the difference isolates what
the retrieved text SAYS from the fact that a second segment exists at all.

`base` is a first-class arm here rather than a borrowed one. #218's `shared_zero`
returns the same logits but still builds cross_attn/gate/span_gate/encoder and
hands ~11.8M scalars to a 10x-LR optimizer group, so it is not a matched
comparator -- see `tests/test_span_ner_text_baseline_gate.py` for the measurement.
That same asymmetry is why `trainer_for` exists: a `base` cell has no
`model.inner.` prefix, so `difflr` would train its ENTIRE backbone at 10x.
"""

from __future__ import annotations

from dataclasses import dataclass

# Reuse #218's backbone -> model-config map so the two routes cannot drift onto
# different tokenizers for the same backbone name.
from knowhyper.span_campaign import BACKBONES, DATASETS, GPUS_PER_NODE

__all__ = [
    "ARMS",
    "BACKBONES",
    "DATASETS",
    "GPUS_PER_NODE",
    "SpanTextCampaignError",
    "SpanTextCell",
    "expand",
    "experiment_for",
    "pack",
    "trainer_for",
]

# Ordered so `base` is planned (and dispatched) first: if a node dies partway, the
# shared comparator is the arm most expensive to be missing.
ARMS: tuple[str, ...] = ("base", "text_real", "text_sham")

_DATASET_SUFFIX: dict[str, str] = {
    "bc5cdr_chem": "",
    "bc2gm": "_bc2gm",
    "bc5cdr_disease": "_bc5cdr_disease",
    "jnlpba": "_jnlpba",
    "ncbi_disease": "_ncbi_disease",
}

_ARM_STEM: dict[str, str] = {
    "base": "span_ner_base",
    "text_real": "span_ner_retrieved_text",
    "text_sham": "span_ner_text_sham",
}

# EVERY arm here takes `benchmark_nolog`. None of them may take `difflr`.
#
# The text route has NO knowledge branch: its configs set `injector.encoder: null`
# and `integrator: null`, and all three arms instantiate the same plain
# `ieforge.models.hf_span_classifier.HFSpanClassifier`. Retrieved text enters as a
# second segment built by the collator, not through a cross-attention adapter, so
# there is no separate module for a differential LR to target.
#
# `difflr` splits on `BACKBONE_PREFIX = "model.inner."`, which exists only on the
# fusion wrapper. A plain classifier's parameters are `model.encoder.*`, so the
# backbone group comes out EMPTY and all 216 tensors (130.7M scalars) land in the
# "knowledge" group at 10x the base LR -- 2e-4 instead of 2e-5 on a pretrained
# BERT. Measured, not inferred: `build_param_groups` returns
# `[(216, 2e-05)]` at mult=1 and `[(216, 0.0002)]` at mult=10.
#
# That is what Phase 2 ran, and it is the most likely cause of the text arms
# collapsing to the empty prediction on some seeds while `base` -- identical
# model, identical data, 1x LR -- never did.
#
# `tests/test_span_ner_text_baseline_gate.py` already stated this corollary ("a
# plain span baseline has no 'inner.' prefix, so difflr would train its entire
# backbone at 10x -- it must run with trainer=nolog"); the text arms were
# nevertheless assigned difflr by classifying them on intent ("knowledge arm")
# rather than on what they instantiate. Classify by the model, not the label.
_ARM_TRAINER: dict[str, str] = {
    "base": "benchmark_nolog",
    "text_real": "benchmark_nolog",
    "text_sham": "benchmark_nolog",
}


class SpanTextCampaignError(RuntimeError):
    """A cell selection that must not be submitted."""


def _check_arm(arm: str) -> None:
    if arm not in _ARM_STEM:
        msg = f"unknown arm {arm!r}; the #220 text campaign covers {list(ARMS)}"
        raise SpanTextCampaignError(msg)


def _check_dataset(dataset: str) -> None:
    if dataset not in _DATASET_SUFFIX:
        msg = f"unknown dataset {dataset!r}; the span campaign covers {list(DATASETS)}"
        raise SpanTextCampaignError(msg)


def _check_backbone(backbone: str) -> None:
    if backbone not in BACKBONES:
        msg = f"unknown backbone {backbone!r}; expected {sorted(BACKBONES)}"
        raise SpanTextCampaignError(msg)


def experiment_for(arm: str, dataset: str) -> str:
    """Return the experiment config for an arm x dataset pair."""
    _check_arm(arm)
    _check_dataset(dataset)
    return f"{_ARM_STEM[arm]}{_DATASET_SUFFIX[dataset]}"


def trainer_for(arm: str) -> str:
    """Return the trainer group for an arm.

    Not a caller's choice: pairing `base` with `difflr_*` silently trains the
    whole pretrained backbone at 10x the base LR and still reports a plausible
    score.
    """
    _check_arm(arm)
    return _ARM_TRAINER[arm]


def model_config_for(backbone: str) -> str:
    """Return the fusion model config for a backbone (text arms)."""
    _check_backbone(backbone)
    return BACKBONES[backbone]


@dataclass(frozen=True)
class SpanTextCell:
    """One training cell of the #220 text campaign."""

    backbone: str
    dataset: str
    arm: str
    seed: int
    gpu: int = 0

    @property
    def experiment(self) -> str:
        return experiment_for(self.arm, self.dataset)

    @property
    def trainer(self) -> str:
        return trainer_for(self.arm)

    @property
    def run_dir(self) -> str:
        """Unique per cell, and scoped to #220 rather than #218's `span_ner/`."""
        return (
            f"ner_span_text/{self.backbone}/{self.dataset}/{self.arm}/seed_{self.seed}"
        )


def expand(
    *,
    backbones: list[str],
    datasets: list[str],
    arms: list[str],
    seeds: list[int],
) -> list[SpanTextCell]:
    """Expand a selection into cells.

    Validates the whole selection up front so a typo dies here rather than after
    a node has been queued.
    """
    for backbone in backbones:
        _check_backbone(backbone)
    for dataset in datasets:
        _check_dataset(dataset)
    for arm in arms:
        _check_arm(arm)

    return [
        SpanTextCell(backbone=backbone, dataset=dataset, arm=arm, seed=seed)
        for backbone in backbones
        for dataset in datasets
        for seed in seeds
        for arm in arms
    ]


def pack(
    cells: list[SpanTextCell], *, cells_per_node: int = GPUS_PER_NODE
) -> list[list[SpanTextCell]]:
    """Pack cells onto rt_HF nodes, one GPU per cell.

    Note rt_HF has been observed delivering 4 GPUs rather than 8 on this
    reservation, so size `cells_per_node` from what the node actually reports.
    """
    if cells_per_node < 1 or cells_per_node > GPUS_PER_NODE:
        msg = f"cells_per_node must be 1..{GPUS_PER_NODE}, got {cells_per_node}"
        raise SpanTextCampaignError(msg)
    nodes: list[list[SpanTextCell]] = []
    for start in range(0, len(cells), cells_per_node):
        chunk = cells[start : start + cells_per_node]
        nodes.append(
            [
                SpanTextCell(
                    backbone=cell.backbone,
                    dataset=cell.dataset,
                    arm=cell.arm,
                    seed=cell.seed,
                    gpu=index,
                )
                for index, cell in enumerate(chunk)
            ]
        )
    return nodes
