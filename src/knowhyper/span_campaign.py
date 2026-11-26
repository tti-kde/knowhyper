"""Cell plan for the #218 span-level NER knowledge campaign.

Deliberately separate from `knowhyper.matrix`: that module's methods are the
token-fusion family, and #218 Phase 1 requires a dedicated span driver rather than
extra condition branches bolted onto the old one.

Paper terminology maps onto the config families as:

| paper            | condition | config family              |
|------------------|-----------|----------------------------|
| Structure only   | `edge`    | `khp_span_ner_edge*`       |
| +attributes      | `ev`      | `khp_span_ner_ev*`         |
| +pretraining     | `ev_warm` | `khp_span_ner_ev_warm*`    |

The `zero` control returns the span baseline before the gate, so it is identical
across conditions; `shared_zero=True` collapses it to one arm per
backbone x dataset x seed and labels its condition `shared` rather than claiming a
condition it did not exercise. `shuffled` is NEVER shared: it must pair with the
graph its own condition retrieved.
"""

from __future__ import annotations

from dataclasses import dataclass

BACKBONES: dict[str, str] = {
    "pubmedbert": "knowhyper_span_cross_attention_pubmedbert",
    "biolinkbert": "knowhyper_span_cross_attention_biolinkbert",
}

# The knowledge-free first pass that produces candidate proposals.
PROPOSAL_MODELS: dict[str, str] = {
    "pubmedbert": "pubmedbert_span_classifier",
    "biolinkbert": "biolinkbert_span_classifier",
}

DATASETS: tuple[str, ...] = ("bc5cdr_chem", "bc2gm", "bc5cdr_disease")

# Paper-table extension datasets.  Keep ``DATASETS`` as the completed three-dataset
# campaign so old planners and aggregators do not silently grow, while accepting these
# only when a caller selects them explicitly.
EXTENSION_DATASETS: tuple[str, ...] = ("jnlpba", "ncbi_disease")
ALL_DATASETS: tuple[str, ...] = DATASETS + EXTENSION_DATASETS

CONDITIONS: tuple[str, ...] = ("edge", "ev", "ev_warm")

CONTROLS: tuple[str, ...] = ("real", "shuffled", "zero")

# Dataset suffix on the experiment config name. Chem is the root config.
_DATASET_SUFFIX: dict[str, str] = {
    "bc5cdr_chem": "",
    "bc2gm": "_bc2gm",
    "bc5cdr_disease": "_bc5cdr_disease",
    "jnlpba": "_jnlpba",
    "ncbi_disease": "_ncbi_disease",
}

_CONDITION_STEM: dict[str, str] = {
    "edge": "khp_span_ner_edge",
    "ev": "khp_span_ner_ev",
    "ev_warm": "khp_span_ner_ev_warm",
}

GPUS_PER_NODE = 8

# The condition label carried by a collapsed zero arm.
SHARED = "shared"


class SpanCampaignError(RuntimeError):
    """A cell selection that must not be submitted."""


def model_config_for(backbone: str) -> str:
    """Return the fusion model config for a backbone."""
    try:
        return BACKBONES[backbone]
    except KeyError:
        msg = (
            f"unknown backbone {backbone!r}; the span campaign covers "
            f"{sorted(BACKBONES)}"
        )
        raise SpanCampaignError(msg) from None


def experiment_for(condition: str, dataset: str) -> str:
    """Return the experiment config for a condition x dataset pair."""
    # A shared zero runs the cheapest condition that reaches the span baseline;
    # every condition's zero arm returns before the gate, so they agree.
    stem = _CONDITION_STEM.get("edge" if condition == SHARED else condition)
    if stem is None:
        msg = f"unknown condition {condition!r}; expected one of {list(CONDITIONS)}"
        raise SpanCampaignError(msg)
    if dataset not in _DATASET_SUFFIX:
        msg = f"unknown dataset {dataset!r}; the span campaign covers {list(ALL_DATASETS)}"
        raise SpanCampaignError(msg)
    return f"{stem}{_DATASET_SUFFIX[dataset]}"


@dataclass(frozen=True)
class SpanCell:
    """One training cell."""

    backbone: str
    dataset: str
    condition: str
    control: str
    seed: int
    gpu: int = 0

    @property
    def experiment(self) -> str:
        return experiment_for(self.condition, self.dataset)

    @property
    def model_config(self) -> str:
        return model_config_for(self.backbone)

    @property
    def run_dir(self) -> str:
        """Unique per cell: two live writers in one directory lose both runs."""
        return (
            f"span_ner/{self.backbone}/{self.dataset}/"
            f"{self.condition}_{self.control}/seed_{self.seed}"
        )

    @property
    def needs_encoder(self) -> bool:
        return self.condition == "ev_warm"


@dataclass(frozen=True)
class ProposalCell:
    """One first-pass proposal-generation cell.

    `fold is None` is the full-train model that predicts validation/test; a fold
    index is an OOF model that predicts its own holdout part of train.
    """

    backbone: str
    dataset: str
    fold: int | None
    num_folds: int
    fold_seed: int = 42
    proposal_seed: int = 0

    @property
    def model_config(self) -> str:
        try:
            return PROPOSAL_MODELS[self.backbone]
        except KeyError:
            msg = f"unknown backbone {self.backbone!r}"
            raise SpanCampaignError(msg) from None

    @property
    def splits(self) -> str:
        return "train" if self.fold is not None else "validation,test"

    @property
    def output_dir(self) -> str:
        """Backbone-scoped: candidate JSONL must never be shared across backbones."""
        leaf = "full" if self.fold is None else f"oof_fold{self.fold}"
        return f"span_ner_predictions/{self.backbone}/{self.dataset}/{leaf}"


def expand(
    *,
    backbones: list[str],
    datasets: list[str],
    conditions: list[str],
    controls: list[str],
    seeds: list[int],
    shared_zero: bool,
    encoder_root: str | None = None,
    require_encoder: bool = True,
) -> list[SpanCell]:
    """Expand a selection into cells, collapsing the zero arm when requested.

    `require_encoder` defaults to True so a warm arm cannot be planned without an
    encoder root by accident: the safe value is the one you must opt OUT of. Pass
    False only to enumerate cells for inspection, never to submit them.
    """
    for backbone in backbones:
        model_config_for(backbone)
    for dataset in datasets:
        if dataset not in _DATASET_SUFFIX:
            msg = (
                f"unknown dataset {dataset!r}; the span campaign covers "
                f"{list(ALL_DATASETS)}"
            )
            raise SpanCampaignError(msg)
    for condition in conditions:
        if condition not in CONDITIONS:
            msg = f"unknown condition {condition!r}; expected {list(CONDITIONS)}"
            raise SpanCampaignError(msg)
    for control in controls:
        if control not in CONTROLS:
            msg = f"unknown control {control!r}; expected {list(CONTROLS)}"
            raise SpanCampaignError(msg)

    if require_encoder and "ev_warm" in conditions and not encoder_root:
        msg = (
            "ev_warm cells need a cache-matched pretrained encoder root; refusing "
            "to plan a warm arm that would silently run cold"
        )
        raise SpanCampaignError(msg)

    cells: list[SpanCell] = []
    for backbone in backbones:
        for dataset in datasets:
            for seed in seeds:
                for control in controls:
                    if control == "zero" and shared_zero:
                        continue
                    cells.extend(
                        SpanCell(
                            backbone=backbone,
                            dataset=dataset,
                            condition=condition,
                            control=control,
                            seed=seed,
                        )
                        for condition in conditions
                    )
                if shared_zero and "zero" in controls:
                    cells.append(
                        SpanCell(
                            backbone=backbone,
                            dataset=dataset,
                            condition=SHARED,
                            control="zero",
                            seed=seed,
                        )
                    )
    return cells


def proposal_cells(
    *, backbones: list[str], datasets: list[str], num_folds: int = 5
) -> list[ProposalCell]:
    """Expand the §0.2 proposal-generation plan: `n_folds` OOF plus one full-train."""
    if num_folds < 2:
        msg = f"OOF needs at least 2 folds, got {num_folds}"
        raise SpanCampaignError(msg)
    cells: list[ProposalCell] = []
    for backbone in backbones:
        if backbone not in PROPOSAL_MODELS:
            msg = f"unknown backbone {backbone!r}"
            raise SpanCampaignError(msg)
        for dataset in datasets:
            if dataset not in _DATASET_SUFFIX:
                msg = f"unknown dataset {dataset!r}"
                raise SpanCampaignError(msg)
            cells.extend(
                ProposalCell(
                    backbone=backbone,
                    dataset=dataset,
                    fold=fold,
                    num_folds=num_folds,
                )
                for fold in range(num_folds)
            )
            cells.append(
                ProposalCell(
                    backbone=backbone,
                    dataset=dataset,
                    fold=None,
                    num_folds=num_folds,
                )
            )
    return cells


def pack(
    cells: list[SpanCell], *, cells_per_node: int = GPUS_PER_NODE
) -> list[list[SpanCell]]:
    """Pack cells onto rt_HF nodes, one GPU per cell.

    rt_HG is a 1/8 slice PBS co-schedules other users' jobs onto, so its
    DataLoaders contend; owning a whole rt_HF node and packing it yourself measured
    ~13 min/cell against ~28 min at 20 concurrent slices.
    """
    if cells_per_node < 1 or cells_per_node > GPUS_PER_NODE:
        msg = f"cells_per_node must be 1..{GPUS_PER_NODE}, got {cells_per_node}"
        raise SpanCampaignError(msg)
    nodes: list[list[SpanCell]] = []
    for start in range(0, len(cells), cells_per_node):
        chunk = cells[start : start + cells_per_node]
        nodes.append(
            [
                SpanCell(
                    backbone=cell.backbone,
                    dataset=cell.dataset,
                    condition=cell.condition,
                    control=cell.control,
                    seed=cell.seed,
                    gpu=index,
                )
                for index, cell in enumerate(chunk)
            ]
        )
    return nodes
