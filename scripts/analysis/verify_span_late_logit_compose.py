#!/usr/bin/env python3
"""Assert every #231 Variant 1 cell composes and the pins are intact.

Run this in-job BEFORE dispatching GPU work. It catches an upstream default drift and,
critically, a **wrong-directory submission**: `abci submit` snapshots the cwd, so
submitting from the main repo ships a source tree without this branch's model config,
and Hydra then reports the experiment name as unknown while sibling arms that exist on
main succeed -- which reads like a config bug rather than a wrong-directory bug.

    uv run python scripts/analysis/verify_span_late_logit_compose.py
    uv run python scripts/analysis/verify_span_late_logit_compose.py --proposals

The #218 verifier checks the `khp_span_ner_*` family; this one checks
`khp_span_late_*` + `knowhyper_span_late_logit_*`. Pointing the #231 driver at the
#218 verifier would pass while composing an entirely different architecture.
"""

from __future__ import annotations

import argparse
import sys

from hydra import compose, initialize_config_dir
from ieforge import CONFIGS_DIR as IEFORGE_CONFIGS_DIR

from knowhyper.ieforge._configs import CONFIGS_DIR
from knowhyper.pins import (
    CORE_REVISION,
    FEATURES_REVISION,
    PACKAGE_REV,
)
from knowhyper.span_late_campaign import (
    ALL_DATASETS,
    BACKBONES,
    CONDITIONS,
    CONTROLS,
    PROPOSAL_MODELS,
    experiment_for,
    model_config_for,
)

EXPECTED_CORE = "0.12.13"
EXPECTED_FEATURES = "0.12.15"
EXPECTED_PACKAGE = "v0.12.14"


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(IEFORGE_CONFIGS_DIR)):
        return compose(
            config_name="train",
            overrides=[*overrides, f"hydra.searchpath=[file://{CONFIGS_DIR}]"],
        )


def _check_pins(failures: list[str]) -> None:
    for name, actual, expected in (
        ("CORE", CORE_REVISION, EXPECTED_CORE),
        ("FEATURES", FEATURES_REVISION, EXPECTED_FEATURES),
        ("package rev", PACKAGE_REV, EXPECTED_PACKAGE),
    ):
        if actual != expected:
            failures.append(f"pin drift: {name} is {actual!r}, expected {expected!r}")


def _verify_knowledge(failures: list[str]) -> int:
    checked = 0
    for backbone in BACKBONES:
        model = model_config_for(backbone)
        checkpoint = None
        for dataset in ALL_DATASETS:
            for condition in CONDITIONS:
                for control in CONTROLS:
                    experiment = experiment_for(condition, dataset)
                    overrides = [
                        f"experiment={experiment}",
                        f"model={model}",
                        f"model.knowledge_control={control}",
                        "trainer=difflr_benchmark_nolog",
                        "logger=none",
                    ]
                    if condition == "ev_warm":
                        overrides.append(
                            "model.pretrained_encoder_path=/nonexistent/encoder.pt"
                        )
                    try:
                        cfg = _compose(*overrides)
                    except Exception as error:
                        failures.append(f"{experiment}/{backbone}/{control}: {error}")
                        continue
                    checked += 1

                    if not cfg.model._target_.endswith("SpanLateLogitKnowledgeModel"):
                        failures.append(
                            f"{experiment}/{backbone}: model is {cfg.model._target_}, "
                            "not the #231 late-logit wrapper"
                        )
                    if cfg.model.knowledge_control != control:
                        failures.append(
                            f"{experiment}/{backbone}: knowledge_control did not take"
                        )
                    if cfg.model.tokenizer_name != cfg.model.inner.model_name:
                        failures.append(
                            f"{experiment}/{backbone}: tokenizer "
                            f"{cfg.model.tokenizer_name} != encoder "
                            f"{cfg.model.inner.model_name}"
                        )
                    if checkpoint is None:
                        checkpoint = cfg.model.tokenizer_name
                    elif cfg.model.tokenizer_name != checkpoint:
                        failures.append(
                            f"{backbone}: checkpoint varies across cells "
                            f"({checkpoint} vs {cfg.model.tokenizer_name})"
                        )
                    if cfg.task.max_span_width != 12:
                        failures.append(f"{experiment}: max_span_width is not 12")
                    if cfg.task.decoder != "max_total_score":
                        failures.append(f"{experiment}: decoder is not max_total_score")
                    expected_strict = dataset != "jnlpba"
                    if (
                        bool(cfg.knowledge.pipeline.injector.linker.strict_type_match)
                        != expected_strict
                    ):
                        failures.append(
                            f"{experiment}: strict_type_match must be "
                            f"{expected_strict} for {dataset}"
                        )
                    if not cfg.loss._target_.endswith("SpanFocalLoss"):
                        failures.append(f"{experiment}: loss is not focal")
                    checkpoint_cb = cfg.trainer.callbacks[0]
                    if checkpoint_cb.save_top_k != 1 or checkpoint_cb.save_last:
                        failures.append(f"{experiment}: checkpoints are not bounded")
                    if not cfg.trainer.callbacks[3]._target_.endswith(
                        "KnowledgeLRCallback"
                    ):
                        failures.append(
                            f"{experiment}: callbacks.3 is not KnowledgeLRCallback, so "
                            "trainer.callbacks.3.knowledge_lr_mult targets the wrong one"
                        )
    return checked


def _verify_proposals(failures: list[str]) -> int:
    checked = 0
    for backbone, model in PROPOSAL_MODELS.items():
        for dataset in ALL_DATASETS:
            for fold in (None, 0):
                overrides = [
                    "experiment=span_ner",
                    f"dataset=blurb_ner_{dataset}",
                    f"model={model}",
                    "task.max_span_width=12",
                    "trainer=benchmark_nolog",
                    "logger=none",
                ]
                if fold is not None:
                    overrides += [
                        "train_dataset=blurb_ner_oof",
                        f"train_dataset.fold={fold}",
                        "train_dataset.num_folds=5",
                        "train_dataset.fold_seed=42",
                        "train_dataset.role=train",
                    ]
                try:
                    cfg = _compose(*overrides)
                except Exception as error:
                    failures.append(f"proposal {backbone}/{dataset}/{fold}: {error}")
                    continue
                checked += 1

                # The knowledge-free first pass must NOT be the fusion wrapper: the
                # model override is load-bearing because experiment=span_ner defaults
                # to modernbert_span_classifier.
                if not cfg.model._target_.endswith("HFSpanClassifier"):
                    failures.append(
                        f"proposal {backbone}/{dataset}: model is {cfg.model._target_}"
                    )
                if cfg.model.tokenizer_name != cfg.model.model_name:
                    failures.append(
                        f"proposal {backbone}/{dataset}: tokenizer "
                        f"{cfg.model.tokenizer_name} != encoder {cfg.model.model_name}"
                    )
                if cfg.task.max_span_width != 12:
                    failures.append(f"proposal {backbone}/{dataset}: msw is not 12")
                if fold is not None and cfg.train_dataset.role != "train":
                    failures.append(
                        f"proposal {backbone}/{dataset}: OOF fit role is "
                        f"{cfg.train_dataset.role!r}, not 'train' -- that would leak"
                    )
    return checked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--proposals",
        action="store_true",
        help="verify the §0.2 proposal cells instead of the knowledge cells",
    )
    args = ap.parse_args()

    failures: list[str] = []
    _check_pins(failures)
    if args.proposals:
        checked = _verify_proposals(failures)
        label = "proposal"
    else:
        checked = _verify_knowledge(failures)
        label = "knowledge"

    # A pass over nothing is how a verifier passes while looking at the wrong artifact.
    if checked == 0:
        print(
            "FATAL verified 0 cells; the plan or config tree is wrong", file=sys.stderr
        )
        return 2

    if failures:
        print(
            f"FAILED {len(failures)} checks over {checked} {label} cells:",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        f"OK {checked} {label} cells compose; "
        f"CORE {CORE_REVISION} FEATURES {FEATURES_REVISION} package {PACKAGE_REV}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
