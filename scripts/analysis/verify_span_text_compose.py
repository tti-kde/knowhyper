#!/usr/bin/env python
"""Pre-dispatch gate for the #220 span text campaign.

Every cell must COMPOSE before a node spends GPU time. This catches three failures
that all read as something else:

1. upstream default drift (span_rep_mode, decoder, loss) moving one route only;
2. a wrong-directory submission -- `abci submit` snapshots cwd, so submitting from
   the main repo ships a tree without this branch's configs and Hydra reports the
   experiment as unknown, which reads like a config bug;
3. a text arm whose retrieval fingerprint has drifted off #218's caches, which
   would silently build a SECOND cache and compare the two Table 6 routes against
   different graphs.

Exit 0 = safe to dispatch. Exit 1 = do not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import ieforge
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import knowhyper
from knowhyper.span_campaign import experiment_for as graph_experiment_for
from knowhyper.span_text_campaign import (
    ARMS,
    DATASETS,
    experiment_for,
    trainer_for,
)

EXTENSION_DATASETS = ("jnlpba", "ncbi_disease")
VERIFY_DATASETS = DATASETS + EXTENSION_DATASETS

IEFORGE_CONFIGS = str(Path(ieforge.__file__).parent / "configs")
KNOWHYPER_CONFIGS = str(Path(knowhyper.__file__).parent / "configs")

# #218's retrieval identities. The text arms MUST land on these: #220 names #218 the
# canonical producer of the prediction-scoped caches and forbids building others.
EXPECTED_FINGERPRINTS = {
    "bc5cdr_chem": "87eac63095440a4c",
    "bc2gm": "9db2f6246eb038b5",
    "bc5cdr_disease": "a2216e900fff8a5c",
}

BACKBONE_MODELS = {
    "pubmedbert": "pubmedbert_span_classifier",
    "biolinkbert": "biolinkbert_span_classifier",
}


def _compose(experiment: str, model: str):
    with initialize_config_dir(version_base=None, config_dir=IEFORGE_CONFIGS):
        return compose(
            config_name="train",
            overrides=[
                f"experiment={experiment}",
                f"model={model}",
                f"hydra.searchpath=[file://{KNOWHYPER_CONFIGS}]",
            ],
        )


def _check_cells() -> list[str]:
    """Compose all 2 x 3 x 3 = 18 arm configurations."""
    problems: list[str] = []
    for backbone, model in sorted(BACKBONE_MODELS.items()):
        for dataset in VERIFY_DATASETS:
            for arm in ARMS:
                experiment = experiment_for(arm, dataset)
                label = f"{backbone}:{dataset}:{arm}"
                try:
                    cfg = _compose(experiment, model)
                except Exception as exc:
                    problems.append(f"{label}: compose failed ({exc})")
                    continue

                if cfg.dataset.name != f"blurb_ner_{dataset}":
                    problems.append(
                        f"{label}: dataset is {cfg.dataset.name}, expected "
                        f"blurb_ner_{dataset}"
                    )
                if cfg.model.name != f"{backbone}_span":
                    problems.append(
                        f"{label}: model is {cfg.model.name}, expected "
                        f"{backbone}_span -- a dropped model= override selects "
                        "ModernBERT silently"
                    )
                if int(cfg.task.max_span_width) != 12:
                    problems.append(
                        f"{label}: max_span_width={cfg.task.max_span_width}, want 12"
                    )
                if str(cfg.task.decoder) != "max_total_score":
                    problems.append(
                        f"{label}: decoder={cfg.task.decoder}, want max_total_score"
                    )
                if str(cfg.model.span_rep_mode) != "start_end":
                    problems.append(
                        f"{label}: span_rep_mode={cfg.model.span_rep_mode}, "
                        "want start_end"
                    )
                if "SpanFocalLoss" not in str(cfg.loss._target_):
                    problems.append(f"{label}: loss is {cfg.loss._target_}")

                has_knowledge = OmegaConf.select(cfg, "knowledge") is not None
                if arm == "base" and has_knowledge:
                    problems.append(f"{label}: base must have NO knowledge branch")
                if arm != "base" and not has_knowledge:
                    problems.append(f"{label}: text arm lost its knowledge branch")
                if arm != "base":
                    expected_strict = dataset != "jnlpba"
                    actual_strict = bool(
                        cfg.knowledge.pipeline.injector.linker.strict_type_match
                    )
                    if actual_strict != expected_strict:
                        problems.append(
                            f"{label}: strict_type_match={actual_strict}, "
                            f"want {expected_strict}"
                        )
    return problems


def _check_trainers() -> list[str]:
    """NO arm may take difflr: all three are plain HFSpanClassifiers.

    The text route has no knowledge branch, so difflr's `model.inner.` split
    matches nothing and the entire pretrained backbone trains at 10x the base LR.
    """
    return [
        f"{arm} trainer is {trainer_for(arm)}, not benchmark_nolog -- difflr "
        "would train its whole pretrained backbone at 10x LR, silently"
        for arm in ARMS
        if trainer_for(arm) != "benchmark_nolog"
    ]


def _check_fingerprints() -> list[str]:
    """The text arms must still resolve onto #218's caches."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from resolve_cache_fingerprint import resolve
    except Exception as exc:
        return [f"cannot import resolve_cache_fingerprint: {exc}"]

    problems: list[str] = []
    expected_by_dataset = dict(EXPECTED_FINGERPRINTS)
    for dataset in EXTENSION_DATASETS:
        try:
            expected_by_dataset[dataset] = resolve(
                graph_experiment_for("edge", dataset)
            )[0]
        except Exception as exc:
            problems.append(f"{dataset}: edge fingerprint resolve failed ({exc})")
    for dataset, expected in expected_by_dataset.items():
        for arm in ("text_real", "text_sham"):
            experiment = experiment_for(arm, dataset)
            try:
                got, _, _ = resolve(experiment)
            except Exception as exc:
                problems.append(f"{experiment}: fingerprint resolve failed ({exc})")
                continue
            if got != expected:
                problems.append(
                    f"{experiment}: fingerprint {got} != #218's {expected} -- this "
                    "arm would build a SECOND cache and read a different graph"
                )
    return problems


def main() -> int:
    problems: list[str] = []
    problems += _check_cells()
    problems += _check_trainers()
    problems += _check_fingerprints()

    n_cells = len(BACKBONE_MODELS) * len(VERIFY_DATASETS) * len(ARMS)
    if problems:
        print(f"FAIL span-text compose verification ({len(problems)} problems):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        f"OK {n_cells} span-text cell configurations compose; "
        "trainers correct; text arms match #218 edge fingerprints "
        f"(3 pinned core profiles + {len(EXTENSION_DATASETS)} resolved extension profiles)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
