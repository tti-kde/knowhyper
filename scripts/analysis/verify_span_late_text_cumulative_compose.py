#!/usr/bin/env python3
"""Fail-closed compose gate for the fixed #220 text + #231 ev_warm campaign.

The cumulative label is valid only when one composed cell simultaneously contains
#220's second-segment serializer/collator and #231's graph incidence channels. In an
earlier draft ``feature_channels`` was silently swallowed by ``**kwargs`` and the run
would have been text-only despite a graph experiment name; this gate checks the resolved
objects, not just the config filenames.
"""

from __future__ import annotations

import sys

from hydra import compose, initialize_config_dir
from ieforge import CONFIGS_DIR as IEFORGE_CONFIGS_DIR
from omegaconf import OmegaConf

from knowhyper.ieforge._configs import CONFIGS_DIR
from knowhyper.pins import CORE_REVISION, FEATURES_REVISION, PACKAGE_REV

BACKBONES = ("pubmedbert", "biolinkbert")
DATASETS = {
    "bc5cdr_chem": "",
    "bc2gm": "_bc2gm",
    "bc5cdr_disease": "_bc5cdr_disease",
    "jnlpba": "_jnlpba",
    "ncbi_disease": "_ncbi_disease",
}
CONTROLS = ("real", "shuffled")
EXPECTED_PINS = ("0.12.13", "0.12.15", "v0.12.14")


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(IEFORGE_CONFIGS_DIR)):
        return compose(
            config_name="train",
            overrides=[*overrides, f"hydra.searchpath=[file://{CONFIGS_DIR}]"],
        )


def _plain(value):
    return OmegaConf.to_container(value) if OmegaConf.is_config(value) else value


def main() -> int:
    failures: list[str] = []
    actual_pins = (CORE_REVISION, FEATURES_REVISION, PACKAGE_REV)
    if actual_pins != EXPECTED_PINS:
        failures.append(f"pin drift: got {actual_pins!r}, expected {EXPECTED_PINS!r}")

    checked = 0
    for backbone in BACKBONES:
        for dataset, suffix in DATASETS.items():
            graph = _compose(
                f"experiment=khp_span_late_ev_warm{suffix}",
                f"model=knowhyper_span_late_logit_{backbone}",
                "model.pretrained_baseline_path=/frozen/base.ckpt",
                "model.pretrained_encoder_path=/frozen/encoder.pt",
                "trainer=difflr_benchmark_nolog",
                "logger=none",
            )
            text = _compose(
                f"experiment=span_ner_retrieved_text{suffix}",
                f"model={backbone}_span_classifier",
                "trainer=benchmark_nolog",
                "logger=none",
            )
            for control in CONTROLS:
                experiment = f"khp_span_late_text_ev_warm{suffix}"
                try:
                    cfg = _compose(
                        f"experiment={experiment}",
                        f"model=knowhyper_span_late_logit_{backbone}",
                        f"model.knowledge_control={control}",
                        "model.pretrained_baseline_path=/frozen/text-real.ckpt",
                        "model.pretrained_encoder_path=/frozen/encoder.pt",
                        "trainer=difflr_benchmark_nolog",
                        "logger=none",
                    )
                except Exception as error:
                    failures.append(f"{experiment}/{backbone}/{control}: {error}")
                    continue
                checked += 1
                label = f"{backbone}/{dataset}/{control}"
                injector = cfg.knowledge.pipeline.injector
                collator = cfg.knowledge.collator

                if (
                    cfg.dataset.name != f"blurb_ner_{dataset}"
                    or cfg.dataset.batch_size != 16
                ):
                    failures.append(f"{label}: wrong dataset or batch size")
                if not cfg.model._target_.endswith("SpanLateLogitKnowledgeModel"):
                    failures.append(f"{label}: not the late-logit model")
                if not cfg.model.freeze_inner or cfg.model.evidence_source != "graph":
                    failures.append(
                        f"{label}: first pass is not frozen graph evidence mode"
                    )
                if injector.text_serializer_mode != "graph_sidecar":
                    failures.append(f"{label}: serializer is not a graph sidecar")
                expected_strict = dataset != "jnlpba"
                if bool(injector.linker.strict_type_match) != expected_strict:
                    failures.append(
                        f"{label}: strict_type_match must be {expected_strict}"
                    )
                if injector.get("emit_a0_metadata", False):
                    failures.append(
                        f"{label}: closed candidate-text A0 route is enabled"
                    )

                channels = [entry._target_ for entry in collator.feature_channels]
                if channels != [
                    "knowhyper.ieforge.channels.IncidenceChannel",
                    "knowhyper.ieforge.channels.NERMentionSpanChannel",
                ]:
                    failures.append(
                        f"{label}: graph channels missing or reordered: {channels}"
                    )
                text_fields = (
                    "max_length",
                    "max_knowledge_tokens",
                    "min_source_tokens",
                    "mode",
                )
                failures.extend(
                    f"{label}: #220 collator field drift: {field}"
                    for field in text_fields
                    if collator.get(field) != text.knowledge.collator.get(field)
                )
                if _plain(injector.text_serializer) != _plain(
                    text.knowledge.pipeline.injector.text_serializer
                ):
                    failures.append(f"{label}: #220 serializer drift")
                graph_fields = (
                    "top_k",
                    "mention_type_map",
                    "mention_provenance_schema",
                    "retriever",
                    "encoder",
                    "integrator",
                    "entity_view_augmenter",
                )
                failures.extend(
                    f"{label}: #231 graph field drift: {field}"
                    for field in graph_fields
                    if _plain(injector.get(field))
                    != _plain(graph.knowledge.pipeline.injector.get(field))
                )
                checkpoint_cb = cfg.trainer.callbacks[0]
                if checkpoint_cb.save_top_k != 1 or checkpoint_cb.save_last:
                    failures.append(f"{label}: checkpoints are not bounded")
                if not cfg.trainer.callbacks[3]._target_.endswith(
                    "KnowledgeLRCallback"
                ):
                    failures.append(f"{label}: LR callback index 3 drifted")

    if checked == 0:
        print("FATAL verified 0 cumulative cells", file=sys.stderr)
        return 2
    if failures:
        print(f"FAILED {len(failures)} checks over {checked} cells:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(
        f"OK {checked} cumulative cells compose; #220 text + #231 ev_warm; "
        f"CORE {CORE_REVISION} FEATURES {FEATURES_REVISION} package {PACKAGE_REV}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
