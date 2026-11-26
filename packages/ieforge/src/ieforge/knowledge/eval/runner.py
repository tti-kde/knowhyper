"""Hydra-backed knowledge evaluation runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from ieforge import CONFIGS_DIR
from ieforge.knowledge.core.config import strip_names
from ieforge.knowledge.core.pipeline import KnowledgePipeline
from ieforge.knowledge.eval.metrics import compute_example_metrics, summarize_metrics
from ieforge.knowledge.eval.schema import (
    KnowledgeEvalArtifacts,
    KnowledgeEvalError,
    KnowledgeEvalExample,
    KnowledgeEvalExampleResult,
    KnowledgeEvalReport,
    KnowledgeEvalSummary,
    load_eval_examples,
    write_report_json,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ieforge.knowledge.core.schema import KnowledgeContext, Query, RetrievalResult


@dataclass(frozen=True)
class KnowledgeEvalRunConfig:
    """Configuration for one knowledge evaluation run."""

    config_name: str
    overrides: list[str]
    dataset: Path
    output: Path
    k: int = 5


@dataclass(frozen=True)
class _EvalPipelineExample:
    id: str
    query: str
    metadata: dict[str, Any]


class _KnowledgeEvalPipeline(Protocol):
    def retrieve(
        self, example: _EvalPipelineExample
    ) -> tuple[Query, RetrievalResult]: ...

    def inject(self, query: Query, retrieval: RetrievalResult) -> KnowledgeContext: ...


def run_knowledge_eval(run_config: KnowledgeEvalRunConfig) -> KnowledgeEvalReport:
    """Evaluate a configured knowledge pipeline over JSONL examples."""
    examples = load_eval_examples(run_config.dataset)
    cfg = _compose_hydra_config(run_config.config_name, run_config.overrides)
    pipeline = _instantiate_eval_pipeline(cfg)
    report = _eval_pipeline_examples(pipeline, examples, run_config.k)
    write_report_json(report, run_config.output)
    return report


def _compose_hydra_config(config_name: str, overrides: list[str]) -> DictConfig:
    """Compose an IEForge Hydra config for knowledge evaluation."""
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def _instantiate_eval_pipeline(cfg: DictConfig) -> KnowledgePipeline:
    """Instantiate cfg.knowledge.pipeline with eval query fields."""
    if (
        "knowledge" not in cfg
        or cfg.knowledge is None
        or "pipeline" not in cfg.knowledge
        or cfg.knowledge.pipeline is None
    ):
        msg = "cfg.knowledge.pipeline is required for knowledge eval"
        raise KnowledgeEvalError(msg)

    pipeline_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.knowledge.pipeline, resolve=False)
    )
    with open_dict(pipeline_cfg):
        pipeline_cfg.query_builder = {
            "_target_": (
                "ieforge.knowledge.integration.query_builders.AttributeQueryBuilder"
            ),
            "field": "query",
            "metadata_fields": ["id"],
        }

    # pipeline_cfg above is an unresolved container so the query_builder override
    # can be injected via open_dict; here we resolve it and strip registry-only
    # ``name:`` keys before instantiation (same as the train paths).
    pipeline_clean = strip_names(OmegaConf.to_container(pipeline_cfg, resolve=True))
    pipeline = instantiate(OmegaConf.create(pipeline_clean))
    if not isinstance(pipeline, KnowledgePipeline):
        msg = "cfg.knowledge.pipeline must instantiate KnowledgePipeline"
        raise KnowledgeEvalError(msg)
    return pipeline


def _eval_pipeline_examples(
    pipeline: _KnowledgeEvalPipeline,
    examples: list[KnowledgeEvalExample],
    k: int,
) -> KnowledgeEvalReport:
    """Run retrieval/injection and compute report metrics."""
    results: list[KnowledgeEvalExampleResult] = []
    example_metrics: list[dict[str, float | None]] = []

    for example in examples:
        pipeline_example = _EvalPipelineExample(
            id=example.id,
            query=example.query,
            metadata=dict(example.metadata),
        )
        query, retrieval = pipeline.retrieve(pipeline_example)
        context = pipeline.inject(query, retrieval)
        full_retrieved_fact_ids = [fact.id for fact in retrieval.facts]
        retrieved_fact_ids = full_retrieved_fact_ids[: max(k, 0)]
        metrics = compute_example_metrics(
            retrieved_ids=full_retrieved_fact_ids,
            expected_ids=example.expected_fact_ids,
            text_context=context.text_context,
            expected_text=example.expected_text,
            k=k,
        )
        example_metrics.append(metrics)
        results.append(
            KnowledgeEvalExampleResult(
                id=example.id,
                query=example.query,
                expected_fact_ids=list(example.expected_fact_ids),
                retrieved_fact_ids=retrieved_fact_ids,
                metrics=metrics,
                artifacts=KnowledgeEvalArtifacts(
                    has_text_context=context.text_context is not None,
                    has_prompt_embeds=context.prompt_embeds is not None,
                    has_prompt_attention_mask=(
                        context.prompt_attention_mask is not None
                    ),
                    graph_feature_keys=sorted(context.graph_features),
                ),
                metadata=dict(example.metadata),
            )
        )

    summary_metrics, num_scored_examples = summarize_metrics(example_metrics)
    return KnowledgeEvalReport(
        summary=KnowledgeEvalSummary(
            num_examples=len(examples),
            metrics=summary_metrics,
            num_scored_examples=num_scored_examples,
        ),
        examples=results,
    )
