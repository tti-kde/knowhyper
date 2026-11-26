"""IEForge command-line interface."""

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import typer

from ieforge.knowledge.eval.compare import compare_reports, write_comparison_json
from ieforge.knowledge.eval.runner import KnowledgeEvalRunConfig, run_knowledge_eval
from ieforge.knowledge.eval.schema import KnowledgeEvalError
from ieforge.metrics.biored import (
    biored_pubtator_score,
    merge_relation_and_novelty_predictions,
    read_biored_predictions_jsonl,
    relation_predictions_to_pubtator_lines,
)
from ieforge.registry import KINDS, list_components, load_entrypoints

app = typer.Typer(help="IEForge: information extraction research framework")
knowledge_app = typer.Typer(help="Knowledge-layer utilities")
knowledge_eval_app = typer.Typer(help="Evaluate knowledge pipelines")
app.add_typer(knowledge_app, name="knowledge")
knowledge_app.add_typer(knowledge_eval_app, name="eval")

sweep_app = typer.Typer(help="wandb sweep hyperparameter search")
app.add_typer(sweep_app, name="sweep")


@sweep_app.command("create")
def sweep_create(
    name: str = typer.Argument(..., help="sweep config name under configs/sweep/"),
    project: str = typer.Option("ieforge", "--project"),
    entity: str | None = typer.Option(None, "--entity"),
) -> None:
    """Create a wandb sweep from a configs/sweep/<name>.yaml definition."""
    from ieforge.sweep.runner import create_sweep

    sweep_id = create_sweep(name, project=project, entity=entity)
    typer.echo(sweep_id)


@sweep_app.command("agent")
def sweep_agent(
    sweep_id: str = typer.Argument(..., help="wandb sweep id"),
    count: int | None = typer.Option(None, "--count", help="trials to run"),
    project: str = typer.Option("ieforge", "--project"),
    entity: str | None = typer.Option(None, "--entity"),
) -> None:
    """Start a wandb agent running trials for a sweep."""
    from ieforge.sweep.runner import run_agent

    run_agent(sweep_id, count=count, project=project, entity=entity)


@sweep_app.command(
    "run",
    hidden=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def sweep_run(ctx: typer.Context) -> None:
    """Run a single sweep trial (invoked by the wandb agent, not by users)."""
    from ieforge.sweep.runner import run_trial_from_args

    run_trial_from_args(list(ctx.args))


def _run_hydra(config_name: str, overrides: list[str]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "ieforge.cli._hydra_entry",
        f"--config-name={config_name}",
        *overrides,
    ]
    result = subprocess.run(cmd, check=False)
    raise SystemExit(result.returncode)


@app.command()
def train(
    overrides: list[str] = typer.Argument(
        default=None, help="Hydra overrides, e.g. model=bert"
    ),
) -> None:
    """Train an IE model."""
    _run_hydra("train", overrides or [])


@app.command()
def eval(
    overrides: list[str] = typer.Argument(
        default=None, help="Hydra overrides, e.g. ckpt_path=..."
    ),
) -> None:
    """Evaluate a trained IE model."""
    _run_hydra("eval", overrides or [])


@knowledge_eval_app.command("run")
def knowledge_eval_run(
    config_name: str = typer.Option("train", "--config-name"),
    overrides: list[str] | None = typer.Option(None, "--override", "-o"),
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(..., "--output", dir_okay=False, writable=True),
    k: int = typer.Option(5, "--k", min=1),
) -> None:
    """Run a knowledge pipeline evaluation."""
    try:
        run_knowledge_eval(
            KnowledgeEvalRunConfig(
                config_name=config_name,
                overrides=overrides or [],
                dataset=dataset,
                output=output,
                k=k,
            )
        )
    except KnowledgeEvalError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@knowledge_eval_app.command("compare")
def knowledge_eval_compare(
    baseline: Path = typer.Option(
        ...,
        "--baseline",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    candidate: Path = typer.Option(
        ...,
        "--candidate",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(..., "--output", dir_okay=False, writable=True),
) -> None:
    """Compare two knowledge evaluation reports."""
    try:
        comparison = compare_reports(baseline, candidate)
        write_comparison_json(comparison, output)
    except KnowledgeEvalError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command(name="train_re")
def train_re(
    overrides: list[str] = typer.Argument(
        default=None, help="Hydra overrides, e.g. dataset=ddi"
    ),
) -> None:
    """Train an RE model. Deprecated: use `ieforge train experiment=re ...`."""
    _run_hydra("train", ["experiment=re", *(overrides or [])])


@app.command(name="eval_re")
def eval_re(
    overrides: list[str] = typer.Argument(
        default=None, help="Hydra overrides, e.g. ckpt_path=..."
    ),
) -> None:
    """Evaluate a trained RE model. Deprecated: use `ieforge eval experiment=re ...`."""
    _run_hydra("eval", ["experiment=re", *(overrides or [])])


def _gold_relation_lines(path: Path) -> list[str]:
    lines = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and len(stripped.split("\t")) == 5:
                lines.append(stripped)
    return lines


def _print_scores(scores: dict[str, float], keys: Iterable[str]) -> None:
    for key in keys:
        typer.echo(f"{key}: {scores[key]:.6f}")


@app.command(name="eval-biored")
def eval_biored(
    relation_predictions: Path = typer.Option(
        ...,
        "--relation-predictions",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Relation prediction JSONL from IEForge eval export.",
    ),
    novelty_predictions: Path = typer.Option(
        ...,
        "--novelty-predictions",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Novelty prediction JSONL from IEForge eval export.",
    ),
    gold_pubtator: Path = typer.Option(
        ...,
        "--gold-pubtator",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Gold PubTator file containing five-field BioRED relation lines.",
    ),
) -> None:
    """Evaluate combined BioRED relation and novelty predictions."""
    relation = read_biored_predictions_jsonl(relation_predictions)
    novelty = read_biored_predictions_jsonl(novelty_predictions)
    merged = merge_relation_and_novelty_predictions(relation, novelty)
    pred_lines = relation_predictions_to_pubtator_lines(merged)
    scores = biored_pubtator_score(_gold_relation_lines(gold_pubtator), pred_lines)
    _print_scores(scores, ("precision", "recall", "f1"))


def _load_builtins() -> None:
    import ieforge.datasets.blurb_doc_cls
    import ieforge.datasets.blurb_ner
    import ieforge.datasets.blurb_pico
    import ieforge.datasets.blurb_qa
    import ieforge.datasets.blurb_re
    import ieforge.datasets.blurb_sentence_similarity
    import ieforge.knowledge.backends.corpus
    import ieforge.knowledge.backends.features
    import ieforge.knowledge.backends.prompts
    import ieforge.knowledge.backends.retrievers
    import ieforge.knowledge.channels.features
    import ieforge.knowledge.channels.text
    import ieforge.knowledge.integration.collators
    import ieforge.knowledge.integration.query_builders
    import ieforge.losses.mse
    import ieforge.losses.token_cross_entropy
    import ieforge.metrics.accuracy
    import ieforge.metrics.biored
    import ieforge.metrics.entity_f1
    import ieforge.metrics.multilabel_f1
    import ieforge.metrics.pearson
    import ieforge.metrics.pico_macro_f1
    import ieforge.metrics.relation_f1
    import ieforge.models.hf_sequence_classifier
    import ieforge.models.hf_sequence_regressor
    import ieforge.models.hf_token_classifier
    import ieforge.tasks.biored
    import ieforge.tasks.doc_cls
    import ieforge.tasks.ner
    import ieforge.tasks.pico
    import ieforge.tasks.qa
    import ieforge.tasks.re
    import ieforge.tasks.sentence_similarity  # noqa: F401


@app.command(name="list")
def list_cmd(
    kind: str = typer.Argument(
        default=None,
        help="Component kind, e.g. model, metric, dataset, knowledge_injector",
    ),
) -> None:
    """List registered components. Pass a kind to filter (e.g. loss, model)."""
    _load_builtins()
    load_entrypoints()
    if kind is not None:
        if kind not in KINDS:
            typer.echo(f"Unknown kind {kind!r}. Choose from: {sorted(KINDS)}", err=True)
            raise typer.Exit(1)
        for comp_name in list_components(kind):
            typer.echo(comp_name)
    else:
        for k in sorted(KINDS):
            typer.echo(f"{k}:")
            for comp_name in list_components(k):
                typer.echo(f"  {comp_name}")


if __name__ == "__main__":
    # Enables `python -m ieforge.cli.main ...`, which the wandb sweep agent uses
    # to launch each trial (`python -m ieforge.cli.main sweep run ...`). Without
    # this the module imports but never invokes the app, so trials silently no-op.
    app()
