"""Command-line interface for the eval harness.

Wires the dataset loader, experiment config, judge panel, and runner into a few
Typer commands. Matches the parent project's Typer-based CLI convention.

Commands:
    run                Run an experiment end-to-end against guideline-gpt.
    validate-dataset   Load and schema-validate an eval set.
    list-experiments   List the experiment configs in a directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from eval_harness.config import get_settings
from eval_harness.dataset.loader import EvalDatasetError, load_eval_set
from eval_harness.judges.base import get_judge_client
from eval_harness.judges.correctness_judge import CorrectnessJudge
from eval_harness.judges.faithfulness_judge import FaithfulnessJudge
from eval_harness.judges.relevance_judge import RelevanceJudge
from eval_harness.logging_setup import configure_logging
from eval_harness.runner.experiment import ExperimentConfigError, load_experiment
from eval_harness.runner.runner import run_experiment
from eval_harness.runner.scoring import JudgePanel

app = typer.Typer(
    add_completion=False,
    help="Evaluate guideline-gpt: run experiments and score them.",
)


@app.command()
def run(
    experiment: Annotated[Path, typer.Argument(help="Path to an experiment YAML.")],
    dataset: Annotated[
        Path | None, typer.Option(help="Eval set JSONL (defaults to EVAL_DATASET_PATH).")
    ] = None,
    results_dir: Annotated[
        Path | None, typer.Option(help="Output root (defaults to RESULTS_DIR).")
    ] = None,
) -> None:
    """Run an experiment's variants over the eval set and persist results.

    Requires a working ``guideline-gpt`` corpus and the relevant API keys.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    config = load_experiment(experiment)
    questions = load_eval_set(dataset or settings.eval_dataset_path)

    client = get_judge_client(settings)
    panel = JudgePanel(
        faithfulness=FaithfulnessJudge(client),
        relevance=RelevanceJudge(client),
        correctness=CorrectnessJudge(client),
    )

    results = run_experiment(
        config,
        questions,
        panel=panel,
        results_dir=results_dir or settings.results_dir,
    )

    typer.echo(
        f"Ran {len(results)} variant(s) for {config.experiment_id} on {len(questions)} questions."
    )
    for result in results:
        variant_name = result.config.get("variant", {})
        name = variant_name.get("name") if isinstance(variant_name, dict) else variant_name
        correctness = result.aggregate_metrics.get("correctness_mean", 0.0)
        typer.echo(f"  {name}: correctness {correctness:.3f}, cost ${result.total_cost_usd:.4f}")


@app.command(name="validate-dataset")
def validate_dataset(
    dataset: Annotated[
        Path | None, typer.Option(help="Eval set JSONL (defaults to EVAL_DATASET_PATH).")
    ] = None,
) -> None:
    """Load an eval set and report how many questions validated."""
    settings = get_settings()
    path = dataset or settings.eval_dataset_path
    try:
        questions = load_eval_set(path)
    except EvalDatasetError as exc:
        typer.secho(f"Invalid eval set: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho(f"OK: {len(questions)} questions validated in {path}", fg=typer.colors.GREEN)


@app.command(name="list-experiments")
def list_experiments(
    directory: Annotated[Path, typer.Option(help="Directory of experiment YAMLs.")] = Path(
        "experiments"
    ),
) -> None:
    """List the experiments defined in a directory."""
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        typer.secho(f"No experiment YAMLs found in {directory}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)
    for path in paths:
        try:
            config = load_experiment(path)
        except ExperimentConfigError as exc:
            typer.secho(f"{path.name}: INVALID — {exc}", fg=typer.colors.RED, err=True)
            continue
        typer.echo(f"{config.experiment_id}: {config.name} ({len(config.variants)} variants)")


if __name__ == "__main__":  # pragma: no cover
    app()
