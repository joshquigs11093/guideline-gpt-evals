"""Run an experiment: every variant over every question, scored and persisted.

For each variant the runner builds a pipeline (via an injected
:class:`PipelineFactory`), queries it once per eval question, maps each response
to a :class:`Prediction`, scores it with the judge panel + retrieval metrics,
aggregates, and writes ``predictions.jsonl`` / ``metrics.json`` / ``manifest.json``
under ``results/{experiment_id}/{variant}/``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import guideline_gpt

from eval_harness.logging_setup import get_logger
from eval_harness.report.aggregator import aggregate_scores
from eval_harness.runner.experiment import ExperimentConfig, Variant
from eval_harness.runner.scoring import JudgePanel, score_question, to_prediction
from eval_harness.runner.variant import PipelineFactory, default_pipeline_factory
from eval_harness.types import EvalQuestion, ExperimentResults, Prediction

log = get_logger(__name__)


def _json_default(obj: object) -> str:
    """JSON serializer for the non-primitive types we persist."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialize {type(obj).__name__}")


def variant_results_dir(results_dir: Path, experiment_id: str, variant_name: str) -> Path:
    """Return the output directory for one variant's results."""
    return results_dir / experiment_id / variant_name


def run_variant(
    config: ExperimentConfig,
    variant: Variant,
    questions: list[EvalQuestion],
    *,
    factory: PipelineFactory,
    panel: JudgePanel,
    timestamp: datetime,
    guideline_gpt_version: str,
) -> tuple[ExperimentResults, list[Prediction]]:
    """Run and score a single variant over the eval set."""
    pipeline = factory(variant, config.shared_config)
    predictions: list[Prediction] = []
    scores = []
    total_cost = 0.0
    total_latency = 0

    for index, question in enumerate(questions, start=1):
        response = pipeline.query(question.question)
        prediction = to_prediction(question.question_id, response)
        predictions.append(prediction)
        scores.append(score_question(question, prediction, panel))
        total_cost += prediction.cost_usd
        total_latency += prediction.latency_ms
        log.info(
            "question_scored",
            experiment=config.experiment_id,
            variant=variant.name,
            progress=f"{index}/{len(questions)}",
        )

    results = ExperimentResults(
        experiment_id=config.experiment_id,
        experiment_name=config.name,
        config={"shared_config": config.shared_config, "variant": variant.model_dump(mode="json")},
        run_timestamp=timestamp,
        guideline_gpt_version=guideline_gpt_version,
        question_scores=scores,
        aggregate_metrics=aggregate_scores(questions, scores),
        total_cost_usd=total_cost,
        total_latency_ms=total_latency,
    )
    return results, predictions


def persist_variant_results(
    results_dir: Path,
    variant_name: str,
    results: ExperimentResults,
    predictions: list[Prediction],
) -> Path:
    """Write predictions/metrics/manifest for one variant; return its directory."""
    out_dir = variant_results_dir(results_dir, results.experiment_id, variant_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_jsonl = "\n".join(
        json.dumps(asdict(prediction), default=_json_default) for prediction in predictions
    )
    (out_dir / "predictions.jsonl").write_text(predictions_jsonl + "\n", encoding="utf-8")

    (out_dir / "metrics.json").write_text(
        json.dumps(asdict(results), indent=2, default=_json_default), encoding="utf-8"
    )

    manifest = {
        "experiment_id": results.experiment_id,
        "experiment_name": results.experiment_name,
        "variant": variant_name,
        "run_timestamp": results.run_timestamp,
        "guideline_gpt_version": results.guideline_gpt_version,
        "num_questions": len(predictions),
        "total_cost_usd": results.total_cost_usd,
        "total_latency_ms": results.total_latency_ms,
        "config": results.config,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
    )
    return out_dir


def run_experiment(
    config: ExperimentConfig,
    questions: list[EvalQuestion],
    *,
    panel: JudgePanel,
    results_dir: Path,
    factory: PipelineFactory = default_pipeline_factory,
    timestamp: datetime | None = None,
    guideline_gpt_version: str | None = None,
) -> list[ExperimentResults]:
    """Run every variant of an experiment and persist comparable results.

    Args:
        config: The experiment definition.
        questions: The eval set to run through each variant.
        panel: The three judges used to score every prediction.
        results_dir: Root directory for ``{experiment_id}/{variant}/`` outputs.
        factory: Builds a pipeline per variant; defaults to the real
            ``guideline-gpt`` pipeline. Injectable for testing.
        timestamp: Run timestamp recorded in results (defaults to now).
        guideline_gpt_version: Version recorded in results (defaults to the
            installed parent version).

    Returns:
        One :class:`ExperimentResults` per variant, in config order.
    """
    run_time = timestamp if timestamp is not None else datetime.now()
    version = (
        guideline_gpt_version if guideline_gpt_version is not None else guideline_gpt.__version__
    )

    all_results: list[ExperimentResults] = []
    for variant in config.variants:
        log.info("variant_start", experiment=config.experiment_id, variant=variant.name)
        results, predictions = run_variant(
            config,
            variant,
            questions,
            factory=factory,
            panel=panel,
            timestamp=run_time,
            guideline_gpt_version=version,
        )
        persist_variant_results(results_dir, variant.name, results, predictions)
        all_results.append(results)

    return all_results
