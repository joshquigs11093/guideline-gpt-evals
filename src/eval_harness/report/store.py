"""Read persisted experiment results back into typed objects.

The runner writes ``results/{experiment_id}/{variant}/metrics.json`` (an
``ExperimentResults`` serialized via :func:`dataclasses.asdict`). This module is
the inverse: it reconstructs those dataclasses so the dashboard and notebook can
work with typed data instead of raw dicts. The dashboard reads pre-computed
results only — no live experiments, no API keys.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from eval_harness.types import ExperimentResults, Prediction, QuestionScore

METRICS_FILENAME = "metrics.json"
PREDICTIONS_FILENAME = "predictions.jsonl"


class ResultsStoreError(ValueError):
    """Raised when persisted results are missing or malformed."""


def _question_score(data: dict[str, Any]) -> QuestionScore:
    return QuestionScore(**data)


def _experiment_results(data: dict[str, Any]) -> ExperimentResults:
    raw_scores = data.get("question_scores", [])
    if not isinstance(raw_scores, list):
        raise ResultsStoreError("question_scores must be a list")
    scores = [_question_score(item) for item in raw_scores]
    return ExperimentResults(
        experiment_id=str(data["experiment_id"]),
        experiment_name=str(data["experiment_name"]),
        config=data.get("config", {}),
        run_timestamp=datetime.fromisoformat(str(data["run_timestamp"])),
        guideline_gpt_version=str(data["guideline_gpt_version"]),
        question_scores=scores,
        aggregate_metrics=data.get("aggregate_metrics", {}),
        total_cost_usd=float(data["total_cost_usd"]),
        total_latency_ms=int(data["total_latency_ms"]),
    )


def load_variant_results(variant_dir: Path) -> ExperimentResults:
    """Load one variant's :class:`ExperimentResults` from its ``metrics.json``."""
    metrics_path = variant_dir / METRICS_FILENAME
    if not metrics_path.exists():
        raise ResultsStoreError(f"no {METRICS_FILENAME} in {variant_dir}")
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        return _experiment_results(data)
    except (KeyError, ValueError, TypeError) as exc:
        raise ResultsStoreError(f"{metrics_path}: {exc}") from exc


def load_predictions(variant_dir: Path) -> list[Prediction]:
    """Load one variant's predictions from its ``predictions.jsonl`` (if present)."""
    predictions_path = variant_dir / PREDICTIONS_FILENAME
    if not predictions_path.exists():
        return []
    predictions: list[Prediction] = []
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            predictions.append(Prediction(**json.loads(line)))
    return predictions


def load_results(results_dir: Path) -> dict[str, dict[str, ExperimentResults]]:
    """Load every experiment's variant results under ``results_dir``.

    Args:
        results_dir: Root holding ``{experiment_id}/{variant}/metrics.json``.

    Returns:
        ``{experiment_id: {variant_name: ExperimentResults}}``, sorted by id then
        variant. Empty if ``results_dir`` does not exist.
    """
    if not results_dir.exists():
        return {}

    results: dict[str, dict[str, ExperimentResults]] = {}
    for experiment_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        variants: dict[str, ExperimentResults] = {}
        for variant_dir in sorted(p for p in experiment_dir.iterdir() if p.is_dir()):
            if (variant_dir / METRICS_FILENAME).exists():
                variants[variant_dir.name] = load_variant_results(variant_dir)
        if variants:
            results[experiment_dir.name] = variants
    return results
