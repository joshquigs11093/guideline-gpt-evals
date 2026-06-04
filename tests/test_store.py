"""Round-trip tests: persist results with the runner, read them back."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from eval_harness.report.store import (
    ResultsStoreError,
    load_predictions,
    load_results,
    load_variant_results,
)
from eval_harness.runner.runner import persist_variant_results, variant_results_dir
from eval_harness.types import ExperimentResults, Prediction, QuestionScore


def _score(qid: str) -> QuestionScore:
    return QuestionScore(
        question_id=qid,
        precision_at_5=0.4,
        recall_at_5=1.0,
        mrr=1.0,
        ndcg_at_5=0.9,
        faithfulness=0.9,
        relevance=0.8,
        correctness=0.7,
        faithfulness_rationale="f",
        relevance_rationale="r",
        correctness_rationale="c",
    )


def _prediction(qid: str) -> Prediction:
    return Prediction(
        question_id=qid,
        retrieved_chunk_ids=["a", "b"],
        retrieved_chunks_text=["ta", "tb"],
        answer="ans",
        citations=["a"],
        latency_ms=500,
        input_tokens=1000,
        output_tokens=100,
        cost_usd=0.0015,
    )


def _results() -> ExperimentResults:
    return ExperimentResults(
        experiment_id="exp_test",
        experiment_name="Test",
        config={"variant": {"name": "baseline"}, "shared_config": {"retrieval_top_k": 10}},
        run_timestamp=datetime(2026, 6, 4, 12, 0, 0),
        guideline_gpt_version="0.1.0-test",
        question_scores=[_score("q1"), _score("q2")],
        aggregate_metrics={"correctness_mean": 0.7},
        total_cost_usd=0.003,
        total_latency_ms=1000,
    )


def _persist(tmp_path: Path) -> Path:
    results = _results()
    persist_variant_results(tmp_path, "baseline", results, [_prediction("q1"), _prediction("q2")])
    return variant_results_dir(tmp_path, "exp_test", "baseline")


def test_round_trips_variant_results(tmp_path: Path) -> None:
    loaded = load_variant_results(_persist(tmp_path))
    assert loaded.experiment_id == "exp_test"
    assert loaded.run_timestamp == datetime(2026, 6, 4, 12, 0, 0)
    assert len(loaded.question_scores) == 2
    assert loaded.question_scores[0].correctness == pytest.approx(0.7)
    assert loaded.aggregate_metrics["correctness_mean"] == pytest.approx(0.7)


def test_round_trips_predictions(tmp_path: Path) -> None:
    predictions = load_predictions(_persist(tmp_path))
    assert len(predictions) == 2
    assert predictions[0].retrieved_chunk_ids == ["a", "b"]
    assert predictions[0].cost_usd == pytest.approx(0.0015)


def test_load_results_indexes_by_experiment_and_variant(tmp_path: Path) -> None:
    _persist(tmp_path)
    results = load_results(tmp_path)
    assert "exp_test" in results
    assert "baseline" in results["exp_test"]
    assert results["exp_test"]["baseline"].total_cost_usd == pytest.approx(0.003)


def test_load_results_missing_dir_is_empty(tmp_path: Path) -> None:
    assert load_results(tmp_path / "nope") == {}


def test_missing_metrics_file_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ResultsStoreError, match="no metrics.json"):
        load_variant_results(tmp_path / "empty")
