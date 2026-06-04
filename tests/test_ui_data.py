"""Tests for the dashboard's pure dataframe builders (no Streamlit runtime)."""

from __future__ import annotations

from datetime import datetime

from eval_harness.types import EvalQuestion, ExperimentResults, QuestionScore
from eval_harness.ui.data import (
    SYNTHETIC_VERSION,
    aggregate_long_df,
    is_synthetic,
    question_long_df,
    variant_summary_df,
)


def _score(qid: str, correctness: float) -> QuestionScore:
    return QuestionScore(
        question_id=qid,
        precision_at_5=0.5,
        recall_at_5=0.6,
        mrr=0.7,
        ndcg_at_5=0.65,
        faithfulness=0.8,
        relevance=0.85,
        correctness=correctness,
        faithfulness_rationale="f",
        relevance_rationale="r",
        correctness_rationale="c",
    )


def _results(version: str = SYNTHETIC_VERSION) -> dict[str, dict[str, ExperimentResults]]:
    def make(variant: str, correctness_mean: float) -> ExperimentResults:
        return ExperimentResults(
            experiment_id="exp1",
            experiment_name="Experiment 1",
            config={},
            run_timestamp=datetime(2026, 6, 4),
            guideline_gpt_version=version,
            question_scores=[_score("q1", correctness_mean), _score("q2", correctness_mean)],
            aggregate_metrics={"correctness_mean": correctness_mean, "recall_at_5_mean": 0.6},
            total_cost_usd=0.01,
            total_latency_ms=1000,
        )

    return {"exp1": {"a": make("a", 0.7), "b": make("b", 0.5)}}


QUESTIONS = [
    EvalQuestion("q1", "?", "a", ["c"], ["d.pdf"], "easy", "factoid"),
    EvalQuestion("q2", "?", "a", ["c"], ["d.pdf"], "hard", "synthesis", is_hard_case=True),
]


def test_variant_summary_has_row_per_variant() -> None:
    df = variant_summary_df(_results())
    assert len(df) == 2
    assert set(df["variant"]) == {"a", "b"}
    assert df.loc[df["variant"] == "a", "correctness"].iloc[0] == 0.7


def test_aggregate_long_df_is_melted() -> None:
    df = aggregate_long_df(_results())
    # 2 variants * 7 metrics.
    assert len(df) == 14
    assert set(df["metric"]) >= {"correctness", "recall_at_5"}


def test_question_long_df_joins_metadata() -> None:
    df = question_long_df(_results(), QUESTIONS)
    assert len(df) == 4  # 2 variants * 2 questions
    hard = df[df["question_id"] == "q2"]
    assert (hard["difficulty"] == "hard").all()
    assert hard["is_hard_case"].all()


def test_is_synthetic_detects_marker() -> None:
    assert is_synthetic(_results(SYNTHETIC_VERSION)) is True
    assert is_synthetic(_results("0.1.0")) is False


def test_builders_handle_empty() -> None:
    assert variant_summary_df({}).empty
    assert aggregate_long_df({}).empty
