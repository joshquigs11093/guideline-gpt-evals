"""Tests for rolling per-question scores into experiment aggregates."""

from __future__ import annotations

import pytest

from eval_harness.report.aggregator import aggregate_scores
from eval_harness.types import EvalQuestion, QuestionScore


def _question(qid: str, difficulty: str, qtype: str) -> EvalQuestion:
    return EvalQuestion(
        question_id=qid,
        question="q",
        ground_truth_answer="a",
        relevant_chunk_ids=["c"],
        source_documents=["d.pdf"],
        difficulty=difficulty,  # type: ignore[arg-type]
        question_type=qtype,  # type: ignore[arg-type]
    )


def _score(qid: str, correctness: float) -> QuestionScore:
    return QuestionScore(
        question_id=qid,
        precision_at_5=0.0,
        recall_at_5=0.0,
        mrr=0.0,
        ndcg_at_5=0.0,
        faithfulness=0.0,
        relevance=0.0,
        correctness=correctness,
        faithfulness_rationale="",
        relevance_rationale="",
        correctness_rationale="",
    )


QUESTIONS = [
    _question("q1", "easy", "factoid"),
    _question("q2", "hard", "synthesis"),
    _question("q3", "hard", "factoid"),
]
SCORES = [_score("q1", 1.0), _score("q2", 0.0), _score("q3", 0.5)]


def test_empty_scores_yield_empty() -> None:
    assert aggregate_scores(QUESTIONS, []) == {}


def test_overall_mean_and_std() -> None:
    agg = aggregate_scores(QUESTIONS, SCORES)
    assert agg["correctness_mean"] == pytest.approx(0.5)
    assert agg["correctness_std"] > 0.0
    # A metric that is uniformly zero has zero spread.
    assert agg["faithfulness_mean"] == 0.0
    assert agg["faithfulness_std"] == 0.0


def test_per_difficulty_breakdown() -> None:
    agg = aggregate_scores(QUESTIONS, SCORES)
    assert agg["correctness_mean[difficulty=easy]"] == pytest.approx(1.0)
    # hard = mean(q2=0.0, q3=0.5)
    assert agg["correctness_mean[difficulty=hard]"] == pytest.approx(0.25)


def test_per_question_type_breakdown() -> None:
    agg = aggregate_scores(QUESTIONS, SCORES)
    # factoid = mean(q1=1.0, q3=0.5)
    assert agg["correctness_mean[type=factoid]"] == pytest.approx(0.75)
    assert agg["correctness_mean[type=synthesis]"] == pytest.approx(0.0)
