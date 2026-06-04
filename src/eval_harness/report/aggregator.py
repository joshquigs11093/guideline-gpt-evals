"""Roll per-question scores into experiment-level aggregate metrics.

Produces a flat ``dict[str, float]`` (the shape :class:`ExperimentResults`
expects) holding, for each metric: its mean and population standard deviation
overall, plus per-difficulty and per-question-type means. Breakdown keys are
namespaced, e.g. ``correctness_mean[difficulty=hard]``.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from eval_harness.types import EvalQuestion, QuestionScore

#: The numeric metric fields on :class:`QuestionScore`, in report order.
METRIC_FIELDS: tuple[str, ...] = (
    "precision_at_5",
    "recall_at_5",
    "mrr",
    "ndcg_at_5",
    "faithfulness",
    "relevance",
    "correctness",
)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _values(scores: list[QuestionScore], metric: str) -> list[float]:
    return [float(getattr(score, metric)) for score in scores]


def aggregate_scores(
    questions: list[EvalQuestion], scores: list[QuestionScore]
) -> dict[str, float]:
    """Aggregate per-question scores into experiment-level metrics.

    Args:
        questions: The eval questions, used for difficulty/type breakdowns.
        scores: The per-question scores to aggregate.

    Returns:
        A flat mapping of metric name to value (means, std devs, and namespaced
        per-difficulty / per-question-type means). Empty if there are no scores.
    """
    if not scores:
        return {}

    question_by_id = {question.question_id: question for question in questions}
    aggregates: dict[str, float] = {}

    for metric in METRIC_FIELDS:
        values = _values(scores, metric)
        aggregates[f"{metric}_mean"] = _mean(values)
        aggregates[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0

    for dimension, attribute in (("difficulty", "difficulty"), ("type", "question_type")):
        groups: dict[str, list[QuestionScore]] = defaultdict(list)
        for score in scores:
            question = question_by_id.get(score.question_id)
            if question is not None:
                groups[str(getattr(question, attribute))].append(score)
        for key, group in groups.items():
            for metric in METRIC_FIELDS:
                aggregates[f"{metric}_mean[{dimension}={key}]"] = _mean(_values(group, metric))

    return aggregates
