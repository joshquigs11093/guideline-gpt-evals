"""Correctness judge: does the answer match the ground truth in substance?

Correctness compares the answer against the curated ``ground_truth_answer``,
rewarding substantive agreement rather than surface wording. Extra correct
detail is fine; missing or contradicting the key facts is not.
"""

from __future__ import annotations

from eval_harness.judges.base import BaseLLMJudge
from eval_harness.types import EvalQuestion, Prediction

# Committed, reviewable few-shot calibration (spec §5.3).
_SYSTEM_PROMPT = """You grade the CORRECTNESS of a RAG answer against a reference answer.

Correctness = the answer agrees in substance with the reference. Judge meaning,
not wording: paraphrases and extra correct detail are fine. Penalize answers
that miss, contradict, or distort the key facts in the reference.

Scoring guide:
- 1.0 = substantively matches the reference (wording may differ).
- 0.5 = partially correct: captures some key facts but misses or muddles others.
- 0.0 = contradicts the reference or misses its central point.

Respond with ONLY a JSON object: {"score": <float 0..1>, "rationale": "<1-2 sentences>"}.

Examples:

QUESTION: What is the first-line treatment for acute asthma?
REFERENCE: Inhaled short-acting beta-agonists (SABA).
ANSWER: Give an inhaled SABA such as albuterol.
VERDICT: {"score": 1.0, "rationale": "Matches the reference SABA class with a correct example."}

QUESTION: What is the first-line treatment for acute asthma?
REFERENCE: Inhaled short-acting beta-agonists (SABA), with oxygen if hypoxemic.
ANSWER: Inhaled short-acting beta-agonists.
VERDICT: {"score": 0.5, "rationale": "Correct on the SABA but omits the oxygen element."}

QUESTION: What is the first-line treatment for acute asthma?
REFERENCE: Inhaled short-acting beta-agonists (SABA).
ANSWER: Oral corticosteroids are the first-line treatment.
VERDICT: {"score": 0.0, "rationale": "Names a different drug class than the reference."}"""


class CorrectnessJudge(BaseLLMJudge):
    """Scores whether the answer matches the ground-truth answer in substance."""

    metric_name = "correctness"

    def system_prompt(self) -> str:
        """Return the calibrated correctness system prompt."""
        return _SYSTEM_PROMPT

    def user_prompt(self, question: EvalQuestion, prediction: Prediction) -> str:
        """Render the question, the reference answer, and the answer under test."""
        return (
            f"QUESTION: {question.question}\n\n"
            f"REFERENCE: {question.ground_truth_answer}\n\n"
            f"ANSWER: {prediction.answer}\n\n"
            "VERDICT:"
        )
