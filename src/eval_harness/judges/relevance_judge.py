"""Relevance judge: does the answer address what the question actually asked?

Relevance is independent of correctness and faithfulness: a relevant answer
engages the question directly, without dodging, padding, or answering a
different question. It may still be wrong (that is the correctness judge's job).
"""

from __future__ import annotations

from eval_harness.judges.base import BaseLLMJudge
from eval_harness.types import EvalQuestion, Prediction

# Committed, reviewable few-shot calibration (spec §5.3).
_SYSTEM_PROMPT = """You grade the RELEVANCE of a RAG answer to the question asked.

Relevance = the answer directly addresses what was asked, without evasion or
off-topic content. Do NOT judge factual accuracy here — only whether the answer
engages the actual question. Penalize non-answers ("I don't know"), answers to a
different question, and answers buried in irrelevant padding.

Scoring guide:
- 1.0 = squarely answers the question.
- 0.5 = partially on-topic, or answers only part of the question.
- 0.0 = evasive, off-topic, or answers a different question.

Respond with ONLY a JSON object: {"score": <float 0..1>, "rationale": "<1-2 sentences>"}.

Examples:

QUESTION: What is the first-line treatment for acute asthma?
ANSWER: First-line treatment is an inhaled short-acting beta-agonist.
VERDICT: {"score": 1.0, "rationale": "Directly answers the treatment question."}

QUESTION: What is the first-line treatment for acute asthma, and when is intubation indicated?
ANSWER: First-line treatment is an inhaled short-acting beta-agonist.
VERDICT: {"score": 0.5, "rationale": "Answers treatment but ignores the intubation question."}

QUESTION: What is the first-line treatment for acute asthma?
ANSWER: Asthma is a chronic inflammatory disease affecting millions worldwide.
VERDICT: {"score": 0.0, "rationale": "Describes the disease instead of naming a treatment."}"""


class RelevanceJudge(BaseLLMJudge):
    """Scores whether the answer addresses the question that was asked."""

    metric_name = "relevance"

    def system_prompt(self) -> str:
        """Return the calibrated relevance system prompt."""
        return _SYSTEM_PROMPT

    def user_prompt(self, question: EvalQuestion, prediction: Prediction) -> str:
        """Render the question and the answer under evaluation."""
        return f"QUESTION: {question.question}\n\nANSWER: {prediction.answer}\n\nVERDICT:"
