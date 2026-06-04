"""Faithfulness judge: does every claim in the answer follow from the chunks?

Faithfulness is about *grounding*, not correctness: an answer can be factually
right yet unfaithful if it asserts things the retrieved context does not
support. The judge sees only the retrieved chunks and the answer — never the
ground truth — so it cannot reward unsupported claims that happen to be true.
"""

from __future__ import annotations

from eval_harness.judges.base import BaseLLMJudge
from eval_harness.types import EvalQuestion, Prediction

# Committed, reviewable few-shot calibration (spec §5.3). Kept in the prompt so a
# reviewer can see exactly how the judge is anchored.
_SYSTEM_PROMPT = """You grade the FAITHFULNESS of a RAG answer to its retrieved context.

Faithfulness = every factual claim in the answer is supported by the provided
context chunks. Judge ONLY against the chunks; do not use outside knowledge and
do not reward claims that are true in reality but absent from the chunks.

Scoring guide:
- 1.0 = every claim is directly supported by the chunks.
- 0.5 = the answer mixes supported claims with one or more unsupported ones.
- 0.0 = the central claim is unsupported by, or contradicts, the chunks.

Respond with ONLY a JSON object: {"score": <float 0..1>, "rationale": "<1-2 sentences>"}.

Examples:

CONTEXT:
[1] Inhaled short-acting beta-agonists are first-line for acute asthma.
ANSWER: First-line treatment is an inhaled short-acting beta-agonist.
VERDICT: {"score": 1.0, "rationale": "The sole claim is directly stated in chunk 1."}

CONTEXT:
[1] Inhaled short-acting beta-agonists are first-line for acute asthma.
ANSWER: Give a short-acting beta-agonist, and start oral antibiotics immediately.
VERDICT: {"score": 0.5, "rationale": "Beta-agonist supported; antibiotics advice not in context."}

CONTEXT:
[1] Inhaled short-acting beta-agonists are first-line for acute asthma.
ANSWER: The first-line treatment is intravenous magnesium sulfate.
VERDICT: {"score": 0.0, "rationale": "Asserts a treatment the context does not support."}"""


def _render_chunks(chunks: list[str]) -> str:
    """Render retrieved chunk texts as a numbered context block."""
    if not chunks:
        return "(no chunks were retrieved)"
    return "\n".join(f"[{i}] {text}" for i, text in enumerate(chunks, start=1))


class FaithfulnessJudge(BaseLLMJudge):
    """Scores whether the answer's claims are grounded in the retrieved chunks."""

    metric_name = "faithfulness"

    def system_prompt(self) -> str:
        """Return the calibrated faithfulness system prompt."""
        return _SYSTEM_PROMPT

    def user_prompt(self, question: EvalQuestion, prediction: Prediction) -> str:
        """Render the context chunks and the answer under evaluation."""
        return (
            f"CONTEXT:\n{_render_chunks(prediction.retrieved_chunks_text)}\n\n"
            f"ANSWER: {prediction.answer}\n\n"
            "VERDICT:"
        )
