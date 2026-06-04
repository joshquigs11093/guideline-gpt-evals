"""Turn a pipeline response into a scored :class:`QuestionScore`.

Two steps: map the parent's :class:`QueryResponse` into our serializable
:class:`Prediction`, then score that prediction with the deterministic retrieval
metrics and the three LLM judges.

We deliberately import only the parent's lightweight ``types`` module (no
chromadb) and own a small cost table here, so this layer stays fast and testable
without spinning up the retrieval stack.
"""

from __future__ import annotations

from dataclasses import dataclass

from guideline_gpt.types import QueryResponse, QueryTrace, RetrievalHit

from eval_harness.judges.base import Judge
from eval_harness.metrics.retrieval import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from eval_harness.types import EvalQuestion, Prediction, QuestionScore

#: Retrieval metrics are reported at this cutoff (matches the QuestionScore fields).
RETRIEVAL_K = 5

# Per-million-token USD prices, matched by model-name prefix. Mirrors the parent
# project's table; owned here so scoring needs no heavyweight imports. Update as
# provider pricing changes.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-haiku": (1.00, 5.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-opus": (15.00, 75.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a completion; ``0.0`` if the model is unknown."""
    for prefix, (in_price, out_price) in _PRICES.items():
        if model.startswith(prefix):
            return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
    return 0.0


@dataclass(frozen=True)
class JudgePanel:
    """The three LLM judges applied to every prediction."""

    faithfulness: Judge
    relevance: Judge
    correctness: Judge


def _context_hits(trace: QueryTrace) -> list[RetrievalHit]:
    """Return the hits whose chunks were the LLM's context.

    Prefers the reranked set; falls back through fusion and the single-arm
    results so a stage failure upstream still yields the chunks actually used.
    """
    hits: list[RetrievalHit] = (
        trace.reranked_hits or trace.fused_hits or trace.vector_hits or trace.bm25_hits
    )
    return hits


def to_prediction(question_id: str, response: QueryResponse) -> Prediction:
    """Map a parent :class:`QueryResponse` into a serializable :class:`Prediction`."""
    trace = response.trace
    hits = _context_hits(trace)
    return Prediction(
        question_id=question_id,
        retrieved_chunk_ids=[hit.chunk.chunk_id for hit in hits],
        retrieved_chunks_text=[hit.chunk.text for hit in hits],
        answer=response.answer,
        citations=[chunk.chunk_id for chunk in response.citations],
        latency_ms=trace.llm_latency_ms,
        input_tokens=trace.llm_input_tokens,
        output_tokens=trace.llm_output_tokens,
        cost_usd=estimate_cost(trace.llm_model, trace.llm_input_tokens, trace.llm_output_tokens),
    )


def score_question(
    question: EvalQuestion, prediction: Prediction, panel: JudgePanel
) -> QuestionScore:
    """Score one prediction with retrieval metrics and the judge panel."""
    relevant = question.relevant_chunk_ids
    retrieved = prediction.retrieved_chunk_ids

    faithfulness, faithfulness_rationale = panel.faithfulness.score(
        question=question, prediction=prediction
    )
    relevance, relevance_rationale = panel.relevance.score(question=question, prediction=prediction)
    correctness, correctness_rationale = panel.correctness.score(
        question=question, prediction=prediction
    )

    return QuestionScore(
        question_id=question.question_id,
        precision_at_5=precision_at_k(retrieved, relevant, RETRIEVAL_K),
        recall_at_5=recall_at_k(retrieved, relevant, RETRIEVAL_K),
        mrr=reciprocal_rank(retrieved, relevant),
        ndcg_at_5=ndcg_at_k(retrieved, relevant, RETRIEVAL_K),
        faithfulness=faithfulness,
        relevance=relevance,
        correctness=correctness,
        faithfulness_rationale=faithfulness_rationale,
        relevance_rationale=relevance_rationale,
        correctness_rationale=correctness_rationale,
    )
