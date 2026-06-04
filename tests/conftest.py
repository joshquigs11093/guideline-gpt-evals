"""Shared pytest fixtures for the eval-harness test suite."""

from __future__ import annotations

import pytest

from eval_harness.types import EvalQuestion, Prediction


@pytest.fixture
def sample_question() -> EvalQuestion:
    """A minimal, valid EvalQuestion for use across tests."""
    return EvalQuestion(
        question_id="q-0001",
        question="What is the first-line treatment for acute asthma exacerbation?",
        ground_truth_answer="Inhaled short-acting beta-agonists (SABA).",
        relevant_chunk_ids=["chunk-a", "chunk-b"],
        source_documents=["asthma_guideline.pdf"],
        difficulty="medium",
        question_type="factoid",
    )


@pytest.fixture
def sample_prediction() -> Prediction:
    """A minimal, valid Prediction for use across tests."""
    return Prediction(
        question_id="q-0001",
        retrieved_chunk_ids=["chunk-a", "chunk-c"],
        retrieved_chunks_text=["...", "..."],
        answer="Give inhaled SABA such as albuterol.",
        citations=["1"],
        latency_ms=820,
        input_tokens=1200,
        output_tokens=64,
        cost_usd=0.0012,
    )
