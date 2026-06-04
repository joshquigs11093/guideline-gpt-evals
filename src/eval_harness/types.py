"""Shared data types — the contract between every layer of the eval harness.

These dataclasses are serialized directly to the JSONL/JSON files under
``results/`` and consumed by the report and dashboard layers. No untyped dicts
cross a module boundary (see the engineering principles in the spec). Mirrors
the convention used by the parent ``guideline_gpt.types`` module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Difficulty = Literal["easy", "medium", "hard"]
QuestionType = Literal["factoid", "synthesis", "comparison", "ambiguous"]


@dataclass(frozen=True)
class EvalQuestion:
    """A single question in the eval dataset."""

    question_id: str
    question: str
    ground_truth_answer: str
    relevant_chunk_ids: list[str]
    source_documents: list[str]
    difficulty: Difficulty
    question_type: QuestionType
    is_hard_case: bool = False
    rationale: str = ""


@dataclass(frozen=True)
class Prediction:
    """A single RAG system's response to one question."""

    question_id: str
    retrieved_chunk_ids: list[str]
    retrieved_chunks_text: list[str]
    answer: str
    citations: list[str]
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class QuestionScore:
    """All metrics for one question, one experiment variant."""

    question_id: str
    precision_at_5: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    faithfulness: float
    relevance: float
    correctness: float
    faithfulness_rationale: str
    relevance_rationale: str
    correctness_rationale: str


@dataclass(frozen=True)
class ExperimentResults:
    """Aggregated results for one experiment variant."""

    experiment_id: str
    experiment_name: str
    config: dict[str, object]
    run_timestamp: datetime
    guideline_gpt_version: str
    question_scores: list[QuestionScore]
    aggregate_metrics: dict[str, float]
    total_cost_usd: float
    total_latency_ms: int
