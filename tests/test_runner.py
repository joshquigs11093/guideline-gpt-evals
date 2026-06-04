"""End-to-end runner tests with a stub pipeline and fake judges (no API, no
chromadb, no corpus) over a 3-question mini eval set (spec §10)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from guideline_gpt.types import (
    Chunk,
    DocumentMetadata,
    QueryResponse,
    QueryTrace,
    RetrievalHit,
)

from eval_harness.runner.experiment import ExperimentConfig, Variant
from eval_harness.runner.runner import run_experiment
from eval_harness.runner.scoring import JudgePanel, score_question, to_prediction
from eval_harness.types import EvalQuestion, Prediction


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
def _chunk(chunk_id: str) -> Chunk:
    meta = DocumentMetadata(source_path="p.pdf", source_name="p", page_number=1)
    return Chunk(chunk_id=chunk_id, text=f"text-{chunk_id}", metadata=meta, token_count=10)


def make_response(retrieved_ids: list[str], *, answer: str = "an answer") -> QueryResponse:
    hits = [
        RetrievalHit(chunk=_chunk(cid), score=1.0 / rank, source="rerank", rank=rank)
        for rank, cid in enumerate(retrieved_ids, start=1)
    ]
    trace = QueryTrace(query="q", timestamp=datetime(2026, 1, 1))
    trace.reranked_hits = hits
    trace.llm_model = "claude-haiku-4-5"
    trace.llm_input_tokens = 1000
    trace.llm_output_tokens = 100
    trace.llm_latency_ms = 500
    return QueryResponse(answer=answer, citations=[h.chunk for h in hits[:1]], trace=trace)


class StubPipeline:
    def __init__(self, responder):
        self._responder = responder
        self.queries: list[str] = []

    def query(self, question: str) -> QueryResponse:
        self.queries.append(question)
        return self._responder(question)


class RecordingFactory:
    def __init__(self, responder):
        self._responder = responder
        self.built: list[StubPipeline] = []

    def __call__(self, variant: Variant, shared_config: dict) -> StubPipeline:
        pipeline = StubPipeline(self._responder)
        self.built.append(pipeline)
        return pipeline


class FixedJudge:
    def __init__(self, score: float, rationale: str = "ok"):
        self._score = score
        self._rationale = rationale

    def score(self, *, question: EvalQuestion, prediction: Prediction) -> tuple[float, str]:
        return self._score, self._rationale


# --------------------------------------------------------------------------- #
# Mini eval set + responder
# --------------------------------------------------------------------------- #
def _q(qid: str, relevant: list[str], difficulty: str = "easy") -> EvalQuestion:
    return EvalQuestion(
        question_id=qid,
        question=f"question {qid}",
        ground_truth_answer="truth",
        relevant_chunk_ids=relevant,
        source_documents=["p.pdf"],
        difficulty=difficulty,  # type: ignore[arg-type]
        question_type="factoid",
    )


MINI_SET = [
    _q("q1", ["a", "b"]),  # perfect retrieval
    _q("q2", ["c"], difficulty="hard"),  # relevant at rank 2
    _q("q3", ["z"], difficulty="hard"),  # retrieval miss
]

_RESPONSES = {
    "question q1": make_response(["a", "b", "d"]),
    "question q2": make_response(["x", "c", "y"]),
    "question q3": make_response(["m", "n", "o"]),
}


def _responder(question: str) -> QueryResponse:
    return _RESPONSES[question]


def _panel() -> JudgePanel:
    return JudgePanel(
        faithfulness=FixedJudge(0.9),
        relevance=FixedJudge(0.8),
        correctness=FixedJudge(0.7),
    )


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="exp_test",
        name="Test experiment",
        variants=[Variant(name="baseline"), Variant(name="variant_b")],
        shared_config={"retrieval_top_k": 10},
    )


# --------------------------------------------------------------------------- #
# Unit: mapping + scoring
# --------------------------------------------------------------------------- #
def test_to_prediction_maps_trace_fields() -> None:
    pred = to_prediction("q1", make_response(["a", "b"]))
    assert pred.retrieved_chunk_ids == ["a", "b"]
    assert pred.input_tokens == 1000
    assert pred.output_tokens == 100
    assert pred.latency_ms == 500
    # claude-haiku price: (1000*1.0 + 100*5.0) / 1e6
    assert pred.cost_usd == pytest.approx(0.0015)


def test_score_question_combines_retrieval_and_judges() -> None:
    pred = to_prediction("q1", make_response(["a", "b", "d"]))
    score = score_question(_q("q1", ["a", "b"]), pred, _panel())
    assert score.mrr == pytest.approx(1.0)  # first relevant at rank 1
    assert score.recall_at_5 == pytest.approx(1.0)  # both relevant retrieved
    assert score.precision_at_5 == pytest.approx(2 / 5)
    assert (score.faithfulness, score.relevance, score.correctness) == (0.9, 0.8, 0.7)


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def test_run_experiment_scores_all_variants(tmp_path: Path) -> None:
    factory = RecordingFactory(_responder)
    results = run_experiment(
        _config(),
        MINI_SET,
        panel=_panel(),
        results_dir=tmp_path,
        factory=factory,
        timestamp=datetime(2026, 6, 4, 12, 0, 0),
        guideline_gpt_version="0.1.0-test",
    )

    assert len(results) == 2
    for res in results:
        assert len(res.question_scores) == 3
        assert res.aggregate_metrics["correctness_mean"] == pytest.approx(0.7)
        assert res.guideline_gpt_version == "0.1.0-test"
        # 3 questions * 0.0015 each.
        assert res.total_cost_usd == pytest.approx(0.0045)

    # Each variant's stub was queried once per question.
    assert all(len(p.queries) == 3 for p in factory.built)


def test_run_experiment_persists_files(tmp_path: Path) -> None:
    run_experiment(
        _config(),
        MINI_SET,
        panel=_panel(),
        results_dir=tmp_path,
        factory=RecordingFactory(_responder),
        timestamp=datetime(2026, 6, 4, 12, 0, 0),
        guideline_gpt_version="0.1.0-test",
    )

    variant_dir = tmp_path / "exp_test" / "baseline"
    assert (variant_dir / "predictions.jsonl").exists()
    assert (variant_dir / "metrics.json").exists()
    assert (variant_dir / "manifest.json").exists()

    predictions = [
        json.loads(line)
        for line in (variant_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(predictions) == 3
    assert {p["question_id"] for p in predictions} == {"q1", "q2", "q3"}

    metrics = json.loads((variant_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["experiment_id"] == "exp_test"
    assert len(metrics["question_scores"]) == 3

    manifest = json.loads((variant_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_questions"] == 3
    assert manifest["run_timestamp"] == "2026-06-04T12:00:00"


def test_per_difficulty_aggregates_present(tmp_path: Path) -> None:
    results = run_experiment(
        _config(),
        MINI_SET,
        panel=_panel(),
        results_dir=tmp_path,
        factory=RecordingFactory(_responder),
    )
    agg = results[0].aggregate_metrics
    assert "correctness_mean[difficulty=hard]" in agg
    assert "recall_at_5_mean[difficulty=easy]" in agg
