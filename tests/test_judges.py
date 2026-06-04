"""Tests for the LLM-as-judge implementations, driven by a fake LLM client so no
real API calls happen (spec §10/§11)."""

from __future__ import annotations

import pytest
from guideline_gpt.generation.llm_client import (
    AnthropicClient,
    CompletionResult,
    OpenAIClient,
)

from eval_harness.config import Settings
from eval_harness.judges.base import FALLBACK_RATIONALE, get_judge_client
from eval_harness.judges.correctness_judge import CorrectnessJudge
from eval_harness.judges.faithfulness_judge import FaithfulnessJudge
from eval_harness.judges.relevance_judge import RelevanceJudge


class FakeLLMClient:
    """Returns canned completion texts in sequence; repeats the last one."""

    def __init__(self, responses: list[str]) -> None:
        assert responses, "need at least one canned response"
        self._responses = responses
        self.calls = 0
        self.last_system = ""
        self.last_user = ""

    def complete(self, system: str, user: str) -> CompletionResult:
        self.calls += 1
        self.last_system = system
        self.last_user = user
        text = self._responses[0]
        if len(self._responses) > 1:
            self._responses.pop(0)
        return CompletionResult(
            text=text, input_tokens=10, output_tokens=5, latency_ms=1, model="judge-test"
        )


def _settings(**overrides: object) -> Settings:
    # Ignore any local .env so tests are hermetic.
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


JUDGES = [FaithfulnessJudge, RelevanceJudge, CorrectnessJudge]


@pytest.mark.parametrize("judge_cls", JUDGES)
class TestHappyPath:
    def test_parses_score_and_rationale(self, judge_cls, sample_question, sample_prediction):
        client = FakeLLMClient(['{"score": 0.8, "rationale": "looks good"}'])
        judge = judge_cls(client)
        score, rationale = judge.score(question=sample_question, prediction=sample_prediction)
        assert score == pytest.approx(0.8)
        assert rationale == "looks good"
        assert client.calls == 1

    def test_extracts_json_from_prose_and_fence(
        self, judge_cls, sample_question, sample_prediction
    ):
        wrapped = 'Sure, here is my verdict:\n```json\n{"score": 0.4, "rationale": "partial"}\n```'
        judge = judge_cls(FakeLLMClient([wrapped]))
        score, rationale = judge.score(question=sample_question, prediction=sample_prediction)
        assert score == pytest.approx(0.4)
        assert rationale == "partial"


class TestScoreClamping:
    def test_overshoot_clamped_to_one(self, sample_question, sample_prediction):
        judge = FaithfulnessJudge(FakeLLMClient(['{"score": 1.5, "rationale": "x"}']))
        score, _ = judge.score(question=sample_question, prediction=sample_prediction)
        assert score == 1.0

    def test_undershoot_clamped_to_zero(self, sample_question, sample_prediction):
        judge = RelevanceJudge(FakeLLMClient(['{"score": -0.2, "rationale": "x"}']))
        score, _ = judge.score(question=sample_question, prediction=sample_prediction)
        assert score == 0.0


class TestRetryAndFallback:
    def test_retries_then_succeeds(self, sample_question, sample_prediction):
        client = FakeLLMClient(["not json at all", '{"score": 0.7, "rationale": "ok"}'])
        judge = CorrectnessJudge(client, max_retries=2)
        score, rationale = judge.score(question=sample_question, prediction=sample_prediction)
        assert score == pytest.approx(0.7)
        assert rationale == "ok"
        assert client.calls == 2

    def test_persistent_failure_falls_back(self, sample_question, sample_prediction):
        client = FakeLLMClient(["never valid json"])
        judge = CorrectnessJudge(client, max_retries=2)
        score, rationale = judge.score(question=sample_question, prediction=sample_prediction)
        assert score == 0.0
        assert rationale == FALLBACK_RATIONALE
        # 1 initial attempt + 2 retries.
        assert client.calls == 3

    def test_missing_required_field_is_treated_as_failure(self, sample_question, sample_prediction):
        # Valid JSON but no "rationale" -> validation fails -> fallback.
        client = FakeLLMClient(['{"score": 0.9}'])
        judge = RelevanceJudge(client, max_retries=0)
        score, rationale = judge.score(question=sample_question, prediction=sample_prediction)
        assert (score, rationale) == (0.0, FALLBACK_RATIONALE)
        assert client.calls == 1


class TestPromptContent:
    def test_correctness_prompt_includes_reference(self, sample_question, sample_prediction):
        client = FakeLLMClient(['{"score": 1.0, "rationale": "r"}'])
        CorrectnessJudge(client).score(question=sample_question, prediction=sample_prediction)
        assert sample_question.ground_truth_answer in client.last_user

    def test_faithfulness_prompt_includes_chunks_not_reference(
        self, sample_question, sample_prediction
    ):
        client = FakeLLMClient(['{"score": 1.0, "rationale": "r"}'])
        FaithfulnessJudge(client).score(question=sample_question, prediction=sample_prediction)
        # Faithfulness must judge against retrieved chunks, never the ground truth.
        assert sample_question.ground_truth_answer not in client.last_user

    def test_faithfulness_prompt_handles_retrieval_miss(self, sample_question, sample_prediction):
        # A total retrieval miss (no chunks) must still produce a well-formed prompt.
        from dataclasses import replace

        client = FakeLLMClient(['{"score": 0.0, "rationale": "r"}'])
        empty = replace(sample_prediction, retrieved_chunk_ids=[], retrieved_chunks_text=[])
        FaithfulnessJudge(client).score(question=sample_question, prediction=empty)
        assert "(no chunks were retrieved)" in client.last_user


class TestGetJudgeClient:
    def test_anthropic_requires_key(self):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            get_judge_client(_settings(judge_provider="anthropic", anthropic_api_key=None))

    def test_openai_requires_key(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_judge_client(_settings(judge_provider="openai", openai_api_key=None))

    def test_builds_anthropic_client(self):
        client = get_judge_client(_settings(judge_provider="anthropic", anthropic_api_key="sk-x"))
        assert isinstance(client, AnthropicClient)

    def test_builds_openai_client(self):
        client = get_judge_client(_settings(judge_provider="openai", openai_api_key="sk-x"))
        assert isinstance(client, OpenAIClient)
