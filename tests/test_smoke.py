"""Smoke tests verifying the M0 scaffolding: contract types, config, and that
the system under test (guideline-gpt) is importable as a dependency."""

from __future__ import annotations

import eval_harness
from eval_harness.config import Settings, get_settings
from eval_harness.types import EvalQuestion, Prediction


def test_package_has_version() -> None:
    assert eval_harness.__version__ == "0.1.0"


def test_settings_defaults() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    # The judge defaults to a strong Anthropic model (spec §5.3 / ADR-002).
    assert settings.judge_provider == "anthropic"
    assert settings.judge_model.startswith("claude-opus")


def test_eval_question_is_frozen(sample_question: EvalQuestion) -> None:
    assert sample_question.is_hard_case is False
    assert sample_question.relevant_chunk_ids == ["chunk-a", "chunk-b"]


def test_prediction_roundtrips_fields(sample_prediction: Prediction) -> None:
    assert sample_prediction.question_id == "q-0001"
    assert sample_prediction.cost_usd > 0


def test_guideline_gpt_importable() -> None:
    """M0 acceptance: the system under test imports as a dependency."""
    from guideline_gpt.config import Settings as GgSettings
    from guideline_gpt.pipeline import QueryPipeline

    assert QueryPipeline is not None
    assert GgSettings is not None
