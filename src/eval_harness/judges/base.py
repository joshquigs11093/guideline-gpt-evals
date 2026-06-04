"""Judge protocol and the shared LLM-as-judge base implementation.

The base class encapsulates the parts every judge shares: prompt the judge
model for a structured JSON verdict, validate it, retry a bounded number of
times on malformed output, and fall back to a safe ``0.0`` score on persistent
failure rather than letting a bad judge call abort a whole experiment.

Concrete judges supply only their system/user prompts (see the judge modules in
this package). The judge model is reached through the parent project's
provider-agnostic ``LLMClient`` so we do not reinvent provider plumbing.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Protocol

from guideline_gpt.generation.llm_client import AnthropicClient, LLMClient, OpenAIClient
from pydantic import BaseModel

from eval_harness.config import Settings
from eval_harness.logging_setup import get_logger
from eval_harness.types import EvalQuestion, Prediction

log = get_logger(__name__)

# A judge response may arrive wrapped in prose or a ```json fence; grab the
# outermost brace-delimited object and parse that.
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

FALLBACK_RATIONALE = "judge failed to produce valid output"


class Judge(Protocol):
    """Scores one prediction against one question on a single metric."""

    def score(self, *, question: EvalQuestion, prediction: Prediction) -> tuple[float, str]:
        """Return ``(score in [0, 1], rationale)``."""
        ...


class JudgeVerdict(BaseModel):
    """The structured response we require from the judge model."""

    score: float
    rationale: str


def _clamp_unit(value: float) -> float:
    """Clamp a score into ``[0, 1]`` (judges occasionally over/undershoot)."""
    return max(0.0, min(1.0, value))


def get_judge_client(settings: Settings) -> LLMClient:
    """Construct the judge LLM client from eval-harness settings.

    Reuses the parent project's concrete clients but drives them from the
    eval-harness ``JUDGE_PROVIDER`` / ``JUDGE_MODEL`` rather than the parent's own
    LLM selection, so the judge can be a stronger model than the system tested.

    Args:
        settings: Eval-harness configuration.

    Returns:
        A provider-agnostic :class:`LLMClient` bound to the judge model.

    Raises:
        ValueError: If the API key for the selected judge provider is missing.
    """
    if settings.judge_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when JUDGE_PROVIDER=anthropic.")
        return AnthropicClient(settings.anthropic_api_key, settings.judge_model)

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required when JUDGE_PROVIDER=openai.")
    return OpenAIClient(settings.openai_api_key, settings.judge_model)


class BaseLLMJudge(ABC):
    """Shared scaffolding for LLM-as-judge metrics.

    Subclasses implement :meth:`system_prompt` and :meth:`user_prompt`; this base
    handles the call/parse/retry/fallback loop.
    """

    #: Short metric name used in logs (e.g. ``"faithfulness"``).
    metric_name: str = "judge"

    def __init__(self, client: LLMClient, *, max_retries: int = 2) -> None:
        """Initialize the judge.

        Args:
            client: The LLM client used to reach the judge model.
            max_retries: Extra attempts after the first on malformed output.
        """
        self._client = client
        self._max_retries = max_retries

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt, including committed few-shot examples."""

    @abstractmethod
    def user_prompt(self, question: EvalQuestion, prediction: Prediction) -> str:
        """Render the per-item user prompt for this question/prediction."""

    def score(self, *, question: EvalQuestion, prediction: Prediction) -> tuple[float, str]:
        """Score the prediction, returning ``(score in [0, 1], rationale)``.

        On persistent malformed output or call failure, returns
        ``(0.0, FALLBACK_RATIONALE)`` after logging a warning, rather than
        raising — a single bad judge call must not abort an experiment.
        """
        system = self.system_prompt()
        user = self.user_prompt(question, prediction)

        for attempt in range(self._max_retries + 1):
            try:
                result = self._client.complete(system, user)
                verdict = self._parse_verdict(result.text)
            except Exception as exc:  # noqa: BLE001 - retry then fall back per spec §5.3
                log.warning(
                    "judge_attempt_failed",
                    metric=self.metric_name,
                    attempt=attempt,
                    error=str(exc),
                )
                continue
            return _clamp_unit(verdict.score), verdict.rationale

        log.warning("judge_failed", metric=self.metric_name, question_id=question.question_id)
        return 0.0, FALLBACK_RATIONALE

    @staticmethod
    def _parse_verdict(text: str) -> JudgeVerdict:
        """Extract and validate the JSON verdict from a judge response.

        Raises:
            ValueError: If no JSON object is present or it is malformed.
            pydantic.ValidationError: If required fields are missing.
        """
        match = _JSON_OBJECT.search(text)
        if match is None:
            raise ValueError("no JSON object found in judge response")
        data = json.loads(match.group(0))
        return JudgeVerdict.model_validate(data)
