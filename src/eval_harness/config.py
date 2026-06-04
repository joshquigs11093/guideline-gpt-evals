"""Centralized, type-safe configuration loaded from environment variables.

Mirrors the parent project's ``guideline_gpt.config`` pattern: everything
tunable lives here, and no module reads ``os.environ`` directly. See spec §9 for
the full variable table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

JudgeProvider = Literal["anthropic", "openai"]


class Settings(BaseSettings):
    """Runtime configuration, populated from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM-as-judge -----------------------------------------------------
    # Critical design rule (spec §5.3 / ADR-002): the judge must be a *stronger*
    # model than the system under test. Default to a strong Anthropic model.
    judge_provider: JudgeProvider = "anthropic"
    judge_model: str = "claude-opus-4-7"

    # --- Provider credentials --------------------------------------------
    anthropic_api_key: str | None = None
    # OpenAI key is required for embeddings via guideline-gpt regardless of the
    # judge provider chosen.
    openai_api_key: str | None = None

    # --- Paths ------------------------------------------------------------
    results_dir: Path = Path("./results")
    eval_dataset_path: Path = Path("./eval_dataset/questions.jsonl")
    judge_cache_dir: Path = Path("./.judge_cache")

    # --- Logging ----------------------------------------------------------
    log_level: str = "INFO"


def get_settings() -> Settings:
    """Construct a :class:`Settings` instance from the current environment."""
    return Settings()
