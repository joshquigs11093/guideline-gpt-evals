"""Declarative experiment definitions loaded from YAML.

A spec §5.4 experiment lists variants that each tweak the system under test.
Most tweaks are plain ``guideline-gpt`` ``Settings`` overrides, but three of the
six experiments cannot be expressed that way (the parent forbids
``rerank_top_k=0``, has no retrieval-mode flag, and bakes prompts in as module
constants). So each variant also carries a small set of *component overrides*
that the variant builder maps to injected pipeline components.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RerankerMode(StrEnum):
    """How the variant's reranking stage behaves."""

    DEFAULT = "default"
    """Use the parent's cross-encoder reranker."""
    IDENTITY = "identity"
    """Bypass reranking (the spec's ``rerank_top_k=0`` ablation)."""


class RetrievalMode(StrEnum):
    """Which retrieval arms feed fusion."""

    HYBRID = "hybrid"
    """Vector + BM25 fused (the parent default)."""
    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"


class PromptMode(StrEnum):
    """Which generation prompt the variant uses."""

    BASELINE = "baseline"
    CITE_SOURCES = "cite_sources"
    REASON_FIRST = "reason_first"


class ComponentOverrides(BaseModel):
    """Variant tweaks that require injecting pipeline components, not settings."""

    model_config = ConfigDict(extra="forbid")

    reranker: RerankerMode = RerankerMode.DEFAULT
    retrieval: RetrievalMode = RetrievalMode.HYBRID
    prompt: PromptMode = PromptMode.BASELINE


class Variant(BaseModel):
    """One configuration of the system under test within an experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str
    overrides: dict[str, object] = Field(default_factory=dict)
    """Parent ``Settings`` field overrides (e.g. ``chunk_size: 256``)."""
    components: ComponentOverrides = Field(default_factory=ComponentOverrides)


class ExperimentConfig(BaseModel):
    """A full experiment: shared config plus the variants to compare."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    name: str
    description: str = ""
    variants: list[Variant] = Field(min_length=1)
    shared_config: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_variant_names(self) -> ExperimentConfig:
        names = [v.name for v in self.variants]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate variant names: {sorted(duplicates)}")
        return self


class ExperimentConfigError(ValueError):
    """Raised when an experiment YAML is missing or invalid."""


def load_experiment(path: Path) -> ExperimentConfig:
    """Load and validate an experiment definition from YAML.

    Args:
        path: Path to the experiment ``.yaml`` file.

    Returns:
        The validated :class:`ExperimentConfig`.

    Raises:
        ExperimentConfigError: If the file is missing or fails validation.
    """
    if not path.exists():
        raise ExperimentConfigError(f"experiment config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return ExperimentConfig.model_validate(data)
    except ValueError as exc:
        raise ExperimentConfigError(f"{path}: {exc}") from exc
