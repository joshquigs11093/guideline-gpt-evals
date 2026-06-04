"""Realise an experiment :class:`Variant` as a runnable pipeline.

Two kinds of variation are supported:

- **Settings overrides** — merged onto the parent ``Settings`` and validated
  against its real fields, so a typo fails loudly instead of being ignored.
- **Component overrides** — the cases the parent cannot express through settings
  (no-rerank, single-arm retrieval, prompt swaps). These resolve to components
  injected into :class:`guideline_gpt.pipeline.QueryPipeline`.

The heavyweight ``QueryPipeline`` (and thus chromadb) is imported lazily inside
:func:`default_pipeline_factory`, so the pure logic here stays import-light and
testable. The runner depends only on the :class:`PipelineFactory` protocol, so
tests can inject a stub.
"""

from __future__ import annotations

from typing import Protocol

from guideline_gpt.config import Settings as GgSettings
from guideline_gpt.types import QueryResponse, RetrievalHit

from eval_harness.runner.experiment import PromptMode, RerankerMode, RetrievalMode, Variant


class VariantError(ValueError):
    """Raised when a variant cannot be realised (bad override or unsupported mode)."""


def build_settings(
    variant: Variant,
    shared_config: dict[str, object],
    *,
    base: GgSettings | None = None,
) -> GgSettings:
    """Merge shared + variant overrides onto the parent settings, validated.

    Args:
        variant: The variant whose ``overrides`` to apply.
        shared_config: Experiment-wide settings applied to every variant first.
        base: Starting settings (defaults to a fresh ``Settings()``).

    Returns:
        A parent ``Settings`` with the merged overrides, re-validated.

    Raises:
        VariantError: If an override names a field the parent settings lack, or
            an override value is rejected by the parent's validation.
    """
    base_settings = base if base is not None else GgSettings()
    merged: dict[str, object] = {**shared_config, **variant.overrides}

    unknown = set(merged) - set(GgSettings.model_fields)
    if unknown:
        raise VariantError(
            f"variant {variant.name!r}: unknown settings override(s): {sorted(unknown)}"
        )

    data = base_settings.model_dump()
    data.update(merged)
    try:
        return GgSettings(**data)
    except ValueError as exc:
        raise VariantError(f"variant {variant.name!r}: invalid override: {exc}") from exc


class _IdentityReranker:
    """A pass-through reranker: returns the top ``k`` hits unchanged.

    Lets us express the spec's "no rerank" ablation, which the parent settings
    forbid via ``rerank_top_k > 0``.
    """

    def rerank(self, query: str, hits: list[RetrievalHit], k: int) -> list[RetrievalHit]:  # noqa: ARG002
        """Return the first ``k`` hits without reordering."""
        return hits[:k]


class _EmptyRetriever:
    """A retriever that returns nothing, disabling one arm of hybrid retrieval."""

    def search(self, query: str, k: int) -> list[RetrievalHit]:  # noqa: ARG002
        """Return no hits."""
        return []


def resolve_components(variant: Variant) -> dict[str, object]:
    """Map a variant's component overrides to ``QueryPipeline`` constructor kwargs.

    Args:
        variant: The variant whose component overrides to realise.

    Returns:
        Keyword arguments (``reranker`` / ``vector`` / ``bm25``) to inject; empty
        when the variant uses the parent defaults.

    Raises:
        VariantError: For component modes not yet wired (prompt variations need a
            prompt-injectable pipeline; see the M5 feasibility notes).
    """
    components = variant.components
    kwargs: dict[str, object] = {}

    if components.reranker is RerankerMode.IDENTITY:
        kwargs["reranker"] = _IdentityReranker()

    if components.retrieval is RetrievalMode.VECTOR_ONLY:
        kwargs["bm25"] = _EmptyRetriever()
    elif components.retrieval is RetrievalMode.BM25_ONLY:
        kwargs["vector"] = _EmptyRetriever()

    if components.prompt is not PromptMode.BASELINE:
        raise VariantError(
            f"variant {variant.name!r}: prompt mode {components.prompt.value!r} is not yet "
            "wired — needs a prompt-injectable pipeline (tracked for the prompt-variations "
            "experiment)."
        )

    return kwargs


class QueryRunner(Protocol):
    """The minimal surface the runner needs from a built pipeline."""

    def query(self, question: str) -> QueryResponse:
        """Answer one question end-to-end."""
        ...


class PipelineFactory(Protocol):
    """Builds a :class:`QueryRunner` for a given variant."""

    def __call__(self, variant: Variant, shared_config: dict[str, object]) -> QueryRunner:
        """Construct the runnable pipeline for ``variant``."""
        ...


def default_pipeline_factory(variant: Variant, shared_config: dict[str, object]) -> QueryRunner:
    """Build a real ``guideline-gpt`` pipeline for a variant.

    Imports ``QueryPipeline`` lazily (it pulls in the retrieval stack / chromadb),
    so importing this module stays cheap. Requires an ingested corpus and API
    keys at call time.
    """
    from guideline_gpt.pipeline import QueryPipeline

    settings = build_settings(variant, shared_config)
    return QueryPipeline(settings, **resolve_components(variant))  # type: ignore[no-any-return]
