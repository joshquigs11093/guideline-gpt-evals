"""Tests for variant settings merging and component resolution."""

from __future__ import annotations

import pytest
from guideline_gpt.config import Settings as GgSettings

from eval_harness.runner.experiment import ComponentOverrides, Variant
from eval_harness.runner.variant import (
    VariantError,
    _EmptyRetriever,
    _IdentityReranker,
    build_settings,
    resolve_components,
)


def _base() -> GgSettings:
    # Hermetic: ignore any local .env.
    return GgSettings(_env_file=None)  # type: ignore[call-arg]


def test_build_settings_merges_shared_then_variant() -> None:
    variant = Variant(name="chunks_256", overrides={"chunk_size": 256})
    settings = build_settings(variant, {"retrieval_top_k": 10}, base=_base())
    assert settings.chunk_size == 256
    assert settings.retrieval_top_k == 10


def test_variant_overrides_win_over_shared() -> None:
    variant = Variant(name="v", overrides={"retrieval_top_k": 5})
    settings = build_settings(variant, {"retrieval_top_k": 20}, base=_base())
    assert settings.retrieval_top_k == 5


def test_unknown_override_key_raises() -> None:
    variant = Variant(name="v", overrides={"not_a_setting": 1})
    with pytest.raises(VariantError, match="unknown settings override"):
        build_settings(variant, {}, base=_base())


def test_invalid_override_value_raises() -> None:
    # The parent forbids rerank_top_k <= 0 (Field gt=0).
    variant = Variant(name="v", overrides={"rerank_top_k": 0})
    with pytest.raises(VariantError, match="invalid override"):
        build_settings(variant, {}, base=_base())


def test_resolve_components_defaults_empty() -> None:
    assert resolve_components(Variant(name="v")) == {}


def test_resolve_identity_reranker() -> None:
    variant = Variant(name="v", components=ComponentOverrides(reranker="identity"))
    kwargs = resolve_components(variant)
    assert isinstance(kwargs["reranker"], _IdentityReranker)
    assert "vector" not in kwargs and "bm25" not in kwargs


def test_resolve_vector_only_disables_bm25() -> None:
    variant = Variant(name="v", components=ComponentOverrides(retrieval="vector_only"))
    kwargs = resolve_components(variant)
    assert isinstance(kwargs["bm25"], _EmptyRetriever)


def test_resolve_bm25_only_disables_vector() -> None:
    variant = Variant(name="v", components=ComponentOverrides(retrieval="bm25_only"))
    kwargs = resolve_components(variant)
    assert isinstance(kwargs["vector"], _EmptyRetriever)


def test_prompt_variation_not_yet_supported() -> None:
    variant = Variant(name="v", components=ComponentOverrides(prompt="cite_sources"))
    with pytest.raises(VariantError, match="not yet wired"):
        resolve_components(variant)


def test_identity_reranker_truncates() -> None:
    reranker = _IdentityReranker()
    assert reranker.rerank("q", ["h1", "h2", "h3"], 2) == ["h1", "h2"]  # type: ignore[arg-type]


def test_empty_retriever_returns_nothing() -> None:
    assert _EmptyRetriever().search("q", 5) == []
