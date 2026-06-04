"""Tests for experiment-config parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_harness.runner.experiment import (
    ExperimentConfigError,
    PromptMode,
    RerankerMode,
    RetrievalMode,
    load_experiment,
)

_YAML = """
experiment_id: "02_reranker_ablation"
name: "Reranker ablation"
description: "rerank on vs off"
variants:
  - name: "with_rerank"
  - name: "no_rerank"
    components:
      reranker: identity
shared_config:
  retrieval_top_k: 20
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_and_defaults(tmp_path: Path) -> None:
    config = load_experiment(_write(tmp_path / "e.yaml", _YAML))
    assert config.experiment_id == "02_reranker_ablation"
    assert [v.name for v in config.variants] == ["with_rerank", "no_rerank"]
    # Defaults on the first variant.
    assert config.variants[0].components.reranker is RerankerMode.DEFAULT
    assert config.variants[0].components.retrieval is RetrievalMode.HYBRID
    assert config.variants[0].components.prompt is PromptMode.BASELINE
    # Override on the second.
    assert config.variants[1].components.reranker is RerankerMode.IDENTITY
    assert config.shared_config == {"retrieval_top_k": 20}


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ExperimentConfigError, match="not found"):
        load_experiment(tmp_path / "nope.yaml")


def test_duplicate_variant_names_rejected(tmp_path: Path) -> None:
    text = """
experiment_id: x
name: x
variants:
  - name: dup
  - name: dup
"""
    with pytest.raises(ExperimentConfigError, match="duplicate variant names"):
        load_experiment(_write(tmp_path / "e.yaml", text))


def test_no_variants_rejected(tmp_path: Path) -> None:
    text = "experiment_id: x\nname: x\nvariants: []\n"
    with pytest.raises(ExperimentConfigError):
        load_experiment(_write(tmp_path / "e.yaml", text))


def test_unknown_component_key_rejected(tmp_path: Path) -> None:
    text = """
experiment_id: x
name: x
variants:
  - name: v
    components:
      typo: identity
"""
    with pytest.raises(ExperimentConfigError):
        load_experiment(_write(tmp_path / "e.yaml", text))


def test_invalid_enum_value_rejected(tmp_path: Path) -> None:
    text = """
experiment_id: x
name: x
variants:
  - name: v
    components:
      retrieval: telepathy
"""
    with pytest.raises(ExperimentConfigError):
        load_experiment(_write(tmp_path / "e.yaml", text))
