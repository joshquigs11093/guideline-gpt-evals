"""Tests for the dataset corpus-consistency validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_harness.dataset.validator import (
    DatasetValidationError,
    load_corpus_chunk_ids,
    validate_dataset_files,
    validate_eval_set,
)
from eval_harness.types import EvalQuestion


def _q(qid: str, chunks: list[str]) -> EvalQuestion:
    return EvalQuestion(qid, "?", "a", chunks, ["d.pdf"], "easy", "factoid")


def test_consistent_set_passes() -> None:
    validate_eval_set([_q("q1", ["a"]), _q("q2", ["b", "c"])], {"a", "b", "c"})


def test_unknown_chunk_id_raises() -> None:
    with pytest.raises(DatasetValidationError, match="not in corpus"):
        validate_eval_set([_q("q1", ["z"])], {"a"})


def test_empty_relevant_raises() -> None:
    with pytest.raises(DatasetValidationError, match="no relevant_chunk_ids"):
        validate_eval_set([_q("q1", [])], {"a"})


def test_duplicate_id_raises() -> None:
    with pytest.raises(DatasetValidationError, match="duplicate"):
        validate_eval_set([_q("q1", ["a"]), _q("q1", ["a"])], {"a"})


def test_load_corpus_chunk_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"chunk_ids": ["a", "b"]}), encoding="utf-8")
    assert load_corpus_chunk_ids(manifest) == {"a", "b"}


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="not found"):
        load_corpus_chunk_ids(tmp_path / "nope.json")


def test_manifest_without_chunk_ids_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"documents": {}}), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="no 'chunk_ids'"):
        load_corpus_chunk_ids(manifest)


def test_committed_dataset_validates() -> None:
    """Spec §11: the committed eval set validates against its corpus manifest."""
    count = validate_dataset_files(
        Path("eval_dataset/questions.jsonl"), Path("eval_dataset/corpus_manifest.json")
    )
    assert count == 75
