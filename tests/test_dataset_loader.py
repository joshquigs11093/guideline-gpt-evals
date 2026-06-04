"""Tests for the JSONL eval-set loader and its validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_harness.dataset.loader import EvalDatasetError, load_eval_set

_VALID = {
    "question_id": "q1",
    "question": "What is first-line for acute asthma?",
    "ground_truth_answer": "Inhaled SABA.",
    "relevant_chunk_ids": ["a", "b"],
    "source_documents": ["asthma.pdf"],
    "difficulty": "easy",
    "question_type": "factoid",
}


def _write(path: Path, *rows: object) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_loads_valid_questions(tmp_path: Path) -> None:
    second = {**_VALID, "question_id": "q2", "difficulty": "hard", "question_type": "synthesis"}
    questions = load_eval_set(_write(tmp_path / "q.jsonl", _VALID, second))
    assert [q.question_id for q in questions] == ["q1", "q2"]
    assert questions[1].difficulty == "hard"


def test_blank_lines_ignored(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(_VALID) + "\n\n", encoding="utf-8")
    assert len(load_eval_set(path)) == 1


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(EvalDatasetError, match="not found"):
        load_eval_set(tmp_path / "nope.jsonl")


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(EvalDatasetError, match="no questions"):
        load_eval_set(path)


def test_malformed_json_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_VALID) + "\nnot json\n", encoding="utf-8")
    with pytest.raises(EvalDatasetError, match=r":2:"):
        load_eval_set(path)


def test_invalid_difficulty_rejected(tmp_path: Path) -> None:
    bad = {**_VALID, "difficulty": "impossible"}
    with pytest.raises(EvalDatasetError):
        load_eval_set(_write(tmp_path / "q.jsonl", bad))


def test_missing_required_field_rejected(tmp_path: Path) -> None:
    bad = {k: v for k, v in _VALID.items() if k != "ground_truth_answer"}
    with pytest.raises(EvalDatasetError):
        load_eval_set(_write(tmp_path / "q.jsonl", bad))


def test_duplicate_question_id_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvalDatasetError, match="duplicate"):
        load_eval_set(_write(tmp_path / "q.jsonl", _VALID, _VALID))
