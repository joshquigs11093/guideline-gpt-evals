"""Tests for the Typer CLI (commands that need no API keys or corpus)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from eval_harness.cli import app

runner = CliRunner()

_VALID_QUESTION = {
    "question_id": "q1",
    "question": "What is first-line for acute asthma?",
    "ground_truth_answer": "Inhaled SABA.",
    "relevant_chunk_ids": ["a"],
    "source_documents": ["asthma.pdf"],
    "difficulty": "easy",
    "question_type": "factoid",
}


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "validate-dataset", "list-experiments"):
        assert command in result.output


def test_validate_dataset_ok(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(_VALID_QUESTION) + "\n", encoding="utf-8")
    result = runner.invoke(app, ["validate-dataset", "--dataset", str(path)])
    assert result.exit_code == 0
    assert "1 questions validated" in result.output


def test_validate_dataset_bad_exits_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    result = runner.invoke(app, ["validate-dataset", "--dataset", str(path)])
    assert result.exit_code == 1


def test_list_experiments(tmp_path: Path) -> None:
    yaml_text = "experiment_id: demo\nname: Demo experiment\nvariants:\n  - name: a\n  - name: b\n"
    (tmp_path / "demo.yaml").write_text(yaml_text, encoding="utf-8")
    result = runner.invoke(app, ["list-experiments", "--directory", str(tmp_path)])
    assert result.exit_code == 0
    assert "demo: Demo experiment (2 variants)" in result.output


def test_list_experiments_empty_dir_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["list-experiments", "--directory", str(tmp_path)])
    assert result.exit_code == 1


def test_list_real_experiments_directory() -> None:
    # The committed experiments/ directory should hold all six configs.
    result = runner.invoke(app, ["list-experiments", "--directory", "experiments"])
    assert result.exit_code == 0
    assert result.output.count("\n") >= 6
