"""Validate the eval dataset against the corpus it was authored on.

Schema validity is enforced by the loader; this adds corpus-consistency checks
that run in CI (spec §5.1/§11): every ``relevant_chunk_ids`` entry exists in the
corpus manifest, no duplicate question ids, and every question names at least one
relevant chunk.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval_harness.dataset.loader import load_eval_set
from eval_harness.types import EvalQuestion


class DatasetValidationError(ValueError):
    """Raised when the eval dataset is inconsistent with the corpus."""


def load_corpus_chunk_ids(manifest_path: Path) -> set[str]:
    """Return the set of valid chunk ids declared in a corpus manifest.

    Raises:
        DatasetValidationError: If the manifest is missing or lacks ``chunk_ids``.
    """
    if not manifest_path.exists():
        raise DatasetValidationError(f"corpus manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_ids = data.get("chunk_ids")
    if not isinstance(chunk_ids, list):
        raise DatasetValidationError(f"{manifest_path}: manifest has no 'chunk_ids' list")
    return set(chunk_ids)


def validate_eval_set(questions: list[EvalQuestion], valid_chunk_ids: set[str]) -> None:
    """Check every question is consistent with the corpus.

    Args:
        questions: The loaded eval questions.
        valid_chunk_ids: Chunk ids known to exist in the corpus.

    Raises:
        DatasetValidationError: Listing every problem found.
    """
    issues: list[str] = []
    seen: set[str] = set()
    for question in questions:
        if question.question_id in seen:
            issues.append(f"{question.question_id}: duplicate question_id")
        seen.add(question.question_id)
        if not question.relevant_chunk_ids:
            issues.append(f"{question.question_id}: no relevant_chunk_ids")
        missing = [cid for cid in question.relevant_chunk_ids if cid not in valid_chunk_ids]
        if missing:
            issues.append(f"{question.question_id}: chunk ids not in corpus: {missing}")
    if issues:
        raise DatasetValidationError(f"{len(issues)} dataset issue(s):\n  " + "\n  ".join(issues))


def validate_dataset_files(dataset_path: Path, manifest_path: Path) -> int:
    """Load and fully validate the dataset against its corpus manifest.

    Returns:
        The number of validated questions.
    """
    questions = load_eval_set(dataset_path)
    validate_eval_set(questions, load_corpus_chunk_ids(manifest_path))
    return len(questions)
