"""Load and schema-validate the eval dataset from JSONL.

Each line is one :class:`EvalQuestion`. Validation happens on load via a Pydantic
:class:`TypeAdapter` over the dataclass, so a malformed line fails fast with a
file/line-qualified error rather than surfacing deep in an experiment run.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from eval_harness.types import EvalQuestion

_QUESTION_ADAPTER: TypeAdapter[EvalQuestion] = TypeAdapter(EvalQuestion)


class EvalDatasetError(ValueError):
    """Raised when the eval dataset is missing, malformed, or inconsistent."""


def load_eval_set(path: Path) -> list[EvalQuestion]:
    """Load and validate every question in a JSONL eval set.

    Args:
        path: Path to a ``.jsonl`` file, one ``EvalQuestion`` per line. Blank
            lines are ignored.

    Returns:
        The questions in file order.

    Raises:
        EvalDatasetError: If the file is missing, a line is invalid JSON or
            fails schema validation, a ``question_id`` is duplicated, or the set
            is empty.
    """
    if not path.exists():
        raise EvalDatasetError(f"eval dataset not found: {path}")

    questions: list[EvalQuestion] = []
    seen_ids: set[str] = set()

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            question = _QUESTION_ADAPTER.validate_python(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise EvalDatasetError(f"{path}:{lineno}: {exc}") from exc
        if question.question_id in seen_ids:
            raise EvalDatasetError(
                f"{path}:{lineno}: duplicate question_id {question.question_id!r}"
            )
        seen_ids.add(question.question_id)
        questions.append(question)

    if not questions:
        raise EvalDatasetError(f"{path}: no questions found")
    return questions
