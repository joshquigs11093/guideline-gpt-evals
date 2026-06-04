"""Verify the dashboard import graph needs none of the heavy `experiments` deps.

Runs a fresh interpreter with guideline_gpt/torch/chromadb/ragas/deepeval import
blocked, then imports the dashboard's data layer and every page component. If
that succeeds, the lean (torch-free) dashboard image is viable.
"""

from __future__ import annotations

import subprocess
import sys

_GUARD = """
import builtins

_blocked = {"guideline_gpt", "torch", "chromadb", "ragas", "deepeval"}
_real_import = builtins.__import__


def _guarded(name, *args, **kwargs):
    if name.split(".")[0] in _blocked:
        raise ImportError(f"heavy dep '{name}' must not be imported by the dashboard")
    return _real_import(name, *args, **kwargs)


builtins.__import__ = _guarded

# The dashboard entry imports these; importing them must not pull a heavy dep.
import eval_harness.ui.data  # noqa: F401, E402
from eval_harness.ui.components import (  # noqa: F401, E402
    comparison_view,
    experiment_view,
    failure_analysis,
    methodology,
    overview,
    question_explorer,
)

print("ok")
"""


def test_dashboard_imports_without_heavy_deps() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _GUARD],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"dashboard pulled a heavy dep:\n{result.stderr}"
    assert "ok" in result.stdout
