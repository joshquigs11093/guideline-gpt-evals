"""Smoke tests: every dashboard page renders without raising, against the
committed synthetic results. Uses Streamlit's in-process AppTest (no server).

Each page is run via a tiny import-and-call script so the page executes inside
its own module namespace (where `st`, `px`, etc. are imported).
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

PAGE_MODULES = [
    "overview",
    "experiment_view",
    "comparison_view",
    "question_explorer",
    "failure_analysis",
    "methodology",
]


@pytest.mark.parametrize("module", PAGE_MODULES)
def test_page_renders_without_exception(module: str) -> None:
    script = f"from eval_harness.ui.components import {module}\n{module}.render()\n"
    app = AppTest.from_string(script).run(timeout=60)
    assert not app.exception, f"{module} raised: {app.exception}"
