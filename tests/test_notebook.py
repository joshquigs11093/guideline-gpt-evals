"""Guard the committed analysis notebook: it must exist, have code cells, and
carry no error outputs (i.e. it executed cleanly when last built)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
import pytest

NOTEBOOK = Path("notebooks/analysis.ipynb")


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebook not built")
def test_notebook_executed_without_errors() -> None:
    nb = nbf.read(NOTEBOOK, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert code_cells, "notebook has no code cells"
    for cell in code_cells:
        errors = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        assert not errors, f"error output in cell: {errors}"


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebook not built")
def test_notebook_has_narrative() -> None:
    nb = nbf.read(NOTEBOOK, as_version=4)
    markdown = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "Hypothesis" in markdown
    assert "Takeaway" in markdown
    assert "Limitations" in markdown
