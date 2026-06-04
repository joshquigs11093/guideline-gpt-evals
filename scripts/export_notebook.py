"""Export the analysis notebook to standalone HTML for GitHub viewing.

Converts the (already-executed) ``notebooks/analysis.ipynb`` to
``notebooks/analysis.html`` without re-executing, so the HTML is a deterministic
function of the committed notebook. CI can re-run this and diff to confirm the
HTML is up to date.

    python scripts/export_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbconvert import HTMLExporter

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO / "notebooks" / "analysis.ipynb"
HTML_PATH = REPO / "notebooks" / "analysis.html"


def main() -> None:
    if not NOTEBOOK_PATH.exists():
        raise SystemExit(f"{NOTEBOOK_PATH} not found — run scripts/build_notebook.py first.")
    nb = nbf.read(NOTEBOOK_PATH, as_version=4)
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(nb)
    HTML_PATH.write_text(body, encoding="utf-8")
    print(f"Wrote {HTML_PATH} ({len(body):,} bytes)")


if __name__ == "__main__":
    main()
