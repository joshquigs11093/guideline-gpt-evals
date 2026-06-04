"""Methodology page: render the methodology and dataset docs inline."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def _render_markdown_file(path: Path, fallback_heading: str) -> None:
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.subheader(fallback_heading)
        st.info(f"`{path.name}` has not been written yet.")


def render() -> None:
    """Render the methodology page."""
    st.title("📖 Methodology")
    st.caption(
        "How the evaluation is done and why — the reference for the rigor behind the numbers."
    )

    tab_method, tab_dataset = st.tabs(["Methodology", "Dataset"])
    with tab_method:
        _render_markdown_file(Path("METHODOLOGY.md"), "Methodology")
    with tab_dataset:
        _render_markdown_file(Path("DATASET.md"), "Dataset")
