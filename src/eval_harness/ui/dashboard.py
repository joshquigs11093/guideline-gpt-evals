"""guideline-gpt-evals results dashboard (Streamlit entry point).

Reads pre-computed results from ``results/`` — no API keys, no live experiments.
Launch with::

    streamlit run src/eval_harness/ui/dashboard.py
"""

from __future__ import annotations

import streamlit as st

from eval_harness.ui import data
from eval_harness.ui.components import (
    comparison_view,
    experiment_view,
    failure_analysis,
    methodology,
    overview,
    question_explorer,
)


def _synthetic_banner() -> None:
    results, _ = data.load_all()
    if data.is_synthetic(results):
        st.warning(
            "⚠️ **Synthetic demo data.** These results are generated placeholders "
            "(`SYNTHETIC-DEMO`), not real measurements — they exist so the dashboard can be "
            "browsed before a real corpus + API runs. See `results/SYNTHETIC_DATA.md`.",
            icon="⚠️",
        )


def main() -> None:
    """Configure the app and run the multi-page navigation."""
    st.set_page_config(page_title="guideline-gpt-evals", page_icon="📊", layout="wide")
    _synthetic_banner()

    # Explicit url_path per page: every page callable is named `render`, so
    # without this Streamlit infers the same pathname for all and raises.
    pages = [
        st.Page(overview.render, title="Overview", icon="📊", url_path="overview", default=True),
        st.Page(experiment_view.render, title="Experiments", icon="🧪", url_path="experiments"),
        st.Page(comparison_view.render, title="Compare", icon="⚖️", url_path="compare"),
        st.Page(
            question_explorer.render, title="Question Explorer", icon="🔍", url_path="explorer"
        ),
        st.Page(failure_analysis.render, title="Failure Analysis", icon="❌", url_path="failures"),
        st.Page(methodology.render, title="Methodology", icon="📖", url_path="methodology"),
    ]
    st.navigation(pages).run()


main()
