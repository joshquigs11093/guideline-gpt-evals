"""Failure Analysis page: surface and cluster the worst-scoring questions."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from eval_harness.ui import data


def _pattern(rationale: str) -> str:
    """Bucket a judge rationale by its leading phrase (before ':')."""
    head = rationale.split(":", 1)[0].strip()
    return head or "Other"


def render() -> None:
    """Render the failure analysis page."""
    st.title("❌ Failure Analysis")
    st.caption(
        "A repo that surfaces what it gets wrong is more credible than one that only shows wins."
    )

    results, questions = data.load_all()
    if not results:
        st.warning("No results found.")
        return

    experiment_id = st.sidebar.selectbox("Experiment", sorted(results))
    variant = st.sidebar.selectbox("Variant", sorted(results[experiment_id]))

    question_df = data.question_long_df(results, questions)
    subset = question_df[
        (question_df["experiment_id"] == experiment_id) & (question_df["variant"] == variant)
    ].copy()
    if subset.empty:
        st.warning("No scored questions for this variant.")
        return

    # Failure pattern clustering from correctness rationales of low scorers.
    failures = subset[subset["correctness"] < 0.6].copy()
    if not failures.empty:
        failures["pattern"] = failures["correctness_rationale"].map(_pattern)
        counts = failures["pattern"].value_counts().reset_index()
        counts.columns = ["pattern", "count"]
        data.takeaway(
            f"{len(failures)} of {len(subset)} questions scored below 0.6 on correctness — "
            "clustered by failure mode below."
        )
        fig = px.bar(
            counts,
            x="count",
            y="pattern",
            orientation="h",
            color_discrete_sequence=data.PALETTE,
            title="Failure patterns (correctness < 0.6)",
        )
        st.plotly_chart(data.style_fig(fig), use_container_width=True)

    st.subheader("Top 10 worst-scoring questions")
    worst = subset.nsmallest(10, "correctness")
    for _, row in worst.iterrows():
        with st.expander(
            f"{row['question_id']} — correctness {row['correctness']:.2f} ({row['difficulty']})"
        ):
            st.markdown(f"> **Correctness:** {row['correctness_rationale']}")
            st.markdown(
                f"> **Faithfulness ({row['faithfulness']:.2f}):** {row['faithfulness_rationale']}"
            )
            st.markdown(f"> **Relevance ({row['relevance']:.2f}):** {row['relevance_rationale']}")
