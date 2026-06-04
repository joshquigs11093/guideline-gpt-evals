"""Compare page: contrast variants across experiments."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from eval_harness.types import EvalQuestion
from eval_harness.ui import data


def _label(experiment_id: str, variant: str) -> str:
    return f"{experiment_id} / {variant}"


def render() -> None:
    """Render the comparison page."""
    st.title("⚖️ Compare")
    results, questions = data.load_all()
    if not results:
        st.warning("No results found.")
        return

    options = [
        _label(experiment_id, variant)
        for experiment_id, variants in sorted(results.items())
        for variant in variants
    ]
    chosen = st.sidebar.multiselect("Variants to compare", options, default=options[:2])
    if len(chosen) < 2:
        st.info("Pick at least two variants from the sidebar to compare.")
        return

    summary = data.variant_summary_df(results)
    summary["label"] = summary["experiment_id"] + " / " + summary["variant"]
    selected = summary[summary["label"].isin(chosen)]

    long = selected.melt(
        id_vars=["label"],
        value_vars=list(data.ALL_METRICS),
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].map(data.METRIC_LABELS)
    data.takeaway("Side-by-side mean metrics for the selected variants.")
    fig = px.bar(
        long,
        x="metric",
        y="value",
        color="label",
        barmode="group",
        color_discrete_sequence=data.PALETTE,
        title="Metric comparison",
    )
    st.plotly_chart(data.style_fig(fig), use_container_width=True)

    if len(chosen) == 2:
        _pairwise(results, questions, chosen[0], chosen[1])


def _pairwise(
    results: data.Results, questions: list[EvalQuestion], label_a: str, label_b: str
) -> None:
    question_df = data.question_long_df(results, questions)
    question_df["label"] = question_df["experiment_id"] + " / " + question_df["variant"]

    a = question_df[question_df["label"] == label_a][["question_id", "correctness"]]
    b = question_df[question_df["label"] == label_b][["question_id", "correctness"]]
    merged = a.merge(b, on="question_id", suffixes=("_a", "_b"))
    if merged.empty:
        return

    wins = int((merged["correctness_a"] > merged["correctness_b"]).sum())
    losses = int((merged["correctness_a"] < merged["correctness_b"]).sum())
    ties = int((merged["correctness_a"] == merged["correctness_b"]).sum())

    cols = st.columns(3)
    cols[0].metric(f"{label_a} wins", wins)
    cols[1].metric(f"{label_b} wins", losses)
    cols[2].metric("Ties", ties)

    data.takeaway(
        "Each point is a question; points below the diagonal favour the first variant, "
        "above favour the second."
    )
    fig = px.scatter(
        merged,
        x="correctness_a",
        y="correctness_b",
        hover_name="question_id",
        color_discrete_sequence=data.PALETTE,
        labels={"correctness_a": label_a, "correctness_b": label_b},
        title="Per-question correctness",
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dot", "color": "gray"})
    st.plotly_chart(data.style_fig(fig), use_container_width=True)
