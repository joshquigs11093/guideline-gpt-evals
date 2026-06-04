"""Experiments page: per-experiment variant metrics and breakdowns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from eval_harness.runner.experiment import ExperimentConfigError, load_experiment
from eval_harness.types import ExperimentResults
from eval_harness.ui import data


def _description(experiment_id: str) -> str:
    path = Path("experiments") / f"{experiment_id}.yaml"
    try:
        return load_experiment(path).description
    except ExperimentConfigError:
        return ""


def _breakdown_records(
    variants: dict[str, ExperimentResults], metric: str, dimension: str
) -> pd.DataFrame:
    prefix = f"{metric}_mean[{dimension}="
    records: list[dict[str, object]] = []
    for variant_name, result in variants.items():
        for key, value in result.aggregate_metrics.items():
            if key.startswith(prefix):
                records.append(
                    {"variant": variant_name, dimension: key[len(prefix) : -1], metric: value}
                )
    return pd.DataFrame(records)


def render() -> None:
    """Render the experiment detail page."""
    st.title("🧪 Experiments")
    results, _ = data.load_all()
    if not results:
        st.warning("No results found.")
        return

    experiment_id = st.sidebar.selectbox("Experiment", sorted(results))
    variants = results[experiment_id]
    sample = next(iter(variants.values()))
    st.header(sample.experiment_name)
    description = _description(experiment_id)
    if description:
        st.markdown(f"_{description}_")

    summary = data.variant_summary_df({experiment_id: variants})

    # Aggregate metrics by variant.
    long = summary.melt(
        id_vars=["variant"],
        value_vars=list(data.ALL_METRICS),
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].map(data.METRIC_LABELS)
    data.takeaway("Mean metric per variant — taller is better across all seven metrics.")
    fig = px.bar(
        long,
        x="metric",
        y="value",
        color="variant",
        barmode="group",
        color_discrete_sequence=data.PALETTE,
        title="Aggregate metrics by variant",
    )
    st.plotly_chart(data.style_fig(fig), use_container_width=True)

    # Per-difficulty correctness breakdown.
    difficulty_df = _breakdown_records(variants, "correctness", "difficulty")
    if not difficulty_df.empty:
        data.takeaway("Correctness by difficulty exposes where each variant struggles.")
        fig = px.bar(
            difficulty_df,
            x="difficulty",
            y="correctness",
            color="variant",
            barmode="group",
            category_orders={"difficulty": ["easy", "medium", "hard"]},
            color_discrete_sequence=data.PALETTE,
            title="Correctness by difficulty",
        )
        st.plotly_chart(data.style_fig(fig), use_container_width=True)

    # Cost & latency.
    left, right = st.columns(2)
    with left:
        fig = px.bar(
            summary,
            x="variant",
            y="total_cost_usd",
            color="variant",
            color_discrete_sequence=data.PALETTE,
            title="Total eval cost (USD)",
        )
        st.plotly_chart(data.style_fig(fig), use_container_width=True)
    with right:
        fig = px.bar(
            summary,
            x="variant",
            y="total_latency_ms",
            color="variant",
            color_discrete_sequence=data.PALETTE,
            title="Total latency (ms)",
        )
        st.plotly_chart(data.style_fig(fig), use_container_width=True)

    with st.expander("Variant config + raw metrics"):
        st.dataframe(summary, use_container_width=True)
