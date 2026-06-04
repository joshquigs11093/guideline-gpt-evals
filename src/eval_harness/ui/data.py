"""Data layer for the dashboard: cached loaders, dataframes, and chart helpers.

Pure dataframe builders (no Streamlit) are unit-tested; the cached loaders and
``takeaway`` / ``style_fig`` helpers wrap Streamlit/Plotly for the UI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from eval_harness.config import get_settings
from eval_harness.dataset.loader import load_eval_set
from eval_harness.report.store import load_results
from eval_harness.types import EvalQuestion, ExperimentResults

SYNTHETIC_VERSION = "SYNTHETIC-DEMO"

RETRIEVAL_METRICS = ("precision_at_5", "recall_at_5", "mrr", "ndcg_at_5")
JUDGE_METRICS = ("faithfulness", "relevance", "correctness")
ALL_METRICS = RETRIEVAL_METRICS + JUDGE_METRICS

METRIC_LABELS = {
    "precision_at_5": "Precision@5",
    "recall_at_5": "Recall@5",
    "mrr": "MRR",
    "ndcg_at_5": "nDCG@5",
    "faithfulness": "Faithfulness",
    "relevance": "Relevance",
    "correctness": "Correctness",
}

# Okabe-Ito colourblind-safe palette.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#999999"]

Results = dict[str, dict[str, ExperimentResults]]


@st.cache_data(show_spinner=False)
def load_dashboard_results(results_dir: str) -> Results:
    """Load all experiment results from ``results_dir`` (cached)."""
    return load_results(Path(results_dir))


@st.cache_data(show_spinner=False)
def load_dashboard_questions(dataset_path: str) -> list[EvalQuestion]:
    """Load the eval set if present (cached); empty list if missing."""
    path = Path(dataset_path)
    if not path.exists():
        return []
    return load_eval_set(path)


def load_all() -> tuple[Results, list[EvalQuestion]]:
    """Load results and questions using the configured paths."""
    settings = get_settings()
    results = load_dashboard_results(str(settings.results_dir))
    questions = load_dashboard_questions(str(settings.eval_dataset_path))
    return results, questions


def is_synthetic(results: Results) -> bool:
    """True if any loaded results are flagged as synthetic placeholders."""
    return any(
        variant.guideline_gpt_version == SYNTHETIC_VERSION
        for experiment in results.values()
        for variant in experiment.values()
    )


# --------------------------------------------------------------------------- #
# Pure dataframe builders (no Streamlit) — unit-tested.
# --------------------------------------------------------------------------- #
def variant_summary_df(results: Results) -> pd.DataFrame:
    """One row per (experiment, variant) with mean metrics, cost, and latency."""
    records: list[dict[str, object]] = []
    for experiment_id, variants in results.items():
        for variant_name, result in variants.items():
            record: dict[str, object] = {
                "experiment_id": experiment_id,
                "experiment_name": result.experiment_name,
                "variant": variant_name,
                "total_cost_usd": result.total_cost_usd,
                "total_latency_ms": result.total_latency_ms,
                "n_questions": len(result.question_scores),
            }
            for metric in ALL_METRICS:
                record[metric] = result.aggregate_metrics.get(f"{metric}_mean", 0.0)
            records.append(record)
    return pd.DataFrame(records)


def aggregate_long_df(results: Results) -> pd.DataFrame:
    """Long-form: one row per (experiment, variant, metric) with its mean value."""
    summary = variant_summary_df(results)
    if summary.empty:
        return summary
    return summary.melt(
        id_vars=["experiment_id", "experiment_name", "variant"],
        value_vars=list(ALL_METRICS),
        var_name="metric",
        value_name="value",
    )


def question_long_df(results: Results, questions: list[EvalQuestion]) -> pd.DataFrame:
    """One row per (experiment, variant, question) with metrics and metadata."""
    by_id = {q.question_id: q for q in questions}
    records: list[dict[str, object]] = []
    for experiment_id, variants in results.items():
        for variant_name, result in variants.items():
            for score in result.question_scores:
                question = by_id.get(score.question_id)
                records.append(
                    {
                        "experiment_id": experiment_id,
                        "variant": variant_name,
                        "question_id": score.question_id,
                        "difficulty": question.difficulty if question else "unknown",
                        "question_type": question.question_type if question else "unknown",
                        "is_hard_case": question.is_hard_case if question else False,
                        "precision_at_5": score.precision_at_5,
                        "recall_at_5": score.recall_at_5,
                        "mrr": score.mrr,
                        "ndcg_at_5": score.ndcg_at_5,
                        "faithfulness": score.faithfulness,
                        "relevance": score.relevance,
                        "correctness": score.correctness,
                        "faithfulness_rationale": score.faithfulness_rationale,
                        "relevance_rationale": score.relevance_rationale,
                        "correctness_rationale": score.correctness_rationale,
                    }
                )
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def takeaway(text: str) -> None:
    """Render a one-sentence takeaway above a chart."""
    st.info(text, icon="💡")


def style_fig(fig: object) -> object:
    """Apply consistent layout to a Plotly figure."""
    fig.update_layout(  # type: ignore[attr-defined]
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        height=420,
    )
    return fig
