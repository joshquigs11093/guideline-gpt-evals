"""Question Explorer page: filter the eval set and inspect per-question scores."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from eval_harness.ui import data


def render() -> None:
    """Render the question explorer page."""
    st.title("🔍 Question Explorer")
    results, questions = data.load_all()
    if not questions:
        st.warning("No eval questions found.")
        return

    questions_df = pd.DataFrame(
        [
            {
                "question_id": q.question_id,
                "question": q.question,
                "difficulty": q.difficulty,
                "question_type": q.question_type,
                "is_hard_case": q.is_hard_case,
                "source": ", ".join(q.source_documents),
                "ground_truth_answer": q.ground_truth_answer,
            }
            for q in questions
        ]
    )

    st.sidebar.header("Filters")
    difficulties = st.sidebar.multiselect(
        "Difficulty", sorted(questions_df["difficulty"].unique()), default=[]
    )
    types = st.sidebar.multiselect(
        "Question type", sorted(questions_df["question_type"].unique()), default=[]
    )
    hard_only = st.sidebar.toggle("Hard cases only", value=False)
    sources = st.sidebar.multiselect("Source document", sorted(questions_df["source"].unique()))

    view = questions_df
    if difficulties:
        view = view[view["difficulty"].isin(difficulties)]
    if types:
        view = view[view["question_type"].isin(types)]
    if hard_only:
        view = view[view["is_hard_case"]]
    if sources:
        view = view[view["source"].isin(sources)]

    st.caption(f"{len(view)} of {len(questions_df)} questions match.")
    st.dataframe(
        view[["question_id", "difficulty", "question_type", "is_hard_case", "source"]],
        use_container_width=True,
        hide_index=True,
    )

    if view.empty:
        return

    selected: str = st.selectbox("Inspect a question", view["question_id"])
    row = questions_df[questions_df["question_id"] == selected].iloc[0]
    st.markdown(f"**Question:** {row['question']}")
    st.markdown(f"**Ground-truth answer:** {row['ground_truth_answer']}")
    st.markdown(
        f"**Source:** {row['source']} · **Difficulty:** {row['difficulty']} · "
        f"**Type:** {row['question_type']}"
    )

    question_df = data.question_long_df(results, questions)
    scores = question_df[question_df["question_id"] == selected]
    if scores.empty:
        return
    st.markdown("**Per-variant scores**")
    st.dataframe(
        scores[["experiment_id", "variant", *data.ALL_METRICS]],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Judge rationales"):
        for _, score_row in scores.iterrows():
            st.markdown(f"**{score_row['experiment_id']} / {score_row['variant']}**")
            st.markdown(f"> Correctness: {score_row['correctness_rationale']}")
            st.markdown(f"> Faithfulness: {score_row['faithfulness_rationale']}")
