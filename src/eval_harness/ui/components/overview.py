"""Overview page: headline stats and three lead findings."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from eval_harness.ui import data


def _chunk_size(variant: str) -> int:
    digits = "".join(ch for ch in variant if ch.isdigit())
    return int(digits) if digits else 0


def render() -> None:
    """Render the landing page."""
    st.title("📊 guideline-gpt-evals")
    st.caption(
        "A rigorous evaluation of "
        "[guideline-gpt](https://github.com/joshquigs11093/guideline-gpt) — retrieval quality, "
        "faithfulness, relevance, and correctness across six experiments."
    )

    results, questions = data.load_all()
    if not results:
        st.warning("No results found. Run `python scripts/generate_synthetic_data.py` first.")
        return

    summary = data.variant_summary_df(results)
    total_spend = float(summary["total_cost_usd"].sum())
    total_predictions = int(summary["n_questions"].sum())

    cols = st.columns(4)
    cols[0].metric("Eval Questions", len(questions) or "—")
    cols[1].metric("Experiments", len(results))
    cols[2].metric("LLM Spend (eval)", f"${total_spend:,.2f}")
    cols[3].metric("Predictions Scored", f"{total_predictions:,}")

    st.divider()
    st.subheader("Headline findings")

    left, right = st.columns(2)

    # Finding 1: reranking improves correctness.
    with left:
        rerank = summary[summary["experiment_id"] == "02_reranker_ablation"]
        if not rerank.empty:
            with_r = rerank.loc[rerank["variant"] == "with_rerank", "correctness"]
            without_r = rerank.loc[rerank["variant"] == "no_rerank", "correctness"]
            if not with_r.empty and not without_r.empty and without_r.iloc[0] > 0:
                lift = (with_r.iloc[0] - without_r.iloc[0]) / without_r.iloc[0] * 100
                data.takeaway(f"Reranking improves mean correctness by {lift:.0f}%.")
            fig = px.bar(
                rerank,
                x="variant",
                y="correctness",
                color="variant",
                color_discrete_sequence=data.PALETTE,
                title="Correctness: reranking on vs off",
            )
            st.plotly_chart(data.style_fig(fig), use_container_width=True)

    # Finding 2: chunk size shows diminishing returns.
    with right:
        chunks = summary[summary["experiment_id"] == "01_chunk_size_sweep"].copy()
        if not chunks.empty:
            chunks["chunk_size"] = chunks["variant"].map(_chunk_size)
            chunks = chunks.sort_values("chunk_size")
            data.takeaway("Correctness peaks around chunk size 512, then flattens.")
            fig = px.line(
                chunks,
                x="chunk_size",
                y="correctness",
                markers=True,
                color_discrete_sequence=data.PALETTE,
                title="Correctness vs chunk size",
            )
            st.plotly_chart(data.style_fig(fig), use_container_width=True)

    # Finding 3: hard cases surface failures.
    question_df = data.question_long_df(results, questions)
    if not question_df.empty:
        by_difficulty = question_df.groupby("difficulty")["correctness"].mean().reset_index()
        data.takeaway(
            "Hard cases score well below easy/medium — they catch failures the rest miss."
        )
        fig = px.bar(
            by_difficulty,
            x="difficulty",
            y="correctness",
            color="difficulty",
            category_orders={"difficulty": ["easy", "medium", "hard"]},
            color_discrete_sequence=data.PALETTE,
            title="Mean correctness by question difficulty",
        )
        st.plotly_chart(data.style_fig(fig), use_container_width=True)

    st.divider()
    st.subheader("What we measure")
    st.markdown(
        "- **Retrieval quality** — precision@5, recall@5, MRR, nDCG@5 (deterministic)\n"
        "- **Faithfulness** — does every claim follow from the retrieved chunks? (LLM judge)\n"
        "- **Relevance** — does the answer address the question? (LLM judge)\n"
        "- **Correctness** — does the answer match ground truth? (LLM judge)"
    )
