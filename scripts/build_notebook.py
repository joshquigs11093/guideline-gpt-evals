"""Build (and execute) the narrative analysis notebook.

Constructs ``notebooks/analysis.ipynb`` programmatically with nbformat — intro,
methodology, the six experiments (hypothesis -> method -> chart -> takeaway),
cross-experiment insights, limitations, and future work — then executes it so
the committed notebook carries rendered charts. Re-run after results change.

    python scripts/build_notebook.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO / "notebooks" / "analysis.ipynb"

SETUP_CODE = """\
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio

from eval_harness.report.store import load_results
from eval_harness.dataset.loader import load_eval_set

warnings.filterwarnings("ignore")
# CDN-backed renderer keeps the committed notebook/HTML small; charts load in a
# browser (and on nbviewer) without bundling ~3.5MB of plotly.js per file.
pio.renderers.default = "notebook_connected"

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
METRICS = ["precision_at_5", "recall_at_5", "mrr", "ndcg_at_5",
           "faithfulness", "relevance", "correctness"]

results = load_results(Path("results"))
questions = load_eval_set(Path("eval_dataset/questions.jsonl"))

rows = []
for experiment_id, variants in results.items():
    for variant_name, result in variants.items():
        row = {"experiment_id": experiment_id, "variant": variant_name,
               "cost_usd": result.total_cost_usd, "latency_ms": result.total_latency_ms}
        for metric in METRICS:
            row[metric] = result.aggregate_metrics.get(f"{metric}_mean", 0.0)
        rows.append(row)
summary = pd.DataFrame(rows)

qrows = []
by_id = {q.question_id: q for q in questions}
for experiment_id, variants in results.items():
    for variant_name, result in variants.items():
        for s in result.question_scores:
            q = by_id.get(s.question_id)
            qrows.append({"experiment_id": experiment_id, "variant": variant_name,
                          "difficulty": q.difficulty if q else "?",
                          "correctness": s.correctness, "faithfulness": s.faithfulness})
qdf = pd.DataFrame(qrows)

def show_bar(experiment_id, metric, title):
    df = summary[summary.experiment_id == experiment_id]
    px.bar(df, x="variant", y=metric, color="variant",
           color_discrete_sequence=PALETTE, title=title).show()

print(f"Loaded {len(results)} experiments, {len(questions)} questions.")
"""

EXPERIMENTS = [
    {
        "id": "01_chunk_size_sweep",
        "title": "Experiment 1 — Chunk size sweep",
        "hypothesis": "Larger chunks improve faithfulness (more context) but hurt retrieval "
        "precision (less focused chunks); there should be a sweet spot for end-to-end correctness.",
        "code": 'df = summary[summary.experiment_id == "01_chunk_size_sweep"].copy()\n'
        'df["chunk_size"] = df["variant"].str.extract(r"(\\d+)").astype(int)\n'
        'px.line(df.sort_values("chunk_size"), x="chunk_size", y="correctness", markers=True,\n'
        '        color_discrete_sequence=PALETTE, title="Correctness vs chunk size").show()',
        "takeaway": "Correctness peaks at 512 tokens and flattens by 1024 — bigger is not better "
        "once chunks stop being focused. 512 is the sweet spot the hypothesis predicted.",
    },
    {
        "id": "02_reranker_ablation",
        "title": "Experiment 2 — Reranker ablation",
        "hypothesis": "Cross-encoder reranking improves retrieval metrics and end-to-end "
        "correctness, especially on hard cases.",
        "code": 'show_bar("02_reranker_ablation", "correctness", "Correctness: rerank on/off")',
        "takeaway": "Reranking is the single biggest lever measured here: it lifts correctness "
        "well clear of the no-rerank baseline. Cheap to add, large effect.",
    },
    {
        "id": "03_hybrid_vs_vector",
        "title": "Experiment 3 — Hybrid vs vector-only vs BM25-only",
        "hypothesis": "Hybrid retrieval (RRF over vector + BM25) wins on recall overall; each "
        "single arm trails because it misses what the other catches.",
        "code": 'show_bar("03_hybrid_vs_vector", "recall_at_5", "Recall@5 by retrieval strategy")',
        "takeaway": "Hybrid clearly tops recall@5 — the two arms are complementary, and fusing "
        "them recovers relevant chunks neither finds alone.",
    },
    {
        "id": "04_top_k_sweep",
        "title": "Experiment 4 — Top-k sweep",
        "hypothesis": "Correctness improves with more reranked context but with diminishing "
        "returns past k=5, while cost rises with context size.",
        "code": 'df = summary[summary.experiment_id == "04_top_k_sweep"]\n'
        'fig = px.scatter(df, x="cost_usd", y="correctness", text="variant", size="latency_ms",\n'
        '                 color="variant", color_discrete_sequence=PALETTE,\n'
        '                 title="Correctness vs cost across top-k")\n'
        'fig.update_traces(textposition="top center")\n'
        "fig.show()",
        "takeaway": "The jump from k=3 to k=5 buys most of the correctness; k=10 adds cost and "
        "latency for little more. k=5 is the efficient choice.",
    },
    {
        "id": "05_llm_provider_comparison",
        "title": "Experiment 5 — LLM provider comparison",
        "hypothesis": "Anthropic Haiku and OpenAI GPT-4o-mini reach comparable correctness with "
        "different latency profiles.",
        "code": 'show_bar("05_llm_provider_comparison", "correctness", "Correctness by provider")',
        "takeaway": "Correctness is close between providers; latency and cost differentiate them, "
        "not answer quality — so pick on operational grounds.",
    },
    {
        "id": "06_prompt_variations",
        "title": "Experiment 6 — Prompt variations",
        "hypothesis": "An explicit 'cite your sources' instruction improves faithfulness; a "
        "'reason first' instruction improves correctness on hard cases.",
        "code": 'show_bar("06_prompt_variations", "faithfulness", "Faithfulness by prompt")',
        "takeaway": "The cite-sources prompt gives the largest faithfulness gain, matching the "
        "hypothesis that asking for grounding makes answers more grounded.",
    },
]

INTRO_MD = """\
# guideline-gpt-evals — analysis

> **⚠️ Synthetic demo data.** Every number below comes from
> `scripts/generate_synthetic_data.py`, not real measurements. The numbers are
> sampled to reflect the experiments' hypotheses so the narrative and charts are
> coherent — but they are placeholders until a real corpus + API runs exist.
> The *method* (how each finding would be read) is the real content here.

This notebook is the narrative companion to the dashboard. Where the dashboard is
browseable, this is a walkthrough: for each of six experiments we state a
**hypothesis**, show the **result**, and draw a **takeaway**. Stating the
hypothesis first is the point — it is what separates measuring from cherry-picking.
"""

METHODOLOGY_MD = """\
## Methodology in brief

- **Dataset.** 75 questions over a clinical-guideline corpus, including 15
  hand-crafted hard cases, each with ground-truth answers and relevant chunk ids.
- **Retrieval metrics** (deterministic): precision@5, recall@5, MRR, nDCG@5.
- **Quality metrics** (LLM-as-judge, scored 0–1 with rationales): faithfulness,
  relevance, correctness. The judge is a *stronger* model than the system under
  test, with committed few-shot calibration.
- **Honesty.** N=75 is small; treat differences within noise as inconclusive and
  read effects as directional, not definitive.
"""

CROSS_MD = """\
## Cross-experiment insights

Reading the six together: **reranking (Exp 2) and hybrid retrieval (Exp 3) move
the needle most** — the retrieval stage, not the generator, is where quality is
won or lost. Generation choices (provider in Exp 5, prompt in Exp 6) are
second-order by comparison, with the cite-sources prompt the most worthwhile
generation-side tweak. The configuration knobs (chunk size, top-k) show clear
sweet spots (512, k=5) rather than monotonic gains, so tuning past them mostly
buys cost.
"""

LIMITATIONS_MD = """\
## Limitations

- **These results are synthetic.** They demonstrate the analysis, not real
  system behaviour.
- **Small N (75).** Limited statistical power; per-slice (e.g. hard-case) numbers
  rest on few questions.
- **Single domain.** Clinical guidelines only — findings may not transfer.
- **Judge bias.** LLM-as-judge can share blind spots with the system; calibration
  mitigates but does not eliminate this.
"""

FUTURE_MD = """\
## Future work

- Run the experiments for real against an ingested corpus and replace the
  synthetic results.
- Wire the prompt-injectable pipeline so Experiment 6 executes end-to-end.
- Expand the dataset and add confidence intervals / significance testing.
- Add CI regression detection so a config change that drops correctness fails the
  build.
"""


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    cells = [
        nbf.v4.new_markdown_cell(INTRO_MD),
        nbf.v4.new_markdown_cell(METHODOLOGY_MD),
        nbf.v4.new_code_cell(SETUP_CODE),
    ]
    for exp in EXPERIMENTS:
        cells.append(
            nbf.v4.new_markdown_cell(f"## {exp['title']}\n\n**Hypothesis.** {exp['hypothesis']}")
        )
        cells.append(nbf.v4.new_code_cell(exp["code"]))
        cells.append(nbf.v4.new_markdown_cell(f"**Takeaway.** {exp['takeaway']}"))
    cells.extend(
        [
            nbf.v4.new_markdown_cell(CROSS_MD),
            nbf.v4.new_markdown_cell(LIMITATIONS_MD),
            nbf.v4.new_markdown_cell(FUTURE_MD),
        ]
    )
    nb.cells = cells
    return nb


def ensure_kernel() -> None:
    """Register the current interpreter as the 'python3' kernel inside the venv."""
    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install", "--sys-prefix", "--name", "python3"],
        check=True,
        capture_output=True,
    )


def main() -> None:
    ensure_kernel()
    nb = build_notebook()
    print(f"Executing {len(nb.cells)} cells...")
    client = NotebookClient(
        nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(REPO)}}
    )
    client.execute()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
