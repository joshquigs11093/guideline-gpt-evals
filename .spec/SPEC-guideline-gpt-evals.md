# guideline-gpt-evals — Build Specification

**Version:** 1.0
**Status:** Ready to build
**Owner:** Joshua Quigley
**Companion to:** [`guideline-gpt`](https://github.com/joshquigs11093/guideline-gpt)

---

## 1. Project overview

Build a rigorous evaluation framework for RAG systems, demonstrated against `guideline-gpt`. The framework measures retrieval quality, faithfulness, answer relevance, and end-to-end correctness; runs systematic experiments comparing architecture variants; and publishes results as a **browseable dashboard** and a **rendered analysis notebook** so anyone can explore findings without running code.

This is a companion repository to `guideline-gpt`. It imports `guideline-gpt` as a dependency, runs experiments against it, and produces published results. The two repos tell a complementary story: *I build* + *I measure*.

### 1.1 Goals

- A high-quality, hand-curated **evaluation dataset** of ~75 questions covering the default clinical guideline corpus, including 15 hand-crafted "hard cases" requiring domain expertise
- A **config-driven experiment harness** that runs multiple RAG variants through the eval set and produces comparable results
- A **published results dashboard** (Streamlit) anyone can browse without API keys, showing pre-computed experiment results
- A **rendered analysis notebook** walking through 6 experiments and the design conclusions drawn from each
- Documentation strong enough to stand alone as a portfolio piece

### 1.2 Non-goals

- Not competing with Ragas, DeepEval, Promptfoo, or commercial platforms — this **uses** them as primitives
- No real-time eval-in-production tracing (that's observability, different problem)
- No human-eval crowdsourcing infrastructure
- No automatic regression detection in CI (planned future work, not v1)
- No model training, fine-tuning, or RLHF
- No general-purpose framework — this is purpose-built for guideline-gpt and similar systems

### 1.3 Target audience

1. Engineering hiring managers evaluating the author's portfolio
2. Practitioners learning how to evaluate RAG systems rigorously
3. Healthcare AI researchers interested in clinical-domain RAG benchmarks

### 1.4 What "easier to view and read" means for this project

The user has been explicit that readability of results is a top priority. Concretely this means:

- Pre-computed results shipped in the repo so anyone can browse without setup
- Streamlit dashboard as the primary results interface
- Rendered analysis notebook (committed as `.ipynb` + exported HTML) viewable directly on GitHub
- Charts, not tables, wherever possible
- Every chart has a one-sentence takeaway above it
- README has screenshots/GIFs of the dashboard

---

## 2. Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Eval dataset (offline)                      │
│                                                                  │
│   Corpus chunks ──► LLM-generated Q&A ──► Manual curation       │
│                                                  │               │
│   Domain expertise ──► Hand-crafted hard cases ──┤               │
│                                                  ▼               │
│                                          questions.jsonl         │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                  Experiment runner (offline)                    │
│                                                                  │
│   experiment.yaml ──► Variant builder ──► RAG pipeline          │
│                                                │                 │
│                                                ▼                 │
│   eval set ──────────────────────────► Predictions              │
│                                                │                 │
│                                                ▼                 │
│                                    Metrics + LLM-as-judge       │
│                                                │                 │
│                                                ▼                 │
│                                       results/{experiment}.json │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                    Results viewer (online)                      │
│                                                                  │
│   results/*.json ──► Streamlit dashboard                        │
│                  ──► Jupyter notebook ──► HTML export           │
└────────────────────────────────────────────────────────────────┘
```

### 2.1 Tech stack

| Layer | Choice | Justification |
|---|---|---|
| Language | Python 3.11+ | Match parent project |
| Package mgmt | `uv` | Match parent project |
| RAG dependency | `guideline-gpt` (editable install) | The system under test |
| Eval primitives | Ragas + DeepEval | Don't reinvent; build on standards |
| LLM-as-judge | Provider-abstracted (Anthropic or OpenAI) | Reuse parent's `LLMClient` |
| Data | Pandas + JSONL | Standard; JSONL is diffable in git |
| Experiment config | YAML | Human-readable, declarative |
| Dashboard | Streamlit | Match parent project; fast to build |
| Charts | Plotly | Interactive; renders nicely in Streamlit and notebooks |
| Notebook | Jupyter | Standard; exported to HTML for GitHub viewing |
| CLI | Typer | Match parent project |
| Config | Pydantic Settings | Match parent project |
| Testing | pytest + pytest-cov | Match parent project |
| Linting/types | ruff + mypy --strict | Match parent project |
| CI | GitHub Actions | Match parent project |

**Principle: identical tooling to `guideline-gpt`.** A reviewer browsing both repos should see consistent conventions. Senior signal.

---

## 3. Repository structure

```
guideline-gpt-evals/
├── README.md
├── METHODOLOGY.md
├── DATASET.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── docs/
│   ├── images/
│   └── decisions/
│       ├── 001-ragas-as-primitive.md
│       ├── 002-llm-judge-model-choice.md
│       ├── 003-dataset-construction.md
│       └── 004-pre-computed-results.md
├── eval_dataset/
│   ├── README.md
│   ├── questions.jsonl
│   ├── hard_cases.jsonl
│   └── corpus_manifest.json
├── experiments/
│   ├── 01_chunk_size_sweep.yaml
│   ├── 02_reranker_ablation.yaml
│   ├── 03_hybrid_vs_vector.yaml
│   ├── 04_top_k_sweep.yaml
│   ├── 05_llm_provider_comparison.yaml
│   └── 06_prompt_variations.yaml
├── results/
│   └── {experiment_id}/
│       ├── manifest.json
│       ├── predictions.jsonl
│       └── metrics.json
├── notebooks/
│   ├── analysis.ipynb
│   └── analysis.html
├── src/eval_harness/
│   ├── __init__.py
│   ├── config.py
│   ├── cli.py
│   ├── types.py
│   ├── dataset/
│   │   ├── generator.py
│   │   ├── loader.py
│   │   └── validator.py
│   ├── metrics/
│   │   ├── retrieval.py
│   │   ├── faithfulness.py
│   │   ├── relevance.py
│   │   └── correctness.py
│   ├── judges/
│   │   ├── base.py
│   │   ├── faithfulness_judge.py
│   │   ├── relevance_judge.py
│   │   └── correctness_judge.py
│   ├── runner/
│   │   ├── experiment.py
│   │   ├── runner.py
│   │   └── variant.py
│   ├── report/
│   │   ├── aggregator.py
│   │   ├── comparator.py
│   │   └── exporter.py
│   └── ui/
│       ├── dashboard.py
│       └── components/
│           ├── overview.py
│           ├── experiment_view.py
│           ├── comparison_view.py
│           ├── question_explorer.py
│           └── failure_analysis.py
├── scripts/
│   ├── generate_dataset.py
│   ├── run_all_experiments.py
│   └── export_notebook.py
└── tests/
    ├── conftest.py
    ├── test_metrics_retrieval.py
    ├── test_metrics_faithfulness.py
    ├── test_judges.py
    ├── test_runner.py
    └── test_aggregator.py
```

---

## 4. Core data types

Define in `src/eval_harness/types.py`. Use throughout.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Difficulty = Literal["easy", "medium", "hard"]
QuestionType = Literal["factoid", "synthesis", "comparison", "ambiguous"]


@dataclass(frozen=True)
class EvalQuestion:
    """A single question in the eval dataset."""
    question_id: str
    question: str
    ground_truth_answer: str
    relevant_chunk_ids: list[str]
    source_documents: list[str]
    difficulty: Difficulty
    question_type: QuestionType
    is_hard_case: bool = False
    rationale: str = ""


@dataclass(frozen=True)
class Prediction:
    """A single RAG system's response to one question."""
    question_id: str
    retrieved_chunk_ids: list[str]
    retrieved_chunks_text: list[str]
    answer: str
    citations: list[str]
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class QuestionScore:
    """All metrics for one question, one experiment."""
    question_id: str
    precision_at_5: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    faithfulness: float
    relevance: float
    correctness: float
    faithfulness_rationale: str
    relevance_rationale: str
    correctness_rationale: str


@dataclass(frozen=True)
class ExperimentResults:
    """Aggregated results for one experiment."""
    experiment_id: str
    experiment_name: str
    config: dict
    run_timestamp: datetime
    guideline_gpt_version: str
    question_scores: list[QuestionScore]
    aggregate_metrics: dict[str, float]
    total_cost_usd: float
    total_latency_ms: int
```

**These dataclasses are the contract** between every layer. JSONL files in `results/` serialize these directly. No untyped dicts allowed.

---

## 5. Component specifications

### 5.1 Dataset construction

**`dataset/generator.py`** — used once, then retired.

- `generate_questions(corpus_dir: Path, n_per_chunk: int, llm: LLMClient) -> list[EvalQuestion]`
- For each chunk, prompts a strong LLM (Claude Opus or GPT-4o) to generate Q&A pairs grounded in that chunk
- Filters out questions whose ground-truth answer is trivially in the question phrasing or contain phrases like "according to the passage"
- Outputs JSONL for manual review

**Manual curation workflow** (documented in `DATASET.md`):
1. Run generator to produce ~150 candidates
2. Reviewer (the author) reviews each, marking keep/edit/reject
3. Target: 60 medium-quality questions retained from synthetic generation
4. Author hand-writes 15 hard cases drawing on RT domain expertise
5. Final set: 75 questions, with `is_hard_case=True` on the 15 hand-crafted ones

**`dataset/loader.py`** — `load_eval_set(path: Path) -> list[EvalQuestion]`; validates schema on load via Pydantic.

**`dataset/validator.py`** — runs as part of CI. Confirms all `relevant_chunk_ids` exist in the current corpus, no duplicates, required fields populated.

### 5.2 Metrics

**`metrics/retrieval.py`** — deterministic, no LLM:
- `precision_at_k`, `recall_at_k`, `mean_reciprocal_rank`, `ndcg_at_k`
- Standard implementations; cite formulas in docstrings

**`metrics/faithfulness.py`** — LLM-as-judge. "Does every claim in the answer follow from the retrieved chunks?" Returns score in [0,1] + rationale.

**`metrics/relevance.py`** — LLM-as-judge. "Does the answer address what was asked?" Returns score in [0,1] + rationale.

**`metrics/correctness.py`** — LLM-as-judge. "Does the answer match the ground-truth answer in substance?" Returns score in [0,1] + rationale. Uses `ground_truth_answer` as reference.

### 5.3 Judges

**Critical design rule: use a stronger model as judge than the model under test.** When testing GPT-4o-mini, judge with GPT-4o or Claude Opus. Document this in ADR-002.

**`judges/base.py`**
```python
from typing import Protocol

class Judge(Protocol):
    def score(self, *, question: EvalQuestion, prediction: Prediction) -> tuple[float, str]:
        """Return (score in [0,1], rationale)."""
```

Implementation pattern:
- Each judge has a carefully designed prompt that asks for a structured response (JSON with `score` and `rationale`)
- Uses Pydantic to validate the JSON response
- Retries up to 2x on parse failure
- Falls back to score=0.0 with rationale="judge failed to produce valid output" on persistent failure (logged as warning)

**Calibration:** Each judge prompt includes 3–5 few-shot examples of how to score. Examples committed to the repo and reviewable.

### 5.4 Experiment runner

**`runner/experiment.py`** — YAML format:

```yaml
experiment_id: "01_chunk_size_sweep"
name: "Chunk size sweep"
description: "Compare retrieval quality at chunk sizes 256, 512, 1024"
variants:
  - name: "chunks_256"
    overrides:
      chunk_size: 256
      chunk_overlap: 32
  - name: "chunks_512"
    overrides:
      chunk_size: 512
      chunk_overlap: 64
  - name: "chunks_1024"
    overrides:
      chunk_size: 1024
      chunk_overlap: 128
shared_config:
  retrieval_top_k: 20
  rerank_top_k: 5
  llm_provider: "anthropic"
```

**`runner/variant.py`** — `build_variant(name, overrides, shared)` constructs a `guideline-gpt` `QueryPipeline` with override settings. Re-runs ingestion if chunk-related settings changed (cached by config hash).

**`runner/runner.py`** — `run_experiment(experiment_path, eval_set_path) -> ExperimentResults`. For each variant × each question: build pipeline, run query, score with all metrics and judges. Persists `predictions.jsonl`, `metrics.json`, and `manifest.json` to `results/{experiment_id}/`. Progress bar; resumable if interrupted. Caches LLM judge calls by hash of (question_id, prediction text).

### 5.5 Report

**`report/aggregator.py`** — rolls per-question scores into experiment-level aggregates (mean, std, percentiles, per-difficulty and per-question-type breakdowns).

**`report/comparator.py`** — given two or more experiments, produces a comparison table and statistical comparisons (paired t-tests on per-question scores; flag changes likely to be noise).

**`report/exporter.py`** — renders to HTML and markdown for embedding in README and notebook.

---

## 6. The results dashboard

**This is the differentiating artifact for this project.** Most eval repos have results buried in a CSV. This one has a browseable Streamlit app.

### 6.1 Dashboard layout

Single Streamlit app, multi-page (`st.navigation`):

```
guideline-gpt-evals dashboard

[Sidebar]
  Pages:
    📊 Overview
    🧪 Experiments
    ⚖️ Compare
    🔍 Question Explorer
    ❌ Failure Analysis
    📖 Methodology
```

### 6.2 Page: Overview

Landing page; renders even without selecting an experiment.

- Header: project name, one-line description, "View on GitHub" link
- Stats row (large numbers, SaaS-landing-page style):
  - `75` Eval Questions
  - `6` Experiments Run
  - `~$X.XX` Total LLM Spend on Evaluation
  - `XX,XXX` Total Predictions Scored
- "Headline findings" section: 3 takeaways from the experiments, each with a small chart
  - "Reranking improves correctness by 18%" + bar chart
  - "Chunk size shows diminishing returns above 512" + line chart
  - "Hard cases catch failures the synthetic set misses" + scatter
- Brief explainer of RAG evaluation and the four metric families

### 6.3 Page: Experiments

A grid/list of all 6 experiments. Each is a card showing:
- Experiment name
- One-line description
- Run timestamp
- Headline metric (e.g., "Best variant: chunks_512, correctness 0.78")
- "View details" button

Detail view:
- Full description
- Config table (the YAML, rendered)
- Per-variant aggregate metrics (table + bar charts)
- Per-difficulty breakdown (faceted chart)
- Per-question-type breakdown (faceted chart)
- Cost and latency comparison

### 6.4 Page: Compare

User selects 2+ experiments (or 2+ variants across experiments) and sees:
- Side-by-side aggregate metrics
- Win/loss/tie counts per metric
- Statistical significance flags (paired t-test, p < 0.05)
- Scatter plot of per-question scores (variant A on x, variant B on y) showing which questions move which direction
- "Questions where they disagree" — expandable comparison

### 6.5 Page: Question Explorer

Searchable, filterable view of every question in the eval set.

Filters:
- Difficulty (easy/medium/hard)
- Question type (factoid/synthesis/comparison/ambiguous)
- Hard case only (toggle)
- Source document (multi-select)
- Variant performance (e.g., "questions where chunks_256 scored < 0.5")

For each question:
- The question text
- Ground-truth answer
- Source documents
- Per-experiment scores (sparkline across variants)
- Expandable: each variant's prediction, retrieved chunks, judge rationales

### 6.6 Page: Failure Analysis

The most interesting page for senior reviewers — shows you can *learn* from failures.

For a selected experiment:
- "Top 10 worst-scoring questions" — list with judge rationales
- Failure categorization: cluster failures by judge rationale themes
- Common failure pattern cards: "Retrieval miss," "Hallucination," "Partial answer," "Confidence inflation"
- "What I learned" notes (curated commentary from author — markdown files in `docs/failure_notes/`)

### 6.7 Page: Methodology

Renders `METHODOLOGY.md` and `DATASET.md` inline. Reference for visitors who want to understand the rigor.

### 6.8 Pre-computed results

**The dashboard reads from `results/*/*.json` — no live experiments, no API keys required.** Anyone who clones the repo and runs `streamlit run` sees the full pre-computed results. This is the "easier to view and read" promise made real.

To re-run experiments, users follow `docs/running-experiments.md` and provide their own API keys.

### 6.9 Visual design notes

- Wide layout (`st.set_page_config(layout="wide")`)
- All charts use Plotly with a consistent color palette (colorblind-safe)
- Every chart has a one-sentence takeaway in `st.info` above it
- Numbers use locale formatting; costs in `$0.0000` format
- Failure rationales rendered as quote blocks
- Sticky filters in `st.sidebar` per page

---

## 7. The analysis notebook

`notebooks/analysis.ipynb` is the long-form complement to the dashboard. Where the dashboard is browseable, the notebook is **narrative**.

Structure:
1. **Introduction** — what we set out to learn
2. **Methodology summary** — the dataset, the metrics, the judges
3. **Experiment 1: Chunk size sweep** — hypothesis, results, takeaway
4. **Experiment 2: Reranker ablation** — hypothesis, results, takeaway
5. **Experiment 3: Hybrid vs vector-only** — hypothesis, results, takeaway
6. **Experiment 4: Top-k sweep** — hypothesis, results, takeaway
7. **Experiment 5: LLM provider comparison** — hypothesis, results, takeaway
8. **Experiment 6: Prompt variations** — hypothesis, results, takeaway
9. **Cross-experiment insights** — what patterns emerged across all six
10. **Limitations of this evaluation** — explicitly named (small dataset, single domain, judge bias risks)
11. **Future work** — what's next

Every section: hypothesis → method → results (chart) → takeaway. Senior practitioners write this way; junior practitioners just show results.

`scripts/export_notebook.py` exports to `notebooks/analysis.html` for direct GitHub viewing. CI verifies HTML is up to date.

---

## 8. The six experiments

### 8.1 Chunk size sweep
- **Variants:** chunk_size ∈ {256, 512, 1024}; overlap = chunk_size / 8
- **Hypothesis:** larger chunks improve faithfulness (more context) but hurt retrieval precision (less focused chunks)
- **Primary metric:** correctness
- **Secondary:** precision@5, faithfulness

### 8.2 Reranker ablation
- **Variants:** rerank_top_k from {0 (no rerank), 5}; retrieval_top_k = 20
- **Hypothesis:** reranking improves all retrieval metrics and especially helps on hard cases
- **Primary metric:** precision@5
- **Secondary:** correctness on hard_cases slice

### 8.3 Hybrid vs vector-only
- **Variants:** {vector_only, bm25_only, hybrid_rrf}
- **Hypothesis:** hybrid wins overall; BM25 wins on questions with rare technical terms; vector wins on paraphrased questions
- **Primary metric:** recall@5
- **Secondary:** per-question-type breakdown

### 8.4 Top-k sweep
- **Variants:** rerank_top_k ∈ {3, 5, 10}
- **Hypothesis:** diminishing returns past 5; cost scales linearly
- **Primary metric:** correctness vs cost tradeoff curve

### 8.5 LLM provider comparison
- **Variants:** Anthropic Haiku vs OpenAI GPT-4o-mini, same retrieval config
- **Hypothesis:** comparable correctness, different latency profiles
- **Primary metric:** correctness, faithfulness, latency

### 8.6 Prompt variations
- **Variants:** baseline prompt vs explicit "cite your sources" vs "explain reasoning before answering"
- **Hypothesis:** explicit citation instruction improves faithfulness; reasoning-first improves correctness on hard cases
- **Primary metric:** faithfulness
- **Secondary:** correctness on hard_cases slice

---

## 9. Configuration

All via env vars; identical pattern to parent project.

| Variable | Default | Purpose |
|---|---|---|
| `JUDGE_PROVIDER` | `anthropic` | Provider for LLM-as-judge calls |
| `JUDGE_MODEL` | `claude-opus-4-7` | Strong model for judging |
| `ANTHROPIC_API_KEY` | — | Required if judge provider is anthropic |
| `OPENAI_API_KEY` | — | Required for embeddings via guideline-gpt |
| `RESULTS_DIR` | `./results` | Where experiment outputs land |
| `EVAL_DATASET_PATH` | `./eval_dataset/questions.jsonl` | The eval set |
| `JUDGE_CACHE_DIR` | `./.judge_cache` | Cached judge responses |
| `LOG_LEVEL` | `INFO` | structlog level |

---

## 10. Testing

- **Unit test coverage target: 75%+** for `src/eval_harness/` excluding `ui/`
- Every metric in `metrics/retrieval.py` tested against hand-computed expected values
- Judges tested with mocked LLM responses (canned JSON) — no real API calls in CI
- Runner tested end-to-end with a stub `QueryPipeline` and a 3-question mini eval set
- Dataset validator runs in CI against committed `eval_dataset/questions.jsonl`
- `mypy --strict` passes on `src/`

---

## 11. CI

Identical pattern to parent project. Additionally:
- Validates `eval_dataset/questions.jsonl` schema on every PR
- Validates that `notebooks/analysis.html` is up to date with `analysis.ipynb` (fails if diff)
- Validates that every `experiments/*.yaml` has a corresponding `results/*/` folder

---

## 12. Docker

Single-purpose: run the dashboard. Multi-stage Dockerfile. Image runs `streamlit run src/eval_harness/ui/dashboard.py`. No API keys needed — dashboard reads pre-computed results from the mounted `results/` volume.

```
docker compose up
```
Opens dashboard at http://localhost:8501 with full pre-computed results visible.

---

## 13. Architecture Decision Records

Write all four before declaring shippable:
1. `001-ragas-as-primitive.md` — why we build on Ragas/DeepEval rather than from scratch
2. `002-llm-judge-model-choice.md` — stronger-than-tested judge model; how we mitigate judge bias
3. `003-dataset-construction.md` — hybrid synthetic+hand-crafted; why ~75 questions; how hard cases differ
4. `004-pre-computed-results.md` — why ship results in the repo rather than always running live

---

## 14. Milestones

| Milestone | Deliverable | Acceptance criteria |
|---|---|---|
| M0 | Scaffolding + CI | Empty repo passes CI; `uv pip install -e .` succeeds; imports `guideline-gpt` |
| M1 | Eval dataset v1 | `eval_dataset/questions.jsonl` exists with ≥50 curated questions; validator passes |
| M2 | Eval dataset complete | All 75 questions including 15 hand-crafted hard cases; `DATASET.md` complete |
| M3 | Retrieval metrics | `metrics/retrieval.py` complete with full test coverage |
| M4 | LLM-as-judge | All 3 judges working; tested against mocked responses; few-shot examples committed |
| M5 | Runner + first experiment | `01_chunk_size_sweep` runs end-to-end and persists results |
| M6 | All 6 experiments run | All experiments completed; `results/` populated |
| M7 | Dashboard | All 6 dashboard pages working against pre-computed results |
| M8 | Analysis notebook | Complete narrative notebook + HTML export + screenshots |
| M9 | Polish + ship | README, METHODOLOGY, all 4 ADRs, Docker, screenshots, deploy to demo |

---

## 15. Style conventions

Identical to parent project. See `guideline-gpt/SPEC.md` §15.

---

## 16. Engineering principles

1. **Pre-computed is the default.** Anyone should be able to browse the full results without an API key or running anything but Streamlit. The repo is a published artifact, not just code.
2. **Judges are calibrated.** Every judge prompt has few-shot examples. Document them. Never trust an uncalibrated judge.
3. **Compare like-for-like.** When comparing variants, only one thing changes at a time. Document confounds.
4. **Statistical honesty.** Small datasets (N=75) limit statistical power. Flag effects that might be noise. Report confidence intervals.
5. **Failure cases are first-class.** A repo that surfaces what it gets wrong is more credible than one that only shows what it gets right.
6. **Reuse, don't reinvent.** Ragas, DeepEval, scikit-learn metrics — use them. Credit them in README. Build the layer of analysis on top that they don't provide.

---

## 17. Definition of "done"

- [ ] CI is green on `main`
- [ ] All 75 questions in `eval_dataset/questions.jsonl` validated and committed
- [ ] All 6 experiments have committed results in `results/`
- [ ] Dashboard renders all 6 pages from pre-computed data
- [ ] `notebooks/analysis.ipynb` walks through all 6 experiments with charts and takeaways
- [ ] `notebooks/analysis.html` exported and current
- [ ] All 4 ADRs written
- [ ] `METHODOLOGY.md` and `DATASET.md` complete
- [ ] README has dashboard screenshots/GIF
- [ ] `docker compose up` launches the dashboard
- [ ] Test coverage ≥ 75% on non-UI code
- [ ] `mypy --strict` passes
- [ ] At least one experiment finding has been turned into a LinkedIn post (counts as ship)

---

## 18. What this project demonstrates to a reviewer

A senior engineer or hiring manager reviewing this repo should walk away knowing the author can:

- Design and curate an evaluation dataset with rigor
- Implement and calibrate LLM-as-judge methodology
- Run systematic, comparable experiments
- Distinguish signal from noise statistically
- Communicate findings in both interactive (dashboard) and narrative (notebook) form
- Build on existing primitives (Ragas, DeepEval) without reinventing them
- Write honest documentation including limitations and failure modes
- Apply the same engineering standards (types, tests, CI) to evaluation code as to production code

This is what "I think rigorously about AI systems" looks like in artifacts.
