# Methodology

How `guideline-gpt-evals` measures a RAG system, and why each choice was made.
This is the reference behind the numbers in the dashboard and notebook.

> The committed numbers are currently **synthetic** (`SYNTHETIC-DEMO`); see
> [`DATASET.md`](DATASET.md) and [ADR-004](docs/decisions/004-pre-computed-results.md).
> The methodology below is what a real run executes.

## The four metric families

Each prediction is scored on seven metrics across two families.

### Retrieval quality (deterministic, no LLM)

Computed from the retrieved chunk ids vs. the ground-truth relevant ids, at a
cutoff of k=5. Formulas are in the docstring of `metrics/retrieval.py` and each
is tested against hand-computed values.

| Metric | Question it answers |
|---|---|
| precision@5 | Of the top-5 retrieved, how many are relevant? |
| recall@5 | Of the relevant chunks, how many made the top-5? |
| MRR | How high is the *first* relevant chunk ranked? |
| nDCG@5 | Are relevant chunks ranked near the top (rank-weighted)? |

Relevance is binary; duplicate retrieved ids are collapsed so a repeat cannot
inflate a score.

### Answer quality (LLM-as-judge)

Three judges, each returning a score in `[0, 1]` **and a rationale**:

- **Faithfulness** — does every claim in the answer follow from the *retrieved
  chunks*? (judged against the chunks only, never the ground truth)
- **Relevance** — does the answer address the question that was asked?
- **Correctness** — does the answer match the ground-truth answer in substance?

## The judges

See [ADR-002](docs/decisions/002-llm-judge-model-choice.md). In short: the judge
is a **stronger** model than the system under test, configured independently of
the generator, with **committed few-shot calibration** in every prompt. Judges
parse a structured JSON verdict, retry on malformed output, and fall back to a
safe `0.0` rather than aborting a run. Rationales are persisted and surfaced in
the dashboard so every score is auditable.

## The experiment runner

An experiment (`experiments/*.yaml`) lists variants that each tweak the system.
For every variant × every question the runner builds the pipeline, runs the
query, maps the response to a `Prediction`, and scores it. Results are written to
`results/{experiment_id}/{variant}/`.

Most variants are plain settings overrides, but three experiments need
**component injection** (no-rerank, single-arm retrieval, prompt swaps) because
the parent cannot express them through settings — see the variant builder and
the experiment configs.

## Statistical honesty

- **N = 75 is small.** Per-slice numbers (e.g. the 15 hard cases) rest on few
  questions; treat differences within noise as inconclusive.
- Aggregates report mean **and** standard deviation, plus per-difficulty and
  per-question-type breakdowns, so spread is visible rather than hidden behind a
  single number.
- Comparisons are read as **directional**, not definitive, until a larger set and
  significance testing are added (see the notebook's future-work section).

## What this is not

Not a general-purpose framework, not a leaderboard, and not a substitute for
Ragas/DeepEval — it is a purpose-built, transparent harness for `guideline-gpt`
that you can read end to end.
