# guideline-gpt-evals

> A rigorous evaluation framework for RAG systems, demonstrated against
> [`guideline-gpt`](https://github.com/joshquigs11093/guideline-gpt).

[![CI](https://github.com/joshquigs11093/guideline-gpt-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/joshquigs11093/guideline-gpt-evals/actions/workflows/ci.yml)

This is the companion repository to `guideline-gpt`. It imports `guideline-gpt`
as a dependency, runs systematic experiments against it, and publishes the
results as a **browseable Streamlit dashboard** and a **rendered analysis
notebook** — so anyone can explore the findings without running code or
supplying API keys.

The two repos tell a complementary story: *I build* (`guideline-gpt`) +
*I measure* (this).

> **Status:** 🚧 Under construction. M0 (scaffolding + CI) in progress. See
> [`.spec/SPEC-guideline-gpt-evals.md`](.spec/SPEC-guideline-gpt-evals.md) for
> the full build specification and milestones.

## What it measures

Four metric families, applied across six controlled experiments:

- **Retrieval quality** — precision@k, recall@k, MRR, nDCG (deterministic)
- **Faithfulness** — does every claim follow from the retrieved chunks? (LLM-as-judge)
- **Answer relevance** — does the answer address the question? (LLM-as-judge)
- **Correctness** — does the answer match ground truth? (LLM-as-judge)

## Quick start

```bash
# Install (pulls guideline-gpt from GitHub by default).
uv venv
uv pip install -e ".[dev]"

# Browse pre-computed results — no API keys required.
streamlit run src/eval_harness/ui/dashboard.py
```

For local development of both repos in tandem, install the sibling clone
editable so changes to `guideline-gpt` are picked up immediately:

```bash
uv pip install -e ../guideline-gpt
```

To re-run experiments yourself (requires `ANTHROPIC_API_KEY` +
`OPENAI_API_KEY`), copy `.env.example` to `.env` and fill in your keys.

## Documentation

- [`METHODOLOGY.md`](METHODOLOGY.md) — how evaluation is done and why *(planned)*
- [`DATASET.md`](DATASET.md) — how the eval set was constructed *(planned)*
- [`docs/decisions/`](docs/decisions/) — architecture decision records *(planned)*

## License

[MIT](LICENSE)
