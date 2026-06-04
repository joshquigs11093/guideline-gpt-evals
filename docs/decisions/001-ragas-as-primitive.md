# ADR-001: Eval primitives — build the core, keep Ragas/DeepEval as options

**Status:** Accepted

## Context

The spec's principle was "reuse, don't reinvent" — build on Ragas / DeepEval
rather than from scratch. In practice the two metric families pulled in
different directions:

- **Retrieval metrics** (precision@k, recall@k, MRR, nDCG) are short, standard,
  deterministic formulas. Wrapping a library here adds a dependency and a layer
  of indirection for ~60 lines of code we want full test control over.
- **LLM-as-judge metrics** are where libraries add real value, but also where we
  most need transparency: the prompt, the few-shot calibration, the retry/fallback
  behaviour, and the exact model are all things a reviewer should be able to read.

## Decision

Implement the core ourselves and keep the libraries as an optional cross-check:

- Hand-roll `metrics/retrieval.py` with formulas documented in the docstring and
  tested against hand-computed values.
- Hand-roll the three judges on a shared base (`judges/`) with committed
  few-shot examples and an explicit retry/fallback contract.
- Declare `ragas` and `deepeval` in the `experiments` extra so they are available
  to cross-validate our scores, without forcing them (and their heavy transitive
  deps) on anyone who only browses the dashboard.

## Consequences

- **Transparency.** Every metric and judge is readable in-repo; nothing hides in
  a library default.
- **Lighter base install.** The dashboard image needs none of this (see ADR-005).
- **Tradeoff.** We own the maintenance of standard metrics. Mitigated by their
  simplicity and full test coverage.
- **Follow-up.** Wire a Ragas/DeepEval cross-check pass to sanity-check our judge
  scores against an independent implementation.
