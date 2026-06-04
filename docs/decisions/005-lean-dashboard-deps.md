# ADR-005: Split deps so the dashboard stays torch-free

**Status:** Accepted

## Context

The package depends on `guideline-gpt` (the system under test), which pulls in
`sentence-transformers` → torch and `chromadb` (~2 GB). But the **dashboard never
uses any of it** — it only reads pre-computed JSON. Bundling torch into a
results-viewer image is ~2 GB of dead weight, and forcing it on anyone who just
wants to browse contradicts the zero-setup goal (ADR-004).

## Decision

- **Base dependencies** = only what the dashboard / report layer / CLI IO need
  (pandas, plotly, streamlit, pydantic, pyyaml, …). Torch-free.
- **`experiments` extra** = `guideline-gpt`, `ragas`, `deepeval`, `chromadb` —
  everything needed to *run* experiments. Installed via `.[experiments]`.
- The **Docker image installs base deps only** and serves the dashboard; CI
  installs `.[dev,experiments]` to run the full test suite.
- A test (`test_dashboard_lean_deps.py`) blocks `guideline_gpt`/`torch`/
  `chromadb`/`ragas`/`deepeval` imports and asserts the dashboard import graph
  still loads — guarding the split from regressions.

## Consequences

- **Small image, fast cold start** for the published artifact.
- **`chromadb>=1.0` pin** lives in the extra: the parent only pins `>=0.5`, and
  old 0.5.x builds bundle an OpenTelemetry whose generated protobufs break
  against modern protobuf ("Descriptors cannot be created directly").
- **Sharp edge.** `eval-harness run` and the judge/runner code need the
  `experiments` extra; importing them in a base-only environment fails by design.
