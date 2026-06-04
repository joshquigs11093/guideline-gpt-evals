# ADR-004: Ship pre-computed results in the repo

**Status:** Accepted

## Context

Most eval repos bury results in a CSV or require you to run the pipeline (and pay
for API calls) before you can see anything. The goal here is the opposite: a
reviewer should browse the full findings in seconds, with no setup and no keys.

## Decision

- Commit results under `results/{experiment_id}/{variant}/` as
  `predictions.jsonl` + `metrics.json` + `manifest.json`. These are **not**
  gitignored — they are the published artifact.
- The dashboard and notebook read only from `results/`; neither runs an
  experiment or needs an API key.
- Re-running experiments is opt-in (`eval-harness run`, with your own keys); it
  overwrites the committed results in place.

## Consequences

- **Zero-setup browsing.** `docker compose up` or `streamlit run` shows
  everything immediately.
- **Reproducibility tension.** Committed results can drift from the code that
  produced them. Mitigated by recording the config, timestamp, and
  `guideline_gpt_version` in every manifest, and by CI checking that every
  experiment config has a results directory.
- **Honesty requirement.** Because results are committed and browseable, their
  provenance must be unmistakable — hence the `SYNTHETIC-DEMO` version stamp and
  the `results/SYNTHETIC_DATA.md` marker while the data is synthetic.
