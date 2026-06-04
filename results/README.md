# Results

**Pre-computed experiment results, committed to the repo.** This is the core
of the "easier to view and read" promise (spec §16.1): anyone who clones the
repo can browse the full results in the dashboard without API keys or running
anything.

Layout, one directory per experiment:

```
results/{experiment_id}/
├── manifest.json      # experiment config, run timestamp, guideline-gpt version
├── predictions.jsonl  # one Prediction per (variant, question)
└── metrics.json       # aggregated ExperimentResults
```

These files serialize the dataclasses in `eval_harness.types`.

> _Populated at milestone M6._
