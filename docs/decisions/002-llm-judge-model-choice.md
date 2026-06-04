# ADR-002: Judge with a stronger model than the system under test

**Status:** Accepted

## Context

LLM-as-judge scoring is only trustworthy if the judge is at least as capable as
the system it grades. A judge that is weaker than the model under test will
systematically misjudge the cases that matter most — the hard ones. There is also
a self-preference risk: a model judging its own family can inflate scores.

## Decision

- The judge defaults to a **stronger** model than the systems under test
  (`JUDGE_MODEL=claude-opus-4-7` by default, configurable). The systems under
  test in the experiments are smaller/cheaper models (Haiku, GPT-4o-mini).
- The judge provider is configured independently of the system under test
  (`JUDGE_PROVIDER`), so the judge can sit in a different model family from the
  generator to reduce self-preference.
- Every judge prompt ships **committed few-shot calibration** (3 examples each)
  so its scoring rubric is explicit and reviewable, not implied.
- Judges return a score **and a rationale**; the rationale is persisted and shown
  in the dashboard so scores are auditable rather than opaque.
- Parsing is defensive: structured-JSON verdict, bounded retries, and a safe
  `(0.0, "judge failed…")` fallback so one bad call never aborts an experiment.

## Consequences

- **Cost.** The judge is the most expensive model in the loop; judge calls are
  cached by `(question, prediction)` to avoid re-paying on re-runs.
- **Residual bias.** A stronger judge is not an unbiased judge — it can share
  blind spots with the system. Calibration and cross-checking (ADR-001) mitigate
  but do not eliminate this; it is named explicitly in the notebook's limitations.
