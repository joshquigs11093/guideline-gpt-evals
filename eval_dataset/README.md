# Eval dataset

The hand-curated evaluation dataset for `guideline-gpt`. See [`DATASET.md`](../DATASET.md)
for the full construction methodology.

| File | Contents |
|---|---|
| `questions.jsonl` | The full eval set (~75 questions), one `EvalQuestion` per line. |
| `hard_cases.jsonl` | The 15 hand-crafted hard cases (also flagged `is_hard_case` in the full set). |
| `corpus_manifest.json` | The corpus snapshot the questions were authored against. |

Each line in `questions.jsonl` serializes an `eval_harness.types.EvalQuestion`.
The dataset validator (`eval_harness.dataset.validator`) runs in CI to confirm
schema validity and that every `relevant_chunk_ids` entry exists in the corpus.

> _Populated at milestones M1–M2._
