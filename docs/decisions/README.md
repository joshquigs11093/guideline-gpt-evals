# Architecture Decision Records

Short records of the consequential design choices for this project (spec §13).

1. `001-ragas-as-primitive.md` — build the metric/judge core; keep Ragas/DeepEval as options
2. `002-llm-judge-model-choice.md` — stronger-than-tested judge model; mitigating judge bias
3. `003-dataset-construction.md` — hybrid synthetic + hand-crafted; why ~75 questions
4. `004-pre-computed-results.md` — why ship results in the repo rather than always running live
5. `005-lean-dashboard-deps.md` — split deps so the dashboard image stays torch-free
