# Experiments

Declarative YAML definitions for each controlled experiment. The runner
(`eval_harness.runner`) reads one of these, builds each variant against
`guideline-gpt`, runs the eval set through it, and writes results to
`results/{experiment_id}/`.

The six experiments (spec §8):

1. `01_chunk_size_sweep.yaml` — chunk_size ∈ {256, 512, 1024}
2. `02_reranker_ablation.yaml` — rerank on/off
3. `03_hybrid_vs_vector.yaml` — vector / BM25 / hybrid RRF
4. `04_top_k_sweep.yaml` — rerank_top_k ∈ {3, 5, 10}
5. `05_llm_provider_comparison.yaml` — Anthropic Haiku vs OpenAI GPT-4o-mini
6. `06_prompt_variations.yaml` — baseline vs cite-sources vs reason-first

> _Populated at milestones M5–M6._
