"""Generate a synthetic eval set and synthetic experiment results.

This populates ``eval_dataset/`` and ``results/`` with believable, internally
consistent **placeholder** data so the dashboard and notebook can be built and
browsed before a real corpus + API runs exist. Numbers are sampled to reflect
the spec's hypotheses (e.g. reranking helps; chunk 512 is a sweet spot) but are
NOT real measurements.

Everything is clearly labelled synthetic: ``guideline_gpt_version`` is set to
``SYNTHETIC-DEMO`` and a ``results/SYNTHETIC_DATA.md`` marker is written.

Run from the repo root:

    python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

from eval_harness.report.aggregator import aggregate_scores
from eval_harness.runner.experiment import load_experiment
from eval_harness.runner.runner import persist_variant_results
from eval_harness.types import EvalQuestion, ExperimentResults, Prediction, QuestionScore

SEED = 20260604
SYNTHETIC_VERSION = "SYNTHETIC-DEMO"
REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "eval_dataset"
RESULTS_DIR = REPO / "results"
EXPERIMENTS_DIR = REPO / "experiments"
TIMESTAMP = datetime(2026, 6, 4, 9, 0, 0)

DOCS = [
    "bts_asthma_2024",
    "gold_copd_2024",
    "nice_pneumonia_2023",
    "ats_ards_2023",
    "surviving_sepsis_2021",
]
CHUNKS_PER_DOC = 40
QTYPES = ["factoid", "synthesis", "comparison", "ambiguous"]

METRICS = (
    "precision_at_5",
    "recall_at_5",
    "mrr",
    "ndcg_at_5",
    "faithfulness",
    "relevance",
    "correctness",
)

BASE = {
    "precision_at_5": 0.62,
    "recall_at_5": 0.70,
    "mrr": 0.72,
    "ndcg_at_5": 0.68,
    "faithfulness": 0.80,
    "relevance": 0.85,
    "correctness": 0.66,
}

# Per-(experiment, variant) deltas applied to BASE; unspecified metrics use BASE.
PROFILES: dict[str, dict[str, dict[str, float]]] = {
    "01_chunk_size_sweep": {
        "chunks_256": {"precision_at_5": 0.08, "faithfulness": -0.06, "correctness": -0.03},
        "chunks_512": {"precision_at_5": 0.02, "correctness": 0.06},
        "chunks_1024": {"precision_at_5": -0.08, "faithfulness": 0.06, "correctness": -0.01},
    },
    "02_reranker_ablation": {
        "with_rerank": {
            "precision_at_5": 0.10,
            "ndcg_at_5": 0.10,
            "mrr": 0.10,
            "correctness": 0.07,
        },
        "no_rerank": {
            "precision_at_5": -0.10,
            "ndcg_at_5": -0.10,
            "mrr": -0.08,
            "correctness": -0.05,
        },
    },
    "03_hybrid_vs_vector": {
        "vector_only": {"recall_at_5": -0.05},
        "bm25_only": {"recall_at_5": -0.10, "precision_at_5": -0.03},
        "hybrid_rrf": {"recall_at_5": 0.10, "precision_at_5": 0.05, "correctness": 0.04},
    },
    "04_top_k_sweep": {
        "top_k_3": {"recall_at_5": -0.06, "correctness": -0.04},
        "top_k_5": {"recall_at_5": 0.02, "correctness": 0.03},
        "top_k_10": {"recall_at_5": 0.05, "correctness": 0.035, "faithfulness": -0.03},
    },
    "05_llm_provider_comparison": {
        "anthropic_haiku": {"correctness": 0.02, "faithfulness": 0.03},
        "openai_4o_mini": {},
    },
    "06_prompt_variations": {
        "baseline": {},
        "cite_sources": {"faithfulness": 0.12, "correctness": 0.03},
        "reason_first": {"faithfulness": 0.02, "correctness": 0.07},
    },
}

# Difficulty shifts the mean of the judge/quality metrics.
DIFFICULTY_SHIFT = {"easy": 0.08, "medium": 0.0, "hard": -0.22}

FAILURE_RATIONALES = {
    "faithfulness": [
        "Hallucination: asserted a dosage not present in any retrieved chunk.",
        "Confidence inflation: stated a contraindication the context only hedges.",
        "Unsupported synthesis across two chunks that do not connect.",
    ],
    "relevance": [
        "Partial answer: addressed mechanism but not the asked-for first-line agent.",
        "Off-topic: described epidemiology instead of management.",
    ],
    "correctness": [
        "Retrieval miss: the relevant guideline section was never retrieved.",
        "Names a second-line therapy where the reference gives first-line.",
        "Omits the threshold value the reference answer specifies.",
    ],
}
GOOD_RATIONALE = {
    "faithfulness": "Every claim traces to a retrieved chunk.",
    "relevance": "Directly answers the question asked.",
    "correctness": "Substantively matches the reference answer.",
}


def all_chunk_ids() -> list[str]:
    return [f"{doc}::p{1 + i // 4}::c{i}" for doc in DOCS for i in range(CHUNKS_PER_DOC)]


def build_eval_set(rng: random.Random) -> list[EvalQuestion]:
    """Build 75 synthetic questions, 15 of them hand-crafted-style hard cases."""
    questions: list[EvalQuestion] = []
    # 30 easy, 30 medium, 15 hard (the hard ones are the "hard cases").
    plan = ["easy"] * 30 + ["medium"] * 30 + ["hard"] * 15
    for index, difficulty in enumerate(plan, start=1):
        doc = DOCS[index % len(DOCS)]
        base = (index * 7) % CHUNKS_PER_DOC
        relevant = [
            f"{doc}::p{1 + (base + j) // 4}::c{(base + j) % CHUNKS_PER_DOC}"
            for j in range(rng.randint(1, 3))
        ]
        is_hard = difficulty == "hard"
        questions.append(
            EvalQuestion(
                question_id=f"q{index:03d}",
                question=f"[synthetic] Clinical question {index} about {doc.replace('_', ' ')}?",
                ground_truth_answer=f"[synthetic] Reference answer {index}.",
                relevant_chunk_ids=relevant,
                source_documents=[f"{doc}.pdf"],
                difficulty=difficulty,  # type: ignore[arg-type]
                question_type=QTYPES[index % len(QTYPES)],  # type: ignore[arg-type]
                is_hard_case=is_hard,
                rationale="Requires domain reasoning beyond surface retrieval." if is_hard else "",
            )
        )
    return questions


def write_eval_set(questions: list[EvalQuestion]) -> None:
    from dataclasses import asdict

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "questions.jsonl").write_text(
        "\n".join(json.dumps(asdict(q)) for q in questions) + "\n", encoding="utf-8"
    )
    (EVAL_DIR / "hard_cases.jsonl").write_text(
        "\n".join(json.dumps(asdict(q)) for q in questions if q.is_hard_case) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "synthetic": True,
        "documents": dict.fromkeys(DOCS, CHUNKS_PER_DOC),
        "total_chunks": len(all_chunk_ids()),
        "note": "Synthetic corpus manifest — placeholder chunk ids, not a real corpus.",
    }
    (EVAL_DIR / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _variant_means(experiment_id: str, variant_name: str) -> dict[str, float]:
    deltas = PROFILES.get(experiment_id, {}).get(variant_name, {})
    return {metric: BASE[metric] + deltas.get(metric, 0.0) for metric in METRICS}


def _rationale(rng: random.Random, metric: str, score: float) -> str:
    if score >= 0.6 or metric not in FAILURE_RATIONALES:
        return GOOD_RATIONALE.get(metric, "")
    return rng.choice(FAILURE_RATIONALES[metric])


def _score_question(
    rng: random.Random, question: EvalQuestion, means: dict[str, float]
) -> QuestionScore:
    shift = DIFFICULTY_SHIFT[question.difficulty]
    values = {m: _clamp01(rng.gauss(means[m] + shift, 0.12)) for m in METRICS}
    return QuestionScore(
        question_id=question.question_id,
        precision_at_5=round(values["precision_at_5"], 4),
        recall_at_5=round(values["recall_at_5"], 4),
        mrr=round(values["mrr"], 4),
        ndcg_at_5=round(values["ndcg_at_5"], 4),
        faithfulness=round(values["faithfulness"], 4),
        relevance=round(values["relevance"], 4),
        correctness=round(values["correctness"], 4),
        faithfulness_rationale=_rationale(rng, "faithfulness", values["faithfulness"]),
        relevance_rationale=_rationale(rng, "relevance", values["relevance"]),
        correctness_rationale=_rationale(rng, "correctness", values["correctness"]),
    )


def _prediction(
    rng: random.Random, question: EvalQuestion, score: QuestionScore, model: str, rerank_k: int
) -> Prediction:
    n_relevant_found = round(score.recall_at_5 * len(question.relevant_chunk_ids))
    retrieved = list(question.relevant_chunk_ids[:n_relevant_found])
    distractors = [
        f"{DOCS[0]}::p{rng.randint(1, 10)}::c{rng.randint(0, 39)}" for _ in range(rerank_k)
    ]
    retrieved = (retrieved + distractors)[:rerank_k]
    input_tokens = 600 + rerank_k * 180 + rng.randint(-40, 40)
    output_tokens = 110 + rng.randint(-30, 60)
    base_latency = 700 if model.startswith("gpt") else 880
    from eval_harness.runner.scoring import estimate_cost

    return Prediction(
        question_id=question.question_id,
        retrieved_chunk_ids=retrieved,
        retrieved_chunks_text=[f"[synthetic] chunk text for {cid}" for cid in retrieved],
        answer=f"[synthetic] Answer to {question.question_id}.",
        citations=retrieved[:n_relevant_found] or retrieved[:1],
        latency_ms=base_latency + rng.randint(-120, 240),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(estimate_cost(model, input_tokens, output_tokens), 6),
    )


def generate_experiment(rng: random.Random, questions: list[EvalQuestion], path: Path) -> None:
    config = load_experiment(path)
    shared = config.shared_config
    for variant in config.variants:
        merged = {**shared, **variant.overrides}
        rerank_k = int(merged.get("rerank_top_k", 5))  # type: ignore[arg-type]
        model = (
            str(merged.get("openai_model", "gpt-4o-mini"))
            if merged.get("llm_provider") == "openai"
            else "claude-haiku-4-5"
        )
        means = _variant_means(config.experiment_id, variant.name)

        scores = [_score_question(rng, q, means) for q in questions]
        predictions = [
            _prediction(rng, q, s, model, rerank_k) for q, s in zip(questions, scores, strict=True)
        ]
        results = ExperimentResults(
            experiment_id=config.experiment_id,
            experiment_name=config.name,
            config={"shared_config": shared, "variant": variant.model_dump(mode="json")},
            run_timestamp=TIMESTAMP,
            guideline_gpt_version=SYNTHETIC_VERSION,
            question_scores=scores,
            aggregate_metrics=aggregate_scores(questions, scores),
            total_cost_usd=round(sum(p.cost_usd for p in predictions), 6),
            total_latency_ms=sum(p.latency_ms for p in predictions),
        )
        persist_variant_results(RESULTS_DIR, variant.name, results, predictions)
        print(f"  {config.experiment_id}/{variant.name}: {len(scores)} questions scored")


def write_marker() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "SYNTHETIC_DATA.md").write_text(
        "# ⚠️ Synthetic results\n\n"
        "The results in this directory are **synthetic placeholders** generated by "
        "`scripts/generate_synthetic_data.py`, not real measurements. They exist so the "
        "dashboard and notebook can be built and browsed before a real corpus and API runs "
        "exist. Every manifest records `guideline_gpt_version: SYNTHETIC-DEMO`.\n\n"
        "Re-run with real experiments via `eval-harness run` once a corpus is ingested.\n",
        encoding="utf-8",
    )


def main() -> None:
    rng = random.Random(SEED)
    print("Generating synthetic eval set...")
    questions = build_eval_set(rng)
    write_eval_set(questions)
    print(
        f"  wrote {len(questions)} questions ({sum(q.is_hard_case for q in questions)} hard cases)"
    )

    print("Generating synthetic results...")
    for path in sorted(EXPERIMENTS_DIR.glob("*.yaml")):
        generate_experiment(rng, questions, path)

    write_marker()
    print("Done.")


if __name__ == "__main__":
    main()
