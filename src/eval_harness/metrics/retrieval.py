"""Deterministic retrieval metrics (no LLM).

All functions take a *ranked* list of retrieved chunk ids (best first) and the
set of ground-truth relevant chunk ids, and return a score in ``[0, 1]``.
Relevance is treated as binary: a chunk either is or is not in the relevant set.

Conventions for degenerate inputs, chosen to keep aggregates well-defined:

- ``k`` must be a positive integer; otherwise :class:`ValueError` is raised.
- An empty relevant set yields ``0.0`` for recall and nDCG (there is nothing to
  retrieve, so credit cannot be earned) and ``0.0`` for precision and MRR.
- Duplicate ids in ``retrieved`` are collapsed to their first occurrence so a
  repeated hit cannot inflate a score.

Formulas (binary relevance, 1-indexed ranks):

- precision@k = |relevant ∩ retrieved[:k]| / k
- recall@k    = |relevant ∩ retrieved[:k]| / |relevant|
- RR          = 1 / rank of the first relevant hit (0 if none)
- DCG@k       = Σ_{i=1..k} rel_i / log2(i + 1)
- nDCG@k      = DCG@k / IDCG@k, where IDCG@k is DCG@k of the ideal ranking
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _require_positive_k(k: int) -> None:
    """Raise :class:`ValueError` unless ``k`` is a positive integer."""
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")


def _dedupe(retrieved: Iterable[str]) -> list[str]:
    """Return ``retrieved`` with duplicates removed, preserving rank order."""
    seen: set[str] = set()
    unique: list[str] = []
    for chunk_id in retrieved:
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(chunk_id)
    return unique


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-``k`` retrieved chunks that are relevant.

    Args:
        retrieved: Ranked chunk ids, best first.
        relevant: Ground-truth relevant chunk ids.
        k: Cutoff rank (must be positive). Always divides by ``k``, even when
            fewer than ``k`` chunks were retrieved.

    Returns:
        precision@k in ``[0, 1]``.
    """
    _require_positive_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = _dedupe(retrieved)[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_set)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant chunks found within the top-``k`` retrieved.

    Args:
        retrieved: Ranked chunk ids, best first.
        relevant: Ground-truth relevant chunk ids.
        k: Cutoff rank (must be positive).

    Returns:
        recall@k in ``[0, 1]``; ``0.0`` when there are no relevant chunks.
    """
    _require_positive_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = _dedupe(retrieved)[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal of the rank of the first relevant chunk (per-query RR).

    Args:
        retrieved: Ranked chunk ids, best first.
        relevant: Ground-truth relevant chunk ids.

    Returns:
        ``1 / rank`` of the first relevant hit (1-indexed), or ``0.0`` if no
        relevant chunk appears.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for rank, chunk_id in enumerate(_dedupe(retrieved), start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    retrieved_per_query: Iterable[Sequence[str]],
    relevant_per_query: Iterable[Iterable[str]],
) -> float:
    """Mean of the per-query reciprocal ranks (MRR).

    Args:
        retrieved_per_query: One ranked list of chunk ids per query.
        relevant_per_query: One relevant set per query, aligned with
            ``retrieved_per_query``.

    Returns:
        The mean reciprocal rank across queries, or ``0.0`` if there are none.
    """
    rrs = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(retrieved_per_query, relevant_per_query, strict=True)
    ]
    if not rrs:
        return 0.0
    return sum(rrs) / len(rrs)


def _dcg_at_k(retrieved: Sequence[str], relevant_set: set[str], k: int) -> float:
    """Discounted cumulative gain at ``k`` under binary relevance."""
    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalized discounted cumulative gain at ``k`` (binary relevance).

    Rewards placing relevant chunks higher in the ranking, normalized against
    the best achievable ordering so the score lands in ``[0, 1]``.

    Args:
        retrieved: Ranked chunk ids, best first.
        relevant: Ground-truth relevant chunk ids.
        k: Cutoff rank (must be positive).

    Returns:
        nDCG@k in ``[0, 1]``; ``0.0`` when there are no relevant chunks.
    """
    _require_positive_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = _dedupe(retrieved)
    dcg = _dcg_at_k(top_k, relevant_set, k)
    # Ideal DCG: every relevant chunk ranked first, capped at k.
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
