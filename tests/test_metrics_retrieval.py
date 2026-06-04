"""Tests for deterministic retrieval metrics, checked against values computed
by hand from the formulas in the module docstring."""

from __future__ import annotations

import math

import pytest

from eval_harness.metrics.retrieval import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# A reusable ranking: relevant chunks sit at ranks 1 and 3.
RETRIEVED = ["a", "b", "c", "d"]
RELEVANT = ["a", "c"]


class TestPrecisionAtK:
    def test_partial_top_k(self) -> None:
        # top-2 = [a, b]; 1 relevant out of k=2.
        assert precision_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(0.5)

    def test_full_window(self) -> None:
        # top-4 = [a, b, c, d]; 2 relevant out of k=4.
        assert precision_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(0.5)

    def test_divides_by_k_not_retrieved_count(self) -> None:
        # Only one chunk retrieved but k=5 -> 1 hit / 5.
        assert precision_at_k(["a"], RELEVANT, 5) == pytest.approx(0.2)

    def test_no_relevant_is_zero(self) -> None:
        assert precision_at_k(RETRIEVED, [], 3) == 0.0


class TestRecallAtK:
    def test_partial(self) -> None:
        # top-1 = [a]; 1 of 2 relevant found.
        assert recall_at_k(RETRIEVED, RELEVANT, 1) == pytest.approx(0.5)

    def test_full(self) -> None:
        # top-4 captures both relevant chunks.
        assert recall_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(1.0)

    def test_no_relevant_is_zero(self) -> None:
        assert recall_at_k(RETRIEVED, [], 3) == 0.0


class TestReciprocalRank:
    def test_first_hit_rank_one(self) -> None:
        assert reciprocal_rank(["a", "b", "c"], ["a"]) == pytest.approx(1.0)

    def test_first_hit_rank_two(self) -> None:
        assert reciprocal_rank(["b", "a", "c"], ["a"]) == pytest.approx(0.5)

    def test_no_hit_is_zero(self) -> None:
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_no_relevant_is_zero(self) -> None:
        assert reciprocal_rank(["a", "b"], []) == 0.0


class TestMeanReciprocalRank:
    def test_mean_over_queries(self) -> None:
        retrieved = [["a", "b"], ["b", "a", "c"]]
        relevant = [["a"], ["a"]]
        # RRs are 1.0 and 0.5 -> mean 0.75.
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.75)

    def test_empty_is_zero(self) -> None:
        assert mean_reciprocal_rank([], []) == 0.0

    def test_misaligned_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            mean_reciprocal_rank([["a"]], [["a"], ["b"]])


class TestNdcgAtK:
    def test_hand_computed(self) -> None:
        # DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG (2 relevant) = 1/log2(2) + 1/log2(3) = 1.0 + 0.63093 = 1.63093
        expected = 1.5 / (1.0 + 1.0 / math.log2(3))
        assert ndcg_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(expected)

    def test_perfect_ranking_is_one(self) -> None:
        # Both relevant chunks at the very top.
        assert ndcg_at_k(["a", "c", "b", "d"], RELEVANT, 4) == pytest.approx(1.0)

    def test_single_relevant_at_rank_two(self) -> None:
        # DCG = 1/log2(3); IDCG = 1/log2(2) = 1.0.
        assert ndcg_at_k(["b", "a", "c"], ["a"], 3) == pytest.approx(1.0 / math.log2(3))

    def test_no_relevant_is_zero(self) -> None:
        assert ndcg_at_k(RETRIEVED, [], 3) == 0.0


class TestDuplicatesAndValidation:
    def test_duplicates_collapse(self) -> None:
        # Repeated 'a' must not earn double credit.
        assert precision_at_k(["a", "a", "b"], ["a", "b"], 2) == pytest.approx(1.0)
        assert recall_at_k(["a", "a", "b"], ["a", "b"], 2) == pytest.approx(1.0)

    @pytest.mark.parametrize("bad_k", [0, -1, -5])
    def test_non_positive_k_raises(self, bad_k: int) -> None:
        with pytest.raises(ValueError):
            precision_at_k(RETRIEVED, RELEVANT, bad_k)
        with pytest.raises(ValueError):
            recall_at_k(RETRIEVED, RELEVANT, bad_k)
        with pytest.raises(ValueError):
            ndcg_at_k(RETRIEVED, RELEVANT, bad_k)
