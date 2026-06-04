"""Evaluation metrics for the eval harness.

Two families live here: deterministic retrieval metrics (:mod:`retrieval`),
which need no LLM, and LLM-as-judge metrics (faithfulness, relevance,
correctness) which land in their own modules.
"""
