"""LLM-as-judge implementations for the qualitative metrics.

Each judge implements the :class:`base.Judge` protocol — given a question and a
prediction, it returns a score in ``[0, 1]`` plus a rationale. The judge model
is deliberately *stronger* than the system under test (spec §5.3 / ADR-002), and
every judge prompt ships with committed few-shot calibration examples.
"""
