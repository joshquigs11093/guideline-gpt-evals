"""Experiment runner: turn a declarative experiment into scored results.

An experiment (:mod:`experiment`) defines variants; the variant builder
(:mod:`variant`) realises each as a configured pipeline; the runner
(:mod:`runner`) runs the eval set through every variant, scores each prediction,
and persists comparable results.
"""
