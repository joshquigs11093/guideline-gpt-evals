"""Spec §11: every experiment config has committed results to back it."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_harness.runner.experiment import load_experiment

EXPERIMENT_PATHS = sorted(Path("experiments").glob("*.yaml"))


def test_experiments_exist() -> None:
    assert EXPERIMENT_PATHS, "no experiment configs found"


@pytest.mark.parametrize("path", EXPERIMENT_PATHS, ids=lambda p: p.stem)
def test_experiment_has_results(path: Path) -> None:
    config = load_experiment(path)
    results_dir = Path("results") / config.experiment_id
    assert results_dir.is_dir(), f"no results directory for {config.experiment_id}"
    variant_dirs = [d for d in results_dir.iterdir() if (d / "metrics.json").exists()]
    assert variant_dirs, f"no variant results under {results_dir}"
