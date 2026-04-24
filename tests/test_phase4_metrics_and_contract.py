import numpy as np
import torch
import os
import sys

# Align import resolution with existing test suite conventions.
test_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(test_dir)
sys.path.insert(0, repo_dir)
sys.path.insert(0, os.path.dirname(repo_dir))

from rl_files.actor_critic import (
    _safe_simplex_projection,
    _invalid_simplex_row_count,
    _compute_reward_metrics,
)


def test_safe_simplex_projection_handles_invalid_rows():
    x = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [float("nan"), 0.0, 0.0],
            [float("inf"), -1.0, 0.0],
            [-0.2, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )

    y = _safe_simplex_projection(x)

    assert torch.isfinite(y).all()
    assert (y >= 0).all()
    sums = y.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_invalid_simplex_row_count_detects_contract_breaks():
    good = torch.tensor([[0.6, 0.4], [0.0, 1.0]], dtype=torch.float32)
    bad = torch.tensor([[0.6, 0.6], [float("nan"), 0.0]], dtype=torch.float32)

    assert _invalid_simplex_row_count(good) == 0
    assert _invalid_simplex_row_count(bad) == 2


def test_compute_reward_metrics_outputs_expected_keys_and_values():
    rewards = np.array([-300.0, -100.0, 0.0, 100.0, 200.0], dtype=np.float64)

    metrics = _compute_reward_metrics(rewards)

    expected = {
        "num_episodes",
        "mean",
        "std",
        "median",
        "p05",
        "p95",
        "cvar05",
        "min",
        "max",
        "outlier_rate_lt_minus200",
    }
    assert expected.issubset(metrics.keys())
    assert metrics["num_episodes"] == 5
    assert metrics["min"] == -300.0
    assert metrics["max"] == 200.0
    assert metrics["outlier_rate_lt_minus200"] == 0.2


def test_compute_reward_metrics_handles_empty_array():
    metrics = _compute_reward_metrics(np.array([], dtype=np.float64))

    assert metrics["num_episodes"] == 0
    assert np.isnan(metrics["mean"])
    assert np.isnan(metrics["outlier_rate_lt_minus200"])
