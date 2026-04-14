"""Tests for simulation.history_wrapper.HistoryWrapper."""

import os
import sys

import numpy as np
import pytest
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.history_wrapper import HistoryWrapper
from simulation.market_gym import Market


def _make_cfg():
    return {
        "market_env": "flow",
        "execution_agent": "rl_agent",
        "volume": 10,
        "terminal_time": 150,
        "time_delta": 15,
        "drop_feature": "drift",
        "inventory_max": 10,
        "penalty_weight": 0.0,
        "maker_rebate": 0.0002,
        "taker_fee": 0.0003,
        "seed": 42,
    }


def test_reset_fills_buffer():
    env = HistoryWrapper(Market(_make_cfg()), history_len=8)
    obs, _ = env.reset(seed=42)

    assert obs.shape == (8, env.obs_dim)
    for i in range(1, 8):
        assert np.allclose(obs[0], obs[i]), (
            f"Row {i} differs from row 0 right after reset"
        )


def test_step_rolls_buffer():
    env = HistoryWrapper(Market(_make_cfg()), history_len=8)
    obs_reset, _ = env.reset(seed=42)
    initial_row = obs_reset[0].copy()

    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    action[-1] = 1.0

    obs1, _, terminated, truncated, _ = env.step(action)
    assert obs1.shape == (8, env.obs_dim)
    assert np.allclose(obs1[0], initial_row), (
        "After 1 step, buffer[0] should still be the initial observation"
    )

    if terminated or truncated:
        return

    different_rows = sum(
        not np.allclose(obs1[i], initial_row) for i in range(8)
    )
    assert different_rows >= 1, (
        f"At least the newest row should differ from the initial observation, "
        f"got {different_rows} rows differing"
    )


def test_observation_space_shape():
    base_env = Market(_make_cfg())
    wrapped = HistoryWrapper(base_env, history_len=8)

    assert wrapped.observation_space.shape == (8, base_env.observation_space.shape[0])
    assert isinstance(wrapped.observation_space, gym.spaces.Box)


def test_full_episode_no_crash():
    env = HistoryWrapper(Market(_make_cfg()), history_len=8)
    obs, _ = env.reset(seed=42)

    for step_i in range(50):
        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        action[-1] = 1.0
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (8, env.obs_dim)
        assert np.isfinite(obs).all(), f"NaN or inf in obs at step {step_i}"
        assert np.isfinite(reward), f"NaN or inf reward at step {step_i}"

        if terminated or truncated:
            break


def test_different_history_lengths():
    for K in [1, 4, 8, 16]:
        env = HistoryWrapper(Market(_make_cfg()), history_len=K)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (K, env.obs_dim), f"Failed for K={K}"
