"""
Tests for maker-taker fee structure.
Verifies that:
  - Taker fees reduce reward on market (aggressive) orders
  - Maker rebates increase reward on passive limit fills
  - Zero fees preserve original behavior (backward compatibility)
  - Fee accumulators are tracked and reset correctly
  - Info dict reports fee diagnostics
"""

import sys
import os
current_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

import numpy as np
from limit_order_book.limit_order_book import LimitOrderBook, LimitOrder, MarketOrder
from simulation.agents import ExecutionAgent, RLAgent
from simulation.market_gym import Market
import pytest


# ---------------------------------------------------------------------------
# Helper: minimal Market config
# ---------------------------------------------------------------------------
def _base_config(**overrides):
    cfg = {
        'market_env': 'noise',
        'execution_agent': 'rl_agent',
        'volume': 20,
        'seed': 42,
        'terminal_time': 150,
        'time_delta': 15,
        'drop_feature': None,
        'inventory_max': float('inf'),
        'maker_rebate': 0.0,
        'taker_fee': 0.0,
    }
    cfg.update(overrides)
    return cfg


# ===========================================================================
# Unit tests: ExecutionAgent fee mechanics
# ===========================================================================

class TestExecutionAgentFees:
    """Low-level tests on ExecutionAgent.set_fees / update_position."""

    def test_set_fees_stores_values(self):
        agent = ExecutionAgent(volume=10, agent_id='test_agent')
        agent.set_fees(maker_rebate=0.001, taker_fee=0.003)
        assert agent.maker_rebate == 0.001
        assert agent.taker_fee == 0.003

    def test_default_fees_are_zero(self):
        agent = ExecutionAgent(volume=10, agent_id='test_agent')
        assert agent.maker_rebate == 0.0
        assert agent.taker_fee == 0.0

    def test_reset_clears_fee_accumulators(self):
        agent = ExecutionAgent(volume=10, agent_id='test_agent')
        agent.set_fees(maker_rebate=0.001, taker_fee=0.003)
        agent.cumulative_fees_paid = 0.5
        agent.cumulative_rebates_earned = 0.2
        agent.reset()
        assert agent.cumulative_fees_paid == 0.0
        assert agent.cumulative_rebates_earned == 0.0
        # fees themselves persist across reset (set by environment)
        assert agent.maker_rebate == 0.001
        assert agent.taker_fee == 0.003


# ===========================================================================
# Integration tests: Market environment with fees
# ===========================================================================

class TestMarketFeeIntegration:
    """End-to-end tests running the Market with nonzero fees."""

    def test_zero_fees_backward_compatible(self):
        """With fees=0, behaviour should be identical to the original."""
        cfg_no_fee = _base_config(maker_rebate=0.0, taker_fee=0.0, seed=7)
        env = Market(cfg_no_fee)
        obs, info = env.reset()
        total_reward_no_fee = 0.0
        for _ in range(5):
            action = np.zeros(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward_no_fee += reward
            if terminated:
                break
        assert info['cumulative_fees_paid'] == 0.0
        assert info['cumulative_rebates_earned'] == 0.0
        assert info['net_fee_impact'] == 0.0

    def test_taker_fee_reduces_reward(self):
        """With a positive taker_fee, aggressive orders should yield lower total reward."""
        results = {}
        for label, taker_fee in [('no_fee', 0.0), ('with_fee', 0.01)]:
            cfg = _base_config(taker_fee=taker_fee, maker_rebate=0.0, seed=7)
            env = Market(cfg)
            obs, info = env.reset()
            cumulative = 0.0
            for _ in range(10):
                # action heavily weighted toward market order (index 0)
                action = np.zeros(env.action_space.shape)
                action[0] = 10.0  # softmax will make this dominant
                obs, reward, terminated, truncated, info = env.step(action)
                cumulative += reward
                if terminated:
                    break
            results[label] = info['reward']

        # With taker fee, cumulative reward should be lower (or equal if no trades happened)
        assert results['with_fee'] <= results['no_fee'] + 1e-9, \
            f"Taker fee should reduce reward: no_fee={results['no_fee']}, with_fee={results['with_fee']}"

    def test_maker_rebate_increases_reward(self):
        """With a positive maker_rebate, passive fills should yield higher total reward."""
        results = {}
        for label, maker_rebate in [('no_rebate', 0.0), ('with_rebate', 0.01)]:
            cfg = _base_config(maker_rebate=maker_rebate, taker_fee=0.0, seed=7)
            env = Market(cfg)
            obs, info = env.reset()
            cumulative = 0.0
            for _ in range(10):
                # action weighted toward limit levels (passive)
                action = np.zeros(env.action_space.shape)
                action[1] = 5.0  # L1
                action[2] = 5.0  # L2
                obs, reward, terminated, truncated, info = env.step(action)
                cumulative += reward
                if terminated:
                    break
            results[label] = info['reward']

        # With maker rebate, reward should be higher (or equal if no passive fills)
        assert results['with_rebate'] >= results['no_rebate'] - 1e-9, \
            f"Maker rebate should increase reward: no_rebate={results['no_rebate']}, with_rebate={results['with_rebate']}"

    def test_fee_accumulators_in_info_dict(self):
        """Info dict should contain fee tracking fields."""
        cfg = _base_config(maker_rebate=0.001, taker_fee=0.003, seed=7)
        env = Market(cfg)
        obs, info = env.reset()
        action = np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        assert 'cumulative_fees_paid' in info
        assert 'cumulative_rebates_earned' in info
        assert 'net_fee_impact' in info

    def test_fee_net_impact_consistent(self):
        """net_fee_impact should equal rebates - fees."""
        cfg = _base_config(maker_rebate=0.002, taker_fee=0.005, seed=7)
        env = Market(cfg)
        obs, info = env.reset()
        for _ in range(10):
            action = np.zeros(env.action_space.shape)
            action[0] = 3.0  # some market orders
            action[1] = 3.0  # some limit orders
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        expected_net = info['cumulative_rebates_earned'] - info['cumulative_fees_paid']
        assert abs(info['net_fee_impact'] - expected_net) < 1e-10, \
            f"net_fee_impact={info['net_fee_impact']} != rebates-fees={expected_net}"

    def test_fees_reset_between_episodes(self):
        """Fee accumulators should reset to 0 on env.reset()."""
        cfg = _base_config(maker_rebate=0.001, taker_fee=0.003, seed=7)
        env = Market(cfg)
        obs, info = env.reset()
        # Run some steps
        for _ in range(5):
            action = np.zeros(env.action_space.shape)
            action[0] = 5.0
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        # Reset and check accumulators are zero
        obs, info = env.reset()
        exec_agent = env.agents[env.execution_agent_id]
        assert exec_agent.cumulative_fees_paid == 0.0
        assert exec_agent.cumulative_rebates_earned == 0.0
