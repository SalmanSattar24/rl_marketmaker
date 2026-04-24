"""
Unit tests to validate the temporal modeling architectures (LSTM and LiT).
Verifies that the networks can process batched historical observations and compute policy/value outputs without crashing.
"""
import sys
import os
import torch
import pytest
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from rl_files.actor_critic import BilateralAgentLSTM, BilateralAgentLiT, BilateralAgentLSTMLob
import gymnasium as gym
from simulation.history_wrapper import HistoryWrapper

def test_lstm_agent_forward_pass():
    """Verify BilateralAgentLSTM can process batched history sequences."""
    batch_size = 4
    seq_len = 8
    obs_dim = 60
    action_levels = 5
    action_space_length = action_levels + 2  # market, levels..., inactive

    # Create dummy observation batch: (batch_size, seq_len, obs_dim)
    dummy_obs = torch.randn(batch_size, seq_len, obs_dim)

    # Initialize agent
    # Assuming BilateralAgentLSTM signature: __init__(self, observation_space, action_book_levels, ...)
    # Let's mock a simple observation space
    class MockSpace:
        def __init__(self, shape):
            self.shape = shape
            
    class MockEnvs:
        def __init__(self, shape, action_length):
            self.single_observation_space = MockSpace(shape)
            self.single_action_space = MockSpace((action_length,)) # Represents ONE side of the bilateral action space
    
    envs = MockEnvs((seq_len, obs_dim), action_space_length)
    
    # Initialize the agent
    agent = BilateralAgentLSTM(envs)

    # Test get_value
    value = agent.get_value(dummy_obs)
    assert value.shape == (batch_size, 1), f"Value shape mismatch: {value.shape}"

    # Test get_action_and_value
    actions, logprob, entropy, value = agent.get_action_and_value(dummy_obs)
    
    # Bilateral returns a tuple of actions (bid, ask) or a concatenated action?
    # Usually it returns a flattened concatenated action or tuple
    if isinstance(actions, tuple):
        bid_act, ask_act = actions
        assert bid_act.shape == (batch_size, action_space_length)
        assert ask_act.shape == (batch_size, action_space_length)
    else:
        # Flat action shape
        assert actions.shape == (batch_size, 2 * action_space_length)
        
    assert logprob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size, 1)

def test_history_wrapper_integration():
    """Verify HistoryWrapper correctly shapes the environment observations."""
    from simulation.market_gym import Market
    config = {
        'market_env': 'noise',
        'execution_agent': 'rl_agent',
        'volume': 100,
        'seed': 42,
        'terminal_time': 100,
        'time_delta': 10,
        'drop_feature': None,
        'inventory_max': 20,
        'penalty_weight': 1.0,
        'use_ofi': False
    }
    env = Market(config)
    
    seq_len = 5
    wrapped_env = HistoryWrapper(env, history_len=seq_len)
    
    obs, info = wrapped_env.reset()
    # Observation should be (seq_len, original_obs_dim)
    assert obs.shape == (seq_len, env.observation_space.shape[0])
    
    # Step should update the last row and shift history
    action = np.zeros(14)
    action[:] = 1.0 / 7.0
    
    next_obs, _, _, _, _ = wrapped_env.step(action)
    assert next_obs.shape == (seq_len, env.observation_space.shape[0])
    # The last row of previous obs should now be the second-to-last row of next obs
    np.testing.assert_array_equal(obs[-1], next_obs[-2])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
