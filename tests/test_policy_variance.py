import sys, os
import numpy as np
import torch
from types import SimpleNamespace

# Ensure project root is on sys.path so `rl_files` can be imported
test_dir = os.path.dirname(os.path.abspath(__file__))
# repo root (rl_marketmaker)
repo_root = os.path.abspath(os.path.join(test_dir, os.pardir))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.dirname(repo_root))

from rl_files.actor_critic import AgentLogisticNormal, BilateralAgentLogisticNormal


class DummySpace:
    def __init__(self, shape):
        self.shape = tuple(shape)


class DummyEnv:
    def __init__(self, obs_shape=(10,), action_shape=(7,)):
        self.single_observation_space = DummySpace(obs_shape)
        self.single_action_space = DummySpace(action_shape)
        # some agents access .observation_space as well; provide fallback
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space


def test_agent_logisticnormal_variance_api_and_sampling():
    env = DummyEnv(obs_shape=(12,), action_shape=(7,))
    agent = AgentLogisticNormal(env, variance_scaling=True)
    # default variance should exist
    assert isinstance(agent.get_variance(), float)
    orig = agent.get_variance()
    agent.set_variance(0.25)
    assert abs(agent.get_variance() - 0.25) < 1e-6
    # small values get clamped to >= 1e-4
    agent.set_variance(0.0)
    assert agent.get_variance() >= 1e-4

    # sampling shapes and finiteness
    obs = np.zeros((1, env.single_observation_space.shape[0]), dtype=np.float32)
    obs_t = torch.as_tensor(obs)
    action, logprob, entropy, val = agent.get_action_and_value(obs_t)
    assert action.shape[0] == 1
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()


def test_bilateral_agent_variance_api_and_sampling():
    env = DummyEnv(obs_shape=(20,), action_shape=(8,))
    agent = BilateralAgentLogisticNormal(env, variance_scaling=True)
    # default variance should exist
    assert isinstance(agent.get_variance(), float)
    agent.set_variance(0.5)
    assert abs(agent.get_variance() - 0.5) < 1e-6

    obs = np.zeros((2, env.single_observation_space.shape[0]), dtype=np.float32)
    obs_t = torch.as_tensor(obs)
    actions, logprob, entropy, val = agent.get_action_and_value(obs_t)
    assert isinstance(actions, tuple) and len(actions) == 2
    bid_action, ask_action = actions
    assert bid_action.shape[0] == 2 and ask_action.shape[0] == 2
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()
