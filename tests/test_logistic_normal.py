import numpy as np
import torch
from rl_files.actor_critic import AgentLogisticNormal


class MockEnv:
    def __init__(self, obs_dim=10, action_dim=4):
        class Space:
            def __init__(self, shape):
                self.shape = shape
        self.single_observation_space = Space((obs_dim,))
        self.single_action_space = Space((action_dim,))


def test_set_get_variance_and_sampling():
    env = MockEnv(obs_dim=8, action_dim=4)
    agent = AgentLogisticNormal(env, variance_scaling=True)

    # default variance is set
    v0 = agent.get_variance()
    assert isinstance(v0, float)

    # set new variance
    agent.set_variance(0.05)
    assert abs(agent.get_variance() - 0.05) < 1e-8

    # invalid variance raises
    try:
        agent.set_variance(float('nan'))
        raised = False
    except ValueError:
        raised = True
    assert raised

    # create dummy observation batch
    obs = np.zeros(env.single_observation_space.shape, dtype=np.float32)
    obs_batch = torch.as_tensor(np.stack([obs] * 3, axis=0), dtype=torch.float32)

    # sample actions
    actions, logprob, entropy, value = agent.get_action_and_value(obs_batch)
    # actions should be in simplex and have last component
    assert actions.shape[1] == env.single_action_space.shape[0]
    sums = actions.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()
    assert value.shape[0] == 3


def test_deterministic_action_sums_to_one():
    env = MockEnv(obs_dim=8, action_dim=5)
    agent = AgentLogisticNormal(env, variance_scaling=True)
    obs = np.zeros(env.single_observation_space.shape, dtype=np.float32)
    obs_batch = torch.as_tensor(np.stack([obs] * 2, axis=0), dtype=torch.float32)
    action = agent.deterministic_action(obs_batch)
    sums = action.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
