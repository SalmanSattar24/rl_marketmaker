import os
import sys
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from simulation.market_gym import Market
from limit_order_book.limit_order_book import LimitOrder


def _base_config(**overrides):
    cfg = {
        'market_env': 'noise',
        'execution_agent': 'rl_agent',
        'volume': 20,
        'seed': 7,
        'terminal_time': 30,
        'time_delta': 15,
        'drop_feature': None,
        'inventory_max': 50,
    }
    cfg.update(overrides)
    return cfg


def test_terminal_closeout_flattens_positive_inventory():
    m = Market(_base_config())
    m.reset(seed=7)

    exec_id = m.execution_agent_id

    # Inject synthetic net inventory and perform deterministic closeout.
    m.lob.agent_net_inventory[exec_id] = 6
    m.agent_inventory = 6

    reward = m._closeout_terminal_inventory(time=m.agents[exec_id].terminal_time)

    assert isinstance(reward, float)
    assert m.agent_inventory == 0
    assert m.lob.agent_net_inventory[exec_id] == 0


def test_terminal_closeout_cancels_resting_orders():
    m = Market(_base_config())
    m.reset(seed=9)

    exec_id = m.execution_agent_id
    best_bid = m.lob.get_best_price('bid')

    # Add an explicit resting order for the execution agent.
    lo = LimitOrder(agent_id=exec_id, side='ask', price=best_bid + 2, volume=2, time=0)
    m.lob.process_order(lo)

    assert len(m.lob.order_map_by_agent[exec_id]) > 0

    # No inventory, but closeout should still clear active orders.
    m.lob.agent_net_inventory[exec_id] = 0
    m.agent_inventory = 0

    _ = m._closeout_terminal_inventory(time=m.agents[exec_id].terminal_time)

    assert len(m.lob.order_map_by_agent[exec_id]) == 0


def test_reward_components_present_and_consistent():
    m = Market(_base_config(reward_mark_to_mid_weight=0.0))
    obs, info = m.reset(seed=11)

    # A stable action in simplex-like direction after env transform.
    action = np.array([0.0] * m.action_space.shape[0], dtype=np.float32)
    obs, reward, terminated, truncated, info = m.step(action)

    assert 'reward_realized_step' in info
    assert 'reward_inventory_step' in info
    assert 'reward_terminal_step' in info

    reconstructed = info['reward_realized_step'] + info['reward_inventory_step']
    assert np.isclose(reward, reconstructed)

    agent = m.agents[m.execution_agent_id]
    assert 'total' in agent.last_reward_components
    assert np.isclose(
        agent.last_reward_components['total'],
        agent.last_reward_components['realized'] + agent.last_reward_components['inventory'] + agent.last_reward_components['terminal'],
    )
