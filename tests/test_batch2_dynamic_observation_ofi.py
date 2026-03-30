import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from simulation.market_gym import Market
from config.config import rl_agent_config


def _mk_cfg(drop_feature=None, volume=10, use_ofi=False):
    return {
        'market_env': 'flow',
        'execution_agent': 'rl_agent',
        'seed': 42,
        'terminal_time': 150,
        'time_delta': 15,
        'drop_feature': drop_feature,
        'volume': volume,
        'inventory_max': 10,
        'use_ofi': use_ofi,
    }


def test_dynamic_observation_length_matches_actual_for_multiple_configs():
    cases = [
        (None, 10),
        (None, 40),
        ('drift', 10),
        ('drift', 40),
        ('volume', 10),
        ('volume', 40),
        ('order_info', 10),
        ('order_info', 40),
    ]

    for drop_feature, volume in cases:
        m = Market(_mk_cfg(drop_feature=drop_feature, volume=volume, use_ofi=False))
        obs, _ = m.reset(seed=42)
        declared = m.observation_space.shape[0]
        actual = len(obs)
        assert declared == actual, (
            f"Mismatch for drop_feature={drop_feature}, volume={volume}: "
            f"declared={declared}, actual={actual}"
        )


def test_dynamic_observation_length_handles_nondefault_observation_levels():
    original = dict(rl_agent_config)
    try:
        rl_agent_config['observation_book_levels'] = 10
        rl_agent_config['action_book_levels'] = 10

        m = Market(_mk_cfg(drop_feature=None, volume=10, use_ofi=False))
        obs, _ = m.reset(seed=101)

        declared = m.observation_space.shape[0]
        actual = len(obs)
        assert declared == actual, f"declared={declared}, actual={actual}"
    finally:
        for k, v in original.items():
            rl_agent_config[k] = v


def test_ofi_flag_adds_single_feature_when_volume_features_active():
    m_no_ofi = Market(_mk_cfg(drop_feature=None, volume=10, use_ofi=False))
    obs_no_ofi, _ = m_no_ofi.reset(seed=123)

    m_with_ofi = Market(_mk_cfg(drop_feature=None, volume=10, use_ofi=True))
    obs_with_ofi, _ = m_with_ofi.reset(seed=123)

    assert len(obs_with_ofi) == len(obs_no_ofi) + 1
    assert m_with_ofi.observation_space.shape[0] == m_no_ofi.observation_space.shape[0] + 1


def test_ofi_flag_has_no_effect_when_volume_block_dropped():
    m_no_ofi = Market(_mk_cfg(drop_feature='volume', volume=10, use_ofi=False))
    obs_no_ofi, _ = m_no_ofi.reset(seed=321)

    m_with_ofi = Market(_mk_cfg(drop_feature='volume', volume=10, use_ofi=True))
    obs_with_ofi, _ = m_with_ofi.reset(seed=321)

    assert len(obs_with_ofi) == len(obs_no_ofi)
    assert m_with_ofi.observation_space.shape[0] == m_no_ofi.observation_space.shape[0]
