"""Generate attention visualizations from trained Transformer ablation models.

Loads each transformer_baseline/transformer_fees model, collects attention
maps over a batch of environment observations, and saves heatmap figures.
"""
import os
import sys
import glob
import numpy as np
import torch
import gymnasium as gym

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'rl_files'))

from simulation.market_gym import Market
from actor_critic import BilateralAgentAttention
from attention_viz import plot_attention_maps, plot_attention_per_head

FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

ENVS = ['noise', 'flow', 'strategic']
TAGS = ['transformer_baseline', 'transformer_fees']

# Fee settings (match ablation_runner.sh)
FEE_CONFIGS = {
    'transformer_baseline': {'maker_rebate': 0.0, 'taker_fee': 0.0},
    'transformer_fees': {'maker_rebate': 0.0002, 'taker_fee': 0.0003},
}


def make_env(env_type, seed, fee_cfg):
    cfg = {
        'market_env': env_type,
        'execution_agent': 'rl_agent',
        'volume': 20,
        'terminal_time': 150,
        'time_delta': 15,
        'drop_feature': 'drift',
        'maker_rebate': fee_cfg['maker_rebate'],
        'taker_fee': fee_cfg['taker_fee'],
        'seed': seed,
    }
    def _thunk():
        return Market(cfg)
    return _thunk


def collect_observations(env_type, fee_cfg, n_steps=500, seed=100):
    """Run a random-action rollout and collect diverse observations."""
    envs = gym.vector.SyncVectorEnv([make_env(env_type, seed + i, fee_cfg) for i in range(4)])
    obs, _ = envs.reset(seed=seed)
    all_obs = [obs]
    for _ in range(n_steps):
        action = np.array([envs.single_action_space.sample() for _ in range(4)])
        obs, _, terms, truncs, _ = envs.step(action)
        all_obs.append(obs)
    envs.close()
    return np.concatenate(all_obs, axis=0)  # (n_steps*4, obs_dim)


def main():
    device = torch.device('cpu')
    models_dir = os.path.join(PROJECT_ROOT, 'models')

    for tag in TAGS:
        for env_type in ENVS:
            pattern = f'{env_type}_*_bsize_3200_log_normal_{tag}_drift.pt'
            matches = glob.glob(os.path.join(models_dir, pattern))
            if not matches:
                print(f'[skip] no model for {tag} / {env_type}')
                continue
            model_path = matches[0]
            print(f'\nLoading {os.path.basename(model_path)}')

            fee_cfg = FEE_CONFIGS[tag]

            # Build a single env to size the agent
            env_fns = [make_env(env_type, 0, fee_cfg)]
            envs = gym.vector.SyncVectorEnv(env_fns)
            agent = BilateralAgentAttention(envs, n_levels=5, drop_feature='drift',
                                            use_ofi=False).to(device)
            state = torch.load(model_path, map_location=device, weights_only=True)
            agent.load_state_dict(state)
            agent.eval()
            envs.close()

            # Collect observations
            print(f'  collecting observations from {env_type} env...')
            obs = collect_observations(env_type, fee_cfg, n_steps=200, seed=100)
            obs_t = torch.from_numpy(obs).float().to(device)
            print(f'  collected {len(obs)} observations')

            # Extract attention
            with torch.no_grad():
                attn_maps = agent.get_attention_maps(obs_t)

            # Save overview
            save_path = os.path.join(FIGURES_DIR, f'attention_{tag}_{env_type}.png')
            plot_attention_maps(attn_maps, save_path=save_path)

            # Save per-head for layer 0
            save_path2 = os.path.join(FIGURES_DIR, f'attention_{tag}_{env_type}_heads_L1.png')
            plot_attention_per_head(attn_maps, layer_idx=0, save_path=save_path2)

    print('\nDone. Figures saved to figures/')


if __name__ == '__main__':
    main()
