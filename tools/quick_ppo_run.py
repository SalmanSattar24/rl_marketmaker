"""Quick PPO training/validation harness.

Runs a very small PPO-style loop for a few iterations and saves a learning
curve to `quick_ppo_plot.png` and returns to `quick_ppo_results.npz`.

Designed for fast local validation (CPU) only.
"""
import os
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym

import sys
cwd = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.dirname(cwd)
sys.path.append(repo_root)

from rl_files.ppo_continuous_action import Agent, make_env


def quick_run(num_envs=4, num_steps=20, num_iterations=6, seed=0, device='cpu', mode='policy', out_tag=''):
    configs = [{'market_env': 'noise', 'execution_agent': 'rl_agent', 'volume': 20, 'seed': seed + s, 'terminal_time': 50, 'time_delta': 10, 'drop_feature': None} for s in range(num_envs)]
    env_fns = [make_env(cfg) for cfg in configs]
    envs = gym.vector.AsyncVectorEnv(env_fns=env_fns)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)

    # storage
    mean_returns = []

    # initialize
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.as_tensor(np.asarray(next_obs), dtype=torch.float32, device=device)

    for iteration in range(num_iterations):
        returns = []
        # rollouts
        for step in range(num_steps):
            with torch.no_grad():
                if mode == 'random':
                    action = torch.randn((num_envs, *envs.single_action_space.shape), device=device)
                else:
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
            # step envs
            next_obs_np, reward, terminations, truncs, infos = envs.step(action.cpu().numpy())
            next_obs = torch.as_tensor(np.asarray(next_obs_np), dtype=torch.float32, device=device)
            if 'final_info' in infos:
                for info in infos['final_info']:
                    if info is not None and 'cum_reward' in info:
                        returns.append(info['cum_reward'])

        mean_ret = float(np.mean(returns)) if returns else 0.0
        mean_returns.append(mean_ret)
        print(f"iter {iteration+1}/{num_iterations} mean_return={mean_ret:.4f}")

        # very small policy update step (toy): do a single backward on a value-derived
        # tensor so gradients exist but do not change parameters meaningfully.
        optimizer.zero_grad()
        loss_val = agent.get_value(next_obs).mean() * 0.0
        loss_val.backward()
        optimizer.step()

    # save results and plot
    out_dir = os.path.join(repo_root, 'runs_quick')
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{out_tag}" if out_tag else ""
    np.savez(os.path.join(out_dir, f'quick_ppo_results{suffix}.npz'), returns=np.array(mean_returns))

    plt.figure()
    plt.plot(np.arange(1, len(mean_returns)+1), mean_returns, marker='o')
    plt.xlabel('iteration')
    plt.ylabel('mean_return')
    plt.title('Quick PPO mean returns')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f'quick_ppo_plot{suffix}.png'), bbox_inches='tight')
    print('Saved quick run results to', out_dir)


if __name__ == '__main__':
    # Allow quick configuration via environment variables so this script can
    # be executed directly (avoids multiprocessing spawn issues when using
    # python -c or stdin). Useful env vars: QP_SEED, QP_ITERS, QP_STEPS, QP_ENVS
    seed = int(os.environ.get('QP_SEED', '0'))
    num_iterations = int(os.environ.get('QP_ITERS', '6'))
    num_steps = int(os.environ.get('QP_STEPS', '20'))
    num_envs = int(os.environ.get('QP_ENVS', '4'))
    mode = os.environ.get('QP_MODE', 'policy')
    out_tag = os.environ.get('QP_TAG', '')
    t0 = time.time()
    quick_run(num_envs=num_envs, num_steps=num_steps, num_iterations=num_iterations, seed=seed, mode=mode, out_tag=out_tag)
    print('done in', time.time()-t0)

