"""Attention weight visualization for LOB Transformer encoder.

Provides plotting utilities to visualize:
  1. Level-level self-attention heatmaps (per layer, averaged over heads/batch)
  2. Per-head self-attention heatmaps (for a specific layer)
  3. Pooling attention weights (bar chart showing level importance)

Usage (after training):
    import torch
    from actor_critic import BilateralAgentAttention
    from attention_viz import plot_attention_maps, plot_attention_per_head

    agent = BilateralAgentAttention(envs, ...)
    agent.load_state_dict(torch.load('model.pt'))

    obs = ...  # (batch, obs_dim) tensor
    attn_maps = agent.get_attention_maps(obs)
    plot_attention_maps(attn_maps, save_path='figures/attention_overview.png')
    plot_attention_per_head(attn_maps, layer_idx=0, save_path='figures/heads_layer1.png')

Usage (batch collection over many episodes for stable averages):
    all_maps = collect_attention_maps(agent, envs, n_episodes=100)
    plot_attention_maps(all_maps, save_path='figures/attention_avg_100ep.png')
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_attention_maps(attn_maps, save_path=None, figsize=(14, 5)):
    """Plot self-attention heatmaps (head-averaged) and pooling weights.

    Args:
        attn_maps: dict from agent.get_attention_maps() with keys:
            'layer_attention': list of (batch, n_heads, K, K) tensors
            'pool_attention': (batch, 1, K) tensor
        save_path: if provided, save figure to this path
        figsize: figure size

    Returns:
        matplotlib Figure
    """
    layer_attns = attn_maps['layer_attention']
    pool_attn = attn_maps['pool_attention']
    n_layers = len(layer_attns)
    n_levels = layer_attns[0].shape[-1]
    level_labels = [f'L{i+1}' for i in range(n_levels)]

    fig, axes = plt.subplots(1, n_layers + 1, figsize=figsize)
    if n_layers + 1 == 1:
        axes = [axes]

    for i, layer_attn in enumerate(layer_attns):
        # Average over batch and heads: (batch, n_heads, K, K) -> (K, K)
        avg = layer_attn.cpu().numpy().mean(axis=(0, 1))
        im = axes[i].imshow(avg, cmap='Blues', vmin=0, vmax=avg.max())
        axes[i].set_title(f'Layer {i+1} Self-Attn', fontsize=11)
        axes[i].set_xticks(range(n_levels))
        axes[i].set_xticklabels(level_labels)
        axes[i].set_yticks(range(n_levels))
        axes[i].set_yticklabels(level_labels)
        axes[i].set_xlabel('Key (attended to)')
        axes[i].set_ylabel('Query (from)')
        for r in range(n_levels):
            for c in range(n_levels):
                axes[i].text(c, r, f'{avg[r, c]:.2f}', ha='center', va='center',
                             fontsize=8,
                             color='white' if avg[r, c] > 0.5 * avg.max() else 'black')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Pooling weights: (batch, 1, K) -> (K,)
    pool_w = pool_attn.cpu().numpy().mean(axis=0).squeeze()
    axes[-1].bar(level_labels, pool_w, color='steelblue', edgecolor='navy', alpha=0.8)
    axes[-1].set_title('Pooling Attention', fontsize=11)
    axes[-1].set_xlabel('LOB Level')
    axes[-1].set_ylabel('Weight')
    axes[-1].set_ylim(0, max(pool_w) * 1.2)
    for j, w in enumerate(pool_w):
        axes[-1].text(j, w + 0.003, f'{w:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('LOB Transformer Attention (batch-averaged)', fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    return fig


def plot_attention_per_head(attn_maps, layer_idx=0, save_path=None, figsize=(12, 4)):
    """Plot per-head attention heatmaps for a specific Transformer layer.

    Args:
        attn_maps: dict from agent.get_attention_maps()
        layer_idx: which transformer layer (0-indexed)
        save_path: optional save path
        figsize: figure size

    Returns:
        matplotlib Figure
    """
    layer_attn = attn_maps['layer_attention'][layer_idx]
    # (batch, n_heads, K, K) -> (n_heads, K, K)
    avg = layer_attn.cpu().numpy().mean(axis=0)
    n_heads = avg.shape[0]
    n_levels = avg.shape[-1]
    level_labels = [f'L{i+1}' for i in range(n_levels)]

    fig, axes = plt.subplots(1, n_heads, figsize=figsize)
    if n_heads == 1:
        axes = [axes]

    for h in range(n_heads):
        im = axes[h].imshow(avg[h], cmap='Oranges', vmin=0)
        axes[h].set_title(f'Head {h+1}', fontsize=11)
        axes[h].set_xticks(range(n_levels))
        axes[h].set_xticklabels(level_labels)
        axes[h].set_yticks(range(n_levels))
        axes[h].set_yticklabels(level_labels)
        axes[h].set_xlabel('Key')
        axes[h].set_ylabel('Query')
        for r in range(n_levels):
            for c in range(n_levels):
                axes[h].text(c, r, f'{avg[h][r, c]:.2f}', ha='center', va='center',
                             fontsize=8,
                             color='white' if avg[h][r, c] > 0.5 * avg[h].max() else 'black')
        plt.colorbar(im, ax=axes[h], fraction=0.046, pad=0.04)

    plt.suptitle(f'Layer {layer_idx+1} Per-Head Attention (batch-averaged)', fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    return fig


def collect_attention_maps(agent, envs, n_episodes=100, device='cpu'):
    """Collect and average attention maps over multiple episodes.

    Runs the agent deterministically through n_episodes, accumulating
    attention weights at every step for a stable average.

    Args:
        agent: BilateralAgentAttention instance (already loaded with weights)
        envs: gymnasium vector env (or single Market env)
        n_episodes: number of episodes to collect
        device: torch device

    Returns:
        dict with same structure as agent.get_attention_maps(), averaged over all steps
    """
    from simulation.market_gym import Market

    agent.eval()
    is_single = isinstance(envs, Market)

    # Accumulators
    layer_sums = None
    pool_sum = None
    total_steps = 0

    episodes_done = 0
    obs, _ = envs.reset()
    if is_single:
        obs = obs[np.newaxis, :]  # add batch dim

    while episodes_done < n_episodes:
        obs_t = torch.from_numpy(obs).float().to(device)
        maps = agent.get_attention_maps(obs_t)

        if layer_sums is None:
            layer_sums = [a.cpu().numpy().sum(axis=0) for a in maps['layer_attention']]
            pool_sum = maps['pool_attention'].cpu().numpy().sum(axis=0)
        else:
            for i, a in enumerate(maps['layer_attention']):
                layer_sums[i] += a.cpu().numpy().sum(axis=0)
            pool_sum += maps['pool_attention'].cpu().numpy().sum(axis=0)
        total_steps += obs_t.shape[0]

        # Step environment
        with torch.no_grad():
            actions = agent.deterministic_action(obs_t)
        bid_a, ask_a = actions
        if is_single:
            env_action = (bid_a.squeeze(0).cpu().numpy(), ask_a.squeeze(0).cpu().numpy())
        else:
            env_action = torch.cat([bid_a, ask_a], dim=1).cpu().numpy()

        next_obs, _, terms, truncs, infos = envs.step(env_action)
        if is_single:
            next_obs = next_obs[np.newaxis, :]
            if terms or truncs:
                episodes_done += 1
                obs, _ = envs.reset()
                obs = obs[np.newaxis, :]
                continue
        else:
            # Count terminated envs
            done_mask = np.logical_or(terms, truncs)
            episodes_done += done_mask.sum()

        obs = next_obs

    # Average
    layer_avg = [torch.from_numpy(s / total_steps).unsqueeze(0) for s in layer_sums]
    pool_avg = torch.from_numpy(pool_sum / total_steps).unsqueeze(0)

    return {'layer_attention': layer_avg, 'pool_attention': pool_avg}
