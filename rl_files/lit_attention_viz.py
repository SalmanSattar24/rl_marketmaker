"""Attention visualization for trained BilateralAgentLiT.

Loads the best_model.pth checkpoint and visualizes the Transformer's attention
patterns across multiple market states, answering:

1. Is the temporal axis being used? (Are token 1-4 weights different from token 5-8?)
2. Does the context token shift its focus based on inventory / time remaining?
3. Do the 4 attention heads learn different patterns?
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from simulation.market_gym import Market
from simulation.history_wrapper import HistoryWrapper
from rl_files.actor_critic import BilateralAgentLiT

FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")

# Token labels (matches BilateralAgentLiT / LiTLOBEncoder order)
# 0: context, 1-4: ask patches (oldest->newest), 5-8: bid patches (oldest->newest)
TOKEN_LABELS = [
    "CTX",
    "A_t-7", "A_t-5", "A_t-3", "A_t-1",  # ask patches by time window
    "B_t-7", "B_t-5", "B_t-3", "B_t-1",  # bid patches by time window
]

# Simple palette for heads
HEAD_COLORS = ["#2C6FBB", "#E8731C", "#2B8C3C", "#B9302F"]


def _make_env_with_history(seed: int = 42):
    cfg = {
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
        "seed": seed,
    }
    return HistoryWrapper(Market(cfg), history_len=8)


def load_trained_agent():
    env = _make_env_with_history(seed=42)

    class Wrap:
        def __init__(self, e):
            self.single_observation_space = e.observation_space
            self.single_action_space = e.action_space

    agent = BilateralAgentLiT(
        Wrap(env),
        K=8,
        patch_width=2,
        drop_feature="drift",
    )
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    agent.load_state_dict(state)
    agent.eval()
    print(f"[OK] Loaded agent from {CHECKPOINT}")
    n_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"     Trainable params: {n_params:,}")
    return agent, env


def collect_states(env, agent, n_episodes=30, seed_base=1000):
    """Roll out `n_episodes` episodes and return observations bucketed by:
      - step_index (1..10)
      - signed inventory

    Returns a list of tuples (obs, step_idx, inventory).
    """
    samples = []
    for ep in range(n_episodes):
        cfg = env.unwrapped.__dict__ if False else None  # not used
        # New env per episode for fresh seed
        sub_env = _make_env_with_history(seed=seed_base + ep)
        obs, _ = sub_env.reset(seed=seed_base + ep)

        step_i = 0
        current_inventory = 0
        while True:
            samples.append((obs.copy(), step_i, current_inventory))

            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                bid, ask = agent.deterministic_action(obs_t)
            bid_np = bid[0].cpu().numpy()
            ask_np = ask[0].cpu().numpy()
            env_action = (bid_np, ask_np)

            obs, reward, terminated, truncated, info = sub_env.step(env_action)
            step_i += 1
            current_inventory = info.get("net_inventory", 0)
            if terminated or truncated or step_i >= 12:
                break
    print(f"[OK] Collected {len(samples)} state samples from {n_episodes} episodes")
    return samples


def compute_batched_attention(agent, samples):
    """Run all samples through agent.get_attention_maps and return stacked outputs."""
    obs_batch = np.stack([s[0] for s in samples], axis=0)
    obs_t = torch.from_numpy(obs_batch).float()
    maps = agent.get_attention_maps(obs_t)
    # layer_attention: list of (B, H, 9, 9)
    # context_attention: (B, H, 9)
    return maps


def plot_layer_heatmaps(maps, path):
    """Plot mean attention matrix per layer (averaged across heads and samples)."""
    n_layers = len(maps["layer_attention"])
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5.5))
    if n_layers == 1:
        axes = [axes]

    for li, attn in enumerate(maps["layer_attention"]):
        # attn: (B, H, 9, 9) → average across batch and heads
        mean_attn = attn.mean(dim=(0, 1)).numpy()

        ax = axes[li]
        im = ax.imshow(mean_attn, cmap="Blues", vmin=0, vmax=mean_attn.max())
        ax.set_title(f"Layer {li + 1} Self-Attention (mean over batch × heads)",
                     fontsize=11)
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.set_xticklabels(TOKEN_LABELS, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(TOKEN_LABELS, fontsize=9)
        ax.set_xlabel("Key (attended to)")
        ax.set_ylabel("Query (from)")

        # Annotate cells
        for i in range(9):
            for j in range(9):
                v = mean_attn[i, j]
                color = "white" if v > mean_attn.max() * 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=7)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {path}")


def plot_per_head_context_attention(maps, path):
    """Bar chart: context token (row 0) attention to 8 LOB patches, per head.

    Shows whether different heads attend to different patches.
    """
    # Use the final layer's attention (the one most directly feeds into CTX output)
    final_layer_attn = maps["layer_attention"][-1]  # (B, H, 9, 9)
    # Context token is token 0 — extract row 0: what context attends to
    ctx_attn = final_layer_attn[:, :, 0, :]  # (B, H, 9)
    # Average across samples
    mean_ctx = ctx_attn.mean(dim=0).numpy()  # (H, 9)

    n_heads = mean_ctx.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4.5), sharey=True)
    if n_heads == 1:
        axes = [axes]

    for hi in range(n_heads):
        ax = axes[hi]
        bars = ax.bar(
            range(9),
            mean_ctx[hi],
            color=[HEAD_COLORS[hi % 4] if lbl != "CTX" else "#999999"
                   for lbl in TOKEN_LABELS],
            edgecolor="black", linewidth=0.5,
        )
        ax.set_xticks(range(9))
        ax.set_xticklabels(TOKEN_LABELS, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"Head {hi + 1}", fontsize=11)
        ax.set_ylim(0, mean_ctx.max() * 1.15)
        ax.axhline(1 / 9, ls="--", color="red", lw=0.8, alpha=0.6,
                   label="uniform (1/9)")
        if hi == 0:
            ax.set_ylabel("CTX → token attention weight")
            ax.legend(fontsize=8)

    plt.suptitle("Context token's attention to LOB patches (final layer, per head)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {path}")


def plot_inventory_conditioned(agent, samples, path):
    """Compare context token attention between positive vs negative inventory states."""
    pos_inv = [s for s in samples if s[2] > 0]
    neg_inv = [s for s in samples if s[2] < 0]
    zero_inv = [s for s in samples if s[2] == 0]

    print(f"[INFO] Inventory buckets: pos={len(pos_inv)}, "
          f"neg={len(neg_inv)}, zero={len(zero_inv)}")

    if len(pos_inv) == 0 or len(neg_inv) == 0:
        print("[WARN] No positive or negative inventory samples — "
              "skipping inventory-conditioned plot")
        return

    def mean_ctx(samples_subset):
        obs_t = torch.from_numpy(
            np.stack([s[0] for s in samples_subset], axis=0)
        ).float()
        m = agent.get_attention_maps(obs_t)
        # Final layer, row 0, averaged over batch and heads
        return m["layer_attention"][-1][:, :, 0, :].mean(dim=(0, 1)).numpy()

    pos_ctx = mean_ctx(pos_inv)
    neg_ctx = mean_ctx(neg_inv)
    zero_ctx = mean_ctx(zero_inv) if zero_inv else None

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(9)
    width = 0.27

    ax.bar(x - width, neg_ctx, width, label=f"inv < 0 (n={len(neg_inv)})",
           color="#C44E52", edgecolor="black", linewidth=0.5)
    if zero_ctx is not None:
        ax.bar(x, zero_ctx, width, label=f"inv = 0 (n={len(zero_inv)})",
               color="#999999", edgecolor="black", linewidth=0.5)
    ax.bar(x + width, pos_ctx, width, label=f"inv > 0 (n={len(pos_inv)})",
           color="#4C72B0", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(TOKEN_LABELS, rotation=45, ha="right")
    ax.set_ylabel("CTX → token attention (final layer, head-avg)")
    ax.set_title("Does context token shift attention based on inventory?")
    ax.axhline(1 / 9, ls="--", color="black", lw=0.8, alpha=0.6,
               label="uniform (1/9)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {path}")


def plot_temporal_vs_side_summary(maps, path):
    """Summary stats: does the model distinguish ask/bid sides and recent/old?"""
    # Average over all samples, batch, heads for final layer
    final = maps["layer_attention"][-1].mean(dim=(0, 1)).numpy()  # (9, 9)

    # Context token attention (row 0)
    ctx_row = final[0, :]
    ask_total = ctx_row[1:5].sum()     # A_t-7 .. A_t-1
    bid_total = ctx_row[5:9].sum()     # B_t-7 .. B_t-1
    ctx_self = ctx_row[0]

    ask_recent = ctx_row[4]            # A_t-1 (newest ask)
    ask_oldest = ctx_row[1]            # A_t-7 (oldest ask)
    bid_recent = ctx_row[8]            # B_t-1 (newest bid)
    bid_oldest = ctx_row[5]            # B_t-7 (oldest bid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: side allocation
    axes[0].bar(
        ["CTX→self", "Ask side\n(sum)", "Bid side\n(sum)"],
        [ctx_self, ask_total, bid_total],
        color=["#999999", "#4C72B0", "#C44E52"],
        edgecolor="black",
    )
    axes[0].set_title("How context distributes attention: self vs sides")
    axes[0].set_ylabel("Attention weight")
    for i, v in enumerate([ctx_self, ask_total, bid_total]):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    # Right: recent vs oldest per side
    axes[1].bar(
        ["A_oldest\n(t-7)", "A_recent\n(t-1)", "B_oldest\n(t-7)", "B_recent\n(t-1)"],
        [ask_oldest, ask_recent, bid_oldest, bid_recent],
        color=["#A0C0E0", "#2C6FBB", "#E8A8A8", "#B9302F"],
        edgecolor="black",
    )
    axes[1].set_title("Temporal attention: oldest vs newest window")
    axes[1].set_ylabel("Attention weight")
    for i, v in enumerate([ask_oldest, ask_recent, bid_oldest, bid_recent]):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {path}")

    # Also print the numbers for quick inspection
    print("\n[SUMMARY] Context token attention (final layer, head-avg):")
    print(f"  Self (CTX):         {ctx_self:.4f}")
    print(f"  Total to ask side:  {ask_total:.4f}  ({ask_total / 8 * 9:.4f} per-token equivalent)")
    print(f"  Total to bid side:  {bid_total:.4f}")
    print(f"  Ask: oldest={ask_oldest:.4f}, newest={ask_recent:.4f}, "
          f"ratio newest/oldest={ask_recent / (ask_oldest + 1e-8):.2f}x")
    print(f"  Bid: oldest={bid_oldest:.4f}, newest={bid_recent:.4f}, "
          f"ratio newest/oldest={bid_recent / (bid_oldest + 1e-8):.2f}x")
    print(f"  Uniform baseline:   {1 / 9:.4f}")


def main():
    print("=" * 70)
    print("LiT Attention Visualization")
    print("=" * 70)

    agent, env = load_trained_agent()
    samples = collect_states(env, agent, n_episodes=30, seed_base=1000)
    maps = compute_batched_attention(agent, samples)

    print(f"[INFO] Attention shapes:")
    print(f"  layer_attention: {len(maps['layer_attention'])} layers, "
          f"each {tuple(maps['layer_attention'][0].shape)}")
    print(f"  context_attention: {tuple(maps['context_attention'].shape)}")
    print()

    plot_layer_heatmaps(
        maps,
        os.path.join(FIGURES_DIR, "lit_attention_layers.png"),
    )
    plot_per_head_context_attention(
        maps,
        os.path.join(FIGURES_DIR, "lit_attention_per_head.png"),
    )
    plot_inventory_conditioned(
        agent, samples,
        os.path.join(FIGURES_DIR, "lit_attention_by_inventory.png"),
    )
    plot_temporal_vs_side_summary(
        maps,
        os.path.join(FIGURES_DIR, "lit_attention_side_temporal_summary.png"),
    )

    print()
    print("=" * 70)
    print("DONE. Figures saved to figures/:")
    print("  - lit_attention_layers.png")
    print("  - lit_attention_per_head.png")
    print("  - lit_attention_by_inventory.png")
    print("  - lit_attention_side_temporal_summary.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
