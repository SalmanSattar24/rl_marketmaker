# RL Market Maker

**English** | [中文](README_zh.md)

A research implementation of a **bilateral reinforcement learning market maker** on a simulated limit order book (LOB).
The agent simultaneously quotes on both bid and ask sides, manages inventory risk, and is trained end-to-end with Proximal Policy Optimization (PPO).

> **Course**: CSCI 566 — Deep Learning and its Applications
> **Institution**: University of Southern California
> **Reference paper**: Cheridito & Weiss (2026), *Reinforcement Learning for Trade Execution with Market and Limit Orders* (arXiv:2507.06345)
> **Base repository**: [moritzweiss/rlte](https://github.com/moritzweiss/rlte)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Running the Notebook](#running-the-notebook)
- [Running Tests](#running-tests)
- [Recent Changes](#recent-changes)
- [Roadmap](#roadmap)
- [中文說明](README_zh.md)

---

## Overview

At each timestep the agent:

1. Observes a state vector (LOB features + inventory + OFI)
2. Outputs a **Logistic-Normal** action — allocating order volume across price levels on both bid and ask sides
3. Receives reward based on PnL, inventory risk, and maker-taker fees
4. Policy is optimized with **standard PPO** (clipped surrogate loss + gradient clipping)

The simulator supports three market regimes: `noise`, `flow`, and `strategic`.

---

## Key Features

| Feature | Details |
| --- | --- |
| **Bilateral quoting** | Agent posts simultaneous bid and ask limit orders |
| **Logistic-Normal policy** | Continuous action mapped to order-allocation simplex |
| **Maker-taker fee structure** | `maker_rebate=0.2`, `taker_fee=0.3` (reward units, calibrated to ~1bp real spread) |
| **Standard PPO** | Clipped surrogate loss, 4 epochs, 4 minibatches, gradient clipping (`max_grad_norm=0.5`) |
| **OFI feature** | Order Flow Imbalance included as optional observation feature (`use_ofi=True`) |
| **Transformer LOB encoder** | Full Transformer block with sinusoidal PE, Pre-LN, GELU FFN, and attention-weighted pooling |
| **Attention visualization** | Extract and plot per-head, per-layer self-attention heatmaps + pooling weights |
| **Ablation framework** | Automated 4×3 experiment matrix: MLP/Transformer × no-fee/fee × noise/flow/strategic |
| **Parallel training** | 32–128 parallel `SyncVectorEnv` environments |
| **TensorBoard logging** | Loss, KL divergence, clip fraction, episode return |
| **Training health artifacts** | Per-iteration JSON health bundle under `results/training_health/` + action-contract diagnostics |
| **67 passing tests** | Full regression suite covering environment, agent, and training path |

---

## Architecture

Two agent architectures are supported:

### Plan A — MLP Bilateral Agent

```text
Observation (63-dim)
    ├── LOB features: bid/ask volumes at 5 price levels
    ├── Inventory features: current volume, active volume, time-weighted inventory
    ├── Market features: spread, mid-price drift
    └── OFI (optional): order flow imbalance

        ↓
┌─────────────────────────────────┐
│   Shared MLP Trunk (128 units)  │
│   + LayerNorm                   │
└──────────┬──────────────────────┘
           │
    ┌──────┴──────┐
    ↓             ↓
Bid Head       Ask Head        Value Head
(Logistic-    (Logistic-       (scalar)
 Normal)       Normal)

Action = (bid_allocation, ask_allocation)   # each is a 7-dim simplex
```

### Plan B — Transformer Bilateral Agent (NEW)

```text
Observation (63-dim)
    ├── LOB bid/ask volumes → reshaped to (5 levels × 2 features)
    ├── Inventory + market features → global context
    └── OFI (optional)

LOB volumes (5, 2)
        ↓
┌──────────────────────────────────────┐
│  Linear Embedding → d_model=32       │
│  + Sinusoidal Positional Encoding    │
│                                      │
│  Transformer Encoder (2 layers)      │
│  ┌────────────────────────────────┐  │
│  │ Pre-LayerNorm                  │  │
│  │ Multi-Head Self-Attention (2h) │  │
│  │ + Residual + Dropout           │  │
│  │ Pre-LayerNorm                  │  │
│  │ FFN (GELU, dim=64)            │  │
│  │ + Residual + Dropout           │  │
│  └────────────────────────────────┘  │
│                                      │
│  Attention-Weighted Pooling          │
│  (learned query vector)              │
└──────────┬───────────────────────────┘
           │
    concat with global features
           ↓
    ┌──────┴──────┐
    ↓             ↓
Bid Head       Ask Head        Value Head
(Logistic-    (Logistic-       (scalar)
 Normal)       Normal)

Action = (bid_allocation, ask_allocation)   # each is a 7-dim simplex
```

The Transformer encoder treats each LOB price level as a token, enabling the model to learn cross-level relationships (e.g., best-bid vs deep-book dynamics). Attention weights can be extracted for interpretability analysis.

**Reward function:**

```text
r_t = PnL(fills) + maker_rebate × passive_fill_vol / V₀
                 - taker_fee   × aggressive_fill_vol / V₀
                 - inventory_penalty × |inventory|
```

---

## Repository Structure

```text
rl_marketmaker/
├── bilateral_mm_agent.ipynb   # Main experiment notebook
├── config/
│   ├── config.py              # All agent/env/fee configs
│   └── __init__.py
├── limit_order_book/
│   └── limit_order_book.py    # LOB engine (order matching, cancellation, book state)
├── simulation/
│   ├── market_gym.py          # Gym-compatible environment
│   └── agents.py              # All agent classes (RL, baseline, noise, strategic…)
├── rl_files/
│   ├── actor_critic.py        # PPO training loop + agent models (MLP & Transformer)
│   ├── attention_viz.py       # Attention weight visualization utilities
│   └── ablation_runner.sh     # Ablation experiment runner (4 configs × 3 envs)
├── initial_shape/             # Initial LOB shape arrays (.npz)
├── models/                    # Saved model checkpoints (.pt)
├── rewards/                   # Evaluation reward arrays (.npz)
├── tests/                     # Regression and integration tests
├── requirements.txt
├── changes_report.tex         # Overleaf-ready PDF report of all code changes
└── FINAL_PROJECT_PLAN.md      # Project planning document
```

---

## Setup

**Recommended Python**: 3.9 – 3.14

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `gymnasium`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tensorboard`, `tyro`, `sortedcontainers`

---

## Running the Notebook

Open `bilateral_mm_agent.ipynb` in JupyterLab or VS Code.
The notebook runs fully locally — no Google Colab or Google Drive required.

**Sections:**

1. Environment setup and dependency check
2. Repository verification
3. Config — dual-regime (flow → strategic)
4. Agent initialization (bilateral RL + baseline fixed-spread)
5. Vectorized quota projection
6. **Training** — PPO with 128 parallel environments
7. **Evaluation** — RL agent vs baseline comparison
8. Visualization — PnL curves, action distributions, fee impact

> Note: Full training (200 × 128 × 100 = 2.56M timesteps) takes several hours on CPU. For quick testing, reduce `total_timesteps` in `Args`.

---

## Running Tests

```bash
python -m pytest -q tests/
```

Expected result: **67 passed**

Key test files:

| File | Coverage |
| --- | --- |
| `test_bilateral_simulator.py` | End-to-end bilateral episode |
| `test_bilateral_action_space.py` | Action sampling and simplex validity |
| `test_maker_taker_fees.py` | Fee accounting correctness |
| `test_bugfix_audit.py` | Regression checks for known fixed bugs |
| `test_phase2_integration.py` | Full environment integration |
| `test_phase3_training.py` | Training loop forward/backward pass |
| `test_phase4_metrics_and_contract.py` | Simplex contract guards + evaluation metrics bundle validation |

---

## Ablation Results

A 4×3 ablation study was run comparing **MLP vs Transformer** architectures with and without **maker-taker fees** across three market regimes (**noise**, **flow**, **strategic**). Each configuration was trained for 200 PPO iterations with 32 parallel environments (640K timesteps total) and evaluated on 200–1000 deterministic episodes.

### Mean Episode Reward (± standard error)

| Config | noise | flow | strategic |
| --- | --- | --- | --- |
| MLP (no fee) | 0.177 ± 0.033 | 0.693 ± 0.024 | **1.093 ± 0.095** |
| MLP (+ fees) | -0.030 ± 0.075 | **0.981 ± 0.053** | 0.431 ± 0.178 |
| Transformer (no fee) | -0.069 ± 0.078 | 0.783 ± 0.058 | 0.670 ± 0.150 |
| Transformer (+ fees) | **0.356 ± 0.072** | 0.805 ± 0.058 | 0.208 ± 0.135 |

**Bold** = best in column. See [figures/ablation_mean_rewards.png](figures/ablation_mean_rewards.png) and [figures/ablation_boxplots.png](figures/ablation_boxplots.png).

### Statistical Significance (Welch t-test, p-values)

| Comparison | noise | flow | strategic |
| --- | --- | --- | --- |
| MLP vs Transformer (no fee) | **0.004** ** (MLP) | 0.156 | **0.018** * (MLP) |
| MLP vs Transformer (+ fees) | **<0.001** *** (Transformer) | **0.026** * (MLP) | 0.320 |
| MLP: no fee vs + fees | **0.012** * (no fee) | **<0.001** *** (+ fees) | **0.001** ** (no fee) |
| Transformer: no fee vs + fees | **<0.001** *** (+ fees) | 0.789 | **0.023** * (no fee) |

`*` p<0.05, `**` p<0.01, `***` p<0.001. Winner in parentheses.

### Key Findings

1. **Transformer benefits most from maker rebates in the noise environment** — With fees enabled, the Transformer's attention mechanism learns to better exploit passive limit-order fills, flipping from the worst (–0.069) to the best (0.356) configuration on noise. This improvement is highly significant (p < 0.001).

2. **MLP wins on flow with fees** — The simpler MLP architecture achieves the highest mean reward (0.981) on flow with fees, significantly outperforming the Transformer here (p = 0.026). For well-structured directional flow, fewer parameters generalize better.

3. **Fees hurt all models on strategic environments** — Both MLP and Transformer see significant reward drops when fees are added in the strategic regime. Toxic informed flow combined with taker fees creates a double cost that neither architecture overcomes in the current training budget.

4. **MLP is more robust under high variance** — On strategic (the hardest environment, with reward std ~2–3), the MLP baseline has the highest mean. Transformer shows slightly lower variance but cannot match MLP's mean performance.

### Attention Interpretability

Attention heatmaps extracted from the trained Transformer models are saved under [figures/](figures/):

- `attention_transformer_{baseline,fees}_{noise,flow,strategic}.png` — Layer-averaged self-attention + pooling weights per configuration
- `attention_transformer_*_heads_L1.png` — Per-head attention patterns for the first Transformer layer

These can be inspected to see which LOB levels the model attends to most heavily under different market conditions and fee structures.

### Reproducing

```bash
bash rl_files/ablation_runner.sh          # Run the 12-config ablation
python rl_files/analyze_ablation.py       # Generate tables, bar chart, box plots, significance tests
python rl_files/generate_attention_viz.py # Extract attention maps from trained Transformer models
```

---

## Recent Changes

### v0.4 — Transformer Encoder + Ablation Framework (April 2026)

**`rl_files/actor_critic.py`**

- **LOBAttentionEncoder**: Full Transformer encoder block replacing single-layer MHSA
  - Sinusoidal positional encoding (registered buffer, not learned) to encode LOB level ordering
  - 2-layer Pre-LayerNorm TransformerEncoder with GELU FFN (d_model=32, n_heads=2, ffn_dim=64)
  - Attention-weighted pooling with learned query vector (replaces naive mean pooling)
  - `get_attention_maps()` method for extracting per-head, per-layer attention weights
- **BilateralAgentAttention**: New agent class wrapping the Transformer encoder with dual Logistic-Normal heads
  - Activated via `--attention` CLI flag
  - Total params: ~51K (vs ~25K for MLP)

**`rl_files/attention_viz.py`** (NEW)

- `plot_attention_maps()`: Layer self-attention heatmaps (head-averaged) + pooling weight bar chart
- `plot_attention_per_head()`: Per-head attention heatmaps for a specific layer
- `collect_attention_maps()`: Accumulate attention weights over N episodes for stable averages

**`rl_files/ablation_runner.sh`** (NEW)

- Automated ablation experiment runner: 4 agent configs × 3 environments = 12 runs
- Configs: MLP baseline, MLP + fees, Transformer baseline, Transformer + fees
- Environments: noise, flow, strategic
- Supports `--debug` flag for quick smoke tests

**`simulation/agents.py`**

- Fixed crash when bilateral MM agent executes all volume before terminal time (`volume == 0` guard)

**`simulation/market_gym.py`**

- Fixed crash when noise agent events extend past terminal time (graceful termination instead of ValueError)

---

### v0.3 — Maker-Taker Fees + PPO Upgrade (April 2026)

**`config/config.py`**

- Added `fee_config` with `maker_rebate=0.2` and `taker_fee=0.3`
- Fees calibrated to ~1bp real spread at reference price 1000, volume 20

**`simulation/agents.py`**

- `ExecutionAgent.set_fees()`: injects fee rates from environment
- Taker fee deducted on market order fills (both buy and sell)
- Maker rebate credited on passive limit order fills
- Removed assertions incompatible with bilateral short positions

**`simulation/market_gym.py`**

- Fee config propagated into each episode via `set_fees()`
- Episode info dict now reports `cumulative_fees_paid`, `cumulative_rebates_earned`, `net_fee_impact`
- Removed `assert volume == 0` at episode end (bilateral agent can hold non-zero inventory)

**`limit_order_book/limit_order_book.py`**

- Fixed stale order-ID tracking: cancellations now correctly deregister from `agent_bid_orders` / `agent_ask_orders`

**`rl_files/actor_critic.py`**

- Restored production hyperparameters: `num_envs=128`, `num_steps=100`, `num_minibatches=4`, `update_epochs=4`
- Standard PPO clipped surrogate loss (Schulman et al. 2017)
- Gradient clipping: `max_grad_norm=0.5`
- KL divergence tracking + TensorBoard logging (`approx_kl`, `clipfrac`)
- Switched from `AsyncVectorEnv` to `SyncVectorEnv` (fixes info-dict format mismatch)
- Added `get_value()` helper to `AgentLogisticNormal`

---

## Roadmap

- [x] Transformer LOB encoder with sinusoidal PE and attention-weighted pooling
- [x] Attention weight extraction and visualization utilities
- [x] Ablation experiment framework (MLP vs Transformer × fee structure × market regime)
- [ ] Ablation results analysis and comparison tables
- [ ] LSTM temporal backbone (replace MLP trunk with LSTM for sequential LOB modeling)
- [ ] OFI ablation experiments (`use_ofi=True` vs `False` comparison)
- [ ] Action distribution analysis (market order ratio over training)
- [ ] Full bilateral order generation without unilateral fallback paths
