# CSCI 566 Final Project Plan: Deep RL Bilateral Market Maker

> Project baseline: Cheridito & Weiss (2025) Logistic-Normal actor-critic for LOB trade execution
> Timeline: 4 weeks (April 2026)
> Team repo: `rl_marketmaker`

---

## Current State Assessment

### What We Have
- Event-driven LOB simulator with noise / flow / strategic environments
- Bilateral market-making agent (`BilateralAgentLogisticNormal`) with dual bid/ask heads
- Logistic-Normal policy mapping continuous outputs to simplex (order allocation)
- OFI (Order Flow Imbalance) feature **already implemented** (`use_ofi=True`)
- Terminal closeout, reward decomposition, circuit breaker
- 29 passing tests, stable v0.2 checkpoint

### What's Weak for a DL Final Project
| Gap | Why It Matters |
|-----|----------------|
| 2-layer MLP (128 units) only | Not "deep" learning |
| No PPO clipping (vanilla REINFORCE + baseline) | Can't claim PPO in paper |
| No temporal modeling (LSTM/Transformer) | Market is sequential but treated as i.i.d. |
| Zero-friction environment | Unrealistic; no maker-taker incentive structure |
| No ablation experiments | Can't demonstrate contribution of each component |

---

## Plan A: Recommended (High Feasibility, 4 Weeks)

### Week 1: Fix Foundation + OFI Ablation

**Task 1.1 -- Correct PPO Implementation**
- Add clipped surrogate loss: `L = min(r*A, clip(r, 1-e, 1+e)*A)`
- Increase `update_epochs` from 1 to 4, `num_minibatches` from 1 to 4
- Files: `rl_files/actor_critic.py` lines 880-890
- Reference: Schulman et al. (2017) PPO; Lin & Beling (2020) PPO for LOB execution
- Estimated effort: ~50 lines changed

**Task 1.2 -- OFI Ablation Baseline**
- OFI is already coded (`agents.py:1448`), just needs experiments
- Train with `use_ofi=False` vs `use_ofi=True` under `strategic` environment
- Collect: mean return, std, CVaR, inventory violations
- Estimated effort: config changes + training runs only

**Deliverable**: Correct PPO + OFI on/off comparison table

---

### Week 2: Maker-Taker Fee Mechanism

**Task 2.1 -- Implement Fee Structure**
- Add `maker_rebate` and `taker_fee` to config (e.g., +0.001 / -0.003 per lot)
- Modify reward calculation in `market_gym.py` (`_compute_reward` or equivalent)
  - Limit order fill: reward += rebate * filled_volume
  - Market order execution: reward -= fee * executed_volume
- Estimated effort: ~30 lines in `market_gym.py`, ~5 lines in `config/config.py`

**Task 2.2 -- Behavioral Analysis**
- Compare agent's action distribution (market order % vs limit order %) with/without fees
- Hypothesis: agent should shift from aggressive (market orders) to passive (limit orders)
- Generate: action heatmaps, market-order-ratio over training, PnL breakdown by order type
- Three conditions: (a) no fee, (b) symmetric fee, (c) asymmetric maker-taker

**Deliverable**: Fee mechanism + behavioral shift analysis charts

---

### Week 3: LSTM Temporal Backbone

**Task 3.1 -- LSTM Actor-Critic Architecture**
- Replace the shared MLP trunk with LSTM:
  ```
  obs_t -> Linear(128) -> LayerNorm -> LSTM(128, 1 layer) -> hidden_t
  hidden_t -> bid_head (Linear -> Logistic-Normal)
  hidden_t -> ask_head (Linear -> Logistic-Normal)
  hidden_t -> value_head (Linear -> scalar)
  ```
- Must handle hidden state across steps within each episode (reset on episode boundary)
- Modify rollout buffer to store/propagate LSTM hidden states
- Reference: DeepLOB (Zhang et al. 2019), Lin & Beling (2020)
- Estimated effort: ~150 lines new code in `actor_critic.py`

**Task 3.2 -- Full Ablation Matrix**

| # | Architecture | OFI | Fees | Expected Insight |
|---|-------------|-----|------|------------------|
| 1 | MLP | off | off | Baseline (current) |
| 2 | MLP | on | off | Does OFI help? |
| 3 | MLP | off | on | Does fee change behavior? |
| 4 | MLP | on | on | Combined effect |
| 5 | LSTM | off | off | Does memory help? |
| 6 | LSTM | on | off | LSTM + OFI |
| 7 | LSTM | off | on | LSTM + fees |
| 8 | LSTM | on | on | **Full model** |

Run each config across `noise`, `flow`, `strategic` environments = 24 experiments total.

**Deliverable**: Ablation table with statistical significance

---

### Week 4: Report + Visualization + Polish

**Task 4.1 -- Figures for Paper/Presentation**
- Training curves (8 configs overlaid)
- Return distribution violin plots (RL vs AS baseline vs fixed-spread baseline)
- Action composition bar charts (how order type allocation shifts with fees)
- Inventory trajectory plots (MLP vs LSTM under strategic environment)
- OFI signal vs agent withdrawal timing (does agent learn to react to toxic flow?)

**Task 4.2 -- Write-up**
- Abstract, Related Work, Method, Experiments, Ablation, Conclusion
- Emphasize: (1) architectural contribution (LSTM), (2) market realism (fees), (3) signal utilization (OFI)

**Deliverable**: Final report + presentation slides

---

## Plan B: Ambitious (If Ahead of Schedule)

Everything in Plan A, plus:

### Add-on B1: Order Book Attention Module (Week 3-4)

Replace flat LOB volume vectors with attention-based encoding:

```
[level_1, level_2, ..., level_K] (each = [price_offset, volume, queue_pos])
    -> Linear embedding per level
    -> Multi-Head Self-Attention (2 heads)
    -> Mean pooling -> LOB_embedding
    -> Concatenate with other features -> LSTM -> policy heads
```

- Reference: Attn-LOB (Guo et al. 2023), DeepLOB (Zhang et al. 2019)
- Estimated effort: ~100 lines
- Adds another row to ablation: LSTM+Attention vs LSTM-only

### Add-on B2: Logistic-Normal vs Beta Policy Comparison

- Implement Beta distribution policy head (as in Wang et al. 2024)
- Same architecture, only change the final distribution: Beta vs Logistic-Normal
- Directly addresses: "is our policy parameterization justified?"
- Estimated effort: ~80 lines (new agent class)

### Add-on B3: Curriculum Learning

- Train sequentially: `noise` (easy) -> `flow` (medium) -> `strategic` (hard)
- Transfer weights between stages
- Compare with training directly on `strategic`
- Estimated effort: training script changes only (~30 lines)

---

## Plan C: Moonshot (If Very Ahead or Extra Team Members)

### C1: Transformer Trunk (Replace LSTM)

- Use causal Transformer encoder over observation history window (H=10 steps)
- Positional encoding + multi-head self-attention
- Reference: TransLOB (Wallbridge 2020), TLOB (Berti et al. 2025), Decision Transformer
- Estimated effort: ~200 lines
- Risk: may need more training data/time to converge

### C2: Distributional RL (CVaR Optimization)

- Learn the full return distribution using Quantile Regression (QR-DQN style)
- Optimize CVaR instead of expected return -- natural for risk-sensitive market making
- Reference: Dabney et al. (2018), "Distributional RL for Stock Trading" (2025)
- Estimated effort: ~200 lines + significant training changes
- Risk: harder to combine with actor-critic; may need IQN or D4PG

### C3: Multi-Agent Competition

- Two independent RL market makers competing in the same LOB
- Study: emergent spread dynamics, inventory management under competition
- Reference: ABIDES-MARL (2024), Karpe et al. (2020)
- Estimated effort: significant simulator changes
- Risk: training instability, long convergence time

---

## Literature References

### Foundational

| # | Paper | Year | Key Technique | Relevance |
|---|-------|------|---------------|-----------|
| 1 | Cheridito & Weiss, "RL for Trade Execution with Market and Limit Orders" | 2025 | Logistic-Normal policy on simplex | **Direct basis of our project** |
| 2 | Avellaneda & Stoikov, "High-frequency trading in a limit order book" | 2008 | Stochastic optimal control for MM | Analytical baseline for comparison |
| 3 | Schulman et al., "Proximal Policy Optimization Algorithms" | 2017 | PPO clipped surrogate loss | Training algorithm we should actually use |

### LOB Feature Extraction (Architecture References)

| # | Paper | Year | Key Technique | Use Case |
|---|-------|------|---------------|----------|
| 4 | Zhang, Zohren & Roberts, "DeepLOB: Deep CNN for Limit Order Books" (IEEE TSP) | 2019 | CNN + Inception + LSTM | LOB spatial-temporal backbone |
| 5 | Wallbridge, "Transformers for Limit Order Books" (TransLOB) | 2020 | CNN + Transformer replacing LSTM | Alternative temporal encoder |
| 6 | Berti et al., "TLOB: Transformer with Dual Attention for LOB" | 2025 | Separate spatial/temporal attention | SOTA LOB representation |
| 7 | Guo, Ning & Gao, "Market Making with Deep RL from LOB" (Attn-LOB, IJCNN) | 2023 | CNN + Attention + continuous action MM | Closest to our attention upgrade |

### Deep RL Market Making

| # | Paper | Year | Key Technique | Use Case |
|---|-------|------|---------------|----------|
| 8 | Spooner et al., "Market Making via RL" (AAMAS) | 2018 | TD learning + tile coding for MM | Pioneer RL-MM paper |
| 9 | Sadighian, "Deep RL in Cryptocurrency Market Making" | 2019/2020 | PPO for crypto MM, 7 reward functions | PPO validation for MM |
| 10 | Gasperov & Kostanjcar, "Deep RL for MM under Hawkes LOB" (IEEE CSL) | 2022 | Hawkes process LOB + deep RL | Realistic order arrival model |
| 11 | Kumar, "Deep RL for High-Frequency MM" (ACML) | 2022 | DRQN (recurrent Q-network) + multi-agent | LSTM for MM temporal patterns |

### Continuous Action / Policy Distribution

| # | Paper | Year | Key Technique | Use Case |
|---|-------|------|---------------|----------|
| 12 | Wang et al., "Market Making with Learned Beta Policies" (ICAIF) | 2024 | Beta distribution over price levels | **Direct comparison to our Logistic-Normal** |
| 13 | "Continuous Action Policy for LOB Trading" (Neural Networks) | 2023 | Beta distribution for limit prices | Another continuous policy approach |

### PPO / Optimal Execution

| # | Paper | Year | Key Technique | Use Case |
|---|-------|------|---------------|----------|
| 14 | Lin & Beling, "End-to-End Optimal Execution with PPO" (IJCAI) | 2020 | PPO + LSTM on raw Level-2 LOB | **Architecture template: PPO+LSTM+LOB** |
| 15 | "Deep RL in Non-Markov Market-Making" (MDPI Risks) | 2025 | SAC + Hawkes Jump-Diffusion | Alternative to PPO (entropy-regularized) |

### Distributional / Risk-Sensitive RL

| # | Paper | Year | Key Technique | Use Case |
|---|-------|------|---------------|----------|
| 16 | Dabney et al., "Distributional RL with Quantile Regression" (AAAI) | 2018 | QR-DQN, full return distribution | CVaR optimization for tail risk |
| 17 | "Distributional RL for Stock Trading" (Applied Soft Computing) | 2025 | Risk preference adaptation | Tunable risk sensitivity |

### Transformer / LLM for Trading

| # | Paper | Year | Key Technique | Use Case |
|---|-------|------|---------------|----------|
| 18 | "Pretrained LLM + LoRA as Decision Transformer for Offline RL Trading" | 2024 | GPT-2 + LoRA fine-tuning | Frontier: LLM meets RL trading |
| 19 | "Financial Transformer RL (FTRL)" (Neurocomputing) | 2025 | Transformer for temporal + cross-asset | Transformer-based RL trading |

### Multi-Agent / Simulation

| # | Paper | Year | Key Technique | Use Case |
|---|-------|------|---------------|----------|
| 20 | Byrd et al., "ABIDES: High-Fidelity Multi-Agent Market Simulation" | 2020 | Large-scale LOB simulator | Alternative simulation framework |
| 21 | "ABIDES-MARL" | 2024 | Multi-agent RL equilibrium | Competitive MM training |
| 22 | Jerome et al., "mbt-gym: RL for Model-Based LOB Trading" (ICAIF) | 2023 | Gym-compatible LOB environments | Benchmarking with analytical baselines |
| 23 | Marin & Vera, "RL to Improve Avellaneda-Stoikov MM" (PLOS ONE) | 2022 | Double DQN tuning AS parameters | RL layered on classical MM |

### Maker-Taker Fee Structure (Domain References)

| # | Paper | Year | Key Technique | Use Case |
|---|-------|------|---------------|----------|
| 24 | Malinova & Park, "Subsidizing Liquidity: The Impact of Make/Take Fees on Market Quality" (J. Finance) | 2015 | Empirical analysis of fee structures | Justification for our fee mechanism |
| 25 | Colliard & Foucault, "Trading Fees and Efficiency in Limit Order Markets" (RFS) | 2012 | Theoretical model of maker-taker fees | Fee design rationale |

---

## Key Metrics for Evaluation

| Metric | What It Measures |
|--------|-----------------|
| Mean episodic return | Overall profitability |
| Std of returns | Consistency / stability |
| CVaR (5%) | Tail risk -- worst-case performance |
| Max inventory | Risk exposure |
| Market order ratio | Aggressiveness (should decrease with fees) |
| Spread capture rate | Passive income efficiency |
| Circuit breaker triggers | Extreme inventory events |
| Training wall-clock time | Practical efficiency |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LSTM training instability | Keep MLP as fallback; use LayerNorm + gradient clipping |
| Fee mechanism breaks reward scale | Normalize fees relative to typical spread (~0.1% of mid price) |
| Not enough training time for full ablation | Prioritize: PPO fix > fees > OFI > LSTM; drop lower rows |
| Colab GPU timeout | Use checkpointing (already implemented); reduce `num_envs` if needed |
| Team coordination conflicts | Each member owns one component; merge via PR |
