# Bilateral RL Market-Making: Optimization Summary (Phase 4)

## Project Overview
- **Goal:** Train a bilateral RL market-making agent to outperform a simple baseline (SymmetricFixedSpread) in a simulated limit order book environment.
- **Workflow:**
  1. Environment setup and dependency management
  2. Baseline agent implementation
  3. RL agent (PPO-based) implementation and training
  4. Evaluation of both agents on 1000 episodes
  5. Metrics comparison and visualization

## RL Agent Optimizations Attempted

### 1. Reward Shaping & Risk Controls
- **Market order penalty:** Penalized market order usage to reduce aggressive trading.
- **Variance penalty:** Added penalty to loss for high action variance (to reduce erratic behavior).
- **Safety bonus:** Rewarded actions close to the baseline allocation.
- **Action projection:** Projected actions onto a safer simplex region (market/inactive caps).

### 2. PPO Training Enhancements
- **Clipped returns:** Clamped returns to avoid extreme gradients from outliers.
- **Advantage normalization:** Standardized advantages for stable updates.
- **Entropy annealing:** Decreased entropy bonus over time for more exploitation.
- **Dynamic penalty:** Increased market order penalty as training progressed.
- **Early episode cutoff:** Terminated episodes early if cumulative reward dropped below a threshold (to avoid catastrophic tail events).
- **Blending with baseline:** Blended RL actions with baseline actions (weighted average) to stabilize learning.
- **Checkpointing:** Only saved agent checkpoints if outlier rate (r < -200) was below 1%.

### 3. Evaluation & Diagnostics
- **1000-episode evaluation** for both RL and baseline agents, using identical seeds for fair comparison.
- **Comprehensive statistics:** Mean, std, min, max, and outlier rates for returns and terminal inventory.
- **Visualization:**
  - Return distributions
  - Cumulative returns
  - Training curves
  - Inventory boxplots

## Results (Most Recent Run)
- **Bilateral RL Agent:**
  - Mean return: -35.26
  - Std deviation: 79.62
  - Min return: -2180.35
  - Max return: -22.30
  - Terminal inventory: 40.00
  - Outlier rate (r < -500): 0.2%
- **Baseline Agent:**
  - Mean return: -31.53
  - Std deviation: 2.45
  - Min return: -37.98
  - Max return: -24.20
  - Terminal inventory: 40.00
  - Outlier rate (r < -500): 0.0%
- **Performance gap:** RL agent underperforms baseline by 3.73 (mean), with much higher variance and rare catastrophic losses.

## Lessons Learned
- RL agent can match or slightly outperform baseline on median, but rare catastrophic losses dominate the mean.
- Risk controls and reward shaping reduce outlier frequency, but not their magnitude.
- Blending with baseline and early cutoff help, but do not fully solve tail risk.
- Further improvements may require:
  - Stronger catastrophe clipping
  - Reward normalization
  - Inventory penalties
  - Curriculum learning
  - More robust exploration/exploitation strategies

## Next Steps for Future LLMs
- Try additional regularization (catastrophe clipping, reward normalization, inventory penalty).
- Experiment with curriculum training (start easy, increase difficulty).
- Consider ensemble or meta-learning approaches for baseline blending.
- Tune PPO hyperparameters and batch sizes.
- Investigate alternative RL algorithms or architectures.

**All code, results, and plots are in `bilateral_mm_phase4.ipynb`.**
**Last run: None of the cells were executed in the current session—run all cells to reproduce results.**
