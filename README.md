# rl_marketmaker

`rl_marketmaker` is a research-oriented reinforcement learning project for **limit-order-book (LOB) market making and trade execution**.  
It combines a custom microstructure simulator, Gym-style environments, baseline agents, bilateral RL policy code, and reproducible tests.

The goal is to learn (and rigorously validate) market-making behavior that can place and manage orders on **both bid and ask sides** while controlling inventory and execution risk.

---

## What this project is about

At a high level, this project studies the following problem:

- An agent interacts with a simulated LOB over a finite horizon.
- At each decision time, it receives a state vector (market features + inventory features).
- It chooses how to allocate order flow across market/limit/inactive buckets (unilateral or bilateral form depending on setup).
- The environment executes those orders, updates inventory/cash/PnL, and returns reward.
- The policy is optimized with policy-gradient style RL loops.

This repository is structured to support both:

1. **Execution-focused settings** (sell program / liquidation style behavior), and
2. **Bilateral market-making settings** (simultaneous buy/sell quoting and inventory control).

---

## Project status and progress so far

This project has gone through a major hardening pass and is currently at a stable checkpoint.

### ✅ Stability checkpoint completed (`v0.2-stable-batch-fixes`)

Recent validated work includes:

1. **Terminal closeout + reward decomposition**
   - Added deterministic terminal inventory closeout path.
   - Added explicit reward decomposition fields:
     - realized component
     - inventory component
     - terminal component

2. **Dynamic observation sizing + optional OFI**
   - Replaced brittle hardcoded observation-size assumptions.
   - Observation size is now derived from active feature blocks/config.
   - Added optional OFI feature support (`use_ofi`) with runtime mismatch guard.

3. **Codebase cleanup**
   - Removed stale one-off fix scripts and temporary markdown artifacts.
   - Replaced legacy `read-me.md` with this structured `README.md`.

4. **Test hygiene improvements**
   - Cleaned tensor construction warnings in integration tests
     (`np.stack(...)` + `torch.as_tensor(...)`).

### ✅ Current verified quality gate

- Full repository test suite result at this checkpoint: **29 passed**.
- Stable tag available: **`v0.2-stable-batch-fixes`**.

---

## Technical architecture (for CS readers)

### 1) Simulator / microstructure layer

- Core implementation: `limit_order_book/limit_order_book.py`
- Supports order submission, cancellation, matching, and message-based position updates.
- Provides best bid/ask and book state needed by environment features.

### 2) Environment layer

- Main environment: `simulation/market_gym.py` (Gym-compatible API)
- Handles:
  - event stepping
  - agent dispatch
  - reward calculation
  - episode termination/truncation
  - inventory/risk constraints

Risk and end-of-episode mechanics now include deterministic closeout and explicit reward-component tracking for better interpretability.

### 3) Agent layer

- Agent definitions: `simulation/agents.py`
- Includes baseline-style behavior and RL-driven behavior.
- RL agent supports feature toggles and bilateral action handling pathways.

### 4) Learning / optimization layer

- RL code in `rl_files/` (policy/value models, training/eval scripts).
- Bilateral policies and policy-gradient loops are exercised via scripts and notebooks.
- Integration tests validate action sampling, shape assumptions, and forward/backward pass stability.

### 5) Validation layer

- `tests/` provides regression + integration coverage for environment, action-space behavior, and training-path consistency.

---

## Key technical concepts used

- **Finite-horizon MDP** over LOB states
- **Policy-gradient RL** (actor-critic style code paths)
- **Action simplex projection/allocation** for order-flow distribution
- **Inventory-aware state and reward design**
- **Risk controls** (inventory caps, circuit-breaker-like termination behavior)
- **Deterministic closeout semantics** at terminal time

---

## Repository structure

- `config/` — runtime and model configuration
- `initial_shape/` — initial LOB shape assets (`.npz`)
- `limit_order_book/` — LOB engine + plotting + LOB-level tests
- `simulation/` — market environment and agent logic
- `rl_files/` — RL models, training, evaluation utilities
- `tests/` — end-to-end and regression tests
- `notebooks/` + `bilateral_mm_agent.ipynb` — experiment and analysis workflows
- `RELEASE_NOTES.md` — release-style summary of stable checkpoint updates
- `TECHNICAL_IMPROVEMENTS.md` — roadmap of highest-leverage next improvements

---

## Setup

Recommended Python version: **3.9+**.

Install dependencies from repository root:

- `requirements.txt`

If using a virtual environment, activate it before running scripts/tests.

---

## Running tests

From repository root:

- `python -m pytest -q tests`

Expected stable checkpoint result: **29 passed**.

---

## Typical workflows

1. Run/iterate on simulator and agent logic in `simulation/` and `limit_order_book/`.
2. Train/evaluate policies via `rl_files/` scripts and notebook workflows.
3. Validate changes with `pytest` before committing.
4. Track user-facing progress in `RELEASE_NOTES.md` and future work in `TECHNICAL_IMPROVEMENTS.md`.

---

## Near-term roadmap

See `TECHNICAL_IMPROVEMENTS.md` for details. The highest-leverage next work is:

- fully end-to-end bilateral order generation (no hidden unilateral fallback paths), and
- asymmetric inventory-drift stress testing with circuit-breaker/closeout verification.

---

## Acknowledgments and attribution

This project builds on prior academic and open-source work.

- **Research foundation**: The design direction, terminology, and several experimental choices in this repository are informed by the trade-execution literature, including the paper referenced during this project hardening cycle:
   - Cheridito & Weiss (2026), *Reinforcement Learning for Trade Execution with Market and Limit Orders* (arXiv:2507.06345).

- **Base repository**: This codebase was developed using `moritzweiss/rlte` as an initial base/reference:
   - https://github.com/moritzweiss/rlte

Where behavior diverges from the upstream/base implementation, this repository documents changes in `RELEASE_NOTES.md` and related test updates.

---

## Notes

- Active branch: `master`
- GitHub remote: `SalmanSattar24/rl_marketmaker`
