# rl_marketmaker

Reinforcement-learning market-making and trade-execution simulator with a custom limit order book, Gym-compatible environments, baseline agents, and bilateral policy training utilities.

## Project overview

This repository includes:

- A custom LOB simulator (`limit_order_book/`)
- Market environments and agents (`simulation/`)
- RL model/training code (`rl_files/`)
- Notebook workflows for experiments and diagnostics (`notebooks/`, `bilateral_mm_agent.ipynb`)
- Regression and integration tests (`tests/`)

## Environment setup

Recommended Python version: **3.9+** (project has also been run with newer versions in this workspace).

Install dependencies from the repository root:

- `requirements.txt`

If using a virtual environment, activate it before running commands.

## Repository structure

- `config/` — runtime configuration
- `initial_shape/` — initial order-book shape assets (`.npz`)
- `limit_order_book/` — LOB engine and plotting helpers
- `simulation/` — environment dynamics and agents
- `rl_files/` — policy/value models and RL utilities
- `tests/` — automated test suite
- `notebooks/` and `bilateral_mm_agent.ipynb` — analysis/training notebooks

## Running tests

Run all tests from repository root:

- `python -m pytest -q tests`

Current validated checkpoint result: **29 passed**.

## Key recent changes (v0.2 stable batch fixes)

1. **Terminal closeout + reward decomposition**
   - Deterministic terminal inventory closeout in market environment
   - Reward decomposition tracked as realized/inventory/terminal components

2. **Dynamic observation sizing + OFI toggle**
   - RL observation size computed from active feature blocks and config
   - Optional OFI feature (`use_ofi`) integrated with shape guards

3. **Cleanup of stale files**
   - Removed obsolete patch utility scripts and temporary summary docs

4. **Integration-test warning cleanup**
   - Replaced slow tensor construction from list-of-ndarrays with `np.stack(...)` + `torch.as_tensor(...)`

## Typical workflows

- Train/evaluate RL agents via `rl_files/` scripts and/or notebook workflows
- Run environment-level logic via `simulation/market_gym.py`
- Analyze performance with notebook outputs and plotting helpers

## Notes

- This repo is currently on `master`.
- Remote repository name has been updated to: `SalmanSattar24/rl_marketmaker`.
