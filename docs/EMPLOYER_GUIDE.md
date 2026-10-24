# Employer guide

## The engineering problem

Model a finite-horizon agent interacting with a limit order book while tracking cash, inventory, execution and terminal closeout. The repository connects simulator mechanics with Gym-style environments and learning code.

## Reading order

1. [Book engine](../limit_order_book/limit_order_book.py): submission, matching and cancellation mechanics.
2. [Environment](../simulation/market_gym.py): observations, stepping and episode accounting.
3. [Agents](../simulation/agents.py): baseline and learned action paths.
4. [RL code](../rl_files/): policy/value models and training/evaluation.
5. [Tests](../tests/): terminal reward decomposition, dynamic observations, bilateral actions and training-path fixtures.

## Setup and evidence

The declared development path uses `requirements.txt` and `python -m pytest -q tests`. The README's 29-passing-tests result belongs to its documented stable checkpoint; this documentation refresh did not rerun the simulator or learning suite.

The roadmap explicitly distinguishes partially integrated bilateral behavior from fully end-to-end bilateral execution. Simulator quality and passing tensor-shape tests are not evidence of a profitable strategy or real-world fill realism.

## Skills to discuss

State/action design, environment invariants, inventory accounting, reproducible regression tests, simulator–policy interfaces and experiment limitations. This is research software informed by academic work credited in the existing README, not investment advice or a live trading service.
