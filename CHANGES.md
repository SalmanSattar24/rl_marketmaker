Changes made during audit and fixes

- 2026-04-15: Prevent RLAgent premature termination and ensure deterministic terminal closeout
  - Added `RLAgent.update_position_from_message_list` override to disable volume-based auto-termination for RL agents.
  - Fixed `ExecutionAgent.sell_remaining_position` to cancel resting orders and handle zero/negative net volumes correctly (flatten only when net inventory != 0).
  - Centralized observation sanitization in `RLAgent._finalize_observation`.
  - Adjusted PPO training default hyperparameters for stability in `rl_files/ppo_continuous_action.py` (clip, update_epochs, max_grad_norm).

Notes on AUTORESEARCH.md compliance
- The logistic-normal policy implementation was preserved; no changes were made to policy parameterization or action-transform code.
- The `Reinforcement Learning for Trade Execution with Market and Limit Orders` paper (arXiv:2507.06345) was reviewed to align termination and inventory handling with the literature. See repository tests and `tests/test_batch1_terminal_closeout_reward.py` for validation of terminal closeout behavior.

Testing
- Ran full pytest suite: all tests pass (77 passed) after the fixes.

If you want, I can now:
- Continue tuning PPO hyperparameters and run small training experiments (quick rollouts) to validate learning stability.
- Add a small validation script that runs a few training iterations and plots summary metrics.
- Incorporate the full PDFs into the repo (`docs/`) and write a short implementation rationale linked to the papers.
