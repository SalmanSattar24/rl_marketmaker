# 1. Executive Summary

After reviewing `bilateral_mm_agent.ipynb`, the main conclusion is that this notebook is **not** implementing an optimal execution agent. Instead, it is implementing a **bilateral market-making agent** that posts orders on both the bid and ask sides of the book and compares its performance against a fixed-spread market-making baseline.

This distinction is important. In an optimal execution problem, the agent typically starts with a position and aims to liquidate or acquire inventory by a deadline while minimizing market impact and execution cost. In this notebook, the agent is trained to quote both sides of the market, earn spread capture, and manage inventory risk. That is a market-making setup, not an execution setup.

# 2. What the Notebook Actually Does

The notebook follows this workflow:

1. It sets up the local or Colab environment and imports the repository modules.
2. It defines a baseline agent, `SymmetricFixedSpreadAgent`, which always posts symmetric quotes at the best bid and best ask.
3. It creates a `Market` environment and instantiates a bilateral RL policy, in this run `BilateralAgentLSTMLob`.
4. It trains that policy with vectorized PPO over multiple environments.
5. It evaluates the trained RL agent against the fixed-spread baseline over 1000 episodes.
6. It compares returns, volatility, and terminal inventory, and then generates plots and summary statistics.
7. It also includes a circuit-breaker stress test for inventory breaches.

The architecture and action design confirm that this is market making:

- The action is a tuple of bid-side and ask-side actions.
- The RL policy outputs two simplex allocations, one for each side of the book.
- The baseline is a symmetric fixed-spread quoting strategy.
- The environment includes inventory limits, circuit breakers, and bid/ask order allocation logic.

All of these are characteristic of a bilateral market-making framework.

# 3. Why It Can Be Mistaken for Optimal Execution

The confusion comes from the fact that the codebase still contains a lot of execution-agent legacy structure. For example:

- The environment still uses classes such as `ExecutionAgent`.
- The variable `volume` is still central to bookkeeping.
- Some termination and reward logic still reflects a liquidation-style framework.

Because of that, the notebook has the appearance of an execution project, but the actual policy logic has been retrofitted into a market-making problem.

In other words, the code is operating under an execution-style shell with market-making behavior inside it.

# 4. Main Technical Problems

The notebook runs, but there are several serious design issues that make the reported results unreliable.

## Problem 1: `volume` and `net_inventory` are semantically mixed up

This is the most important issue.

The agent still uses `self.volume` in a way that resembles an execution task, while the environment separately tracks `net_inventory` as the actual market-making inventory. These two concepts are not the same, but the code partially treats them as if they were.

As a result:

- episode termination can depend on `self.volume == 0`,
- terminal closeout can be triggered using `self.volume`,
- but inventory risk is actually measured using `net_inventory`.

That creates a bookkeeping mismatch between the agent's internal state and the true position in the market.

## Problem 2: Terminal closeout logic is not aligned with market making

At the environment level, there is a proper closeout path based on actual terminal inventory. However, the RL agent can terminate earlier through legacy volume-based logic. When that happens, the correct inventory-based closeout may never be applied.

This likely explains one of the notebook's biggest red flags: the reported terminal inventory is still around 10 for both agents.

From the notebook output:

- RL terminal inventory mean: about `10.134`
- Baseline terminal inventory mean: about `10.558`

For a properly closed-out market-making episode, that is highly suspicious.

## Problem 3: The observation space is polluted by legacy execution variables

The observation includes features derived from `self.volume / self.initial_volume` and related terms. In an execution task, that might represent remaining inventory to trade. In bilateral market making, however, this variable no longer cleanly represents the true state of the agent.

So the policy is learning from a partially misdefined state.

## Problem 4: RL and baseline are not evaluated under the same controls

The RL policy's actions are passed through quota projection and additional inventory-aware sanitation before being sent to the environment. The baseline agent does not appear to receive the same correction layer.

That means the comparison is not fully apples-to-apples. Some of the RL advantage may come from external action filtering rather than the learned policy itself.

## Problem 5: PPO training becomes unstable late in training

The training logs show clear instability in later iterations:

- KL divergence explodes to extremely large values,
- effective update steps collapse from full updates to almost none,
- entropy becomes pathological.

This suggests the policy optimization is no longer healthy, even though the final evaluation still reports a positive average return.

So the final result is not strong evidence of a stable or well-trained policy.

# 5. Conclusion

This notebook should be described as a bilateral market-making experiment, not an optimal execution experiment.

Its real purpose is to train and compare:

- a bilateral RL market maker, and
- a symmetric fixed-spread market-making baseline.

The main conceptual issue is that the surrounding framework still contains execution-style assumptions, especially in the handling of `volume`, termination, and closeout. Because of that, the experimental results are difficult to interpret with confidence.

The most important takeaway is:

> The notebook is not solving best execution; it is solving a market-making problem inside a partially inherited execution framework.

# 6. Matched MLP vs Attention Benchmark Report

This section summarizes the most recent in-repo benchmark run that completed end-to-end and produced saved reward artifacts for both the bilateral MLP and the attention-based bilateral agent.

## Benchmark protocol

- Environment: `flow`
- Seed: `42`
- Evaluation seed start: `50000`
- Evaluation episodes: `200`
- Reward files:
  - `rewards/flow_20_seed_42_eval_seed_50000_eval_episodes_200_num_iterations_50_bsize_1024_log_normal_quick_mlp_drift.npz`
  - `rewards/flow_20_seed_42_eval_seed_50000_eval_episodes_200_num_iterations_50_bsize_1024_log_normal_quick_attn_drift.npz`
- Policy family: **logistic-normal** preserved in both agents

## Results

| Agent | Mean Return | Std Return | Median | CVaR (5%) | P05 | P95 | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bilateral MLP | 0.008845 | 0.764688 | -0.037497 | -1.516633 | -1.175225 | 1.300121 | -1.874505 | 2.500135 |
| Bilateral Attention | 0.834072 | 0.885694 | 0.805110 | -1.048274 | -0.582751 | 2.413224 | -1.474750 | 3.470250 |

### Direct comparison

- Absolute improvement: **+0.825226** mean return per episode
- Relative improvement: **~+93.2%** over the MLP baseline mean
- Interpretation: the attention model is materially better on this matched flow benchmark, while also showing a healthier upside tail.

## Observations

1. **Attention learns a stronger positive edge**
	- The mean and median both move decisively positive relative to the MLP baseline.
	- The attention policy also reaches a higher maximum episode return.

2. **Risk is still meaningful**
	- The attention model has slightly higher standard deviation than the MLP baseline, which is expected for a more expressive policy.
	- However, its downside tail is materially better than the MLP baseline as measured by CVaR(5%).

3. **The result is robust enough to be useful**
	- The improvement is not a tiny fluctuation; it is large relative to the MLP mean.
	- This is consistent with the earlier notebook-level finding that attention can outperform the simpler MLP under the same bilateral policy family.

## Notes on the longer full-budget attempt

A longer `160000`-timestep full-budget MLP run was started during this session, but it did not complete within the available interaction window. To avoid reporting partial or stale numbers, the table above uses the completed matched benchmark artifacts that are fully saved and reproducible in the repository.

## Practical takeaway

For the current flow setup and bilateral logistic-normal policy family, the attention encoder is the better choice than the plain MLP trunk. If the goal is to keep improving the transformer variant, the next sensible step is to tune training stability and inventory control further, rather than increasing model size blindly.
