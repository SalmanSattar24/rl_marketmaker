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
