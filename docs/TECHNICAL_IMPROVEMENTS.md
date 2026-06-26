# Technical Improvements Roadmap

## Current state

The bilateral pipeline is now **stable and test-validated**, but some behavior can still rely on fallback or approximate paths in edge cases.

This document outlines the highest-leverage next improvements to make the system more faithful to true bilateral market-making under realistic stress conditions.

---

## 1) Make bilateral order generation fully end-to-end

### Goal

Ensure both **bid** and **ask** decisions are honored at every step in order placement and lifecycle handling.

### What this means in practice

- Use the policy output for both sides consistently in execution logic.
- Avoid silent fallback to unilateral behavior unless explicitly configured.
- Handle edge cases (empty book side, cancellation conflicts, partial fills) without degrading to one-sided behavior.

### Why this matters

- The policy is trained as bilateral; execution should match training assumptions.
- Prevents hidden mismatch between learned strategy and live simulator behavior.
- Improves interpretability of training results and debugging.

---

## 2) Add targeted stress tests for inventory drift

### Goal

Validate inventory stability under asymmetric and adverse fill conditions.

### Suggested scenarios

- Repeated buy-side fills with weak/no sell-side fills.
- Repeated sell-side fills with weak/no buy-side fills.
- Volatility bursts where one side gets hit repeatedly.
- Thinner liquidity on one side of the book.

### Test expectations

- Inventory remains bounded or returns toward target ranges.
- No unbounded drift under prolonged asymmetry.
- Risk controls remain active and effective throughout.

---

## 3) Validate circuit-breaker and terminal closeout under skewed pressure

### Goal

Prove that risk controls work correctly in worst-case paths, not just average paths.

### Required checks

- Circuit breaker triggers at configured inventory thresholds.
- Environment termination behavior is deterministic and explainable.
- Terminal closeout consistently flattens inventory and cancels stale resting orders.
- Reward decomposition remains coherent during forced exits.

### Why this matters

- Stress behavior is where production risk actually appears.
- Correct emergency behavior protects strategy integrity.

---

## Why this roadmap is high leverage

1. **Improves realism and risk fidelity**
	- Behavior aligns more closely with real bilateral MM constraints.

2. **Finds subtle bugs early**
	- Stress-focused tests reveal issues that average-case tests often miss.

3. **Increases confidence in training-to-execution consistency**
	- Reduces gap between what the model learns and how the simulator executes.

4. **Strengthens future iteration speed**
	- Better guarantees and diagnostics mean faster safe experimentation.

---

## Recommended implementation approach

Use a **tight test-first patch series**:

1. Write failing scenario tests for bilateral execution parity and asymmetric inventory pressure.
2. Implement minimal code changes to satisfy each scenario.
3. Re-run full tests after each patch.
4. Add brief release-note entries per patch to keep change intent clear.

This keeps risk low while steadily increasing correctness and production readiness.
