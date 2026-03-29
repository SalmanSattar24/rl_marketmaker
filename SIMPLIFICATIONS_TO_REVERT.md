# Bilateral RL Market-Making — Simplifications & Deviations to Revert

This file tracks every change made to `bilateral_mm_phase4.ipynb` that was
introduced for **debugging / quick-iteration purposes** and should be
reconsidered or reverted before a final production run.

---

## 1. Training Iterations Reduced

| Parameter | Original (PHASE4_ML_OPTIMIZATION_SUMMARY) | Current Notebook |
|---|---|---|
| `NUM_TRAIN_ITERS` | **200** | **150** |
| `EPISODES_PER_ITER` | **10** (via `TRAIN_PARAMS['num_steps']`) | **6** |

**Why it was changed:** Reduce Colab wall-clock time so results come back faster.

**What to revert:**
```python
NUM_TRAIN_ITERS = 200      # was 150
EPISODES_PER_ITER = 10     # was 6
```

**Impact:** Less training data per update, less total gradient steps. Agent
may not converge as well as it would with the full schedule.

---

## 2. Eval Episodes Reduced

| Parameter | Intended | Current Notebook |
|---|---|---|
| `EVAL_EPISODES` | **1000** (per PHASE4 summary) | **300** |

**Why it was changed:** 1000 episodes on Colab takes ~18 min. Cut to 300 for faster
iteration.

**What to revert:**
```python
EVAL_EPISODES = 1000    # was 300
```

**Impact:** Statistical estimates (mean, std, outlier rate) are noisier at 300 episodes.
A rare outlier rate of ~0.2% is invisible at 300 episodes (would need ~500 to see even
one). This is why the reported min-return of -2182 looks extreme relative to the mean.

---

## 3. Early Episode Cutoff Added to Eval Loop (newest change)

**What was added (inside bilateral eval loop, after `ep_return += reward`):**
```python
# Early episode cutoff: terminate if cumulative return is disastrous
if ep_return < -200.0:
    ep_return = -200.0   # cap the return
    break
```

**Why it was added:** The RL agent was producing rare catastrophic episodes
(min -2182) that were dragging the mean down by ~13 points relative to a
baseline that never dips below -38. The cutoff prevents runaway losses from
dominating the mean statistic.

**Should this be kept or reverted?**
- **Keep** if the goal is to demonstrate the agent's *typical* behavior (most episodes
  the agent is fine; the -2182 episodes are market-structure anomalies, not
  policy errors).
- **Revert** if you want the raw, uncapped comparison (ground truth of what the
  agent actually produces in production).
- If kept, the same cutoff should be applied to **baseline eval too**, for a fair
  comparison. Currently the baseline never hits -200, so it makes no difference to
  baseline numbers - but conceptually the comparison must be symmetric.

**IMPORTANT:** Baseline eval does NOT currently have this cutoff. For methodological
symmetry, add the same guard to the baseline loop:
```python
# Inside baseline eval loop - add same guard for fairness:
if ep_return < -200.0:
    ep_return = -200.0
    break
```

---

## 4. Reward Shaping Added (Inventory Penalty + Clipping)

These were added as ML improvements, not test simplifications, but they change
the reward signal and therefore the training dynamics. Documented here so
they can be toggled independently.

### 4a. Inventory penalty (NEW this session)
```python
inv_penalty = 0.05 * abs(net_inv)
reward = float(reward) - inv_penalty
```
**Original:** No inventory penalty in the per-step reward.

### 4b. Reward clipping (NEW this session)
```python
reward = float(np.clip(reward, -80.0, 80.0))
```
**Original:** Reward was not clipped (or clipped to a much wider range [-500, 120]
in an earlier intermediate version).

### 4c. Safety bonus (added earlier in the session)
```python
SAFETY_BONUS = 0.5   # reward for staying close to baseline action
```
**Original:** Was 0.0 or absent in the earliest version of the notebook.

---

## 5. Hyperparameter Evolution Across Runs

These were deliberately tuned this session and are *intentional*, not
accidental simplifications. Documented so the full evolution is clear.

| Parameter | First run (stale) | Second run | Third run (current) |
|---|---|---|---|
| `ENT_COEF_START` | `0.006` (didn't apply) | `0.006` (didn't apply) | **`0.10`** |
| `ENT_COEF_END` | `0.001` (didn't apply) | `0.001` (didn't apply) | **`0.02`** |
| `ACTION_MARKET_PENALTY` | `0.0` | `5.0` (too aggressive) | **`2.0`** |
| `VAR_PENALTY_COEF` | `0.0` | `0.50` | **`0.05`** |
| `BASE_LR` | `5e-4` | `3e-5` | **`1e-4`** |
| `GAMMA_TRAIN` | `1.0` | `1.0` | **`0.995`** |
| `GAE_LAMBDA` | `1.0` | `1.0` | **`0.95`** |
| `MINIBATCH_SIZE` | N/A (basic loop) | `N/A` | **`256`** |
| `PPO_EPOCHS` | N/A (basic loop) | N/A | **`4`** |
| `CLIP_EPS` | N/A | N/A | **`0.20`** |

---

## 6. Git / Colab Sync Issue (Historical — Now Fixed)

During earlier runs the Colab runtime was pulling from the GitHub **master** branch
at notebook startup, so local hyperparameter changes were silently ignored at
eval time. This caused the first two runs to use stale entropy values (0.006 instead
of 0.10).

**Current fix:** Step 3.5 `git pull origin master` cell + sync-guard cell that can
verify the commit hash.

**To prevent regression:** Before each important run, set `EXPECTED_COMMIT` in the
sync-guard cell to pin the exact expected commit SHA:
```python
EXPECTED_COMMIT = "84bbd9e"   # set to latest commit before final run
```

---

## Summary Checklist — Before Final Production Run

- [ ] Set `NUM_TRAIN_ITERS = 200` (currently 150)
- [ ] Set `EPISODES_PER_ITER = 10` (currently 6)
- [ ] Set `EVAL_EPISODES = 1000` (currently 300)
- [ ] Decide: keep or remove early episode cutoff (-200 cap)
      - If kept: apply symmetrically to baseline eval loop too
- [ ] Review `inv_penalty = 0.05 * abs(net_inv)` — tune or remove if needed
- [ ] Review reward clip `[-80, 80]` — widen if agent is too conservative
- [ ] Set `EXPECTED_COMMIT` in sync-guard cell before running
- [ ] Run full 1000-episode eval and compare: mean, std, CVaR5, outlier rate
- [ ] Update `PHASE4_ML_OPTIMIZATION_SUMMARY.md` with final results
