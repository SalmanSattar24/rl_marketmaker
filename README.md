# RL Market Maker

**English** | [中文](#中文說明)

A research implementation of a **bilateral reinforcement learning market maker** on a simulated limit order book (LOB).
The agent simultaneously quotes on both bid and ask sides, manages inventory risk, and is trained end-to-end with Proximal Policy Optimization (PPO).

> **Course**: CSCI 566 — Deep Learning and its Applications
> **Institution**: University of Southern California
> **Reference paper**: Cheridito & Weiss (2026), *Reinforcement Learning for Trade Execution with Market and Limit Orders* (arXiv:2507.06345)
> **Base repository**: [moritzweiss/rlte](https://github.com/moritzweiss/rlte)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Running the Notebook](#running-the-notebook)
- [Running Tests](#running-tests)
- [Recent Changes](#recent-changes)
- [Roadmap](#roadmap)
- [中文說明](#中文說明)

---

## Overview

At each timestep the agent:

1. Observes a state vector (LOB features + inventory + OFI)
2. Outputs a **Logistic-Normal** action — allocating order volume across price levels on both bid and ask sides
3. Receives reward based on PnL, inventory risk, and maker-taker fees
4. Policy is optimized with **standard PPO** (clipped surrogate loss + gradient clipping)

The simulator supports three market regimes: `noise`, `flow`, and `strategic`.

---

## Key Features

| Feature | Details |
| --- | --- |
| **Bilateral quoting** | Agent posts simultaneous bid and ask limit orders |
| **Logistic-Normal policy** | Continuous action mapped to order-allocation simplex |
| **Maker-taker fee structure** | `maker_rebate=0.2`, `taker_fee=0.3` (reward units, calibrated to ~1bp real spread) |
| **Standard PPO** | Clipped surrogate loss, 4 epochs, 4 minibatches, gradient clipping (`max_grad_norm=0.5`) |
| **OFI feature** | Order Flow Imbalance included as optional observation feature (`use_ofi=True`) |
| **Parallel training** | 128 parallel `SyncVectorEnv` environments |
| **TensorBoard logging** | Loss, KL divergence, clip fraction, episode return |
| **29 passing tests** | Full regression suite covering environment, agent, and training path |

---

## Architecture

```text
Observation (43-dim)
    ├── LOB features: bid/ask volumes at N price levels (normalized by initial shape)
    ├── Inventory features: current volume, active volume, time-weighted inventory
    ├── Market features: spread, mid-price drift
    └── OFI (optional): order flow imbalance

        ↓
┌─────────────────────────────────┐
│   Shared MLP Trunk (128 units)  │
│   + LayerNorm                   │
└──────────┬──────────────────────┘
           │
    ┌──────┴──────┐
    ↓             ↓
Bid Head       Ask Head        Value Head
(Logistic-    (Logistic-       (scalar)
 Normal)       Normal)

Action = (bid_allocation, ask_allocation)   # each is a 7-dim simplex
```

**Reward function:**

```text
r_t = PnL(fills) + maker_rebate × passive_fill_vol / V₀
                 - taker_fee   × aggressive_fill_vol / V₀
                 - inventory_penalty × |inventory|
```

---

## Repository Structure

```text
rl_marketmaker/
├── bilateral_mm_agent.ipynb   # Main experiment notebook
├── config/
│   ├── config.py              # All agent/env/fee configs
│   └── __init__.py
├── limit_order_book/
│   └── limit_order_book.py    # LOB engine (order matching, cancellation, book state)
├── simulation/
│   ├── market_gym.py          # Gym-compatible environment
│   └── agents.py              # All agent classes (RL, baseline, noise, strategic…)
├── rl_files/
│   └── actor_critic.py        # PPO training loop + BilateralAgentLogisticNormal model
├── initial_shape/             # Initial LOB shape arrays (.npz)
├── tests/                     # Regression and integration tests
├── requirements.txt
├── changes_report.tex         # Overleaf-ready PDF report of all code changes
└── FINAL_PROJECT_PLAN.md      # Project planning document
```

---

## Setup

**Recommended Python**: 3.9 – 3.14

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `gymnasium`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tensorboard`, `tyro`, `sortedcontainers`

---

## Running the Notebook

Open `bilateral_mm_agent.ipynb` in JupyterLab or VS Code.
The notebook runs fully locally — no Google Colab or Google Drive required.

**Sections:**

1. Environment setup and dependency check
2. Repository verification
3. Config — dual-regime (flow → strategic)
4. Agent initialization (bilateral RL + baseline fixed-spread)
5. Vectorized quota projection
6. **Training** — PPO with 128 parallel environments
7. **Evaluation** — RL agent vs baseline comparison
8. Visualization — PnL curves, action distributions, fee impact

> Note: Full training (200 × 128 × 100 = 2.56M timesteps) takes several hours on CPU. For quick testing, reduce `total_timesteps` in `Args`.

---

## Running Tests

```bash
python -m pytest -q tests/
```

Expected result: **29 passed**

Key test files:

| File | Coverage |
| --- | --- |
| `test_bilateral_simulator.py` | End-to-end bilateral episode |
| `test_bilateral_action_space.py` | Action sampling and simplex validity |
| `test_maker_taker_fees.py` | Fee accounting correctness |
| `test_bugfix_audit.py` | Regression checks for known fixed bugs |
| `test_phase2_integration.py` | Full environment integration |
| `test_phase3_training.py` | Training loop forward/backward pass |

---

## Recent Changes

### v0.3 — Maker-Taker Fees + PPO Upgrade (April 2026)

**`config/config.py`**

- Added `fee_config` with `maker_rebate=0.2` and `taker_fee=0.3`
- Fees calibrated to ~1bp real spread at reference price 1000, volume 20

**`simulation/agents.py`**

- `ExecutionAgent.set_fees()`: injects fee rates from environment
- Taker fee deducted on market order fills (both buy and sell)
- Maker rebate credited on passive limit order fills
- Removed assertions incompatible with bilateral short positions

**`simulation/market_gym.py`**

- Fee config propagated into each episode via `set_fees()`
- Episode info dict now reports `cumulative_fees_paid`, `cumulative_rebates_earned`, `net_fee_impact`
- Removed `assert volume == 0` at episode end (bilateral agent can hold non-zero inventory)

**`limit_order_book/limit_order_book.py`**

- Fixed stale order-ID tracking: cancellations now correctly deregister from `agent_bid_orders` / `agent_ask_orders`

**`rl_files/actor_critic.py`**

- Restored production hyperparameters: `num_envs=128`, `num_steps=100`, `num_minibatches=4`, `update_epochs=4`
- Standard PPO clipped surrogate loss (Schulman et al. 2017)
- Gradient clipping: `max_grad_norm=0.5`
- KL divergence tracking + TensorBoard logging (`approx_kl`, `clipfrac`)
- Switched from `AsyncVectorEnv` to `SyncVectorEnv` (fixes info-dict format mismatch)
- Added `get_value()` helper to `AgentLogisticNormal`

---

## Roadmap

- [ ] LSTM temporal backbone (replace MLP trunk with LSTM for sequential LOB modeling)
- [ ] OFI ablation experiments (`use_ofi=True` vs `False` comparison)
- [ ] Asymmetric fee ablation (no fee / symmetric / maker-taker)
- [ ] Action distribution analysis (market order ratio over training)
- [ ] Full bilateral order generation without unilateral fallback paths

---

## 中文說明

**[English](#rl-market-maker)** | 中文

本專案實作了一個基於**強化學習的雙邊做市商（Bilateral Market Maker）**，在模擬的限價委託簿（Limit Order Book, LOB）環境中同時在買賣兩側報價、管理庫存風險，並以 PPO 演算法端對端訓練。

> **課程**：CSCI 566 — Deep Learning and its Applications
> **學校**：University of Southern California
> **參考論文**：Cheridito & Weiss (2026), *Reinforcement Learning for Trade Execution with Market and Limit Orders*
> **基礎 repo**：[moritzweiss/rlte](https://github.com/moritzweiss/rlte)

---

### 專案概述

每個時間步驟中，agent 會：

1. 觀測狀態向量（LOB 特徵 + 庫存 + OFI 訂單流失衡）
2. 輸出 **Logistic-Normal** 動作 — 在買賣兩側的各價格層分配委託量
3. 根據 PnL、庫存風險和 maker-taker 手續費獲得獎勵
4. 以標準 **PPO**（clipped surrogate loss + gradient clipping）優化策略

模擬器支援三種市場環境：`noise`（雜訊交易者）、`flow`（流動性交易者）、`strategic`（策略性交易者）。

---

### 主要功能

| 功能 | 說明 |
| --- | --- |
| **雙邊報價** | Agent 同時在買賣兩側掛出限價單 |
| **Logistic-Normal 策略** | 連續動作映射到委託量分配的單純形（simplex） |
| **Maker-Taker 手續費** | `maker_rebate=0.2`，`taker_fee=0.3`（以 reward 單位計，對應真實市場約 1bp 的價差） |
| **標準 PPO** | Clipped surrogate loss、4 epochs、4 minibatches、梯度裁切 |
| **OFI 特徵** | 訂單流失衡作為可選觀測特徵（`use_ofi=True`） |
| **平行訓練** | 128 個 `SyncVectorEnv` 平行環境 |
| **TensorBoard 記錄** | Loss、KL divergence、clip fraction、episode 回報 |
| **29 個測試通過** | 涵蓋環境、agent、訓練流程的完整回歸測試 |

---

### 系統架構

```text
觀測向量（43 維）
    ├── LOB 特徵：各價格層的買賣委託量（以初始形狀正規化）
    ├── 庫存特徵：當前量、活躍量、時間加權庫存
    ├── 市場特徵：價差、中間價漂移
    └── OFI（可選）：訂單流失衡

        ↓
┌──────────────────────────────────┐
│   共享 MLP 主幹（128 個神經元）    │
│   + LayerNorm                    │
└──────────┬───────────────────────┘
           │
    ┌──────┴──────┐
    ↓             ↓
買側頭部        賣側頭部        價值頭部
(Logistic-    (Logistic-      （純量）
 Normal)       Normal)

動作 = (bid_allocation, ask_allocation)  # 各為 7 維單純形
```

**獎勵函數：**

```text
r_t = PnL（成交）+ maker_rebate × 被動成交量 / 初始量
                 - taker_fee   × 主動成交量 / 初始量
                 - 庫存懲罰 × |庫存|
```

---

### 專案結構

```text
rl_marketmaker/
├── bilateral_mm_agent.ipynb   # 主要實驗 notebook
├── config/
│   ├── config.py              # 所有 agent/環境/手續費設定
│   └── __init__.py
├── limit_order_book/
│   └── limit_order_book.py    # LOB 引擎（委託撮合、取消、簿狀態）
├── simulation/
│   ├── market_gym.py          # Gym 相容環境
│   └── agents.py              # 所有 agent 類別（RL、基準線、雜訊、策略型…）
├── rl_files/
│   └── actor_critic.py        # PPO 訓練迴圈 + BilateralAgentLogisticNormal 模型
├── initial_shape/             # 初始 LOB 形狀陣列（.npz）
├── tests/                     # 回歸與整合測試
├── requirements.txt
├── changes_report.tex         # 可上傳至 Overleaf 的程式碼改動報告
└── FINAL_PROJECT_PLAN.md      # 專案規劃文件
```

---

### 環境安裝

**建議 Python 版本**：3.9 – 3.14

```bash
pip install -r requirements.txt
```

依賴套件：`torch`、`gymnasium`、`numpy`、`pandas`、`matplotlib`、`seaborn`、`tensorboard`、`tyro`、`sortedcontainers`

---

### 執行 Notebook

用 JupyterLab 或 VS Code 開啟 `bilateral_mm_agent.ipynb`。
Notebook 完全支援本地執行，不需要 Google Colab 或 Google Drive。

**各區段說明：**

1. 環境設定與依賴套件確認
2. Repository 驗證
3. 設定 — 雙環境模式（flow → strategic）
4. Agent 初始化（雙邊 RL agent + 固定價差基準線）
5. 向量化配額投影
6. **訓練** — PPO，128 個平行環境
7. **評估** — RL agent vs 基準線比較
8. 視覺化 — PnL 曲線、動作分佈、手續費影響

> 注意：完整訓練（200 × 128 × 100 = 256 萬時間步）在 CPU 上需數小時。快速測試可在 `Args` 中調小 `total_timesteps`。

---

### 執行測試

```bash
python -m pytest -q tests/
```

預期結果：**29 passed**

---

### 近期改動（v0.3）

- 新增 maker-taker 手續費結構，taker fee 在市價單成交時扣除，maker rebate 在限價單被動成交時加入獎勵
- 修復 LOB 取消委託時 `agent_bid_orders`/`agent_ask_orders` 殘留 order ID 的 bug
- PPO 升級為標準 clipped surrogate loss，加入梯度裁切與 KL divergence 追蹤
- 恢復生產用超參數（128 個環境、100 步 rollout）
- 修復 `AsyncVectorEnv` info dict 格式不匹配導致 episode 回報靜默丟失的問題
- Notebook 從 Google Colab 移植至本地執行

---

### 未來規劃

- [ ] LSTM 時序主幹（以 LSTM 取代 MLP 主幹，處理 LOB 的時序依賴）
- [ ] OFI 消融實驗（`use_ofi=True` vs `False` 比較）
- [ ] 手續費消融實驗（無手續費 / 對稱手續費 / maker-taker 不對稱）
- [ ] 動作分佈分析（訓練過程中市價單比例變化）
- [ ] 完整雙邊委託生成（移除單邊降級路徑）
