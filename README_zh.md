# RL Market Maker — 中文說明

**[English](README.md)** | 中文

本專案實作了一個基於**強化學習的雙邊做市商（Bilateral Market Maker）**，在模擬的限價委託簿（Limit Order Book, LOB）環境中同時在買賣兩側報價、管理庫存風險，並以 PPO 演算法端對端訓練。

> **課程**：CSCI 566 — Deep Learning and its Applications
> **學校**：University of Southern California
> **參考論文**：Cheridito & Weiss (2026), *Reinforcement Learning for Trade Execution with Market and Limit Orders*
> **基礎 repo**：[moritzweiss/rlte](https://github.com/moritzweiss/rlte)

---

## 專案概述

每個時間步驟中，agent 會：

1. 觀測狀態向量（LOB 特徵 + 庫存 + OFI 訂單流失衡）
2. 輸出 **Logistic-Normal** 動作 — 在買賣兩側的各價格層分配委託量
3. 根據 PnL、庫存風險和 maker-taker 手續費獲得獎勵
4. 以標準 **PPO**（clipped surrogate loss + gradient clipping）優化策略

模擬器支援三種市場環境：`noise`（雜訊交易者）、`flow`（流動性交易者）、`strategic`（策略性交易者）。

---

## 主要功能

| 功能 | 說明 |
| --- | --- |
| **雙邊報價** | Agent 同時在買賣兩側掛出限價單 |
| **Logistic-Normal 策略** | 連續動作映射到委託量分配的單純形（simplex） |
| **Maker-Taker 手續費** | `maker_rebate=0.2`，`taker_fee=0.3`（以 reward 單位計，對應真實市場約 1bp 的價差） |
| **標準 PPO** | Clipped surrogate loss、4 epochs、4 minibatches、梯度裁切 |
| **OFI 特徵** | 訂單流失衡作為可選觀測特徵（`use_ofi=True`） |
| **Transformer LOB 編碼器** | 完整 Transformer 區塊：正弦 PE、Pre-LN、GELU FFN、注意力加權池化 |
| **注意力視覺化** | 提取並繪製每個 head、每層的自注意力熱圖 + 池化權重 |
| **消融實驗框架** | 自動化 4×3 實驗矩陣：MLP/Transformer × 無手續費/有手續費 × noise/flow/strategic |
| **平行訓練** | 32–128 個 `SyncVectorEnv` 平行環境 |
| **TensorBoard 記錄** | Loss、KL divergence、clip fraction、episode 回報 |
| **29 個測試通過** | 涵蓋環境、agent、訓練流程的完整回歸測試 |

---

## 系統架構

支援兩種 agent 架構：

### 方案 A — MLP 雙邊 Agent

```text
觀測向量（63 維）
    ├── LOB 特徵：各價格層的買賣委託量
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

### 方案 B — Transformer 雙邊 Agent（新增）

```text
LOB 委託量 (5 層 × 2 特徵)
        ↓
┌────────────────────────────────────────┐
│  線性嵌入 → d_model=32                  │
│  + 正弦位置編碼（Sinusoidal PE）         │
│                                        │
│  Transformer 編碼器（2 層）              │
│  ┌──────────────────────────────────┐  │
│  │ Pre-LayerNorm                    │  │
│  │ 多頭自注意力（2 heads）            │  │
│  │ + 殘差連接 + Dropout              │  │
│  │ Pre-LayerNorm                    │  │
│  │ FFN（GELU，dim=64）              │  │
│  │ + 殘差連接 + Dropout              │  │
│  └──────────────────────────────────┘  │
│                                        │
│  注意力加權池化（學習查詢向量）            │
└──────────┬─────────────────────────────┘
           │
    與全局特徵拼接
           ↓
    ┌──────┴──────┐
    ↓             ↓
買側頭部        賣側頭部        價值頭部

動作 = (bid_allocation, ask_allocation)  # 各為 7 維單純形
```

Transformer 編碼器將每個 LOB 價格層視為一個 token，使模型能學習跨層關係（如最優報價 vs 深層委託簿的動態）。注意力權重可提取用於可解釋性分析。

**獎勵函數：**

```text
r_t = PnL（成交）+ maker_rebate × 被動成交量 / 初始量
                 - taker_fee   × 主動成交量 / 初始量
                 - 庫存懲罰 × |庫存|
```

---

## 專案結構

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
│   ├── actor_critic.py        # PPO 訓練迴圈 + agent 模型（MLP 與 Transformer）
│   ├── attention_viz.py       # 注意力權重視覺化工具
│   └── ablation_runner.sh     # 消融實驗 runner（4 configs × 3 envs）
├── initial_shape/             # 初始 LOB 形狀陣列（.npz）
├── models/                    # 儲存的模型 checkpoints（.pt）
├── rewards/                   # 評估獎勵陣列（.npz）
├── tests/                     # 回歸與整合測試
├── requirements.txt
├── changes_report.tex         # 可上傳至 Overleaf 的程式碼改動報告
└── FINAL_PROJECT_PLAN.md      # 專案規劃文件
```

---

## 環境安裝

**建議 Python 版本**：3.9 – 3.14

```bash
pip install -r requirements.txt
```

依賴套件：`torch`、`gymnasium`、`numpy`、`pandas`、`matplotlib`、`seaborn`、`tensorboard`、`tyro`、`sortedcontainers`

---

## 執行 Notebook

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

## 執行測試

```bash
python -m pytest -q tests/
```

預期結果：**29 passed**

---

## 近期改動

### v0.4 — Transformer 編碼器 + 消融實驗框架（2026 年 4 月）

- **Transformer LOB 編碼器**：以完整 Transformer 區塊取代單層多頭注意力
  - 正弦位置編碼（registered buffer）編碼 LOB 層級順序
  - 2 層 Pre-LayerNorm TransformerEncoder，GELU FFN（d_model=32, n_heads=2, ffn_dim=64）
  - 學習式查詢向量的注意力加權池化（取代簡單平均池化）
  - `get_attention_maps()` 提取每 head、每層的注意力權重
- **注意力視覺化工具**（`attention_viz.py`）：熱圖繪製 + 多 episode 平均收集
- **消融實驗 runner**（`ablation_runner.sh`）：4 configs × 3 envs = 12 runs 自動化
- **模擬器修復**：bilateral MM 在 terminal time 前執行完所有量時不再 crash；noise agent 事件超過 terminal time 時優雅終止

### v0.3 — Maker-Taker 手續費 + PPO 升級（2026 年 4 月）

- 新增 maker-taker 手續費結構，taker fee 在市價單成交時扣除，maker rebate 在限價單被動成交時加入獎勵
- 修復 LOB 取消委託時 `agent_bid_orders`/`agent_ask_orders` 殘留 order ID 的 bug
- PPO 升級為標準 clipped surrogate loss，加入梯度裁切與 KL divergence 追蹤
- 恢復生產用超參數（128 個環境、100 步 rollout）
- 修復 `AsyncVectorEnv` info dict 格式不匹配導致 episode 回報靜默丟失的問題
- Notebook 從 Google Colab 移植至本地執行

---

## 未來規劃

- [x] Transformer LOB 編碼器：正弦位置編碼 + 注意力加權池化
- [x] 注意力權重提取與視覺化工具
- [x] 消融實驗框架（MLP vs Transformer × 手續費結構 × 市場環境）
- [ ] 消融結果分析與比較表格
- [ ] LSTM 時序主幹（以 LSTM 取代 MLP 主幹，處理 LOB 的時序依賴）
- [ ] OFI 消融實驗（`use_ofi=True` vs `False` 比較）
- [ ] 動作分佈分析（訓練過程中市價單比例變化）
- [ ] 完整雙邊委託生成（移除單邊降級路徑）
