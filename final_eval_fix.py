"""
Phase D: Final Evaluation and Plotting Alignment
Fixes NameErrors and ensures high-quality comparison between RL and Baseline.
"""
import nbformat
import os

path = 'c:/All-Code/CSCI-566/rtle_parallelized/bilateral_mm_agent.ipynb'
if not os.path.exists(path):
    print(f"Error: {path} not found")
    exit(1)

with open(path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# ============================================================
# 1. CELL 23 & 24: Step 8 - Evaluate Bilateral Agent
# ============================================================
nb.cells[23].source = """## Step 8: Evaluate Bilateral Agent (Final Best Model)
# This performs a large-scale evaluation of the trained agent to compare against baseline.
"""
nb.cells[24].source = """import torch
import numpy as np

print("=" * 70)
print("STEP 8: EVALUATE BILATERAL AGENT (RL)")
print("=" * 70)

# Ensure best weights are loaded (redundant but safe)
if 'best_state_dict' in locals():
    bilateral_agent.load_state_dict(best_state_dict)

EVAL_EPISODES = 1000
bilateral_returns = []
bilateral_inventories = []

for i in range(EVAL_EPISODES):
    cfg = dict(EVAL_CONFIG)
    cfg['seed'] = 50000 + i 
    m_raw = Market(cfg)
    market_wrap = EnvWrapper(m_raw)
    obs, _ = market_wrap.reset()
    ep_ret = 0.0
    current_inventory = 0

    while True:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            # Evaluation uses DETERMINISTIC actions
            bid_action, ask_action = bilateral_agent.deterministic_action(obs_tensor)
            
            # Quota projection is strictly applied for consistency
            bid_action = project_action_quota(bid_action, current_inventory, side="bid", inventory_max=EVAL_CONFIG['inventory_max'])
            ask_action = project_action_quota(ask_action, current_inventory, side="ask", inventory_max=EVAL_CONFIG['inventory_max'])
            
            env_action = (bid_action[0].cpu().numpy(), ask_action[0].cpu().numpy())
            
        obs, reward, terminated, truncated, info = market_wrap.step(env_action)
        ep_ret += float(reward)
        current_inventory = info.get("net_inventory", 0)
        
        if terminated or truncated:
            break
            
    bilateral_returns.append(ep_ret)
    bilateral_inventories.append(abs(current_inventory))
    
    if (i + 1) % 200 == 0:
        print(f"[{i+1:4d}/{EVAL_EPISODES}] Mean return: {np.mean(bilateral_returns):.4f} | Abs Inv: {np.mean(bilateral_inventories):.4f}")

bilateral_returns = np.array(bilateral_returns)
bilateral_inventories = np.array(bilateral_inventories)

print(f"\\n[OK] RL Evaluation complete: {np.mean(bilateral_returns):.4f} +/- {np.std(bilateral_returns):.4f}")
"""

# ============================================================
# 3. CELL 26: Step 9 - Evaluate Baseline Agent (Ensure inventories)
# ============================================================
nb.cells[26].source = """print("=" * 70)
print(f"STEP 9: EVALUATE BASELINE (Fixed Spread)")
print("=" * 70)

baseline_agent = SymmetricFixedSpreadAgent(spread=1)
baseline_returns = []
baseline_inventories = []

for i in range(EVAL_EPISODES):
    cfg = dict(EVAL_CONFIG)
    cfg['seed'] = 50000 + i
    m_raw = Market(cfg)
    market_wrap = EnvWrapper(m_raw)
    obs, _ = market_wrap.reset()
    ep_ret = 0.0
    current_inventory = 0
    
    while True:
        action = baseline_agent.get_action(obs)
        obs, reward, terminated, truncated, info = market_wrap.step(action)
        ep_ret += float(reward)
        current_inventory = info.get("net_inventory", 0)
        if terminated or truncated:
            break
            
    baseline_returns.append(ep_ret)
    baseline_inventories.append(abs(current_inventory))

baseline_returns = np.array(baseline_returns)
baseline_inventories = np.array(baseline_inventories)

print(f"\\n[OK] Baseline Evaluation complete: {np.mean(baseline_returns):.4f} +/- {np.std(baseline_returns):.4f}")
"""

# ============================================================
# 3. CELL 28: Step 10 - Stats
# ============================================================
nb.cells[28].source = """# Compute statistics for comparison
import pandas as pd

stats_data = {
    'Metric': ['Mean Return', 'Std Return', 'CVaR (5%)', 'Outliers (<-200)', 'Mean Abs Inv'],
    'Bilateral RL': [
        np.mean(bilateral_returns),
        np.std(bilateral_returns),
        np.mean(np.sort(bilateral_returns)[:max(1, len(bilateral_returns)//20)]),
        np.mean(bilateral_returns < -200.0),
        np.mean(bilateral_inventories)
    ],
    'Baseline': [
        np.mean(baseline_returns),
        np.std(baseline_returns),
        np.mean(np.sort(baseline_returns)[:max(1, len(baseline_returns)//20)]),
        np.mean(baseline_returns < -200.0),
        np.mean(baseline_inventories)
    ]
}

df_stats = pd.DataFrame(stats_data)
display(df_stats.style.background_gradient(subset=['Bilateral RL', 'Baseline'], cmap='RdYlGn'))
"""

# ============================================================
# 4. CELL 30: Step 11 - Premium Visualization
# ============================================================
nb.cells[30].source = """import matplotlib.pyplot as plt
import seaborn as sns

# Set premium aesthetic
plt.style.use('seaborn-v0_8-muted')
sns.set_context("talk")
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
colors = ['#4C72B0', '#C44E52', '#55A868', '#8172B3']

# 1. Return Distributions
sns.histplot(bilateral_returns, ax=axes[0, 0], color=colors[0], label='Bilateral RL', kde=True, element="step")
sns.histplot(baseline_returns, ax=axes[0, 0], color=colors[1], label='Baseline', kde=True, element="step")
axes[0, 0].set_title('Return Distribution Comparison')
axes[0, 0].set_xlabel('Profit / Loss')
axes[0, 0].legend()

# 2. Cumulative Returns (Box Plot)
axes[0, 1].boxplot([bilateral_returns, baseline_returns], labels=['Bilateral RL', 'Baseline'], patch_artist=True)
axes[0, 1].set_title('Return Range & Outliers')
axes[0, 1].set_ylabel('Profit / Loss')

# 3. Training Curve (if training_returns exists)
if 'training_returns' in locals() and len(training_returns) > 0:
    training_ma = pd.Series(training_returns).rolling(window=20).mean()
    axes[1, 0].plot(training_ma, color=colors[2], linewidth=3, label='Training Moving Avg (20)')
    axes[1, 0].set_title('Training Progress')
    axes[1, 0].set_xlabel('Episode Index')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
else:
    axes[1, 0].text(0.5, 0.5, "No training curve data", ha='center')

# 4. Inventory Management
axes[1, 1].hist(bilateral_inventories, bins=11, alpha=0.7, color=colors[0], label='RL Inventory', density=True)
axes[1, 1].hist(baseline_inventories, bins=11, alpha=0.7, color=colors[1], label='Baseline Inventory', density=True)
axes[1, 1].set_title('Terminal Inventory Density')
axes[1, 1].set_xlabel('Absolute Inventory Value')
axes[1, 1].legend()

plt.suptitle(f"Stabilized Bilateral MM Performance Analysis ({TRAIN_CONFIG['market_env'].upper()})", fontsize=24, y=1.02)
plt.tight_layout()
plt.savefig('bilateral_mm_success_report.png', dpi=150, bbox_inches='tight')
plt.show()

print("[OK] Performance report and plots generated.")
"""

with open(path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("PHASE_D_EVALUATION_FIXED")
