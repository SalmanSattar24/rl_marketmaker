# How to Run Tests & Generate Graphs

## Quick Start (5 min test)

### Step 1: Run the Python Script
```bash
cd /c/All-Code/CSCI-566/rtle_parallelized
python rl_files/actor_critic.py
```

This will:
- ✅ Train for 2 iterations (~2-3 min)
- ✅ Evaluate on 5 episodes (~1-2 min)
- ✅ Generate outputs automatically
- ⏱️ Total time: ~4 minutes

---

## What Gets Generated

### 1. **Console Output** (Real-time updates)
```
starting the training process
environment set up: volume=40, market_env=strategic
batch_size=10, minibatch_size=10, num_iterations=2, ...
Using GPU: cuda:0
...
Iteration 0/2
  Episode Return: 1234.5678
  Global step: 10
iteration=0
  Steps/second: 850
...
Iteration 1/2
  Episode Return: 1289.3456
  Steps/second: 920

starting evaluation
...
evaluation time: 45.23s
reward length: 5

Evaluation Results:
  Mean return: 1256.7890
  Std return: 95.4321
  Min return: 1100.1234
  Max return: 1350.5678

Model saved to models/...
Rewards saved to rewards/...

Training completed!
```

### 2. **Files Generated**

#### A. Model File
```
models/strategic_40_seed_0_eval_seed_100_eval_episodes_5_num_iterations_2_bsize_10_log_normal.pt
```
- Trained neural network weights
- Can be loaded for inference/evaluation later

#### B. Rewards File
```
rewards/strategic_40_seed_0_eval_seed_100_eval_episodes_5_num_iterations_2_bsize_10_log_normal.npz
```
- Contains 5 evaluation returns
- Compressed NumPy format

#### C. TensorBoard Logs
```
tensorboard_logs/strategic_40_seed_0_eval_seed_100_eval_episodes_5_num_iterations_2_bsize_10_log_normal/
├── events.out.tfevents.* (training metrics)
└── Similar structure for each run
```

---

## Viewing Graphs/Visualizations

### Option 1: **TensorBoard** (Interactive Dashboard) ⭐ RECOMMENDED

Open separate terminal:
```bash
cd /c/All-Code/CSCI-566/rtle_parallelized
tensorboard --logdir tensorboard_logs/
```

Output:
```
TensorBoard 2.x.x at http://localhost:6006/
```

Then open browser: **http://localhost:6006**

You'll see:
- ✅ **Training curves** (return, time, drift over iterations)
- ✅ **Loss plots** (policy loss, value loss, total loss)
- ✅ **Charts** (learning rate, entropy, SPS - steps/second)
- ✅ **Real-time updates** while training

**TensorBoard Tabs Available**:
- **charts**: Returns, episode metrics, SPS
- **losses**: Policy/value/total loss, entropy
- **values**: Variance scaling over time

### Option 2: **Jupyter Notebook Analysis**

After training, analyze rewards in a notebook:
```bash
jupyter notebook notebooks/03_evaluate_policy.ipynb
```

Or create a quick analysis:
```python
import numpy as np
import matplotlib.pyplot as plt

# Load rewards
rewards = np.load('rewards/strategic_40_*.npz')['rewards']

# Plot
plt.figure(figsize=(10, 6))
plt.hist(rewards, bins=10)
plt.xlabel('Episode Return')
plt.ylabel('Frequency')
plt.title('Evaluation Returns Distribution')
plt.grid()
plt.savefig('rewards_distribution.png')
plt.show()
```

### Option 3: **Quick Numpy Analysis**

```python
import numpy as np

# Load and analyze
rewards = np.load('rewards/strategic_40_*.npz')['rewards']
print(f"Mean: {np.mean(rewards):.2f}")
print(f"Std: {np.std(rewards):.2f}")
print(f"Min: {np.min(rewards):.2f}")
print(f"Max: {np.max(rewards):.2f}")

# Compare across runs
import glob
all_rewards = {}
for f in glob.glob('rewards/*.npz'):
    key = f.split('/')[-1]
    all_rewards[key] = np.load(f)['rewards']
    print(f"{key}: mean={np.mean(all_rewards[key]):.2f}")
```

---

## Complete Workflow

### Step 1: Run Test
```bash
python rl_files/actor_critic.py
# Generates: model + rewards + TensorBoard logs
# Time: ~4 minutes
```

### Step 2: View TensorBoard (Optional - Real-time during training)
```bash
# In ANOTHER terminal, while training is running:
tensorboard --logdir tensorboard_logs/
# Browse to http://localhost:6006
```

### Step 3: Analyze Results
```bash
# After training finishes:
ls -lh models/              # Check model saved
ls -lh rewards/             # Check rewards file
ls -lh tensorboard_logs/    # Check logs
```

### Step 4: View Graphs
```bash
# Option A: TensorBoard dashboard
tensorboard --logdir tensorboard_logs/

# Option B: Jupyter analysis
jupyter notebook  # and create analysis script
# or use 03_evaluate_policy.ipynb
```

---

## What Each Graph Shows

### TensorBoard Charts

**charts/return**: Episode returns over training
- Y-axis: Mean return per iteration
- X-axis: Global steps
- Shows: Training progress, convergence

**charts/time**: Execution time per episode
- Y-axis: Time (seconds)
- Shows: Training efficiency

**charts/drift**: Price drift per episode
- Y-axis: Drift value
- Shows: Market conditions

**losses/policy_loss**: Policy gradient loss
- Y-axis: Loss
- Shows: Policy improvement

**losses/value_loss**: Value function loss
- Y-axis: Loss
- Shows: Value estimation quality

**charts/SPS**: Steps per second
- Y-axis: Steps/sec
- Shows: GPU utilization, training speed
- **Should show +15-25% improvement with optimization!** ✅

**losses/explained_variance**: How well value function explains returns
- Y-axis: Explained variance (0-1, higher is better)
- Shows: Value network quality

---

## Example: Complete Test Session

```bash
# Terminal 1: Run training
$ cd /c/All-Code/CSCI-566/rtle_parallelized
$ python rl_files/actor_critic.py
starting the training process
environment set up: volume=40, market_env=strategic
...
iteration=0
  Episode Return: 1234.5678
  Steps/second: 850
iteration=1
  Episode Return: 1289.3456
  Steps/second: 920

Evaluation Results:
  Mean return: 1256.7890
  Std return: 95.4321
Model saved to models/strategic_40_seed_0_...
Rewards saved to rewards/strategic_40_seed_0_...
Training completed!
# Takes ~4 minutes
```

```bash
# Terminal 2: Monitor with TensorBoard (while training)
$ tensorboard --logdir tensorboard_logs/
I1234 12:34:56.789012 12345 server.py:42] TensorBoard 2.x.x at http://localhost:6006/
# Open browser to http://localhost:6006
# See live updates as training progresses
```

```bash
# After training completes:
$ ls -lh models/ rewards/ tensorboard_logs/
models/:
drwxr-xr-x  strategic_40_seed_0_...log_normal.pt (5 MB)

rewards/:
strategic_40_seed_0_...log_normal.npz (2 KB)

tensorboard_logs/:
strategic_40_seed_0_...log_normal/
└── events.out.tfevents.*
```

---

## Viewing on Different OS

### Windows:
```bash
# Run Python script
python rl_files/actor_critic.py

# View TensorBoard
# Open separate terminal
tensorboard --logdir tensorboard_logs/
# Browser: http://localhost:6006
```

### Mac/Linux:
```bash
# Same commands work
python rl_files/actor_critic.py
tensorboard --logdir tensorboard_logs/  # in another terminal
```

---

## Troubleshooting

### "No GPU available" warning (OK)
```
No GPU available, using CPU.
```
Script still works, just slower. Optimization still applies to transfers.

### TensorBoard won't start
```bash
# Check if port 6006 is in use
netstat -an | grep 6006

# Use different port
tensorboard --logdir tensorboard_logs/ --port 6007
# Then visit http://localhost:6007
```

### No rewards file generated
```bash
# Check if evaluation ran
grep -i "evaluation" output.log

# If evaluation disabled, check actor_critic.py:
# evaluate: bool = True  (should be True)
```

### File permissions issue
```bash
# Make sure directory is writable
chmod -R 755 /c/All-Code/CSCI-566/rtle_parallelized/models
chmod -R 755 /c/All-Code/CSCI-566/rtle_parallelized/rewards
```

---

## Quick Reference: Command Cheatsheet

```bash
# RUN TRAINING
python rl_files/actor_critic.py

# MONITOR TENSORBOARD
tensorboard --logdir tensorboard_logs/

# VIEW SAVED FILES
ls -lh models/
ls -lh rewards/
ls -lh tensorboard_logs/

# ANALYZE REWARDS
python3 << 'EOF'
import numpy as np
f = sorted(__import__('glob').glob('rewards/*.npz'))[-1]
d = np.load(f)['rewards']
print(f"Mean: {d.mean():.2f}, Std: {d.std():.2f}")
EOF

# CLEANUP OLD RUNS
rm -rf tensorboard_logs/* models/* rewards/*
```

---

## Expected Test Results

For quick 5-min test:
- ✅ **Training**: 2 iterations complete
- ✅ **Model saved**: ~5 MB
- ✅ **5 evaluation episodes**: ~2 KB rewards file
- ✅ **TensorBoard logs**: Real-time metrics
- ✅ **SPS**: Should show proper GPU optimization in action
- ✅ **Console output**: Clear progress indicators

---

**Ready to run?** Start with:
```bash
python rl_files/actor_critic.py
```

Then in another terminal:
```bash
tensorboard --logdir tensorboard_logs/
```

Browse to **http://localhost:6006** to see live graphs! 📊
