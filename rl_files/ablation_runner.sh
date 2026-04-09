#!/bin/bash
# Ablation experiment runner — 12 configurations
# Run from project root: bash rl_files/ablation_runner.sh [--debug]
#
# Matrix: 4 agent configs × 3 environments = 12 runs
#   Agents:  (1) MLP baseline        (2) MLP + fees
#            (3) Transformer          (4) Transformer + fees
#   Envs:    noise, flow, strategic

set -e

DEBUG_MODE=false
if [[ "${1:-}" == "--debug" ]]; then
  DEBUG_MODE=true
  shift
  echo "[DEBUG MODE] Short runs for testing"
fi

PROJECT_ROOT="${PWD}"
PYTHON_SCRIPT="$PROJECT_ROOT/rl_files/actor_critic.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "Error: Python script not found at $PYTHON_SCRIPT"
  echo "Run this script from the project root directory."
  exit 1
fi

# ---- Hyperparameters ----
NUM_STEPS=100
NUM_ENVS=32
NUM_ITERATIONS=200
NUM_EVAL=200

if [[ "$DEBUG_MODE" == true ]]; then
  NUM_STEPS=20
  NUM_ENVS=2
  NUM_ITERATIONS=4
  NUM_EVAL=10
fi

TIMESTEPS=$((NUM_ITERATIONS * NUM_ENVS * NUM_STEPS))

echo "============================================"
echo " ABLATION EXPERIMENT RUNNER"
echo " Timesteps: $TIMESTEPS  Envs: $NUM_ENVS  Steps: $NUM_STEPS  Iters: $NUM_ITERATIONS"
echo "============================================"

ENVS=("noise" "flow" "strategic")

# ---- Config definitions ----
# Format: "tag agent_flags fee_flags"
declare -a CONFIGS=(
  # "mlp_baseline|--bilateral|--maker_rebate 0.0 --taker_fee 0.0"  # DONE
  "mlp_fees|--bilateral|--maker_rebate 0.0002 --taker_fee 0.0003"
  "transformer_baseline|--attention|--maker_rebate 0.0 --taker_fee 0.0"
  "transformer_fees|--attention|--maker_rebate 0.0002 --taker_fee 0.0003"
)

RUN_COUNT=0
TOTAL_RUNS=$((${#CONFIGS[@]} * ${#ENVS[@]}))

for config_str in "${CONFIGS[@]}"; do
  IFS='|' read -r TAG AGENT_FLAGS FEE_FLAGS <<< "$config_str"

  for env in "${ENVS[@]}"; do
    RUN_COUNT=$((RUN_COUNT + 1))
    echo ""
    echo "===== Run $RUN_COUNT/$TOTAL_RUNS: $TAG / $env ====="

    python "$PYTHON_SCRIPT" \
      --env_type "$env" \
      --num_lots 20 \
      --total_timesteps "$TIMESTEPS" \
      --num_envs "$NUM_ENVS" \
      --num_steps "$NUM_STEPS" \
      --n_eval_episodes "$NUM_EVAL" \
      --drop_feature "drift" \
      --tag "$TAG" \
      $AGENT_FLAGS \
      $FEE_FLAGS

    echo "===== Completed: $TAG / $env ====="
  done
done

echo ""
echo "============================================"
echo " ALL $TOTAL_RUNS ABLATION RUNS COMPLETED"
echo "============================================"
