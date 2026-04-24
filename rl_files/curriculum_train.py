#!/usr/bin/env python3
"""
Curriculum Learning Script for Bilateral Market Maker.
Trains sequentially across difficulty regimes: noise -> flow -> strategic.
"""

import os
import subprocess
import sys

def run_training_stage(env_type, total_timesteps, checkpoint_out, checkpoint_in=None, tag_suffix=""):
    """Runs the PPO training script for a specific regime."""
    cmd = [
        "python", "rl_files/ppo_continuous_action.py",
        "--env_type", env_type,
        "--total_timesteps", str(total_timesteps),
        "--num_envs", "32",
        "--num_steps", "100",
        "--bilateral",
        "--attention",
        "--maker_rebate", "0.0002",
        "--taker_fee", "0.0003",
        "--tag", f"curriculum_{env_type}{tag_suffix}"
    ]
    
    if checkpoint_in:
        cmd.extend(["--checkpoint_in", checkpoint_in])
    
    print(f"\n=======================================================")
    print(f"🚀 STARTING CURRICULUM STAGE: {env_type.upper()}")
    print(f"=======================================================")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, env=os.environ.copy())
    if result.returncode != 0:
        print(f"❌ Stage {env_type} failed with code {result.returncode}")
        sys.exit(result.returncode)

def main():
    # We define a curriculum of increasing difficulty
    stages = [
        {"env": "noise", "steps": 500000},
        {"env": "flow", "steps": 500000},
        {"env": "strategic", "steps": 1000000}
    ]
    
    # Normally we would save and pass checkpoints, but since checkpointing
    # requires wandb or local saving (which we assume is handled inside the script if modified),
    # we simulate the curriculum runner.
    # Note: ppo_continuous_action.py might need `--checkpoint_out` and `--checkpoint_in`
    # flags to fully support seamless transition. If they aren't implemented, this script
    # serves as a structural foundation.
    
    print("🎓 Starting Curriculum Learning Runner...")
    
    # Check if we support checkpointing
    help_out = subprocess.run(["python", "rl_files/ppo_continuous_action.py", "--help"], capture_output=True, text=True)
    supports_checkpointing = "--checkpoint_in" in help_out.stdout
    
    last_checkpoint = None
    
    for stage in stages:
        out_ckpt = f"models/curriculum_{stage['env']}.pt"
        run_training_stage(
            env_type=stage['env'], 
            total_timesteps=stage['steps'],
            checkpoint_out=out_ckpt if supports_checkpointing else None,
            checkpoint_in=last_checkpoint if supports_checkpointing else None
        )
        if supports_checkpointing:
            last_checkpoint = out_ckpt

    print("\n🎉 Curriculum Training Complete!")

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    main()
