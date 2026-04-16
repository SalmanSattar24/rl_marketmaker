"""Summarize and plot quick PPO runs.

Scans `runs_quick/quick_ppo_results_seed*.npz`, computes simple summary
statistics, writes a CSV, and saves a combined plot.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs_quick"
CSV_PATH = RUN_DIR / "quick_ppo_summary.csv"
PLOT_PATH = RUN_DIR / "quick_ppo_combined.png"


def load_runs():
    runs = []
    pattern = os.environ.get("QP_PATTERN", "quick_ppo_results_seed*.npz")
    for npz_path in sorted(RUN_DIR.glob(pattern)):
        seed = int(npz_path.stem.split("seed")[-1])
        data = np.load(npz_path)
        returns = np.asarray(data["returns"], dtype=float)
        runs.append((seed, npz_path, returns))
    return runs


def summarize(returns: np.ndarray) -> dict[str, float]:
    if returns.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "final": float("nan"), "area": float("nan")}
    x = np.asarray(returns, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "final": float(x[-1]),
        "area": float(np.trapezoid(x)),
    }


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit(f"No result files found in {RUN_DIR}")

    title = os.environ.get("QP_TITLE", "Quick PPO mean returns by seed")

    rows = []
    max_len = max(len(returns) for _, _, returns in runs)
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, npz_path, returns in runs:
        stats = summarize(returns)
        rows.append({
            "seed": seed,
            "file": npz_path.name,
            "num_iterations": len(returns),
            **stats,
        })
        xs = np.arange(1, len(returns) + 1)
        ax.plot(xs, returns, marker="o", linewidth=1.5, label=f"seed {seed}")

    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean_return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "file", "num_iterations", "mean", "std", "final", "area"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {PLOT_PATH}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
