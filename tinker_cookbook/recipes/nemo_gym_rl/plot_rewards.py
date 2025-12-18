from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_rewards(metrics_path: str | Path) -> None:
    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    print(f"Loading metrics from: {metrics_path}")
    df = pd.read_json(metrics_path, lines=True)

    reward_col = None
    for col in ["reward/total", "env/all/reward/total"]:
        if col in df.columns:
            reward_col = col
            break

    if reward_col is None:
        raise ValueError(f"No reward column found. Available columns: {list(df.columns)}")

    print(f"Loaded {len(df)} training steps")
    print(f"Using reward column: {reward_col}")
    print(f"Reward range: [{df[reward_col].min():.3f}, {df[reward_col].max():.3f}]")

    plt.figure(figsize=(10, 6))
    plt.plot(df[reward_col], label=reward_col, linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Mean Reward")
    title = metrics_path.parent.name.split("/")[-1]
    plt.title(f"Mean Reward - {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = metrics_path.parent / "rewards.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot NeMo Gym RL training rewards")
    parser.add_argument(
        "--log_path",
        type=str,
        default="./outputs/reasoning_gym_knights",
        help="Path to training log directory containing metrics.jsonl",
    )
    args = parser.parse_args()

    metrics_path = Path(args.log_path) / "metrics.jsonl"
    plot_rewards(metrics_path)


if __name__ == "__main__":
    main()
