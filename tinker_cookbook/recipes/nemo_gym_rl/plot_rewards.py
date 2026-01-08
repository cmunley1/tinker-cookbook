import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_json(f"{args.log_path}/metrics.jsonl", lines=True)

    reward_col = next((c for c in ["reward/total", "env/all/reward/total"] if c in df.columns), None)
    if reward_col is None:
        raise ValueError(f"No reward column found. Available: {list(df.columns)}")

    plt.plot(df[reward_col], label=reward_col)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    output_path = f"{args.log_path}/rewards.png"
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
