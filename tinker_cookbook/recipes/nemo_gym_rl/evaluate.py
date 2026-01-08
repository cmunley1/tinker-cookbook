from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List

import chz
import numpy as np

from tinker_cookbook.recipes.nemo_gym_rl.nemo_gym_client import call_nemo_gym_agent
from tinker_cookbook.recipes.nemo_gym_rl.nemo_gym_env import (
    get_agent_server_from_head,
    load_nemo_gym_dataset,
)


def log_results(
    responses: List[Dict[str, Any]],
    dataset_path: str,
    agent_server: str,
    num_examples: int,
    rollouts_per_example: int,
    time_s: float,
):
    print(f"\nEvaluation completed in {time_s:.2f} seconds")
    print("=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Agent: {agent_server}")
    print(f"Examples: {num_examples}")
    print(f"Rollouts per example: {rollouts_per_example}")
    print("-" * 80)

    # Rewards
    rewards = [r.get("reward", 0.0) for r in responses]
    print(f"\nRewards:")
    print(f"  Mean: {np.mean(rewards):.3f}")
    print(f"  Std: {np.std(rewards):.3f}")
    print(f"  Min: {np.min(rewards):.3f}")
    print(f"  Max: {np.max(rewards):.3f}")

    # Per-rollout statistics
    if rollouts_per_example > 1:
        print(f"\nPer-rollout breakdown:")
        n = num_examples
        for i in range(rollouts_per_example):
            rollout_rewards = [rewards[(i * n) + j] for j in range(n)]
            print(f"  Rollout {i+1}: {[round(r, 3) for r in rollout_rewards]}")

    # Turn statistics
    num_turns_list = []
    num_errors = 0
    for response in responses:
        output = response.get("response", {}).get("output", [])
        if "error" in response:
            num_errors += 1
        else:
            turns = sum(1 for item in output if "prompt_token_ids" in item and "generation_token_ids" in item)
            num_turns_list.append(turns)

    if num_turns_list:
        print(f"\nMulti-turn statistics:")
        print(f"  Mean turns: {np.mean(num_turns_list):.1f}")
        print(f"  Min/max turns: {np.min(num_turns_list)}/{np.max(num_turns_list)}")

    if num_errors > 0:
        print(f"\nErrors: {num_errors}/{len(responses)}")

    # Example trajectory
    print(f"\n{'-'*80}")
    print("Example trajectory (first rollout):")
    print(f"{'-'*80}")
    example_response = responses[0]
    example_output = example_response.get("response", {}).get("output", [])
    print(f"Reward: {example_response.get('reward', 0.0):.3f}")
    print(f"Turns: {len([o for o in example_output if 'prompt_token_ids' in o])}")
    for i, item in enumerate(example_output[:5]):
        print(f"\n  Turn {i+1}:")
        print(f"    Type: {item.get('type', 'unknown')}")
        if "generation_token_ids" in item:
            print(f"    Generation tokens: {len(item['generation_token_ids'])}")
        if item.get("type") == "message" and "content" in item:
            content = item["content"]
            if isinstance(content, list) and content:
                text = content[0].get("text", "")
                print(f"    Content: {text}")
    if len(example_output) > 5:
        print(f"\nOnly showing first 5 turns, but there is {len(example_output)} turns in total")
    print(f"{'='*80}\n")


async def evaluate(
    dataset_path: str,
    agent_server: str | None,
    head_server_host: str,
    head_server_port: int,
    num_examples: int,
    rollouts_per_example: int,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout: float,
) -> List[Dict[str, Any]]:
    if agent_server is None:
        agent_server = get_agent_server_from_head(
            head_server_host,
            head_server_port,
        )

    print(f"Using nemo gym agent server: {agent_server}")

    dataset = load_nemo_gym_dataset(dataset_path)
    dataset = dataset[:num_examples]

    print(f"Evaluating on {len(dataset)} examples with {rollouts_per_example} rollouts each")

    expanded_dataset = []
    for item in dataset:
        for _ in range(rollouts_per_example):
            expanded_dataset.append(item.copy())

    print(f"Total requests: {len(expanded_dataset)}")

    start_time = time.time()
    responses = await call_nemo_gym_agent(
        agent_server=agent_server,
        dataset_items=expanded_dataset,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=request_timeout,
    )
    end_time = time.time()

    log_results(
        responses,
        dataset_path,
        agent_server,
        num_examples,
        rollouts_per_example,
        end_time - start_time,
    )

    return responses


@chz.chz
class CLIConfig:
    dataset_path: str
    agent_server: str | None = None
    head_server_host: str = "127.0.0.1"
    head_server_port: int = 11000

    num_examples: int = 5
    rollouts_per_example: int = 3
    max_output_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    request_timeout: float = 600.0

    output_path: str | None = None


async def cli_main(cfg: CLIConfig):
    results = await evaluate(
        dataset_path=cfg.dataset_path,
        agent_server=cfg.agent_server,
        head_server_host=cfg.head_server_host,
        head_server_port=cfg.head_server_port,
        num_examples=cfg.num_examples,
        rollouts_per_example=cfg.rollouts_per_example,
        max_output_tokens=cfg.max_output_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        request_timeout=cfg.request_timeout,
    )

    if cfg.output_path:
        with open(cfg.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {cfg.output_path}")

    return results


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cfg))
