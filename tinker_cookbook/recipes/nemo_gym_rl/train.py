from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, cast

import chz

from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
from tinker_cookbook.recipes.nemo_gym_rl.nemo_gym_client import call_nemo_gym_agent
from tinker_cookbook.recipes.nemo_gym_rl.nemo_gym_env import (
    NemoGymEnvGroupBuilder,
    NemoGymRLDatasetBuilder,
    convert_nemo_gym_responses_to_trajectory_group,
)
from tinker_cookbook.recipes.nemo_gym_rl.tinker_server import serve_in_background, set_client
from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32

    dataset_path: str
    agent_server: str | None = None
    head_server_host: str = "127.0.0.1"
    head_server_port: int = 11000
    tinker_server_port: int = 8000

    dataset_n: int = -1

    group_size: int = 8
    groups_per_batch: int = 32
    num_substeps: int = 1
    learning_rate: float = 1e-5
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    kl_penalty_coef: float = 0.0
    request_timeout: float = 600.0

    eval_every: int = 0
    save_every: int = 10
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig, env: Any | None):
    model_name_short = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"nemo_gym_rl_{model_name_short}_gp{cli_config.groups_per_batch}_gs{cli_config.group_size}"
        f"_lr{cli_config.learning_rate}_rank{cli_config.lora_rank}_{date_and_time}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/nemo_gym_rl/{run_name}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    trajectory_file = os.path.join(log_path, "trajectories.jsonl")
    os.makedirs(log_path, exist_ok=True)

    current_step = [0]

    server_task = asyncio.create_task(
        serve_in_background(host="127.0.0.1", port=cli_config.tinker_server_port)
    )
    logger.info(f"Starting Tinker inference server on port {cli_config.tinker_server_port}")
    await asyncio.sleep(2)

    local_tokenizer: Tokenizer | None = None
    shared_renderer: renderers.Renderer | None = None
    shared_client: TinkerAsyncOpenAIClient | None = None

    async def custom_do_group_rollout(
        builder: EnvGroupBuilder, policy: TokenCompleter
    ) -> TrajectoryGroup:
        nonlocal shared_client, shared_renderer, local_tokenizer

        if local_tokenizer is None:
            local_tokenizer = get_tokenizer(cli_config.model_name)
        if shared_renderer is None:
            renderer_name = model_info.get_recommended_renderer_name(cli_config.model_name)
            shared_renderer = renderers.get_renderer(renderer_name, local_tokenizer)

        sampling_client = cast(TinkerTokenCompleter, policy).sampling_client
        if shared_client is None:
            shared_client = TinkerAsyncOpenAIClient(
                sampling_client, shared_renderer, local_tokenizer
            )
        else:
            shared_client.set_sampling_client(sampling_client)

        set_client(shared_client)

        nemo_gym_builder = cast(NemoGymEnvGroupBuilder, builder)

        dataset_items = [nemo_gym_builder.dataset_item.copy() for _ in range(cli_config.group_size)]

        responses = await call_nemo_gym_agent(
            agent_server=nemo_gym_builder.agent_server,
            dataset_items=dataset_items,
            max_output_tokens=cli_config.max_tokens,
            temperature=cli_config.temperature,
            top_p=cli_config.top_p,
            timeout=cli_config.request_timeout,
        )

        with open(trajectory_file, 'a') as f:
            for i, response in enumerate(responses):
                trajectory_data = {
                    "step": current_step[0],
                    "rollout_idx": i,
                    "reward": response.get("reward", 0.0) if isinstance(response, dict) else 0.0,
                    "output": response.get("response", {}).get("output", []) if isinstance(response, dict) else [],
                    "error": response.get("error") if isinstance(response, dict) else str(response),
                }
                f.write(json.dumps(trajectory_data) + "\n")

        trajectory_group = convert_nemo_gym_responses_to_trajectory_group(responses)

        rewards = trajectory_group.final_rewards_G
        if rewards:
            print(f"\n{'='*80}")
            print(f"[Step {current_step[0]}] Summary:")
            print(f"  Num trajectories: {len(rewards)}")
            print(f"  Mean reward: {sum(rewards)/len(rewards):.3f}")
            print(f"  Min/max reward: {min(rewards):.3f}/{max(rewards):.3f}")

            num_turns = [m.get("num_turns", 0) for m in trajectory_group.metrics_G]
            if num_turns:
                print(f"  Mean turns: {sum(num_turns)/len(num_turns):.1f}")
                print(f"  Min/max turns: {min(num_turns)}/{max(num_turns)}")

            print(f"{'='*80}\n")

        current_step[0] += 1

        return trajectory_group

    train.do_group_rollout = custom_do_group_rollout

    dataset_builder = NemoGymRLDatasetBuilder(
        dataset_path=cli_config.dataset_path,
        agent_server=cli_config.agent_server,
        head_server_host=cli_config.head_server_host,
        head_server_port=cli_config.head_server_port,
        groups_per_batch=cli_config.groups_per_batch,
        dataset_n=cli_config.dataset_n,
    )

    cfg = train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        lora_rank=cli_config.lora_rank,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        log_path=log_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        stream_minibatch_config=None,
    )

    print(f"\n{'='*80}")
    print(f"Starting training")
    print(f"Model: {cli_config.model_name}")
    print(f"Dataset: {cli_config.dataset_path}")
    print(f"Group size: {cli_config.group_size}")
    print(f"Groups per batch: {cli_config.groups_per_batch}")
    print(f"Learning rate: {cli_config.learning_rate}")
    print(f"LoRA rank: {cli_config.lora_rank}")
    print(f"Log path: {log_path}")
    print(f"Trajectory log: {trajectory_file}")
    print(f"{'='*80}\n")

    await train.main(cfg)

if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config, None))
