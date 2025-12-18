from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any, Dict, List, Sequence

import chz
import requests
import tinker
from omegaconf import OmegaConf
import yaml

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)

_nemo_gym_agent_server_ctx: ContextVar[str | None] = ContextVar("nemo_gym_agent_server", default=None)


def set_nemo_gym_agent_server(agent_server: str) -> None:
    _nemo_gym_agent_server_ctx.set(agent_server)


def get_nemo_gym_agent_server() -> str | None:
    return _nemo_gym_agent_server_ctx.get()


def get_agent_server_from_head(
    head_server_host: str = "127.0.0.1",
    head_server_port: int = 11000,
    agent_name: str | None = None,
) -> str:
    try:
        response = requests.get(
            f"http://{head_server_host}:{head_server_port}/global_config_dict_yaml",
            timeout=10
        )
        response.raise_for_status()
        global_config_yaml = response.text
        global_config_dict = OmegaConf.create(yaml.safe_load(global_config_yaml))

        if agent_name:
            for project_name, project_config in global_config_dict.items():
                if hasattr(project_config, 'responses_api_agents'):
                    agents = project_config.responses_api_agents
                    if hasattr(agents, agent_name):
                        agent_config = getattr(agents, agent_name)
                        agent_server = f"http://{agent_config.host}:{agent_config.port}"
                        return agent_server

            raise ValueError(f"Agent '{agent_name}' not found in any project's responses_api_agents")

        # find agent name if not specified
        for _, project_config in global_config_dict.items():
            if hasattr(project_config, 'responses_api_agents'):
                agents = project_config.responses_api_agents
                for name in agents.keys():
                    agent_config = getattr(agents, name)
                    if hasattr(agent_config, 'host') and hasattr(agent_config, 'port'):
                        agent_server = f"http://{agent_config.host}:{agent_config.port}"
                        return agent_server

        raise ValueError("No agents found in global config")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to connect to head server at {head_server_host}:{head_server_port}: {e}")


def convert_nemo_gym_responses_to_trajectory_group(
    responses: List[Dict[str, Any]]
) -> TrajectoryGroup:
    """
    Convert nemo gym responses to tinker TrajectoryGroup.

    Args:
        responses: List of nemo gym agent responses with format:
            {
                "reward": float,
                "response": {
                    "output": [
                        {
                            "type": "message" | "reasoning" | "function_call" | "function_call_output",
                            "prompt_token_ids": List[int],  # Full context
                            "generation_token_ids": List[int],  # This turn's generation
                            "generation_log_probs": List[float],
                            ...
                        },
                        ...
                    ]
                }
            }

    Returns:
        TrajectoryGroup with multi-turn trajectories
    """
    trajectories_G: List[Trajectory] = []
    final_rewards_G: List[float] = []
    metrics_G: List[Dict[str, float | int]] = []

    for response in responses:
        transitions: List[Transition] = []
        output_items = response.get("response", {}).get("output", [])

        # Process each turn that has token information
        for i, item in enumerate(output_items):
            # Only process items with token information (model generations)
            if "prompt_token_ids" not in item or "generation_token_ids" not in item:
                continue

            prompt_ids = item["prompt_token_ids"]
            completion_ids = item["generation_token_ids"]
            completion_logprobs = item.get("generation_log_probs", [])

            ob = tinker.ModelInput.from_ints(prompt_ids)
            ac = TokensWithLogprobs(
                tokens=completion_ids,
                maybe_logprobs=completion_logprobs,
            )

            # Check if this is the last turn with token information
            is_last = True
            for j in range(i + 1, len(output_items)):
                if "prompt_token_ids" in output_items[j] and "generation_token_ids" in output_items[j]:
                    is_last = False
                    break

            transition = Transition(
                ob=ob,
                ac=ac,
                reward=0.0,
                episode_done=is_last,
                metrics={},
            )
            transitions.append(transition)

        if not transitions:
            # No valid transitions, create a dummy one to avoid errors
            # TODO should we error
            trajectory = Trajectory(
                transitions=[
                    Transition(
                        ob=tinker.ModelInput.empty(),
                        ac=TokensWithLogprobs(tokens=[], maybe_logprobs=[]),
                        reward=0.0,
                        episode_done=True,
                        metrics={},
                    )
                ],
                final_ob=tinker.ModelInput.empty()
            )
            trajectories_G.append(trajectory)
            final_rewards_G.append(0.0)
            metrics_G.append({"error": 1})
        else:
            trajectory = Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty())
            trajectories_G.append(trajectory)
            final_rewards_G.append(response.get("reward", 0.0))

            num_turns = len(transitions)
            total_tokens = sum(len(t.ac.tokens) for t in transitions)
            metrics_G.append({
                "num_turns": num_turns,
                "total_tokens": total_tokens,
            })

    return TrajectoryGroup(
        trajectories_G=trajectories_G,
        final_rewards_G=final_rewards_G,
        metrics_G=metrics_G,
    )


def load_nemo_gym_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Expected format:
    {
        "responses_create_params": {...},
        "expected_answers": [...],
        "metadata": {...},  
        "ground_truth": {...},
        ... other fields depending on resources server
    }
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)

    print(f"Loaded {len(data)} examples from {path}")
    return data


class NemoGymRLDataset(RLDataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        agent_server: str,
        groups_per_batch: int,
    ):
        self.rows = rows
        self.agent_server = agent_server
        self.groups_per_batch = groups_per_batch

    def __len__(self) -> int:
        return (len(self.rows) + self.groups_per_batch - 1) // self.groups_per_batch

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.groups_per_batch
        end = min(len(self.rows), start + self.groups_per_batch)
        builders: List[EnvGroupBuilder] = []
        for j in range(start, end):
            row = self.rows[j]
            builders.append(
                NemoGymEnvGroupBuilder(
                    agent_server=self.agent_server,
                    dataset_item=row,
                )
            )
        return builders


@chz.chz
class NemoGymRLDatasetBuilder(RLDatasetBuilder):
    dataset_path: str
    agent_server: str | None = None
    head_server_host: str = "127.0.0.1"
    head_server_port: int = 11000
    agent_name: str | None = None
    groups_per_batch: int = 32
    dataset_n: int = -1

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        agent_server = self.agent_server
        if agent_server is None:
            agent_server = get_nemo_gym_agent_server()
        if agent_server is None:
            agent_server = get_agent_server_from_head(
                self.head_server_host,
                self.head_server_port,
                self.agent_name,
            )
            set_nemo_gym_agent_server(agent_server)

        print(f"Using nemo gym agent server: {agent_server}")

        rows = load_nemo_gym_dataset(self.dataset_path)

        if self.dataset_n > 0:
            rows = rows[:self.dataset_n]
            print(f"Limited dataset to {len(rows)} examples")

        return NemoGymRLDataset(rows, agent_server, self.groups_per_batch), None


class NemoGymEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        agent_server: str,
        dataset_item: Dict[str, Any],
    ):
        self.agent_server = agent_server
        self.dataset_item = dataset_item

    async def make_envs(self):
        return []

    def logging_tags(self) -> List[str]:
        metadata = self.dataset_item.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        task = metadata.get("task", "")
        return [task] if task else []
