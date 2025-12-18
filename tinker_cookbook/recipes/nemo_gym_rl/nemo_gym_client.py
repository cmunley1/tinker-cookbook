from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import aiohttp


async def call_nemo_gym_agent(
    agent_server: str,
    dataset_items: List[Dict[str, Any]],
    max_output_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 1.0,
    timeout: float = 600.0,
) -> List[Dict[str, Any]]:
    """
    Returns:
        List of agent responses with format:
        {
            "reward": float,
            "response": {
                "output": [
                    {
                        "prompt_token_ids": List[int],
                        "generation_token_ids": List[int],
                        "generation_log_probs": List[float],
                        ...
                    }
                ]
            }
        }
    """
    print(f"Calling nemo gym agent: {agent_server}")
    print(f"Number of requests: {len(dataset_items)}")
    print(f"Max output tokens: {max_output_tokens}")
    print(f"Temperature: {temperature}, top_p: {top_p}")

    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar()) as session:
        tasks = []
        for i, item in enumerate(dataset_items):
            request_body = item.copy()

            if "responses_create_params" not in request_body:
                request_body["responses_create_params"] = {
                    "input": [{"role": "user", "content": ""}],
                }

            params = request_body["responses_create_params"]
            params.setdefault("max_output_tokens", max_output_tokens)
            params["temperature"] = temperature
            params["top_p"] = top_p

            if i == 0:
                print(f"First request params keys: {list(params.keys())}")

            task = session.post(
                f"{agent_server}/run",
                json=request_body,
                timeout=aiohttp.ClientTimeout(total=timeout),
            )
            tasks.append(task)

        print(f"Awaiting {len(tasks)} HTTP requests...")
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        print(f"Got {len(responses)} responses")

        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"WARNING: Request {i} failed: {response}")
                results.append({
                    "response": {"output": []},
                    "reward": 0.0,
                    "error": str(response)
                })
            else:
                try:
                    json_data = await response.json()
                    if isinstance(json_data, dict):
                        results.append(json_data)
                    else:
                        print(f"WARNING: Request {i} returned non-dict: {type(json_data)}")
                        results.append({
                            "response": {"output": []},
                            "reward": 0.0,
                            "error": "Non-dict response"
                        })
                except Exception as e:
                    print(f"WARNING: Failed to parse response {i}: {e}")
                    results.append({
                        "response": {"output": []},
                        "reward": 0.0,
                        "error": str(e)
                    })

        return results
