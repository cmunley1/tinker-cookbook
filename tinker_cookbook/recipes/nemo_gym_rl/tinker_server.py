"""
FastAPI server that exposes Tinker's sampling client as an OpenAI-compatible HTTP API.
This allows NeMo-Gym to call Tinker's inference endpoint during training.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient

logger = logging.getLogger(__name__)

app = FastAPI()

_global_client: TinkerAsyncOpenAIClient | None = None


def set_client(client: TinkerAsyncOpenAIClient) -> None:
    global _global_client
    _global_client = client


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    # TODO: Implement on-policy token ID fix
    # Problem: Multi-turn rollouts re-tokenize between turns causing train-generation mismatch
    # Solution: Accept pre-tokenized input (token IDs) and skip renderer/tokenization
    # Reference: https://docs.nvidia.com/nemo-gym/latest/training-framework-integration/on-policy-corrections/
    #
    # Current flow (INCORRECT for multi-turn):
    #   Turn 1: Tinker generates token_ids -> nemo gym converts to text
    #   Turn 2: nemo gym adds tool results -> Tinker re-tokenizes (MISMATCH!)
    #
    # Correct flow (NEEDED):
    #   Turn 1: Tinker generates token_ids -> nemo gym preserves token_ids
    #   Turn 2: nemo gym appends new token_ids -> Tinker uses them directly (NO re-tokenization)
    #
    # Implementation needed in TinkerAsyncOpenAIClient:
    #   - Accept prompt_token_ids in request body
    #   - Skip renderer.build_generation_prompt if token IDs provided
    #   - Use ModelInput.from_ints(prompt_token_ids) directly
    # 
    # But first, lets get running with mismatch.
    
    
    if _global_client is None:
        return JSONResponse(
            status_code=503, content={"error": "Tinker client not initialized"}
        )

    body: Dict[str, Any] = await request.json()

    try:
        response = await _global_client.chat.completions.create(**body)

        response_dict = {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                    },
                    "finish_reason": choice.finish_reason,
                    "logprobs": (
                        {
                            "content": [
                                {
                                    "token": item.token,
                                    "logprob": item.logprob,
                                    "top_logprobs": item.top_logprobs,
                                }
                                for item in choice.logprobs.content
                            ]
                        }
                        if choice.logprobs
                        else None
                    ),
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

        if hasattr(response, "prompt_token_ids"):
            response_dict["prompt_token_ids"] = response.prompt_token_ids
        if hasattr(response.choices[0], "token_ids"):
            response_dict["choices"][0]["token_ids"] = response.choices[0].token_ids

        return JSONResponse(content=response_dict)

    except Exception as e:
        logger.error(f"Error in chat completions: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy", "client_ready": _global_client is not None}


async def serve_in_background(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)

    await server.serve()
