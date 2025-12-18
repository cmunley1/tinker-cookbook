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
    # TODO: Implement on-policy token ID fix https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/generation/vllm/vllm_worker_async.py#L40

    if _global_client is None:
        return JSONResponse(
            status_code=503, content={"error": "Tinker client not initialized"}
        )

    body: Dict[str, Any] = await request.json()
    logger.info(f"Received chat completion request with keys: {list(body.keys())}")

    try:
        response = await _global_client.chat.completions.create(**body)
        logger.info(f"Successfully created chat completion")

        response_dict = {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": choice.message.model_dump(),
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


@app.post("/tokenize")
async def tokenize(request: Request) -> JSONResponse:
    if _global_client is None:
        return JSONResponse(
            status_code=503, content={"error": "Tinker client not initialized"}
        )

    body: Dict[str, Any] = await request.json()
    logger.info(f"Received tokenize request with keys: {list(body.keys())}")

    try:
        from tinker_cookbook.renderers import ToolCall

        messages = body.get("messages", [])

        if not messages:
            logger.warning("No messages provided in tokenize request")
            return JSONResponse(content={"tokens": []})

        for msg in messages:
            if "tool_calls" in msg and msg["tool_calls"]:
                normalized_tool_calls = []
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        normalized_tool_calls.append(
                            ToolCall(
                                id=tc.get("id"),
                                type=tc.get("type", "function"),
                                function=ToolCall.FunctionBody(
                                    name=func.get("name", ""),
                                    arguments=func.get("arguments", "{}"),
                                )
                            )
                        )
                    else:
                        normalized_tool_calls.append(tc)
                msg["tool_calls"] = normalized_tool_calls

        model_input = _global_client.renderer.build_generation_prompt(messages)
        prompt_token_ids = model_input.to_ints()

        logger.info(f"Tokenized {len(messages)} messages -> {len(prompt_token_ids)} tokens")

        # vllm_model format
        response_dict = {
            "tokens": prompt_token_ids
        }

        return JSONResponse(content=response_dict)

    except Exception as e:
        logger.error(f"Error in tokenize: {e}", exc_info=True)
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
