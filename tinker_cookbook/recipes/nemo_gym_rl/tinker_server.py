from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Literal, overload
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import tinker
from openai import AsyncOpenAI
from openai._streaming import AsyncStream
from openai.resources.chat import AsyncChat as OpenAIAsyncChat
from openai.resources.chat.completions import AsyncCompletions as OpenAIAsyncChatCompletions
from openai.resources.completions import AsyncCompletions as OpenAIAsyncCompletions
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.renderers import ToolCall

logger = logging.getLogger(__name__)


class TinkerAsyncOpenAIClient(AsyncOpenAI):
    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        tokenizer: Tokenizer,
    ) -> None:
        super().__init__(api_key="tinker", base_url="http://localhost")
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.tokenizer = tokenizer

    def set_sampling_client(self, sampling_client: tinker.SamplingClient) -> None:
        self.sampling_client = sampling_client

    @property
    def chat(self) -> OpenAIAsyncChat:
        return TinkerAsyncChat(self)

    @property
    def completions(self) -> OpenAIAsyncCompletions:
        return TinkerCompletions(self)


class TinkerChatCompletions(OpenAIAsyncChatCompletions):
    def __init__(self, parent: TinkerAsyncOpenAIClient) -> None:
        self._parent = parent

    @overload
    async def create(
        self, *args: Any, stream: Literal[True], **kwargs: Any
    ) -> AsyncStream[Any]: ...

    @overload
    async def create(
        self, *args: Any, stream: Literal[False] = False, **kwargs: Any
    ) -> ChatCompletion: ...

    @overload
    async def create(self, *args: Any, stream: bool, **kwargs: Any) -> ChatCompletion: ...

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion | AsyncStream[Any]:

        model = kwargs.get("model", "tinker")
        messages = kwargs.get("messages", []).copy()
        tools = kwargs.get("tools", [])

        # dict to ToolCall objects
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

        # If tools are provided, inject them into the system message
        # This is a hack, we should use the renderer.
        # This will fail if we expect a different tool call format.
        if tools:
            import json
            tools_text = "\n\n# Available Tools\n" + json.dumps(tools, indent=2)
            tools_text += '\n\nTo call a tool, use: <tool_call>{"name": "tool_name", "args": {...}}</tool_call>'

            if messages and messages[0].get("role") == "system":
                messages[0] = messages[0].copy()
                messages[0]["content"] = messages[0]["content"] + tools_text
            else:
                messages.insert(0, {"role": "system", "content": "You are a helpful assistant." + tools_text})

        if kwargs.get("stream", False):
            raise ValueError("stream=True not supported by TinkerAsyncOpenAIClient")
        sampling_args = {k: v for k, v in kwargs.items() if k not in ("model", "messages", "tools")}

        stop = sampling_args.get("stop", self._parent.renderer.get_stop_sequences())
        max_tokens = sampling_args.get("max_tokens") or sampling_args.get("max_completion_tokens")

        model_input = self._parent.renderer.build_generation_prompt(messages)
        prompt_token_ids: List[int] = model_input.to_ints()

        sample = await self._parent.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=float(sampling_args.get("temperature", 1.0)),
                max_tokens=int(max_tokens or 128),
                top_p=float(sampling_args.get("top_p", 1.0)),
                top_k=int(sampling_args.get("top_k", -1)),
                stop=stop,
            ),
        )
        seq = sample.sequences[0]
        completion_token_ids: List[int] = seq.tokens
        logprobs: List[float] = seq.logprobs or [0.0] * len(completion_token_ids)

        assistant_message, parse_success = self._parent.renderer.parse_response(
            completion_token_ids
        )
        finish_reason = "stop" if parse_success else "length"

        # ToolCall to dict
        message_dict = assistant_message.copy()
        if "tool_calls" in message_dict and message_dict["tool_calls"]:
            import uuid
            message_dict["tool_calls"] = [
                {
                    "id": tc.id if (hasattr(tc, "id") and tc.id) else f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": tc.function.name if hasattr(tc, "function") else tc.get("function", {}).get("name"),
                        "arguments": tc.function.arguments if hasattr(tc, "function") else tc.get("function", {}).get("arguments"),
                    }
                }
                for tc in message_dict["tool_calls"]
            ]

        response_dict: Dict[str, Any] = {
            "id": "tinker-chatcmpl",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": finish_reason,
                    "logprobs": {
                        "content": [
                            {"token": f"token_id:{tid}", "logprob": lp, "top_logprobs": []}
                            for tid, lp in zip(completion_token_ids, logprobs)
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(completion_token_ids),
                "total_tokens": len(prompt_token_ids) + len(completion_token_ids),
            },
        }
        response = ChatCompletion.model_validate(response_dict)

        setattr(response, "prompt_token_ids", prompt_token_ids)
        setattr(response.choices[0], "token_ids", completion_token_ids)

        return response


class TinkerCompletions(OpenAIAsyncCompletions):
    def __init__(self, parent: TinkerAsyncOpenAIClient) -> None:
        self._parent = parent

    @overload
    async def create(
        self, *args: Any, stream: Literal[True], **kwargs: Any
    ) -> AsyncStream[Completion]: ...

    @overload
    async def create(
        self, *args: Any, stream: Literal[False] = False, **kwargs: Any
    ) -> Completion: ...

    @overload
    async def create(
        self, *args: Any, stream: bool, **kwargs: Any
    ) -> Completion | AsyncStream[Completion]: ...

    async def create(self, *args: Any, **kwargs: Any) -> Completion | AsyncStream[Completion]:
        stream = bool(kwargs.get("stream", False))
        model = kwargs.get("model", "tinker")
        prompt = kwargs.get("prompt", "")
        sampling_args = {k: v for k, v in kwargs.items() if k not in ("model", "prompt")}

        prompt_token_ids: List[int] = self._parent.tokenizer.encode(prompt, add_special_tokens=True)
        model_input = tinker.ModelInput.from_ints(prompt_token_ids)

        sample = await self._parent.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=float(sampling_args.get("temperature", 1.0)),
                max_tokens=int(sampling_args.get("max_tokens", 128)),
                top_p=float(sampling_args.get("top_p", 1.0)),
                top_k=int(sampling_args.get("top_k", -1)),
            ),
        )
        seq = sample.sequences[0]
        completion_token_ids: List[int] = seq.tokens
        logprobs: List[float] = seq.logprobs or [0.0] * len(completion_token_ids)

        text = self._parent.tokenizer.decode(completion_token_ids)
        tokens_str = [f"token_id:{tid}" for tid in completion_token_ids]
        response_dict: Dict[str, Any] = {
            "id": "tinker-cmpl",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop",
                    "logprobs": {
                        "tokens": tokens_str,
                        "token_logprobs": logprobs,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(completion_token_ids),
                "total_tokens": len(prompt_token_ids) + len(completion_token_ids),
            },
        }
        response = Completion.model_validate(response_dict)

        setattr(response.choices[0], "prompt_token_ids", prompt_token_ids)
        setattr(response.choices[0], "token_ids", completion_token_ids)

        if stream:
            return TinkerAsyncCompletionStream(response)
        return response


class TinkerAsyncChat(OpenAIAsyncChat):
    def __init__(self, parent: TinkerAsyncOpenAIClient) -> None:
        self._parent = parent

    @property
    def completions(self) -> OpenAIAsyncChatCompletions:
        return TinkerChatCompletions(self._parent)


class TinkerAsyncCompletionStream(AsyncStream[Completion]):
    def __init__(self, final: Completion) -> None:
        self._final = final

    def __aiter__(self):
        self._done = True
        return self

    async def __anext__(self) -> Completion:
        raise StopAsyncIteration

    def __await__(self):
        async def _await_final():
            return self._final

        return _await_final().__await__()

    async def get_final_response(self) -> Completion:
        return self._final


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
