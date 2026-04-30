from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from .auth import ChatGPTOAuthError, ChatGPTOAuthMissingError, is_auth_locally_available
from .messages import Message, MessageRole, ToolSchema
from .provider import ChatGPTOAuthProvider


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is not None and value.isdigit():
        return int(value)
    return default


def _env_str(name: str, default: str) -> str:
    return os.getenv(name) or default


HOST = _env_str("CODEX_AS_API_HOST", "0.0.0.0")
PORT = _env_int("CODEX_AS_API_PORT", 18080)
MODEL = _env_str("CODEX_AS_API_MODEL", "gpt-5.5")
AUTH_PATH = os.getenv("CODEX_AS_API_AUTH_PATH")

_provider: ChatGPTOAuthProvider | None = None


def _get_provider() -> ChatGPTOAuthProvider:
    global _provider
    if _provider is None:
        _provider = ChatGPTOAuthProvider(
            model=MODEL,
            auth_json_path=AUTH_PATH,
        )
    return _provider


# FastAPI is an optional dependency; fail gracefully if missing.
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field

    app = FastAPI(
        title="codex-as-api",
        description="Local OpenAI-compatible API server backed by ChatGPT/Codex OAuth.",
        version="0.1.0",
    )

    @app.exception_handler(ChatGPTOAuthError)
    async def _chatgpt_oauth_error_handler(_request: Request, exc: ChatGPTOAuthError) -> JSONResponse:
        status = 401 if isinstance(exc, ChatGPTOAuthMissingError) else 500
        return JSONResponse(status_code=status, content={"error": {"message": str(exc), "type": "chatgpt_oauth_error"}})

    # ------------------------------------------------------------------
    # Request/response schemas
    # ------------------------------------------------------------------

    class ChatMessage(BaseModel):
        role: str
        content: str | list[dict[str, Any]] | None = None
        name: str | None = None
        tool_calls: list[dict[str, Any]] | None = None
        tool_call_id: str | None = None

    class ChatCompletionRequest(BaseModel):
        model: str
        messages: list[ChatMessage]
        stream: bool = False
        temperature: float | None = None
        max_tokens: int | None = None
        max_completion_tokens: int | None = None
        stop: str | list[str] | None = None
        tools: list[dict[str, Any]] | None = None
        tool_choice: str | dict[str, Any] | None = None
        reasoning_effort: str | None = None
        prompt_cache_key: str | None = None
        top_p: float | None = None
        frequency_penalty: float | None = None
        presence_penalty: float | None = None
        user: str | None = None
        subagent: str | None = None
        memgen_request: bool | None = None
        previous_response_id: str | None = None
        service_tier: str | None = None
        text: dict[str, Any] | None = None
        client_metadata: dict[str, str] | None = None

    class ImageGenerationRequest(BaseModel):
        model: str
        prompt: str
        size: str | None = "auto"
        reasoning_effort: str | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _openai_model_id(request_model: str | None = None) -> str:
        return f"codex-oauth:{request_model or MODEL}"

    def _request_messages_to_internal(messages: list[ChatMessage]) -> list[Message]:
        result: list[Message] = []
        for msg in messages:
            role = _map_role(msg.role)
            content = _normalize_content(msg.content)
            tool_calls = _parse_tool_calls(msg.tool_calls) if msg.tool_calls else ()
            result.append(
                Message(
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            )
        return result

    def _map_role(role: str) -> MessageRole:
        mapping = {
            "system": MessageRole.SYSTEM,
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "tool": MessageRole.TOOL,
        }
        return mapping.get(role.lower(), MessageRole.USER)

    def _normalize_content(content: str | list[dict[str, Any]] | None) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return str(content)

    def _parse_tool_calls(raw: list[dict[str, Any]] | None) -> tuple[Any, ...]:
        from .messages import ToolCall
        if not raw:
            return ()
        calls = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            call_id = item.get("id") or item.get("call_id") or str(uuid.uuid4().hex)
            func = item.get("function") or {}
            name = func.get("name") if isinstance(func, dict) else item.get("name")
            args = func.get("arguments") if isinstance(func, dict) else item.get("arguments")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    parsed = {"input": args}
            elif isinstance(args, dict):
                parsed = args
            else:
                parsed = {}
            if name:
                calls.append(ToolCall(id=str(call_id), name=str(name), arguments=parsed))
        return tuple(calls)

    def _parse_tools(raw: list[dict[str, Any]] | None) -> list[ToolSchema] | None:
        if not raw:
            return None
        schemas = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            func = item.get("function") or item
            name = func.get("name")
            desc = func.get("description") or ""
            params = func.get("parameters") or {}
            if name:
                schemas.append(ToolSchema(name=str(name), description=str(desc), parameters=params if isinstance(params, dict) else {}))
        return schemas if schemas else None

    def _normalize_stop(stop: str | list[str] | None) -> list[str] | None:
        if stop is None:
            return None
        if isinstance(stop, str):
            return [stop]
        return list(stop)

    def _max_tokens_from_request(req: ChatCompletionRequest) -> int | None:
        if req.max_completion_tokens is not None:
            return req.max_completion_tokens
        return req.max_tokens

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "auth_available": is_auth_locally_available(AUTH_PATH),
            "model": MODEL,
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: ChatCompletionRequest, http_request: Request) -> JSONResponse | StreamingResponse:
        provider = _get_provider()
        messages = _request_messages_to_internal(request.messages)
        tools = _parse_tools(request.tools)
        stop = _normalize_stop(request.stop)
        max_tokens = _max_tokens_from_request(request)

        subagent = request.subagent or http_request.headers.get("x-openai-subagent")
        memgen_request_header = http_request.headers.get("x-openai-memgen-request")
        memgen_request: bool | None = request.memgen_request
        if memgen_request is None and memgen_request_header is not None:
            memgen_request = memgen_request_header.lower() not in ("false", "0", "")
        previous_response_id = request.previous_response_id

        if request.stream:
            async def _stream() -> AsyncIterator[str]:
                request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                created = int(time.time())
                model = _openai_model_id(request.model)

                # SSE preamble
                preamble = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(preamble)}\n\n"

                reasoning_parts: list[str] = []
                content_parts: list[str] = []
                tool_calls_buffer: list[dict[str, Any]] = []
                usage_dict: dict[str, Any] | None = None

                for event in provider.chat_stream(
                    messages,
                    model=request.model,
                    tools=tools,
                    tool_choice=request.tool_choice,
                    temperature=request.temperature,
                    reasoning_effort=request.reasoning_effort,
                    max_tokens=max_tokens,
                    stop=stop,
                    prompt_cache_key=request.prompt_cache_key,
                    subagent=subagent,
                    memgen_request=memgen_request,
                    previous_response_id=previous_response_id,
                    service_tier=request.service_tier,
                    text=request.text,
                    client_metadata=request.client_metadata,
                ):
                    typ = event.get("type")
                    if typ == "content":
                        text = str(event.get("text", ""))
                        content_parts.append(text)
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif typ == "reasoning_delta":
                        text = str(event.get("text", ""))
                        reasoning_parts.append(text)
                        # OpenAI-compatible reasoning field
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"reasoning_content": text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif typ == "reasoning_raw_delta":
                        text = str(event.get("text", ""))
                        reasoning_parts.append(text)
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"reasoning": text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif typ == "tool_call":
                        tc = {
                            "id": event.get("id"),
                            "type": "function",
                            "function": {
                                "name": event.get("name"),
                                "arguments": json.dumps(event.get("arguments") or {}),
                            },
                        }
                        tool_calls_buffer.append(tc)
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"tool_calls": [tc]},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif typ == "finish":
                        usage = event.get("usage")
                        if isinstance(usage, dict):
                            usage_dict = usage
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": event.get("finish_reason") or "stop",
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                # Usage summary chunk if available
                if usage_dict:
                    u = usage_dict
                    finish_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": u.get("prompt_tokens", 0),
                            "completion_tokens": u.get("completion_tokens", 0),
                            "total_tokens": u.get("total_tokens", 0),
                        },
                    }
                    yield f"data: {json.dumps(finish_chunk)}\n\n"

                yield "data: [DONE]\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")

        # Non-streaming
        response = provider.chat(
            messages,
            model=request.model,
            tools=tools,
            tool_choice=request.tool_choice,
            temperature=request.temperature,
            reasoning_effort=request.reasoning_effort,
            max_tokens=max_tokens,
            stop=stop,
            prompt_cache_key=request.prompt_cache_key,
            subagent=subagent,
            memgen_request=memgen_request,
            previous_response_id=previous_response_id,
            service_tier=request.service_tier,
            text=request.text,
            client_metadata=request.client_metadata,
        )

        choices: list[dict[str, Any]] = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response.content,
            },
            "finish_reason": response.finish_reason,
        }]

        if response.tool_calls:
            choices[0]["message"]["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in response.tool_calls
            ]

        if response.reasoning_content:
            choices[0]["message"]["reasoning_content"] = response.reasoning_content

        result: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _openai_model_id(request.model),
            "choices": choices,
        }

        if response.usage:
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens or (response.usage.prompt_tokens + response.usage.completion_tokens),
            }

        return JSONResponse(content=result)

    @app.post("/v1/images/generations")
    async def images_generations(request: ImageGenerationRequest) -> JSONResponse:
        provider = _get_provider()
        images = provider.generate_image(
            request.prompt,
            model=request.model,
            size=request.size,
            reasoning_effort=request.reasoning_effort,
        )
        data = [
            {
                "url": image.get("result"),
                "revised_prompt": image.get("revised_prompt") or request.prompt,
            }
            for image in images
            if image.get("result")
        ]
        return JSONResponse(content={"created": int(time.time()), "data": data})

    # ------------------------------------------------------------------
    # Anthropic Messages API compatible endpoint
    # ------------------------------------------------------------------

    from .anthropic_adapter import (
        anthropic_request_to_internal,
        internal_response_to_anthropic,
        anthropic_stream_adapter,
        format_anthropic_error,
    )

    @app.post("/v1/messages", response_model=None)
    async def anthropic_messages(http_request: Request) -> JSONResponse | StreamingResponse:
        provider = _get_provider()
        body = await http_request.json()

        try:
            messages, tools, tool_choice, stop, reasoning_effort = anthropic_request_to_internal(
                model=body.get("model", MODEL),
                messages=body.get("messages", []),
                system=body.get("system"),
                max_tokens=body.get("max_tokens", 4096),
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                stop_sequences=body.get("stop_sequences"),
                thinking=body.get("thinking"),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content=format_anthropic_error(400, str(exc)))

        stream = body.get("stream", False)
        client_model = body.get("model") or MODEL

        if stream:
            async def _stream() -> AsyncIterator[str]:
                request_id = f"msg_{uuid.uuid4().hex[:24]}"
                for sse_chunk in anthropic_stream_adapter(
                    provider.chat_stream(
                        messages,
                        model=MODEL,
                        tools=tools,
                        tool_choice=tool_choice,
                        reasoning_effort=reasoning_effort,
                        stop=stop,
                    ),
                    model=client_model,
                    request_id=request_id,
                ):
                    yield sse_chunk

            try:
                return StreamingResponse(_stream(), media_type="text/event-stream")
            except ChatGPTOAuthError as exc:
                status = 401 if isinstance(exc, ChatGPTOAuthMissingError) else 500
                return JSONResponse(status_code=status, content=format_anthropic_error(status, str(exc)))

        # Non-streaming
        try:
            response = provider.chat(
                messages,
                model=MODEL,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
                stop=stop,
            )
        except ChatGPTOAuthError as exc:
            status = 401 if isinstance(exc, ChatGPTOAuthMissingError) else 500
            return JSONResponse(status_code=status, content=format_anthropic_error(status, str(exc)))

        request_id = f"msg_{uuid.uuid4().hex[:24]}"
        result = internal_response_to_anthropic(response, client_model, request_id)
        return JSONResponse(content=result)

    # ------------------------------------------------------------------
    # Custom endpoints (not in standard OpenAI API, but exposed for full feature routing)
    # ------------------------------------------------------------------

    @app.post("/v1/inspect")
    async def inspect(request: Request) -> JSONResponse:
        """Inspect images with a text prompt.

        Body: {"prompt": str, "images": [{"image_url": "data:image/..."}, ...], "reasoning_effort": str?}
        """
        provider = _get_provider()
        body = await request.json()
        prompt = str(body.get("prompt", ""))
        images = body.get("images") or []
        reasoning_effort = body.get("reasoning_effort")
        result = provider.inspect_images(prompt, images=images, reasoning_effort=reasoning_effort)
        return JSONResponse(content={"content": result})

    @app.post("/v1/compact")
    async def compact(request: Request) -> JSONResponse:
        """Compact a conversation into a checkpoint for continuation.

        Body: {"messages": [{"role": "system|user|assistant|tool", "content": str, ...}], "reasoning_effort": str?}
        """
        provider = _get_provider()
        body = await request.json()
        raw_messages = body.get("messages") or []
        messages = _request_messages_to_internal([ChatMessage.model_validate(m) for m in raw_messages])
        reasoning_effort = body.get("reasoning_effort")
        checkpoint = provider.compact_messages(messages, reasoning_effort=reasoning_effort)
        return JSONResponse(content={"checkpoint": checkpoint})

    # ------------------------------------------------------------------
    # CLI entry point
    # ------------------------------------------------------------------

    def main() -> None:
        import uvicorn
        uvicorn.run("codex_as_api.server:app", host=HOST, port=PORT, log_level="info")

except ImportError as _import_exc:
    # FastAPI / uvicorn not installed
    app = None  # type: ignore[assignment]

    def main() -> None:  # type: ignore[misc]
        raise ImportError(
            "FastAPI and uvicorn are required to run the server. "
            "Install with: pip install 'codex-as-api[server]'"
        ) from _import_exc
