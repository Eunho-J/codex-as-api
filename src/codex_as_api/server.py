from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from typing import Any, cast

from .auth import (
    ChatGPTOAuthError,
    ChatGPTOAuthInvalidRequestError,
    ChatGPTOAuthMissingError,
    is_auth_locally_available,
)
from .codex_config import load_codex_config
from .messages import Message, MessageRole, ToolSchema
from .model_capabilities import capability_for_model, load_model_capabilities
from .o200k_tokenizer import count_ordinary
from .provider import ChatGPTOAuthProvider, prime_codex_cli_version_cache


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is not None and value.isdigit():
        return int(value)
    return default


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    normalized = value.strip() if value is not None else ""
    return normalized or default


HOST = _env_str("CODEX_AS_API_HOST", "127.0.0.1")
PORT = _env_int("CODEX_AS_API_PORT", 18080)
CODEX_CONFIG = load_codex_config()
MODEL = _env_str("CODEX_AS_API_MODEL", CODEX_CONFIG.model or "gpt-5.5")
AUTH_PATH = os.getenv("CODEX_AS_API_AUTH_PATH")
DEFAULT_CONTEXT_WINDOW = 200_000

_provider: ChatGPTOAuthProvider | None = None


def _get_provider() -> ChatGPTOAuthProvider:
    global _provider
    if _provider is None:
        _provider = ChatGPTOAuthProvider(
            model=MODEL,
            auth_json_path=AUTH_PATH,
        )
    return _provider


def _is_context_window_error(exc: BaseException | str) -> bool:
    return "context window" in str(exc).lower()


def _error_status(exc: BaseException) -> int:
    if isinstance(exc, ChatGPTOAuthInvalidRequestError):
        return 400
    if isinstance(exc, ChatGPTOAuthMissingError):
        return 401
    if _is_context_window_error(exc):
        return 400
    return 500


def _anthropic_output_format_from_body(body: dict[str, Any]) -> dict[str, Any] | None:
    output_format = body.get("output_format")
    if output_format is None:
        return None
    if not isinstance(output_format, dict):
        raise ChatGPTOAuthInvalidRequestError("output_format must be an object")
    return cast(dict[str, Any], output_format)


def _anthropic_backend_model(client_model: object) -> str:
    if isinstance(client_model, str) and client_model in load_model_capabilities():
        return client_model
    return MODEL


def _validate_anthropic_context_management(value: object) -> None:
    if value is None:
        return
    if value == {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}:
        return
    raise ChatGPTOAuthInvalidRequestError(
        "context_management is unsupported except for clear_thinking_20251015 with keep='all'"
    )


def _anthropic_service_tier(body: dict[str, Any]) -> str | None:
    service_tier = body.get("service_tier")
    if service_tier is not None and (not isinstance(service_tier, str) or not service_tier.strip()):
        raise ChatGPTOAuthInvalidRequestError("service_tier must be a non-empty string when provided")

    speed = body.get("speed")
    if speed is None:
        return cast(str | None, service_tier)
    if speed not in {"fast", "standard"}:
        raise ChatGPTOAuthInvalidRequestError("speed must be one of: fast, standard")
    speed_tier = "fast" if speed == "fast" else "default"
    equivalent_tiers = {"fast", "priority"} if speed == "fast" else {"default"}
    if service_tier is not None and service_tier not in equivalent_tiers:
        raise ChatGPTOAuthInvalidRequestError("speed conflicts with service_tier")
    return speed_tier


def _merge_anthropic_text(
    converted: dict[str, Any] | None,
    direct: object,
) -> dict[str, Any] | None:
    if direct is None:
        return converted
    if not isinstance(direct, dict):
        raise ChatGPTOAuthInvalidRequestError("text must be an object")
    merged = dict(converted or {})
    for key, value in direct.items():
        if key in merged and merged[key] != value:
            raise ChatGPTOAuthInvalidRequestError(f"text.{key} conflicts with Anthropic output format")
        merged[key] = value
    return merged


def _error_type(exc: BaseException) -> str:
    if isinstance(exc, ChatGPTOAuthError):
        return "chatgpt_oauth_error"
    return "server_error"


# FastAPI is an optional dependency; fail gracefully if missing.
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(
        title="codex-as-api",
        description="Local OpenAI-compatible API server backed by ChatGPT/Codex OAuth.",
        version="0.6.2",
    )

    @app.exception_handler(ChatGPTOAuthError)
    async def _chatgpt_oauth_error_handler(_request: Request, exc: ChatGPTOAuthError) -> JSONResponse:
        status = _error_status(exc)
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
        reasoning: dict[str, Any] | None = None
        prompt_cache_key: str | None = None
        prompt_cache_options: dict[str, Any] | None = None
        top_p: float | None = None
        frequency_penalty: float | None = None
        presence_penalty: float | None = None
        user: str | None = None
        subagent: str | None = None
        memgen_request: bool | None = None
        previous_response_id: str | None = None
        service_tier: str | None = None
        text: dict[str, Any] | None = None
        verbosity: str | None = None
        safety_identifier: str | None = None
        client_metadata: dict[str, str] | None = None
        codex_metadata: bool | None = None
        responses_lite: bool | str | None = None
        parallel_tool_calls: bool | None = None
        multi_agent: dict[str, Any] | None = None
        programmatic_tool_calling: Any | None = None

    class ImageGenerationRequest(BaseModel):
        model: str
        prompt: str
        size: str | None = "auto"
        reasoning_effort: str | None = None
        reasoning: dict[str, Any] | None = None
        responses_lite: bool | str | None = None
        prompt_cache_options: dict[str, Any] | None = None
        verbosity: str | None = None
        safety_identifier: str | None = None
        multi_agent: dict[str, Any] | None = None
        programmatic_tool_calling: Any | None = None
        tools: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _openai_model_id(request_model: str | None = None) -> str:
        return f"codex-oauth:{request_model or MODEL}"

    def _request_messages_to_internal(messages: list[ChatMessage], *, model: str) -> list[Message]:
        result: list[Message] = []
        for msg in messages:
            role = _map_role(msg.role)
            content, content_parts = _normalize_content(role, msg.content, model=model)
            tool_calls = _parse_tool_calls(msg.tool_calls) if msg.tool_calls else ()
            result.append(
                Message(
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                    content_parts=content_parts,
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

    def _normalize_content(
        role: MessageRole,
        content: str | list[dict[str, Any]] | None,
        *,
        model: str,
    ) -> tuple[str, tuple[dict[str, object], ...]]:
        if content is None:
            return "", ()
        if isinstance(content, str):
            return content, ()
        if isinstance(content, list):
            text_parts: list[str] = []
            wire_parts: list[dict[str, object]] = []
            for index, item in enumerate(content):
                if not isinstance(item, dict):
                    raise ChatGPTOAuthInvalidRequestError(f"message content item {index} must be an object")
                item_type = item.get("type")
                if item.get("prompt_cache_breakpoint") is not None:
                    raise ChatGPTOAuthInvalidRequestError(
                        "prompt_cache_breakpoint is not supported by the ChatGPT Codex OAuth transport"
                    )
                if item_type in {"text", "input_text", "output_text"}:
                    text = item.get("text")
                    if not isinstance(text, str):
                        raise ChatGPTOAuthInvalidRequestError(
                            f"message text content item {index} requires a string text field"
                        )
                    text_parts.append(text)
                    wire: dict[str, object] = {
                        "type": "output_text" if role is MessageRole.ASSISTANT else "input_text",
                        "text": text,
                    }
                    wire_parts.append(wire)
                    continue
                if item_type in {"image_url", "input_image"}:
                    if role is not MessageRole.USER:
                        raise ChatGPTOAuthInvalidRequestError("image content is only supported on user messages")
                    raw_image = item.get("image_url")
                    if isinstance(raw_image, dict):
                        image_url = raw_image.get("url")
                        detail = raw_image.get("detail", item.get("detail"))
                    else:
                        image_url = raw_image
                        detail = item.get("detail")
                    if not isinstance(image_url, str) or image_url == "":
                        raise ChatGPTOAuthInvalidRequestError(
                            f"message image content item {index} requires a non-empty image URL"
                        )
                    if detail is not None and (
                        not isinstance(detail, str) or detail not in {"auto", "low", "high", "original"}
                    ):
                        raise ChatGPTOAuthInvalidRequestError("image detail must be one of: auto, low, high, original")
                    image_wire: dict[str, object] = {
                        "type": "input_image",
                        "image_url": image_url,
                    }
                    if detail is not None:
                        image_wire["detail"] = cast(str, detail)
                    wire_parts.append(image_wire)
                    continue
                raise ChatGPTOAuthInvalidRequestError(
                    f"message content type {item_type!r} is not supported by the Codex Responses adapter"
                )
            return "".join(text_parts), tuple(wire_parts)
        raise ChatGPTOAuthInvalidRequestError("message content must be a string, array, or null")

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
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ChatGPTOAuthInvalidRequestError(f"tools item {index} must be an object")
            if item.get("type") == "programmatic_tool_calling":
                raise ChatGPTOAuthInvalidRequestError(
                    "programmatic_tool_calling requires native Responses program/caller replay "
                    "and is not supported by this Chat facade"
                )
            func = item.get("function") or item
            if not isinstance(func, dict):
                raise ChatGPTOAuthInvalidRequestError(f"tools item {index} function must be an object")
            if any(key in func or key in item for key in ("allowed_callers", "output_schema")):
                raise ChatGPTOAuthInvalidRequestError(
                    "allowed_callers and output_schema require native Programmatic Tool Calling lifecycle support"
                )
            name = func.get("name")
            desc = func.get("description") or ""
            params = func.get("parameters") or {}
            if name:
                schemas.append(
                    ToolSchema(
                        name=str(name), description=str(desc), parameters=params if isinstance(params, dict) else {}
                    )
                )
        return schemas if schemas else None

    def _reject_unsupported_generation_features(body: dict[str, Any]) -> None:
        if body.get("multi_agent") is not None:
            raise ChatGPTOAuthInvalidRequestError(
                "multi_agent requires native Responses beta agent-item lifecycle support"
            )
        if body.get("programmatic_tool_calling") is not None:
            raise ChatGPTOAuthInvalidRequestError(
                "programmatic_tool_calling requires native Responses program/caller replay support"
            )
        raw_tools = body.get("tools")
        if raw_tools is None:
            return
        if not isinstance(raw_tools, list):
            raise ChatGPTOAuthInvalidRequestError("tools must be an array when provided")
        for index, item in enumerate(raw_tools):
            if not isinstance(item, dict):
                raise ChatGPTOAuthInvalidRequestError(f"tools item {index} must be an object")
            if item.get("type") == "programmatic_tool_calling":
                raise ChatGPTOAuthInvalidRequestError(
                    "programmatic_tool_calling requires native Responses program/caller replay support"
                )
            func = item.get("function")
            function_fields = func if isinstance(func, dict) else {}
            if "allowed_callers" in item or "allowed_callers" in function_fields:
                raise ChatGPTOAuthInvalidRequestError(
                    "allowed_callers requires native Programmatic Tool Calling lifecycle support"
                )
            if "output_schema" in item or "output_schema" in function_fields:
                raise ChatGPTOAuthInvalidRequestError(
                    "output_schema requires native Programmatic Tool Calling lifecycle support"
                )

    def _reasoning_fields(
        legacy_effort: str | None,
        reasoning: object,
    ) -> tuple[str | None, str | None, str | None]:
        if reasoning is None:
            return _request_reasoning_effort(legacy_effort), None, None
        if not isinstance(reasoning, dict):
            raise ChatGPTOAuthInvalidRequestError("reasoning must be an object")
        unknown = sorted(set(reasoning) - {"effort", "mode", "context"})
        if unknown:
            raise ChatGPTOAuthInvalidRequestError("reasoning contains unsupported fields: " + ", ".join(unknown))
        nested_effort = reasoning.get("effort")
        if nested_effort is not None and (not isinstance(nested_effort, str) or nested_effort == ""):
            raise ChatGPTOAuthInvalidRequestError("reasoning.effort must be a non-empty string when provided")
        if legacy_effort is not None and nested_effort is not None and legacy_effort != nested_effort:
            raise ChatGPTOAuthInvalidRequestError("reasoning_effort conflicts with reasoning.effort")
        mode = reasoning.get("mode")
        if mode is not None and (not isinstance(mode, str) or mode not in {"standard", "pro"}):
            raise ChatGPTOAuthInvalidRequestError("reasoning.mode must be one of: standard, pro")
        context = reasoning.get("context")
        if context is not None and (
            not isinstance(context, str) or context not in {"auto", "current_turn", "all_turns"}
        ):
            raise ChatGPTOAuthInvalidRequestError("reasoning.context must be one of: auto, current_turn, all_turns")
        requested_effort = cast(str | None, nested_effort or legacy_effort)
        return (
            _request_reasoning_effort(requested_effort),
            cast(str | None, mode),
            cast(str | None, context),
        )

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

    def _context_window(model: str | None = None) -> int:
        selected_model = MODEL if model is None else model
        capability = capability_for_model(selected_model)
        if CODEX_CONFIG.model_context_window is not None:
            if capability.max_context_window is not None:
                return min(CODEX_CONFIG.model_context_window, capability.max_context_window)
            return CODEX_CONFIG.model_context_window
        return capability.context_window or DEFAULT_CONTEXT_WINDOW

    def _auto_compact_token_limit(model: str | None = None) -> int:
        selected_model = MODEL if model is None else model
        has_resolved_context = (
            CODEX_CONFIG.model_context_window is not None
            or capability_for_model(selected_model).context_window is not None
        )
        if CODEX_CONFIG.model_auto_compact_token_limit is not None:
            return min(CODEX_CONFIG.model_auto_compact_token_limit, _context_window(selected_model) * 9 // 10)
        if has_resolved_context:
            return _context_window(selected_model) * 9 // 10
        return _context_window(selected_model) * 4 // 5

    def _request_reasoning_effort(requested: str | None) -> str | None:
        if requested is not None:
            if not isinstance(requested, str) or requested == "":
                raise ChatGPTOAuthInvalidRequestError("reasoning_effort must be a non-empty string when provided")
            return requested
        effort = CODEX_CONFIG.model_reasoning_effort
        if effort is not None and (not isinstance(effort, str) or effort == ""):
            raise ChatGPTOAuthError("reasoning_effort must be a non-empty string when provided")
        return effort

    def _configured_reasoning_effort() -> str | None:
        configured = CODEX_CONFIG.model_reasoning_effort
        effort = configured if configured is not None else capability_for_model(MODEL).default_reasoning_effort
        if effort is not None and (not isinstance(effort, str) or effort == ""):
            raise ChatGPTOAuthError("reasoning_effort must be a non-empty string when provided")
        return effort

    def _messages_from_compact_body(
        body: dict[str, Any],
        *,
        anthropic: bool = False,
    ) -> tuple[list[Message], list[ToolSchema] | None, str | None, dict[str, Any] | None]:
        messages_value = body.get("messages")
        if anthropic or any(key in body for key in ("system", "thinking", "tool_choice", "stop_sequences")):
            _validate_anthropic_context_management(body.get("context_management"))
            anthropic_messages = cast(list[dict[str, Any]], messages_value) if isinstance(messages_value, list) else []
            max_tokens_value = body.get("max_tokens")
            try:
                messages, tools, _tool_choice, _stop, reasoning_effort, text = anthropic_request_to_internal(
                    model=str(body.get("model") or MODEL),
                    messages=anthropic_messages,
                    system=body.get("system"),
                    max_tokens=max_tokens_value if isinstance(max_tokens_value, int) else 4096,
                    tools=body.get("tools") if isinstance(body.get("tools"), list) else None,
                    tool_choice=body.get("tool_choice") if isinstance(body.get("tool_choice"), dict) else None,
                    stop_sequences=body.get("stop_sequences") if isinstance(body.get("stop_sequences"), list) else None,
                    thinking=body.get("thinking") if isinstance(body.get("thinking"), dict) else None,
                    output_format=_anthropic_output_format_from_body(body),
                    output_config=body.get("output_config"),
                )
            except ValueError as exc:
                raise ChatGPTOAuthInvalidRequestError(str(exc)) from exc
            return messages, tools, reasoning_effort, text

        raw_messages: list[Any] = messages_value if isinstance(messages_value, list) else []
        messages = _request_messages_to_internal(
            [ChatMessage.model_validate(m) for m in raw_messages],
            model=MODEL,
        )
        raw_tools = body.get("tools") if isinstance(body.get("tools"), list) else None
        return messages, _parse_tools(raw_tools), None, None

    def _json_token_count(value: Any) -> int:
        serialized = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
        return count_ordinary(serialized)

    def _estimate_input_tokens(messages: list[Message], tools: list[ToolSchema] | None = None) -> int:
        """Count the model-visible request with bundled o200k_base encode_ordinary."""
        total = 8
        image_tokens = 0
        for message in messages:
            total += 3 + count_ordinary(message.role.value) + count_ordinary(message.content)
            image_tokens += len(message.images) * 8500
            if message.tool_calls:
                total += _json_token_count(
                    [
                        {
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }
                        for tool_call in message.tool_calls
                    ]
                )
            if message.tool_call_id:
                total += count_ordinary(message.tool_call_id)
            if message.name:
                total += count_ordinary(message.name)
            if message.reasoning_content:
                total += count_ordinary(message.reasoning_content)
        if tools:
            total += _json_token_count(
                [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    for tool in tools
                ]
            )
        return total + image_tokens

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "auth_available": is_auth_locally_available(AUTH_PATH),
            "model": MODEL,
            "codex_config_path": CODEX_CONFIG.config_path,
            "context_window": _context_window(),
            "auto_compact_token_limit": _auto_compact_token_limit(),
            "reasoning_effort": _configured_reasoning_effort(),
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: ChatCompletionRequest, http_request: Request
    ) -> JSONResponse | StreamingResponse:
        provider = _get_provider()
        _reject_unsupported_generation_features(
            {
                "multi_agent": request.multi_agent,
                "programmatic_tool_calling": request.programmatic_tool_calling,
                "tools": request.tools,
            }
        )
        messages = _request_messages_to_internal(request.messages, model=request.model)
        tools = _parse_tools(request.tools)
        stop = _normalize_stop(request.stop)
        max_tokens = _max_tokens_from_request(request)

        subagent = request.subagent or http_request.headers.get("x-openai-subagent")
        memgen_request_header = http_request.headers.get("x-openai-memgen-request")
        memgen_request: bool | None = request.memgen_request
        if memgen_request is None and memgen_request_header is not None:
            memgen_request = memgen_request_header.lower() not in ("false", "0", "")
        previous_response_id = request.previous_response_id
        reasoning_effort, reasoning_mode, reasoning_context = _reasoning_fields(
            request.reasoning_effort,
            request.reasoning,
        )

        if request.stream:
            prepared_request = provider.preflight_chat(
                messages,
                model=request.model,
                tools=tools,
                tool_choice=request.tool_choice,
                temperature=request.temperature,
                reasoning_effort=reasoning_effort,
                reasoning_mode=reasoning_mode,
                reasoning_context=reasoning_context,
                max_tokens=max_tokens,
                stop=stop,
                prompt_cache_key=request.prompt_cache_key,
                previous_response_id=previous_response_id,
                service_tier=request.service_tier,
                text=request.text,
                client_metadata=request.client_metadata,
                codex_metadata=request.codex_metadata,
                responses_lite=request.responses_lite,
                parallel_tool_calls=request.parallel_tool_calls,
                safety_identifier=request.safety_identifier,
                prompt_cache_options=request.prompt_cache_options,
                verbosity=request.verbosity,
            )

            def _stream() -> Iterator[str]:
                request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                created = int(time.time())
                model = _openai_model_id(request.model)

                # SSE preamble
                preamble = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(preamble)}\n\n"

                reasoning_parts: list[str] = []
                content_parts: list[str] = []
                tool_calls_buffer: list[dict[str, Any]] = []
                usage_dict: dict[str, Any] | None = None

                def _provider_events() -> Iterator[dict[str, Any]]:
                    try:
                        yield from provider.chat_stream(
                            messages,
                            model=request.model,
                            tools=tools,
                            tool_choice=request.tool_choice,
                            temperature=request.temperature,
                            reasoning_effort=reasoning_effort,
                            reasoning_mode=reasoning_mode,
                            reasoning_context=reasoning_context,
                            max_tokens=max_tokens,
                            stop=stop,
                            prompt_cache_key=request.prompt_cache_key,
                            subagent=subagent,
                            memgen_request=memgen_request,
                            previous_response_id=previous_response_id,
                            service_tier=request.service_tier,
                            text=request.text,
                            client_metadata=request.client_metadata,
                            codex_metadata=request.codex_metadata,
                            responses_lite=request.responses_lite,
                            parallel_tool_calls=request.parallel_tool_calls,
                            safety_identifier=request.safety_identifier,
                            prompt_cache_options=request.prompt_cache_options,
                            verbosity=request.verbosity,
                            _prepared_request=prepared_request,
                        )
                    except Exception as exc:  # noqa: BLE001 - serialize runtime stream failures in-band
                        yield {"type": "_stream_error", "error": exc}

                for event in _provider_events():
                    typ = event.get("type")
                    if typ == "_stream_error":
                        exc = event["error"]
                        yield f"data: {json.dumps({'error': {'message': str(exc), 'type': _error_type(exc)}})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    if typ == "content":
                        text = str(event.get("text", ""))
                        content_parts.append(text)
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None,
                                }
                            ],
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
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning_content": text},
                                    "finish_reason": None,
                                }
                            ],
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
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning": text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif typ == "tool_call":
                        tool_index = len(tool_calls_buffer)
                        tc = {
                            "index": tool_index,
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
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"tool_calls": [tc]},
                                    "finish_reason": None,
                                }
                            ],
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
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": event.get("finish_reason") or "stop",
                                }
                            ],
                        }
                        if isinstance(event.get("response_id"), str):
                            chunk["response_id"] = event["response_id"]
                        yield f"data: {json.dumps(chunk)}\n\n"

                # Usage summary chunk if available
                if usage_dict:
                    u = usage_dict
                    prompt_tokens = u.get("prompt_tokens", u.get("input_tokens", 0))
                    completion_tokens = u.get("completion_tokens", u.get("output_tokens", 0))
                    finish_usage: dict[str, Any] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": u.get(
                            "total_tokens",
                            prompt_tokens + completion_tokens,
                        ),
                    }
                    finish_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": finish_usage,
                    }
                    token_details = u.get("input_tokens_details", u.get("prompt_tokens_details"))
                    finish_usage["prompt_tokens_details"] = {
                        key: token_details.get(key, 0) if isinstance(token_details, dict) else 0
                        for key in ("cached_tokens", "cache_write_tokens")
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
            reasoning_effort=reasoning_effort,
            reasoning_mode=reasoning_mode,
            reasoning_context=reasoning_context,
            max_tokens=max_tokens,
            stop=stop,
            prompt_cache_key=request.prompt_cache_key,
            subagent=subagent,
            memgen_request=memgen_request,
            previous_response_id=previous_response_id,
            service_tier=request.service_tier,
            text=request.text,
            client_metadata=request.client_metadata,
            codex_metadata=request.codex_metadata,
            responses_lite=request.responses_lite,
            parallel_tool_calls=request.parallel_tool_calls,
            safety_identifier=request.safety_identifier,
            prompt_cache_options=request.prompt_cache_options,
            verbosity=request.verbosity,
        )

        choices: list[dict[str, Any]] = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.content,
                },
                "finish_reason": response.finish_reason,
            }
        ]

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
        if response.response_id is not None:
            result["response_id"] = response.response_id

        if response.usage:
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
                or (response.usage.prompt_tokens + response.usage.completion_tokens),
            }
            result["usage"]["prompt_tokens_details"] = {
                "cached_tokens": response.usage.cached_tokens,
                "cache_write_tokens": response.usage.cache_write_tokens,
            }

        return JSONResponse(content=result)

    @app.post("/v1/images/generations")
    async def images_generations(request: ImageGenerationRequest) -> JSONResponse:
        provider = _get_provider()
        _reject_unsupported_generation_features(
            {
                "multi_agent": request.multi_agent,
                "programmatic_tool_calling": request.programmatic_tool_calling,
                "tools": request.tools,
            }
        )
        reasoning_effort, reasoning_mode, reasoning_context = _reasoning_fields(
            request.reasoning_effort,
            request.reasoning,
        )
        images = provider.generate_image(
            request.prompt,
            model=request.model,
            size=request.size,
            reasoning_effort=reasoning_effort,
            reasoning_mode=reasoning_mode,
            reasoning_context=reasoning_context,
            responses_lite=request.responses_lite,
            safety_identifier=request.safety_identifier,
            prompt_cache_options=request.prompt_cache_options,
            verbosity=request.verbosity,
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
        anthropic_stream_adapter,
        format_anthropic_error,
        internal_response_to_anthropic,
    )

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(http_request: Request) -> JSONResponse:
        body = await http_request.json()
        for field in ("multi_agent", "programmatic_tool_calling"):
            if body.get(field) is not None:
                return JSONResponse(
                    status_code=400,
                    content=format_anthropic_error(
                        400,
                        f"{field} is not supported by this Anthropic facade",
                    ),
                )
        try:
            _validate_anthropic_context_management(body.get("context_management"))
            messages, tools, _tool_choice, _stop, _reasoning_effort, _text = anthropic_request_to_internal(
                model=body.get("model"),
                messages=body.get("messages") or [],
                system=body.get("system"),
                max_tokens=body.get("max_tokens", 4096),
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                stop_sequences=body.get("stop_sequences"),
                thinking=body.get("thinking"),
                output_format=_anthropic_output_format_from_body(body),
                output_config=body.get("output_config"),
            )
            input_tokens = _estimate_input_tokens(messages, tools)
            request_model = _anthropic_backend_model(body.get("model"))
        except Exception as exc:
            return JSONResponse(status_code=400, content=format_anthropic_error(400, str(exc)))
        return JSONResponse(
            content={
                "input_tokens": input_tokens,
                "context_window": _context_window(request_model),
                "auto_compact_token_limit": _auto_compact_token_limit(request_model),
            }
        )

    @app.post("/v1/messages", response_model=None)
    async def anthropic_messages(http_request: Request) -> JSONResponse | StreamingResponse:
        provider = _get_provider()
        body = await http_request.json()

        if body.get("multi_agent") is not None:
            return JSONResponse(
                status_code=400,
                content=format_anthropic_error(
                    400,
                    "multi_agent requires native Responses beta agent-item lifecycle support "
                    "and is not supported by this Anthropic facade",
                ),
            )
        if body.get("programmatic_tool_calling") is not None:
            return JSONResponse(
                status_code=400,
                content=format_anthropic_error(
                    400,
                    "programmatic_tool_calling requires native Responses program/caller replay support "
                    "and is not supported by this Anthropic facade",
                ),
            )

        try:
            _validate_anthropic_context_management(body.get("context_management"))
            messages, tools, tool_choice, stop, reasoning_effort, text = anthropic_request_to_internal(
                model=body.get("model", MODEL),
                messages=body.get("messages", []),
                system=body.get("system"),
                max_tokens=body.get("max_tokens", 4096),
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                stop_sequences=body.get("stop_sequences"),
                thinking=body.get("thinking"),
                output_format=_anthropic_output_format_from_body(body),
                output_config=body.get("output_config"),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content=format_anthropic_error(400, str(exc)))

        stream = body.get("stream", False)
        responses_lite = body.get("responses_lite")
        client_model = body.get("model") or "claude-sonnet-4-5"
        request_model = _anthropic_backend_model(client_model)
        try:
            explicit_effort = body.get("reasoning_effort")
            if explicit_effort is not None and reasoning_effort is not None and explicit_effort != reasoning_effort:
                raise ChatGPTOAuthInvalidRequestError("reasoning_effort conflicts with Anthropic thinking")
            effective_reasoning_effort, reasoning_mode, reasoning_context = _reasoning_fields(
                cast(str | None, explicit_effort or reasoning_effort),
                body.get("reasoning"),
            )
            service_tier = _anthropic_service_tier(body)
        except ChatGPTOAuthError as exc:
            status = _error_status(exc)
            return JSONResponse(status_code=status, content=format_anthropic_error(status, str(exc)))

        if stream:
            try:
                prepared_request = provider.preflight_chat(
                    messages,
                    model=request_model,
                    tools=tools,
                    tool_choice=tool_choice,
                    reasoning_effort=effective_reasoning_effort,
                    reasoning_mode=reasoning_mode,
                    reasoning_context=reasoning_context,
                    stop=stop,
                    text=text,
                    responses_lite=responses_lite,
                    safety_identifier=body.get("safety_identifier"),
                    prompt_cache_options=body.get("prompt_cache_options"),
                    verbosity=body.get("verbosity"),
                    service_tier=service_tier,
                )
            except ChatGPTOAuthError as exc:
                status = _error_status(exc)
                return JSONResponse(status_code=status, content=format_anthropic_error(status, str(exc)))

            def _stream() -> Iterator[str]:
                request_id = f"msg_{uuid.uuid4().hex[:24]}"
                try:
                    yield from anthropic_stream_adapter(
                        provider.chat_stream(
                            messages,
                            model=request_model,
                            tools=tools,
                            tool_choice=tool_choice,
                            reasoning_effort=effective_reasoning_effort,
                            reasoning_mode=reasoning_mode,
                            reasoning_context=reasoning_context,
                            stop=stop,
                            text=text,
                            responses_lite=responses_lite,
                            safety_identifier=body.get("safety_identifier"),
                            prompt_cache_options=body.get("prompt_cache_options"),
                            verbosity=body.get("verbosity"),
                            service_tier=service_tier,
                            _prepared_request=prepared_request,
                        ),
                        model=client_model,
                        request_id=request_id,
                    )
                except Exception as exc:  # noqa: BLE001 - serialize runtime stream failures in-band
                    status = _error_status(exc) if isinstance(exc, ChatGPTOAuthError) else 500
                    error = format_anthropic_error(status, str(exc))
                    yield f"event: error\ndata: {json.dumps(error, ensure_ascii=False)}\n\n"

            try:
                return StreamingResponse(_stream(), media_type="text/event-stream")
            except ChatGPTOAuthError as exc:
                status = _error_status(exc)
                return JSONResponse(status_code=status, content=format_anthropic_error(status, str(exc)))

        # Non-streaming
        try:
            response = provider.chat(
                messages,
                model=request_model,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=effective_reasoning_effort,
                reasoning_mode=reasoning_mode,
                reasoning_context=reasoning_context,
                stop=stop,
                text=text,
                responses_lite=responses_lite,
                safety_identifier=body.get("safety_identifier"),
                prompt_cache_options=body.get("prompt_cache_options"),
                verbosity=body.get("verbosity"),
                service_tier=service_tier,
            )
        except ChatGPTOAuthError as exc:
            status = _error_status(exc)
            return JSONResponse(status_code=status, content=format_anthropic_error(status, str(exc)))

        request_id = f"msg_{uuid.uuid4().hex[:24]}"
        result = internal_response_to_anthropic(response, client_model, request_id)
        if response.response_id is not None:
            result["response_id"] = response.response_id
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
        _reject_unsupported_generation_features(body)
        prompt = str(body.get("prompt", ""))
        images = body.get("images") or []
        reasoning_effort, reasoning_mode, reasoning_context = _reasoning_fields(
            body.get("reasoning_effort"),
            body.get("reasoning"),
        )
        result = provider.inspect_images(
            prompt,
            model=MODEL,
            images=images,
            reasoning_effort=reasoning_effort,
            reasoning_mode=reasoning_mode,
            reasoning_context=reasoning_context,
            responses_lite=body.get("responses_lite"),
            safety_identifier=body.get("safety_identifier"),
            prompt_cache_options=body.get("prompt_cache_options"),
            verbosity=body.get("verbosity"),
        )
        return JSONResponse(content={"content": result})

    @app.post("/v1/compact")
    @app.post("/v1/messages/compact")
    async def compact(request: Request) -> JSONResponse:
        """Compact a conversation into a checkpoint for continuation.

        Body: {"messages": [{"role": "system|user|assistant|tool", "content": str, ...}], "reasoning_effort": str?}
        Also accepts Anthropic Messages fields at /v1/messages/compact.
        """
        provider = _get_provider()
        body = await request.json()
        _reject_unsupported_generation_features(body)
        for field in ("safety_identifier", "include", "prompt_cache_retention"):
            if body.get(field) is not None:
                raise ChatGPTOAuthInvalidRequestError(f"{field} is not supported by the compact facade")
        messages, tools, reasoning_effort, anthropic_text = _messages_from_compact_body(
            body,
            anthropic=request.url.path == "/v1/messages/compact",
        )
        nested_reasoning = body.get("reasoning")
        nested_effort: object = None
        if nested_reasoning is not None:
            if not isinstance(nested_reasoning, dict):
                raise ChatGPTOAuthInvalidRequestError("reasoning must be an object")
            if any(key in nested_reasoning for key in ("mode", "context")):
                raise ChatGPTOAuthInvalidRequestError("compact does not support reasoning.mode or reasoning.context")
            unknown = sorted(set(nested_reasoning) - {"effort"})
            if unknown:
                raise ChatGPTOAuthInvalidRequestError(
                    "compact reasoning contains unsupported fields: " + ", ".join(unknown)
                )
            nested_effort = nested_reasoning.get("effort")
            if nested_effort is not None and (not isinstance(nested_effort, str) or nested_effort == ""):
                raise ChatGPTOAuthInvalidRequestError("reasoning.effort must be a non-empty string when provided")
        effort_candidates = [
            value for value in (body.get("reasoning_effort"), nested_effort, reasoning_effort) if value is not None
        ]
        if any(not isinstance(value, str) or value == "" for value in effort_candidates):
            raise ChatGPTOAuthInvalidRequestError("reasoning effort fields must be non-empty strings")
        if effort_candidates and any(value != effort_candidates[0] for value in effort_candidates[1:]):
            raise ChatGPTOAuthInvalidRequestError("reasoning effort fields conflict in compact request")
        explicit_effort = cast(str | None, effort_candidates[0] if effort_candidates else None)
        is_anthropic_compact = request.url.path == "/v1/messages/compact"
        compact_options: dict[str, Any] = {
            "model": _anthropic_backend_model(body.get("model")) if is_anthropic_compact else MODEL,
            "tools": tools,
            "reasoning_effort": _request_reasoning_effort(explicit_effort),
            "responses_lite": body.get("responses_lite"),
        }
        if is_anthropic_compact:
            service_tier = _anthropic_service_tier(body)
            if service_tier is not None:
                compact_options["service_tier"] = service_tier
            text = _merge_anthropic_text(anthropic_text, body.get("text"))
            if text is not None:
                compact_options["text"] = text
        for key in (
            "previous_response_id",
            "prompt_cache_key",
            "prompt_cache_options",
            "verbosity",
        ):
            if body.get(key) is not None:
                compact_options[key] = body[key]
        if not is_anthropic_compact and body.get("service_tier") is not None:
            compact_options["service_tier"] = body["service_tier"]
        if not is_anthropic_compact and body.get("text") is not None:
            compact_options["text"] = body["text"]
        checkpoint = provider.compact_messages(messages, **compact_options)
        return JSONResponse(content={"checkpoint": checkpoint})

    # ------------------------------------------------------------------
    # CLI entry point
    # ------------------------------------------------------------------

    def main() -> None:
        import uvicorn

        prime_codex_cli_version_cache()
        uvicorn.run("codex_as_api.server:app", host=HOST, port=PORT, log_level="info")

except ImportError as _import_exc:
    # FastAPI / uvicorn not installed
    _FASTAPI_IMPORT_ERROR = _import_exc
    app = None  # type: ignore[assignment]

    def main() -> None:
        raise ImportError(
            "FastAPI and uvicorn are required to run the server. Install with: pip install 'codex-as-api[server]'"
        ) from _FASTAPI_IMPORT_ERROR
