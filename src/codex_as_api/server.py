from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers

from .anthropic_adapter import (
    anthropic_parallel_tool_calls,
    anthropic_request_to_internal,
    anthropic_stream_adapter,
    format_anthropic_error,
    internal_response_to_anthropic,
)
from .auth import (
    ChatGPTOAuthCatalogUnavailableError,
    ChatGPTOAuthError,
    ChatGPTOAuthInvalidRequestError,
    ChatGPTOAuthMissingError,
    ChatGPTOAuthModelNotFoundError,
    ChatGPTOAuthProtocolError,
    ChatGPTOAuthRefreshError,
    ChatGPTOAuthUpstreamError,
    validate_auth_environment,
)
from .codex_config import load_codex_config
from .messages import Message, MessageRole, ToolSchema
from .model_capabilities import (
    ModelCapability,
    ModelCatalogSnapshot,
    validate_model_capability_environment,
)
from .o200k_tokenizer import count_ordinary
from .provider import (
    ChatGPTOAuthProvider,
    _usage_from_response,
    _validate_image_content_items,
    resolve_model_reasoning_effort,
)
from .strict_json import as_js_safe_integer, strict_json_loads

REQUEST_BODY_LIMIT_BYTES = 50 * 1024 * 1024


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"{name} must contain only ASCII decimal digits")
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return parsed


def _env_str(name: str, default: str | None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if name == "CODEX_AS_API_MODEL" and value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")
    return value


HOST = cast(str, _env_str("CODEX_AS_API_HOST", "127.0.0.1"))
PORT = _env_int("CODEX_AS_API_PORT", 18080)
validate_auth_environment()
validate_model_capability_environment()
CODEX_CONFIG = load_codex_config()
MODEL = _env_str("CODEX_AS_API_MODEL", CODEX_CONFIG.model)
AUTH_PATH = _env_str("CODEX_AS_API_AUTH_PATH", None)
CLAUDE_CODE_SESSION_HEADER = "x-claude-code-session-id"
_CLAUDE_CACHE_KEY_NAMESPACE = "codex-as-api:claude-code-session:"
_ANTHROPIC_CACHE_CONTROL_TTLS = frozenset({"5m", "1h"})
_ANTHROPIC_MESSAGES_FIELDS = frozenset(
    {
        "cache_control",
        "context_management",
        "max_tokens",
        "messages",
        "model",
        "multi_agent",
        "output_config",
        "output_format",
        "previous_response_id",
        "programmatic_tool_calling",
        "prompt_cache_key",
        "prompt_cache_options",
        "reasoning",
        "reasoning_effort",
        "responses_lite",
        "safety_identifier",
        "service_tier",
        "speed",
        "stop_sequences",
        "stream",
        "subagent",
        "system",
        "thinking",
        "tool_choice",
        "tools",
        "verbosity",
        "memgen_request",
    }
)
_ANTHROPIC_COUNT_TOKENS_FIELDS = frozenset(
    {
        "cache_control",
        "context_management",
        "max_tokens",
        "messages",
        "model",
        "multi_agent",
        "output_config",
        "output_format",
        "programmatic_tool_calling",
        "stop_sequences",
        "system",
        "thinking",
        "tool_choice",
        "tools",
    }
)
_INSPECT_FIELDS = frozenset(
    {
        "images",
        "model",
        "multi_agent",
        "programmatic_tool_calling",
        "prompt",
        "prompt_cache_options",
        "reasoning",
        "reasoning_effort",
        "responses_lite",
        "safety_identifier",
        "tools",
        "verbosity",
    }
)
_OPENAI_COMPACT_FIELDS = frozenset(
    {
        "include",
        "messages",
        "model",
        "multi_agent",
        "previous_response_id",
        "programmatic_tool_calling",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "reasoning",
        "reasoning_effort",
        "responses_lite",
        "safety_identifier",
        "service_tier",
        "text",
        "tools",
        "verbosity",
    }
)
_ANTHROPIC_COMPACT_FIELDS = frozenset(
    (
        set(_ANTHROPIC_MESSAGES_FIELDS)
        | {
            "include",
            "prompt_cache_retention",
            "text",
        }
    )
    - {"stream", "subagent", "memgen_request"}
)

_provider: ChatGPTOAuthProvider | None = None


def _get_provider() -> ChatGPTOAuthProvider:
    global _provider
    if _provider is None:
        _provider = ChatGPTOAuthProvider(
            model=MODEL,
            auth_json_path=AUTH_PATH,
        )
    return _provider


def _error_status(exc: BaseException) -> int:
    if isinstance(exc, ChatGPTOAuthUpstreamError):
        return exc.status if 100 <= exc.status <= 599 else 500
    if isinstance(exc, ChatGPTOAuthInvalidRequestError):
        return 400
    if isinstance(exc, ChatGPTOAuthModelNotFoundError):
        return 404
    if isinstance(exc, ChatGPTOAuthMissingError):
        return 401
    if isinstance(exc, ChatGPTOAuthRefreshError):
        return 401
    if isinstance(exc, ChatGPTOAuthCatalogUnavailableError):
        return 503
    if isinstance(exc, ChatGPTOAuthProtocolError):
        return 502
    return 500


def _public_error_message(exc: BaseException) -> str:
    if isinstance(exc, (ChatGPTOAuthMissingError, ChatGPTOAuthRefreshError)):
        return "ChatGPT OAuth credentials are unavailable; rerun codex login"
    if isinstance(exc, (ChatGPTOAuthInvalidRequestError, ChatGPTOAuthModelNotFoundError)):
        return str(exc)
    if isinstance(exc, ChatGPTOAuthCatalogUnavailableError):
        return "authenticated model catalog is unavailable"
    if isinstance(exc, ChatGPTOAuthProtocolError):
        return "upstream protocol validation failed"
    if isinstance(exc, ChatGPTOAuthUpstreamError):
        return "upstream request failed"
    return "internal server error"


def _anthropic_output_format_from_body(body: dict[str, Any]) -> dict[str, Any] | None:
    output_format = body.get("output_format")
    if output_format is None:
        return None
    if not isinstance(output_format, dict):
        raise ChatGPTOAuthInvalidRequestError("output_format must be an object")
    return cast(dict[str, Any], output_format)


def _anthropic_backend_model(
    provider: ChatGPTOAuthProvider,
    client_model: object,
    *,
    snapshot: ModelCatalogSnapshot | None = None,
) -> tuple[ModelCatalogSnapshot, ModelCapability]:
    if client_model is not None and not isinstance(client_model, str):
        raise ChatGPTOAuthInvalidRequestError("model must be a string")
    return provider.resolve_model(
        cast(str | None, client_model),
        anthropic_facade=True,
        snapshot=snapshot,
    )


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


def _validate_anthropic_cache_control(value: object, location: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ChatGPTOAuthInvalidRequestError(f"{location}.cache_control must be an object")
    unknown = sorted(set(value) - {"type", "ttl"})
    if unknown:
        raise ChatGPTOAuthInvalidRequestError(
            f"{location}.cache_control contains unsupported fields: {', '.join(unknown)}"
        )
    if value.get("type") != "ephemeral":
        raise ChatGPTOAuthInvalidRequestError(f"{location}.cache_control.type must be 'ephemeral'")
    if "ttl" in value and (not isinstance(value["ttl"], str) or value["ttl"] not in _ANTHROPIC_CACHE_CONTROL_TTLS):
        raise ChatGPTOAuthInvalidRequestError(f"{location}.cache_control.ttl must be one of: 5m, 1h")


def _reject_unknown_fields(
    body: dict[str, Any],
    allowed: frozenset[str],
) -> None:
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise ChatGPTOAuthInvalidRequestError("request contains unsupported fields: " + ", ".join(unknown))


def _reject_explicit_null_anthropic_fields(body: dict[str, Any]) -> None:
    for field in (
        "system",
        "tools",
        "tool_choice",
        "stop_sequences",
        "thinking",
        "output_config",
        "stream",
        "service_tier",
    ):
        if field in body and body[field] is None:
            raise ChatGPTOAuthInvalidRequestError(f"{field} must not be null")


def _positive_integer_field(
    body: dict[str, Any],
    name: str,
    *,
    required: bool,
) -> int | None:
    if name not in body:
        if required:
            raise ChatGPTOAuthInvalidRequestError(f"{name} is required")
        return None
    value = body[name]
    parsed = as_js_safe_integer(value)
    if parsed is None or parsed <= 0:
        raise ChatGPTOAuthInvalidRequestError(f"{name} must be a positive integer")
    return parsed


def _strip_anthropic_content_cache_controls(value: object, location: str) -> None:
    if not isinstance(value, list):
        return
    for index, block in enumerate(value):
        if not isinstance(block, dict):
            continue
        block_location = f"{location}[{index}]"
        if "cache_control" in block:
            _validate_anthropic_cache_control(block["cache_control"], block_location)
            block.pop("cache_control")
        _strip_anthropic_content_cache_controls(block.get("content"), f"{block_location}.content")


def _strip_anthropic_cache_controls(body: dict[str, Any]) -> None:
    if "cache_control" in body:
        _validate_anthropic_cache_control(body["cache_control"], "request")
        body.pop("cache_control")

    _strip_anthropic_content_cache_controls(body.get("system"), "system")
    messages = body.get("messages")
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            location = f"messages[{index}]"
            _strip_anthropic_content_cache_controls(message.get("content"), f"{location}.content")

    tools = body.get("tools")
    if isinstance(tools, list):
        for index, tool in enumerate(tools):
            if not isinstance(tool, dict) or "cache_control" not in tool:
                continue
            location = f"tools[{index}]"
            _validate_anthropic_cache_control(tool["cache_control"], location)
            tool.pop("cache_control")


def _has_anthropic_cache_control(body: dict[str, Any]) -> bool:
    def content_has_cache_control(value: object) -> bool:
        if not isinstance(value, list):
            return False
        for block in value:
            if not isinstance(block, dict):
                continue
            if (block.get("cache_control") is not None) or content_has_cache_control(block.get("content")):
                return True
        return False

    if body.get("cache_control") is not None:
        return True
    if content_has_cache_control(body.get("system")):
        return True
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if content_has_cache_control(message.get("content")):
                return True
    tools = body.get("tools")
    return isinstance(tools, list) and any(
        isinstance(tool, dict) and tool.get("cache_control") is not None for tool in tools
    )


def _claude_code_session_id(headers: Headers) -> str | None:
    values = headers.getlist(CLAUDE_CODE_SESSION_HEADER)
    if not values:
        return None
    if len(values) != 1:
        raise ChatGPTOAuthInvalidRequestError(f"{CLAUDE_CODE_SESSION_HEADER} must be provided at most once")
    value = values[0]
    if not value.strip():
        raise ChatGPTOAuthInvalidRequestError(f"{CLAUDE_CODE_SESSION_HEADER} must be a non-empty string")
    return value


def _validate_anthropic_compatibility_scope(
    body: dict[str, Any],
    claude_session_id: str | None,
) -> None:
    if claude_session_id is not None:
        return
    if body.get("max_tokens") is not None:
        raise ChatGPTOAuthInvalidRequestError("max_tokens is accepted without forwarding only for Claude Code requests")
    if _has_anthropic_cache_control(body):
        raise ChatGPTOAuthInvalidRequestError(
            "cache_control is accepted without forwarding only for Claude Code requests"
        )


def _anthropic_prompt_cache_key(body: dict[str, Any], claude_session_id: str | None) -> str | None:
    explicit = body.get("prompt_cache_key")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ChatGPTOAuthInvalidRequestError("prompt_cache_key must be a non-empty string")
        return explicit
    if claude_session_id is None:
        return None
    value = f"{_CLAUDE_CACHE_KEY_NAMESPACE}{claude_session_id}".encode()
    return hashlib.sha256(value).hexdigest()


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
    if isinstance(exc, ChatGPTOAuthUpstreamError):
        return "upstream_error"
    if isinstance(exc, ChatGPTOAuthInvalidRequestError):
        return "invalid_request_error"
    if isinstance(exc, ChatGPTOAuthModelNotFoundError):
        return "model_not_found"
    if isinstance(exc, (ChatGPTOAuthMissingError, ChatGPTOAuthRefreshError)):
        return "authentication_error"
    if isinstance(exc, ChatGPTOAuthCatalogUnavailableError):
        return "catalog_unavailable"
    if isinstance(exc, ChatGPTOAuthProtocolError):
        return "upstream_protocol_error"
    return "server_error"


def _health_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def _health_error_message(exc: BaseException) -> str:
    if isinstance(exc, (ChatGPTOAuthMissingError, ChatGPTOAuthRefreshError)):
        return "ChatGPT OAuth credentials are unavailable"
    if isinstance(exc, ChatGPTOAuthCatalogUnavailableError):
        return "authenticated model catalog is unavailable"
    if isinstance(exc, ChatGPTOAuthModelNotFoundError):
        return "configured model is unavailable in the authenticated catalog"
    if isinstance(exc, ChatGPTOAuthInvalidRequestError):
        return "health configuration is invalid"
    if isinstance(exc, ChatGPTOAuthProtocolError):
        return "upstream protocol validation failed"
    if isinstance(exc, ChatGPTOAuthUpstreamError):
        return "upstream request failed"
    return "health preflight failed"


app = FastAPI(
    title="codex-as-api",
    description="Local OpenAI-compatible API server backed by ChatGPT/Codex OAuth.",
    version="0.7.0",
)


@app.middleware("http")
async def _strict_json_request_middleware(request: Request, call_next: Any) -> Any:
    if request.method in {"POST", "PUT", "PATCH"}:
        content_type = request.headers.get("content-type")
        media_type = content_type.split(";", 1)[0].strip().lower() if content_type is not None else ""
        if not (
            media_type == "application/json"
            or (media_type.startswith("application/") and media_type.endswith("+json"))
        ):
            return _request_transport_error_response(
                request,
                415,
                "request Content-Type must be application/json or application/*+json",
            )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > REQUEST_BODY_LIMIT_BYTES:
                    return _request_transport_error_response(request, 413, "request body exceeds 50 MiB")
            except ValueError:
                pass
        raw_buffer = bytearray()
        async for chunk in request.stream():
            raw_buffer.extend(chunk)
            if len(raw_buffer) > REQUEST_BODY_LIMIT_BYTES:
                return _request_transport_error_response(request, 413, "request body exceeds 50 MiB")
        raw = bytes(raw_buffer)
        request._body = raw  # Starlette caches request bodies here for downstream consumers.
        if raw:
            try:
                strict_json_loads(raw)
            except (UnicodeDecodeError, ValueError):
                error = ChatGPTOAuthInvalidRequestError("request body must contain valid JSON")
                if request.url.path.startswith("/v1/messages"):
                    return JSONResponse(
                        status_code=400,
                        content=format_anthropic_error(400, str(error)),
                    )
                error_type = "invalid_request_error"
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": str(error),
                            "type": error_type,
                            "code": error_type,
                        }
                    },
                )
    return await call_next(request)


def _request_transport_error_response(request: Request, status: int, message: str) -> JSONResponse:
    if request.url.path.startswith("/v1/messages"):
        return JSONResponse(status_code=status, content=format_anthropic_error(status, message))
    error_type = "invalid_request_error"
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type, "code": error_type}},
    )


@app.exception_handler(ChatGPTOAuthError)
async def _chatgpt_oauth_error_handler(request: Request, exc: ChatGPTOAuthError) -> JSONResponse:
    status = _error_status(exc)
    if request.url.path.startswith("/v1/messages"):
        return JSONResponse(
            status_code=status,
            content=format_anthropic_error(status, _public_error_message(exc)),
        )
    error_type = _error_type(exc)
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": _public_error_message(exc),
                "type": error_type,
                "code": error_type,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def _request_validation_error_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    error_type = "invalid_request_error"
    details = exc.errors()
    diagnostics = []
    for detail in details:
        location = ".".join(str(part) for part in detail.get("loc", ()))
        error_kind = detail.get("type", "validation_error")
        diagnostics.append(f"{location}: {error_kind}" if location else str(error_kind))
    message = "; ".join(diagnostics) or "request validation failed"
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": error_type,
            }
        },
    )


@app.exception_handler(Exception)
async def _unexpected_error_handler(
    request: Request,
    _exc: Exception,
) -> JSONResponse:
    if request.url.path.startswith("/v1/messages"):
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "internal server error",
                },
            },
        )
    error_type = "server_error"
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "internal server error",
                "type": error_type,
                "code": error_type,
            }
        },
    )


# ------------------------------------------------------------------
# Request/response schemas
# ------------------------------------------------------------------


class _StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ChatMessage(_StrictRequestModel):
    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    audio: dict[str, Any] | None = None
    function_call: dict[str, Any] | None = None
    refusal: str | None = None


class ChatCompletionRequest(_StrictRequestModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool | None = False
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
    audio: dict[str, Any] | None = None
    function_call: dict[str, Any] | str | None = None
    functions: list[dict[str, Any]] | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    metadata: dict[str, str] | None = None
    modalities: list[str] | None = None
    n: int | None = None
    prediction: dict[str, Any] | None = None
    prompt_cache_retention: str | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    store: bool | None = None
    stream_options: dict[str, Any] | None = None
    top_logprobs: int | None = None
    web_search_options: dict[str, Any] | None = None
    multi_agent: dict[str, Any] | None = None
    programmatic_tool_calling: Any | None = None


class ImageGenerationRequest(_StrictRequestModel):
    model: str | None = None
    prompt: str
    reference_images: list[dict[str, Any]] | None = None
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
    background: str | None = None
    moderation: str | None = None
    n: int | None = None
    output_compression: int | None = None
    output_format: str | None = None
    partial_images: int | None = None
    quality: str | None = None
    response_format: str | None = None
    stream: bool | None = None
    style: str | None = None
    user: str | None = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _request_json_object(request: Request) -> dict[str, Any]:
    try:
        body = strict_json_loads(await request.body())
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChatGPTOAuthInvalidRequestError("request body must contain valid JSON") from exc
    if not isinstance(body, dict):
        raise ChatGPTOAuthInvalidRequestError("request body must be an object")
    return body


def _resolve_subagent(
    body_value: object,
    header_value: str | None,
) -> str | None:
    if body_value is not None and (not isinstance(body_value, str) or not body_value.strip()):
        raise ChatGPTOAuthInvalidRequestError("subagent must be a non-empty string when provided")
    if header_value is not None and not header_value.strip():
        raise ChatGPTOAuthInvalidRequestError("x-openai-subagent must be a non-empty string when provided")
    if body_value is not None and header_value is not None and body_value != header_value:
        raise ChatGPTOAuthInvalidRequestError("subagent conflicts with x-openai-subagent")
    resolved = cast(str, body_value) if body_value is not None else header_value
    if resolved is not None and not all("!" <= character <= "~" for character in resolved):
        raise ChatGPTOAuthInvalidRequestError("subagent must contain only visible ASCII characters without spaces")
    return resolved


def _resolve_memgen_request(
    body_value: object,
    header_value: str | None,
) -> bool | None:
    if body_value is not None and not isinstance(body_value, bool):
        raise ChatGPTOAuthInvalidRequestError("memgen_request must be a boolean when provided")
    parsed_header: bool | None = None
    if header_value is not None:
        if header_value == "true":
            parsed_header = True
        elif header_value == "false":
            parsed_header = False
        else:
            raise ChatGPTOAuthInvalidRequestError("x-openai-memgen-request must be 'true' or 'false'")
    if body_value is not None and parsed_header is not None and body_value is not parsed_header:
        raise ChatGPTOAuthInvalidRequestError("memgen_request conflicts with x-openai-memgen-request")
    return cast(bool, body_value) if body_value is not None else parsed_header


def _validated_openai_provider_usage(value: object) -> dict[str, Any] | None:
    _usage_from_response(value)
    if value is None:
        return None
    return cast(dict[str, Any], value)


def _validate_openai_provider_event(event: object) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ChatGPTOAuthProtocolError("provider stream event must be an object")
    typ = event.get("type")
    if typ in {"content", "reasoning_delta", "reasoning_raw_delta"}:
        text = event.get("text")
        if not isinstance(text, str):
            raise ChatGPTOAuthProtocolError(f"provider {typ} event requires string text")
    elif typ == "tool_call":
        if not isinstance(event.get("id"), str):
            raise ChatGPTOAuthProtocolError("provider tool_call event requires a string id")
        if not isinstance(event.get("name"), str):
            raise ChatGPTOAuthProtocolError("provider tool_call event requires a string name")
        if not isinstance(event.get("arguments"), str):
            raise ChatGPTOAuthProtocolError("provider tool_call event arguments must be a string")
    elif typ == "finish":
        if event.get("finish_reason") not in {"stop", "tool_calls"}:
            raise ChatGPTOAuthProtocolError("provider finish event requires a final finish_reason")
        response_id = event.get("response_id")
        if not isinstance(response_id, str) or not response_id:
            raise ChatGPTOAuthProtocolError("provider finish event requires a non-empty response_id")
        _validated_openai_provider_usage(event.get("usage"))
    elif typ == "web_search_call":
        raise ChatGPTOAuthProtocolError("provider web_search_call event cannot be represented by /v1/chat/completions")
    elif typ != "reasoning_section_break":
        raise ChatGPTOAuthProtocolError(f"provider stream event type {typ!r} is unsupported")
    return event


def _openai_model_id(request_model: str) -> str:
    return request_model


def _request_messages_to_internal(messages: list[ChatMessage]) -> list[Message]:
    result: list[Message] = []
    if not messages:
        raise ChatGPTOAuthInvalidRequestError("messages must be a non-empty array")
    for index, msg in enumerate(messages):
        role = _map_role(msg.role)
        role_allowed_fields = {
            MessageRole.SYSTEM: {"role", "content"},
            MessageRole.DEVELOPER: {"role", "content"},
            MessageRole.USER: {"role", "content"},
            MessageRole.ASSISTANT: {"role", "content", "tool_calls", "audio", "function_call", "refusal"},
            MessageRole.TOOL: {"role", "content", "tool_call_id"},
        }[role]
        role_inapplicable = sorted(msg.model_fields_set - role_allowed_fields)
        if role_inapplicable:
            raise ChatGPTOAuthInvalidRequestError(
                f"messages[{index}] contains fields that are not valid for role {msg.role!r}: "
                + ", ".join(role_inapplicable)
            )
        if role is MessageRole.TOOL:
            if not isinstance(msg.tool_call_id, str):
                raise ChatGPTOAuthInvalidRequestError(
                    f"messages[{index}] tool message requires a string tool_call_id"
                )
            if msg.tool_calls:
                raise ChatGPTOAuthInvalidRequestError(f"messages[{index}] tool message cannot contain tool_calls")
        elif msg.tool_call_id is not None:
            raise ChatGPTOAuthInvalidRequestError(f"messages[{index}] tool_call_id is only valid on tool messages")
        if "tool_calls" in msg.model_fields_set and msg.tool_calls is None:
            raise ChatGPTOAuthInvalidRequestError(f"messages[{index}] tool_calls must not be null")
        for field in ("audio", "function_call", "refusal"):
            if field not in msg.model_fields_set:
                continue
            value = getattr(msg, field)
            if role is not MessageRole.ASSISTANT or value is not None:
                raise ChatGPTOAuthInvalidRequestError(
                    f"messages[{index}] {field} is not supported by the private Codex OAuth transport"
                )
        if msg.tool_calls and role is not MessageRole.ASSISTANT:
            raise ChatGPTOAuthInvalidRequestError(f"messages[{index}] tool_calls are only valid on assistant messages")
        tool_calls = _parse_tool_calls(msg.tool_calls) if msg.tool_calls else ()
        content: str
        content_parts: tuple[dict[str, object], ...] | None
        if msg.content is None:
            if role is MessageRole.ASSISTANT and (
                "content" in msg.model_fields_set or "tool_calls" in msg.model_fields_set
            ):
                content, content_parts = "", ()
            else:
                raise ChatGPTOAuthInvalidRequestError(f"messages[{index}] content is required")
        else:
            content, content_parts = _normalize_content(role, msg.content)
        result.append(
            Message(
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=msg.tool_call_id,
                name=None,
                content_parts=content_parts,
            )
        )
    return result


def _map_role(role: str) -> MessageRole:
    mapping = {
        "system": MessageRole.SYSTEM,
        "developer": MessageRole.DEVELOPER,
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
        "tool": MessageRole.TOOL,
    }
    if not isinstance(role, str) or role not in mapping:
        raise ChatGPTOAuthInvalidRequestError(
            "message role must be one of: system, developer, user, assistant, tool"
        )
    return mapping[role]


def _normalize_content(
    role: MessageRole,
    content: str | list[dict[str, Any]] | None,
) -> tuple[str, tuple[dict[str, object], ...] | None]:
    if content is None:
        return "", None
    if isinstance(content, str):
        return content, None
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
                if item_type == "output_text" and role is not MessageRole.ASSISTANT:
                    raise ChatGPTOAuthInvalidRequestError(
                        f"message content item {index} output_text is only valid on assistant messages"
                    )
                if item_type == "input_text" and role is MessageRole.ASSISTANT:
                    raise ChatGPTOAuthInvalidRequestError(
                        f"message content item {index} input_text is not valid on assistant messages"
                    )
                unknown = sorted(set(item) - {"type", "text", "prompt_cache_breakpoint"})
                if unknown:
                    raise ChatGPTOAuthInvalidRequestError(
                        f"message text content item {index} contains unsupported fields: " + ", ".join(unknown)
                    )
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
                unknown = sorted(
                    set(item)
                    - {
                        "type",
                        "image_url",
                        "detail",
                        "prompt_cache_breakpoint",
                    }
                )
                if unknown:
                    raise ChatGPTOAuthInvalidRequestError(
                        f"message image content item {index} contains unsupported fields: " + ", ".join(unknown)
                    )
                if role is not MessageRole.USER:
                    raise ChatGPTOAuthInvalidRequestError("image content is only supported on user messages")
                raw_image = item.get("image_url")
                if isinstance(raw_image, dict):
                    unknown_image_fields = sorted(set(raw_image) - {"url", "detail"})
                    if unknown_image_fields:
                        raise ChatGPTOAuthInvalidRequestError(
                            f"message image content item {index}.image_url contains unsupported fields: "
                            + ", ".join(unknown_image_fields)
                        )
                    if "detail" in raw_image and raw_image["detail"] is None:
                        raise ChatGPTOAuthInvalidRequestError(
                            f"message image content item {index}.image_url.detail must not be null"
                        )
                    image_url = raw_image.get("url")
                    nested_detail = raw_image.get("detail")
                    direct_detail = item.get("detail")
                    if nested_detail is not None and direct_detail is not None and nested_detail != direct_detail:
                        raise ChatGPTOAuthInvalidRequestError(
                            f"message image content item {index} has conflicting detail values"
                        )
                    detail = nested_detail if nested_detail is not None else direct_detail
                else:
                    image_url = raw_image
                    detail = item.get("detail")
                if not isinstance(image_url, str) or not image_url.strip():
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
            if item_type == "input_audio":
                unknown = sorted(set(item) - {"type", "input_audio", "prompt_cache_breakpoint"})
                if unknown:
                    raise ChatGPTOAuthInvalidRequestError(
                        f"message audio content item {index} contains unsupported fields: " + ", ".join(unknown)
                    )
                if role is not MessageRole.USER:
                    raise ChatGPTOAuthInvalidRequestError("audio content is only supported on user messages")
                raw_audio = item.get("input_audio")
                if not isinstance(raw_audio, dict):
                    raise ChatGPTOAuthInvalidRequestError(
                        f"message audio content item {index} requires an input_audio object"
                    )
                unknown_audio_fields = sorted(set(raw_audio) - {"data", "format"})
                if unknown_audio_fields:
                    raise ChatGPTOAuthInvalidRequestError(
                        f"message audio content item {index}.input_audio contains unsupported fields: "
                        + ", ".join(unknown_audio_fields)
                    )
                data = raw_audio.get("data")
                audio_format = raw_audio.get("format")
                if not isinstance(data, str):
                    raise ChatGPTOAuthInvalidRequestError(
                        f"message audio content item {index}.input_audio.data must be a string"
                    )
                if audio_format not in {"wav", "mp3"}:
                    raise ChatGPTOAuthInvalidRequestError(
                        f"message audio content item {index}.input_audio.format must be wav or mp3"
                    )
                wire_parts.append(
                    {
                        "type": "input_audio",
                        "audio_url": f"data:audio/{audio_format};base64,{data}",
                    }
                )
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
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ChatGPTOAuthInvalidRequestError(f"tool_calls item {index} must be an object")
        unknown_item_fields = sorted(set(item) - {"id", "type", "function"})
        if unknown_item_fields:
            raise ChatGPTOAuthInvalidRequestError(
                f"tool_calls item {index} contains unsupported fields: " + ", ".join(unknown_item_fields)
            )
        call_id = item.get("id")
        if not isinstance(call_id, str):
            raise ChatGPTOAuthInvalidRequestError(f"tool_calls item {index} requires a string id")
        if item.get("type") != "function":
            raise ChatGPTOAuthInvalidRequestError(f"tool_calls item {index} type must be 'function'")
        func = item.get("function")
        if not isinstance(func, dict):
            raise ChatGPTOAuthInvalidRequestError(f"tool_calls item {index} function must be an object")
        unknown_function_fields = sorted(set(func) - {"name", "arguments"})
        if unknown_function_fields:
            raise ChatGPTOAuthInvalidRequestError(
                f"tool_calls item {index} function contains unsupported fields: " + ", ".join(unknown_function_fields)
            )
        source = func
        name = source.get("name")
        args = source.get("arguments")
        if not isinstance(name, str):
            raise ChatGPTOAuthInvalidRequestError(f"tool_calls item {index} requires a string function name")
        if not isinstance(args, str):
            raise ChatGPTOAuthInvalidRequestError(f"tool_calls item {index} arguments must be a string")
        calls.append(ToolCall(id=call_id, name=name, arguments=args))
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
        if item.get("type") != "function":
            raise ChatGPTOAuthInvalidRequestError(f"tools item {index} type must be 'function'")
        unknown_item_fields = sorted(
            set(item)
            - {
                "type",
                "function",
                "allowed_callers",
                "output_schema",
                "defer_loading",
                "eager_input_streaming",
            }
        )
        if unknown_item_fields:
            raise ChatGPTOAuthInvalidRequestError(
                f"tools item {index} contains unsupported fields: " + ", ".join(unknown_item_fields)
            )
        func = item.get("function")
        if not isinstance(func, dict):
            raise ChatGPTOAuthInvalidRequestError(f"tools item {index} function must be an object")
        unknown_function_fields = sorted(
            set(func)
            - {
                "name",
                "description",
                "parameters",
                "strict",
                "allowed_callers",
                "output_schema",
                "defer_loading",
                "eager_input_streaming",
            }
        )
        if unknown_function_fields:
            raise ChatGPTOAuthInvalidRequestError(
                f"tools item {index} function contains unsupported fields: " + ", ".join(unknown_function_fields)
            )
        if any(key in item or key in func for key in ("allowed_callers", "output_schema")):
            raise ChatGPTOAuthInvalidRequestError(
                "allowed_callers and output_schema require native Programmatic Tool Calling lifecycle support"
            )
        for key in ("defer_loading", "eager_input_streaming"):
            for owner, fields in (("tool", item), ("function", func)):
                if key in fields:
                    raise ChatGPTOAuthInvalidRequestError(f"tools item {index} {owner}.{key} is not supported")
        name = func.get("name")
        if "description" in func and func["description"] is None:
            raise ChatGPTOAuthInvalidRequestError(f"tools item {index} description must not be null")
        if "parameters" in func and func["parameters"] is None:
            raise ChatGPTOAuthInvalidRequestError(f"tools item {index} parameters must not be null")
        desc = func.get("description")
        params = func.get("parameters", {})
        strict = func.get("strict")
        if strict is None:
            strict = False
        if not isinstance(name, str) or not name:
            raise ChatGPTOAuthInvalidRequestError(f"tools item {index} requires a non-empty function name")
        if desc is not None and not isinstance(desc, str):
            raise ChatGPTOAuthInvalidRequestError(f"tools item {index} description must be a string")
        if not isinstance(params, dict):
            raise ChatGPTOAuthInvalidRequestError(f"tools item {index} parameters must be an object")
        if not isinstance(strict, bool):
            raise ChatGPTOAuthInvalidRequestError(f"tools item {index} strict must be a boolean")
        schemas.append(
            ToolSchema(
                name=name,
                description=desc,
                parameters=params,
                strict=strict,
            )
        )
    return schemas


def _parse_openai_tool_choice(
    value: str | dict[str, Any] | None,
) -> str | dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in {"auto", "none", "required"}:
            raise ChatGPTOAuthInvalidRequestError(
                "tool_choice must be one of: auto, none, required, or a function choice object"
            )
        return value
    unknown = sorted(set(value) - {"type", "function"})
    if unknown:
        raise ChatGPTOAuthInvalidRequestError("tool_choice contains unsupported fields: " + ", ".join(unknown))
    if value.get("type") != "function":
        raise ChatGPTOAuthInvalidRequestError("tool_choice.type must be 'function'")
    function = value.get("function")
    if not isinstance(function, dict):
        raise ChatGPTOAuthInvalidRequestError("tool_choice.function must be an object")
    unknown_function = sorted(set(function) - {"name"})
    if unknown_function:
        raise ChatGPTOAuthInvalidRequestError(
            "tool_choice.function contains unsupported fields: " + ", ".join(unknown_function)
        )
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ChatGPTOAuthInvalidRequestError("tool_choice.function.name must be a non-empty string")
    return {"type": "function", "name": name}


def _reject_unsupported_generation_features(body: dict[str, Any], *, anthropic: bool = False) -> None:
    if "safety_identifier" in body:
        raise ChatGPTOAuthInvalidRequestError(
            "safety_identifier is not supported by the private Codex OAuth HTTP transport"
        )
    if "multi_agent" in body:
        raise ChatGPTOAuthInvalidRequestError("multi_agent requires native Responses beta agent-item lifecycle support")
    if "programmatic_tool_calling" in body:
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
        for key in ("defer_loading", "eager_input_streaming"):
            for fields in (item, function_fields):
                if key in fields and not (anthropic and key == "eager_input_streaming" and fields[key] is None):
                    raise ChatGPTOAuthInvalidRequestError(f"{key} is not supported by the Chat facade")


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
    if context is not None and (not isinstance(context, str) or context not in {"auto", "current_turn", "all_turns"}):
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


def _reject_unmapped_openai_controls(request: ChatCompletionRequest) -> None:
    controls = {
        "audio": request.audio,
        "function_call": request.function_call,
        "functions": request.functions,
        "logit_bias": request.logit_bias,
        "logprobs": request.logprobs,
        "metadata": request.metadata,
        "modalities": request.modalities,
        "n": request.n,
        "prediction": request.prediction,
        "prompt_cache_retention": request.prompt_cache_retention,
        "response_format": request.response_format,
        "seed": request.seed,
        "store": request.store,
        "stream_options": request.stream_options,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "max_completion_tokens": request.max_completion_tokens,
        "top_logprobs": request.top_logprobs,
        "top_p": request.top_p,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
        "user": request.user,
        "web_search_options": request.web_search_options,
    }
    unsupported = [name for name, value in controls.items() if value is not None]
    if unsupported:
        raise ChatGPTOAuthInvalidRequestError("unsupported OpenAI controls: " + ", ".join(unsupported))


def _reject_explicit_null_fields(request: BaseModel, fields: tuple[str, ...]) -> None:
    for field in fields:
        if field in request.model_fields_set and getattr(request, field) is None:
            raise ChatGPTOAuthInvalidRequestError(f"{field} must not be null")


def _reject_unmapped_image_controls(request: ImageGenerationRequest) -> None:
    controls = {
        "background": request.background,
        "moderation": request.moderation,
        "n": request.n,
        "output_compression": request.output_compression,
        "output_format": request.output_format,
        "partial_images": request.partial_images,
        "quality": request.quality,
        "response_format": request.response_format,
        "stream": request.stream,
        "style": request.style,
        "user": request.user,
    }
    unsupported = [name for name, value in controls.items() if value is not None]
    if unsupported:
        raise ChatGPTOAuthInvalidRequestError("unsupported OpenAI image controls: " + ", ".join(unsupported))


def _raw_context_window(capability: ModelCapability) -> int | None:
    resolved_context_window: int | None
    if CODEX_CONFIG.model_context_window is not None:
        maximum = capability.max_context_window
        if maximum is not None and maximum <= 0:
            raise ChatGPTOAuthCatalogUnavailableError(
                f"model {capability.slug!r} publishes a non-positive max context window"
            )
        resolved_context_window = (
            min(CODEX_CONFIG.model_context_window, maximum)
            if maximum is not None
            else CODEX_CONFIG.model_context_window
        )
    else:
        resolved_context_window = (
            capability.context_window if capability.context_window is not None else capability.max_context_window
        )
    if resolved_context_window is not None and resolved_context_window <= 0:
        raise ChatGPTOAuthCatalogUnavailableError(f"model {capability.slug!r} publishes a non-positive context window")
    return resolved_context_window


def _context_window(capability: ModelCapability) -> int | None:
    resolved_context_window = _raw_context_window(capability)
    if resolved_context_window is None:
        return None
    product = resolved_context_window * capability.effective_context_window_percent
    product = max(-(2**63), min(2**63 - 1, product))
    effective = product // 100 if product >= 0 else -((-product) // 100)
    if not 0 < effective <= 2**53 - 1:
        raise ChatGPTOAuthCatalogUnavailableError(
            f"model {capability.slug!r} publishes an unusable effective context window"
        )
    return effective


def _auto_compact_token_limit(capability: ModelCapability) -> int | None:
    context_window = _raw_context_window(capability)
    maximum = context_window * 9 // 10 if context_window is not None else None
    resolved: int | None
    if CODEX_CONFIG.model_auto_compact_token_limit is not None:
        resolved = (
            min(CODEX_CONFIG.model_auto_compact_token_limit, maximum)
            if maximum is not None
            else CODEX_CONFIG.model_auto_compact_token_limit
        )
    elif capability.auto_compact_token_limit is not None:
        resolved = (
            min(capability.auto_compact_token_limit, maximum)
            if maximum is not None
            else capability.auto_compact_token_limit
        )
    else:
        resolved = maximum
    return resolved


def _request_reasoning_effort(requested: str | None) -> str | None:
    if requested is not None:
        if not isinstance(requested, str) or requested == "" or requested != requested.strip():
            raise ChatGPTOAuthInvalidRequestError("reasoning_effort must be a non-empty string when provided")
        return requested
    effort = CODEX_CONFIG.model_reasoning_effort
    if effort is not None and (not isinstance(effort, str) or effort == "" or effort != effort.strip()):
        raise RuntimeError("configured reasoning_effort must be a non-empty string")
    return effort


def _validate_configured_reasoning_effort(
    capability: ModelCapability,
    selected_effort: str | None,
    *,
    request_has_effort: bool,
) -> None:
    if request_has_effort:
        return
    effective_effort = selected_effort or capability.default_reasoning_level
    if effective_effort is None:
        return
    try:
        resolve_model_reasoning_effort(capability, effective_effort)
    except ChatGPTOAuthInvalidRequestError as exc:
        if selected_effort is not None:
            raise RuntimeError("configured reasoning_effort is not supported by the live model") from exc
        raise ChatGPTOAuthCatalogUnavailableError(
            "selected model publishes an unsupported default reasoning effort"
        ) from exc


def _reject_explicit_null_responses_lite(body: dict[str, Any]) -> None:
    if "responses_lite" in body and body["responses_lite"] is None:
        raise ChatGPTOAuthInvalidRequestError("responses_lite must be a boolean or string when provided")


def _messages_from_compact_body(
    body: dict[str, Any],
    *,
    anthropic: bool = False,
    claude_session_id: str | None = None,
) -> tuple[list[Message], list[ToolSchema] | None, str | None, dict[str, Any] | None]:
    messages_value = body.get("messages")
    if not isinstance(messages_value, list) or not messages_value:
        raise ChatGPTOAuthInvalidRequestError("messages must be a non-empty array")
    if anthropic or any(key in body for key in ("system", "thinking", "tool_choice", "stop_sequences")):
        if anthropic:
            _validate_anthropic_compatibility_scope(body, claude_session_id)
        _strip_anthropic_cache_controls(body)
        _validate_anthropic_context_management(body.get("context_management"))
        anthropic_model = body.get("model")
        if not isinstance(anthropic_model, str) or not anthropic_model:
            raise ChatGPTOAuthInvalidRequestError("model must be a non-empty string")
        anthropic_messages = cast(list[dict[str, Any]], messages_value)
        max_tokens = _positive_integer_field(body, "max_tokens", required=True)
        raw_tools = body.get("tools")
        if raw_tools is not None and not isinstance(raw_tools, list):
            raise ChatGPTOAuthInvalidRequestError("tools must be an array")
        raw_tool_choice = body.get("tool_choice")
        if raw_tool_choice is not None and not isinstance(raw_tool_choice, dict):
            raise ChatGPTOAuthInvalidRequestError("tool_choice must be an object")
        raw_stop = body.get("stop_sequences")
        if raw_stop is not None:
            raise ChatGPTOAuthInvalidRequestError(
                "stop_sequences is not supported by the private Codex OAuth compact transport"
            )
        raw_thinking = body.get("thinking")
        if raw_thinking is not None and not isinstance(raw_thinking, dict):
            raise ChatGPTOAuthInvalidRequestError("thinking must be an object")
        try:
            messages, tools, converted_tool_choice, _stop, reasoning_effort, text = anthropic_request_to_internal(
                model=anthropic_model,
                messages=anthropic_messages,
                system=body.get("system"),
                tools=raw_tools,
                tool_choice=raw_tool_choice,
                stop_sequences=raw_stop,
                thinking=raw_thinking,
                max_tokens=max_tokens,
                output_format=_anthropic_output_format_from_body(body),
                output_config=body.get("output_config"),
            )
        except ValueError as exc:
            raise ChatGPTOAuthInvalidRequestError(str(exc)) from exc
        if converted_tool_choice is not None and converted_tool_choice != "auto":
            raise ChatGPTOAuthInvalidRequestError("compact supports only Anthropic tool_choice.type=auto")
        return messages, tools, reasoning_effort, text

    try:
        validated_messages = [ChatMessage.model_validate(message) for message in messages_value]
    except ValidationError as exc:
        raise ChatGPTOAuthInvalidRequestError(str(exc)) from exc
    messages = _request_messages_to_internal(validated_messages)
    raw_tools = body.get("tools")
    if raw_tools is not None and not isinstance(raw_tools, list):
        raise ChatGPTOAuthInvalidRequestError("tools must be an array")
    return messages, _parse_tools(raw_tools), None, None


def _json_token_count(value: Any) -> int:
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
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
                    **({"description": tool.description} if tool.description is not None else {}),
                    "name": tool.name,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                }
                for tool in tools
            ]
        )
    return total + image_tokens


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/health", response_model=None)
async def health() -> JSONResponse:
    provider = _get_provider()
    try:
        snapshot = await run_in_threadpool(provider.get_model_catalog)
    except Exception as exc:  # noqa: BLE001 - health must retain its safe diagnostic schema
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "auth_available": not isinstance(
                    exc,
                    (ChatGPTOAuthMissingError, ChatGPTOAuthRefreshError),
                ),
                "catalog_status": "unavailable",
                "catalog_fetched_at": None,
                "catalog_expires_at": None,
                "model": None,
                "reasoning_effort": None,
                "context_window": None,
                "auto_compact_token_limit": None,
                "error": {
                    "message": _health_error_message(exc),
                    "type": _error_type(exc),
                },
            },
        )
    try:
        _snapshot, capability = await run_in_threadpool(
            provider.resolve_model,
            snapshot=snapshot,
        )
        configured_effort = _request_reasoning_effort(None)
        requested_effort = configured_effort or capability.default_reasoning_level
        try:
            reasoning_effort = (
                resolve_model_reasoning_effort(capability, requested_effort)
                if requested_effort is not None
                else None
            )
        except ChatGPTOAuthInvalidRequestError as exc:
            if configured_effort is not None:
                raise RuntimeError("configured reasoning_effort is not supported by the live model") from exc
            raise ChatGPTOAuthCatalogUnavailableError(
                "selected model publishes an unsupported default reasoning effort"
            ) from exc
        context_window = _context_window(capability)
        auto_compact_token_limit = _auto_compact_token_limit(capability)
    except Exception as exc:  # noqa: BLE001 - health must retain its safe diagnostic schema
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "auth_available": True,
                "catalog_status": "fresh",
                "catalog_fetched_at": _health_timestamp(snapshot.fetched_at),
                "catalog_expires_at": _health_timestamp(snapshot.expires_at),
                "model": None,
                "reasoning_effort": None,
                "context_window": None,
                "auto_compact_token_limit": None,
                "error": {
                    "message": _health_error_message(exc),
                    "type": _error_type(exc),
                },
            },
        )
    return JSONResponse(
        content={
            "status": "ok",
            "auth_available": True,
            "catalog_status": "fresh",
            "catalog_fetched_at": _health_timestamp(snapshot.fetched_at),
            "catalog_expires_at": _health_timestamp(snapshot.expires_at),
            "model": capability.slug,
            "reasoning_effort": reasoning_effort,
            "context_window": context_window,
            "auto_compact_token_limit": auto_compact_token_limit,
        }
    )


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    snapshot = await run_in_threadpool(_get_provider().get_model_catalog)
    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": model.slug,
                    "object": "model",
                    "owned_by": "openai",
                    "display_name": model.display_name,
                    "description": model.description,
                    "priority": model.priority,
                    "visibility": model.visibility,
                    "supported_in_api": model.supported_in_api,
                    "supported_reasoning_levels": [
                        {
                            "effort": level.effort,
                            "description": level.description,
                        }
                        for level in model.supported_reasoning_levels
                    ],
                    "default_reasoning_level": model.default_reasoning_level,
                    "multi_agent_reasoning_effort": (model.multi_agent_reasoning_effort),
                    "supports_reasoning_summary_parameter": (
                        model.supports_reasoning_summary_parameter
                    ),
                    "default_reasoning_summary": model.default_reasoning_summary,
                    "comp_hash": model.comp_hash,
                    "context_window": model.context_window,
                    "max_context_window": model.max_context_window,
                    "effective_context_window_percent": (model.effective_context_window_percent),
                    "auto_compact_token_limit": model.auto_compact_token_limit,
                    "input_modalities": list(model.input_modalities),
                    "service_tiers": [
                        {
                            "id": tier.id,
                            "name": tier.name,
                            "description": tier.description,
                        }
                        for tier in model.service_tiers
                    ],
                    "default_service_tier": model.default_service_tier,
                    "use_responses_lite": model.use_responses_lite,
                    "supports_image_detail_original": (model.supports_image_detail_original),
                    "support_verbosity": model.support_verbosity,
                    "default_verbosity": model.default_verbosity,
                }
                for model in snapshot.models
            ],
        }
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: ChatCompletionRequest, http_request: Request) -> JSONResponse | StreamingResponse:
    provider = _get_provider()
    _reject_explicit_null_fields(
        request,
        (
            "model",
            "function_call",
            "functions",
            "parallel_tool_calls",
            "prompt_cache_key",
            "response_format",
            "safety_identifier",
            "tool_choice",
            "tools",
            "user",
            "web_search_options",
            "multi_agent",
            "programmatic_tool_calling",
        ),
    )
    if "responses_lite" in request.model_fields_set and request.responses_lite is None:
        raise ChatGPTOAuthInvalidRequestError("responses_lite must be a boolean or string when provided")
    _reject_unmapped_openai_controls(request)
    _reject_unsupported_generation_features(
        {
            field: getattr(request, field)
            for field in ("multi_agent", "programmatic_tool_calling", "safety_identifier", "tools")
            if field in request.model_fields_set
        }
    )
    messages = _request_messages_to_internal(request.messages)
    tools = _parse_tools(request.tools)
    tool_choice = _parse_openai_tool_choice(request.tool_choice)
    stop = _normalize_stop(request.stop)
    max_tokens = _max_tokens_from_request(request)

    subagent = _resolve_subagent(
        request.subagent,
        http_request.headers.get("x-openai-subagent"),
    )
    memgen_request = _resolve_memgen_request(
        request.memgen_request,
        http_request.headers.get("x-openai-memgen-request"),
    )
    previous_response_id = request.previous_response_id
    reasoning_effort, reasoning_mode, reasoning_context = _reasoning_fields(
        request.reasoning_effort,
        request.reasoning,
    )
    nested_request_effort = request.reasoning.get("effort") if isinstance(request.reasoning, dict) else None
    resolved_model: tuple[ModelCatalogSnapshot, ModelCapability] | None = None
    if request.reasoning_effort is None and nested_request_effort is None:
        resolved_model = await run_in_threadpool(
            provider.resolve_model,
            request.model,
        )
        _validate_configured_reasoning_effort(
            resolved_model[1],
            reasoning_effort,
            request_has_effort=False,
        )

    prepared_request = await run_in_threadpool(
        provider.preflight_chat,
        messages,
        model=request.model,
        tools=tools,
        tool_choice=tool_choice,
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
        _resolved_model=resolved_model,
    )
    effective_model = prepared_request.capability.slug

    if request.stream:

        def _stream() -> Iterator[str]:
            request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            model = _openai_model_id(effective_model)

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
                    provider_events = provider.chat_stream(
                        messages,
                        model=effective_model,
                        tools=tools,
                        tool_choice=tool_choice,
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
                    for provider_event in provider_events:
                        yield _validate_openai_provider_event(provider_event)
                except Exception as exc:  # noqa: BLE001 - serialize runtime stream failures in-band
                    yield {"type": "_stream_error", "error": exc}

            for event in _provider_events():
                typ = event.get("type")
                if typ == "_stream_error":
                    exc = event["error"]
                    error_type = _error_type(exc)
                    error = {
                        "error": {
                            "message": _public_error_message(exc),
                            "type": error_type,
                            "code": error_type,
                        }
                    }
                    yield f"data: {json.dumps(error)}\n\n"
                    return
                if typ == "content":
                    text = cast(str, event["text"])
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
                    text = cast(str, event["text"])
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
                    text = cast(str, event["text"])
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
                            "arguments": event["arguments"],
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
                    if event.get("usage") is not None:
                        usage_dict = cast(dict[str, Any], event["usage"])
                    finish_reason = cast(str | None, event["finish_reason"])
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    chunk["response_id"] = event["response_id"]
                    yield f"data: {json.dumps(chunk)}\n\n"

            # Usage summary chunk if available
            if usage_dict is not None:
                u = usage_dict
                prompt_tokens = cast(int, u["input_tokens"])
                completion_tokens = cast(int, u["output_tokens"])
                finish_usage: dict[str, Any] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": u["total_tokens"],
                }
                finish_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": finish_usage,
                }
                token_details = u.get("input_tokens_details")
                if isinstance(token_details, dict):
                    prompt_details = {
                        "cached_tokens": token_details["cached_tokens"],
                    }
                    if token_details.get("cache_write_tokens") is not None:
                        prompt_details["cache_write_tokens"] = token_details["cache_write_tokens"]
                    finish_usage["prompt_tokens_details"] = prompt_details
                yield f"data: {json.dumps(finish_chunk)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
        )

    # Non-streaming
    response = await run_in_threadpool(
        provider.chat,
        messages,
        model=effective_model,
        tools=tools,
        tool_choice=tool_choice,
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
    raw_events = response.raw.get("events") if isinstance(response.raw, dict) else None
    if isinstance(raw_events, list) and any(
        isinstance(event, dict) and event.get("type") == "web_search_call" for event in raw_events
    ):
        raise ChatGPTOAuthProtocolError("provider web_search_call event cannot be represented by /v1/chat/completions")
    if response.response_id is None:
        raise ChatGPTOAuthProtocolError("provider response requires a non-empty response_id")
    if response.finish_reason not in {"stop", "tool_calls"}:
        raise ChatGPTOAuthProtocolError("provider response requires a final finish_reason")

    choices: list[dict[str, Any]] = [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response.content,
                "refusal": None,
            },
            "finish_reason": response.finish_reason,
            "logprobs": None,
        }
    ]

    if response.tool_calls:
        choices[0]["message"]["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
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
        "model": _openai_model_id(effective_model),
        "choices": choices,
    }
    result["response_id"] = response.response_id

    if response.usage:
        result["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        prompt_tokens_details: dict[str, int] = {}
        if response.usage.cached_tokens is not None:
            prompt_tokens_details["cached_tokens"] = response.usage.cached_tokens
        if response.usage.cache_write_tokens is not None:
            prompt_tokens_details["cache_write_tokens"] = response.usage.cache_write_tokens
        if prompt_tokens_details:
            result["usage"]["prompt_tokens_details"] = prompt_tokens_details

    return JSONResponse(
        content=result,
    )


@app.post("/v1/images/generations")
async def images_generations(request: ImageGenerationRequest) -> JSONResponse:
    provider = _get_provider()
    _reject_explicit_null_fields(request, ("user", "multi_agent", "programmatic_tool_calling"))
    if "responses_lite" in request.model_fields_set and request.responses_lite is None:
        raise ChatGPTOAuthInvalidRequestError("responses_lite must be a boolean or string when provided")
    _reject_unmapped_image_controls(request)
    _reject_unsupported_generation_features(
        {
            field: getattr(request, field)
            for field in ("multi_agent", "programmatic_tool_calling", "safety_identifier", "tools")
            if field in request.model_fields_set
        }
    )
    if "tools" in request.model_fields_set:
        raise ChatGPTOAuthInvalidRequestError("tools are not supported by the image generation endpoint")
    reasoning_effort, reasoning_mode, reasoning_context = _reasoning_fields(
        request.reasoning_effort,
        request.reasoning,
    )
    reference_images = request.reference_images or []
    _validate_image_content_items(reference_images)
    resolved_model = await run_in_threadpool(provider.resolve_model, request.model)
    nested_request_effort = request.reasoning.get("effort") if isinstance(request.reasoning, dict) else None
    _validate_configured_reasoning_effort(
        resolved_model[1],
        reasoning_effort,
        request_has_effort=request.reasoning_effort is not None or nested_request_effort is not None,
    )
    images = await run_in_threadpool(
        provider.generate_image,
        request.prompt,
        model=request.model,
        reference_images=reference_images,
        size=request.size,
        reasoning_effort=reasoning_effort,
        reasoning_mode=reasoning_mode,
        reasoning_context=reasoning_context,
        responses_lite=request.responses_lite,
        safety_identifier=request.safety_identifier,
        prompt_cache_options=request.prompt_cache_options,
        verbosity=request.verbosity,
        _resolved_model=resolved_model,
    )
    data = []
    for image in images:
        generated = {"url": image["result"]}
        if "revised_prompt" in image:
            generated["revised_prompt"] = image["revised_prompt"]
        data.append(generated)
    return JSONResponse(
        content={"created": int(time.time()), "data": data},
    )


# ------------------------------------------------------------------
# Anthropic Messages API compatible endpoint
# ------------------------------------------------------------------


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(http_request: Request) -> JSONResponse:
    provider = _get_provider()
    try:
        body = await _request_json_object(http_request)
        _reject_unknown_fields(body, _ANTHROPIC_COUNT_TOKENS_FIELDS)
        _reject_explicit_null_anthropic_fields(body)
        max_tokens = _positive_integer_field(body, "max_tokens", required=False)
        claude_session_id = _claude_code_session_id(http_request.headers)
        _validate_anthropic_compatibility_scope(body, claude_session_id)
    except ChatGPTOAuthInvalidRequestError as exc:
        return JSONResponse(
            status_code=400,
            content=format_anthropic_error(400, str(exc)),
        )
    for field in ("multi_agent", "programmatic_tool_calling"):
        if field in body:
            return JSONResponse(
                status_code=400,
                content=format_anthropic_error(
                    400,
                    f"{field} is not supported by this Anthropic facade",
                ),
            )
    try:
        _strip_anthropic_cache_controls(body)
        _validate_anthropic_context_management(body.get("context_management"))
        if body.get("stop_sequences") is not None:
            raise ChatGPTOAuthInvalidRequestError("stop_sequences is not supported by the Anthropic token count facade")
        messages, tools, _tool_choice, _stop, _reasoning_effort, _text = anthropic_request_to_internal(
            model=body.get("model"),
            messages=body.get("messages"),
            system=body.get("system"),
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
            stop_sequences=body.get("stop_sequences"),
            thinking=body.get("thinking"),
            max_tokens=max_tokens,
            output_format=_anthropic_output_format_from_body(body),
            output_config=body.get("output_config"),
        )
        input_tokens = _estimate_input_tokens(messages, tools)
    except (ValueError, ChatGPTOAuthInvalidRequestError) as exc:
        return JSONResponse(status_code=400, content=format_anthropic_error(400, str(exc)))
    try:
        _snapshot, capability = await run_in_threadpool(
            _anthropic_backend_model,
            provider,
            body.get("model"),
        )
        context_window = _context_window(capability)
        auto_compact_token_limit = None if context_window is None else _auto_compact_token_limit(capability)
    except ChatGPTOAuthError as exc:
        status = _error_status(exc)
        return JSONResponse(
            status_code=status,
            content=format_anthropic_error(status, _public_error_message(exc)),
        )
    return JSONResponse(
        content={
            "input_tokens": input_tokens,
            "context_window": context_window,
            "auto_compact_token_limit": auto_compact_token_limit,
        },
    )


@app.post("/v1/messages", response_model=None)
async def anthropic_messages(http_request: Request) -> JSONResponse | StreamingResponse:
    provider = _get_provider()
    try:
        body = await _request_json_object(http_request)
        _reject_explicit_null_responses_lite(body)
        _reject_unknown_fields(body, _ANTHROPIC_MESSAGES_FIELDS)
        _reject_explicit_null_anthropic_fields(body)
        max_tokens = _positive_integer_field(body, "max_tokens", required=True)
        claude_session_id = _claude_code_session_id(http_request.headers)
        _validate_anthropic_compatibility_scope(body, claude_session_id)
    except ChatGPTOAuthInvalidRequestError as exc:
        return JSONResponse(
            status_code=400,
            content=format_anthropic_error(400, str(exc)),
        )

    if "multi_agent" in body:
        return JSONResponse(
            status_code=400,
            content=format_anthropic_error(
                400,
                "multi_agent requires native Responses beta agent-item lifecycle support "
                "and is not supported by this Anthropic facade",
            ),
        )
    if "safety_identifier" in body:
        return JSONResponse(
            status_code=400,
            content=format_anthropic_error(
                400,
                "safety_identifier is not supported by the private Codex OAuth HTTP transport",
            ),
        )
    if "programmatic_tool_calling" in body:
        return JSONResponse(
            status_code=400,
            content=format_anthropic_error(
                400,
                "programmatic_tool_calling requires native Responses program/caller replay support "
                "and is not supported by this Anthropic facade",
            ),
        )

    try:
        if body.get("previous_response_id") is not None:
            raise ChatGPTOAuthInvalidRequestError(
                "previous_response_id is not supported by the Anthropic Messages endpoint"
            )
        subagent = _resolve_subagent(
            body.get("subagent"),
            http_request.headers.get("x-openai-subagent"),
        )
        memgen_request = _resolve_memgen_request(
            body.get("memgen_request"),
            http_request.headers.get("x-openai-memgen-request"),
        )
        _strip_anthropic_cache_controls(body)
        prompt_cache_key = _anthropic_prompt_cache_key(
            body,
            claude_session_id,
        )
        _validate_anthropic_context_management(body.get("context_management"))
        messages, tools, tool_choice, stop, reasoning_effort, text = anthropic_request_to_internal(
            model=body.get("model"),
            messages=body.get("messages"),
            system=body.get("system"),
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
            stop_sequences=body.get("stop_sequences"),
            thinking=body.get("thinking"),
            max_tokens=max_tokens,
            output_format=_anthropic_output_format_from_body(body),
            output_config=body.get("output_config"),
        )
        parallel_tool_calls = anthropic_parallel_tool_calls(cast(dict[str, Any] | None, body.get("tool_choice")))
    except (ValueError, ChatGPTOAuthInvalidRequestError) as exc:
        return JSONResponse(status_code=400, content=format_anthropic_error(400, str(exc)))

    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        return JSONResponse(
            status_code=400,
            content=format_anthropic_error(400, "stream must be a boolean"),
        )
    responses_lite = body.get("responses_lite")
    client_model = body.get("model")
    if not isinstance(client_model, str) or not client_model:
        return JSONResponse(
            status_code=400,
            content=format_anthropic_error(400, "model must be a non-empty string"),
        )
    try:
        explicit_effort = body.get("reasoning_effort")
        if explicit_effort is not None and (not isinstance(explicit_effort, str) or not explicit_effort):
            raise ChatGPTOAuthInvalidRequestError("reasoning_effort must be a non-empty string when provided")
        if explicit_effort is not None and reasoning_effort is not None and explicit_effort != reasoning_effort:
            raise ChatGPTOAuthInvalidRequestError("reasoning_effort conflicts with Anthropic thinking")
        effective_reasoning_effort, reasoning_mode, reasoning_context = _reasoning_fields(
            cast(
                str | None,
                explicit_effort if explicit_effort is not None else reasoning_effort,
            ),
            body.get("reasoning"),
        )
        service_tier = _anthropic_service_tier(body)
    except ChatGPTOAuthError as exc:
        status = _error_status(exc)
        return JSONResponse(status_code=status, content=format_anthropic_error(status, _public_error_message(exc)))

    try:
        resolved_model = await run_in_threadpool(
            _anthropic_backend_model,
            provider,
            client_model,
        )
    except ChatGPTOAuthError as exc:
        status = _error_status(exc)
        return JSONResponse(
            status_code=status,
            content=format_anthropic_error(status, _public_error_message(exc)),
        )
    request_model = resolved_model[1].slug
    nested_request_effort = body.get("reasoning", {}).get("effort") if isinstance(body.get("reasoning"), dict) else None
    _validate_configured_reasoning_effort(
        resolved_model[1],
        effective_reasoning_effort,
        request_has_effort=(
            explicit_effort is not None or reasoning_effort is not None or nested_request_effort is not None
        ),
    )

    try:
        prepared_request = await run_in_threadpool(
            provider.preflight_chat,
            messages,
            model=request_model,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=effective_reasoning_effort,
            reasoning_mode=reasoning_mode,
            reasoning_context=reasoning_context,
            stop=stop,
            prompt_cache_key=prompt_cache_key,
            text=text,
            codex_metadata=False,
            responses_lite=responses_lite,
            parallel_tool_calls=parallel_tool_calls,
            safety_identifier=body.get("safety_identifier"),
            prompt_cache_options=body.get("prompt_cache_options"),
            verbosity=body.get("verbosity"),
            service_tier=service_tier,
            _resolved_model=resolved_model,
        )
    except ChatGPTOAuthError as exc:
        status = _error_status(exc)
        return JSONResponse(
            status_code=status,
            content=format_anthropic_error(status, _public_error_message(exc)),
        )

    if stream:

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
                        prompt_cache_key=prompt_cache_key,
                        subagent=subagent,
                        memgen_request=memgen_request,
                        text=text,
                        codex_metadata=False,
                        responses_lite=responses_lite,
                        parallel_tool_calls=parallel_tool_calls,
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
                error = format_anthropic_error(status, _public_error_message(exc))
                yield f"event: error\ndata: {json.dumps(error, ensure_ascii=False)}\n\n"

        try:
            return StreamingResponse(
                _stream(),
                media_type="text/event-stream",
            )
        except ChatGPTOAuthError as exc:
            status = _error_status(exc)
            return JSONResponse(
                status_code=status,
                content=format_anthropic_error(status, _public_error_message(exc)),
            )

    # Non-streaming
    try:
        response = await run_in_threadpool(
            provider.chat,
            messages,
            model=request_model,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=effective_reasoning_effort,
            reasoning_mode=reasoning_mode,
            reasoning_context=reasoning_context,
            stop=stop,
            prompt_cache_key=prompt_cache_key,
            subagent=subagent,
            memgen_request=memgen_request,
            text=text,
            codex_metadata=False,
            responses_lite=responses_lite,
            parallel_tool_calls=parallel_tool_calls,
            safety_identifier=body.get("safety_identifier"),
            prompt_cache_options=body.get("prompt_cache_options"),
            verbosity=body.get("verbosity"),
            service_tier=service_tier,
            _prepared_request=prepared_request,
        )
    except ChatGPTOAuthError as exc:
        status = _error_status(exc)
        return JSONResponse(status_code=status, content=format_anthropic_error(status, _public_error_message(exc)))

    request_id = f"msg_{uuid.uuid4().hex[:24]}"
    try:
        result = internal_response_to_anthropic(response, client_model, request_id)
    except ChatGPTOAuthError as exc:
        status = _error_status(exc)
        return JSONResponse(
            status_code=status,
            content=format_anthropic_error(status, _public_error_message(exc)),
        )
    return JSONResponse(
        content=result,
    )


# ------------------------------------------------------------------
# Custom endpoints (not in standard OpenAI API, but exposed for full feature routing)
# ------------------------------------------------------------------


@app.post("/v1/inspect")
async def inspect(request: Request) -> JSONResponse:
    """Inspect images with a text prompt.

    Body: {"prompt": str, "images": [{"image_url": "data:image/..."}, ...], "reasoning_effort": str?}
    """
    provider = _get_provider()
    body = await _request_json_object(request)
    _reject_explicit_null_responses_lite(body)
    _reject_unknown_fields(body, _INSPECT_FIELDS)
    _reject_unsupported_generation_features(body)
    if body.get("tools") is not None:
        raise ChatGPTOAuthInvalidRequestError("tools are not supported by the image inspection endpoint")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ChatGPTOAuthInvalidRequestError("prompt must be a non-empty string")
    images = body.get("images")
    if not isinstance(images, list) or not images:
        raise ChatGPTOAuthInvalidRequestError("images must be a non-empty array")
    requested_model = body.get("model")
    if requested_model is not None and not isinstance(requested_model, str):
        raise ChatGPTOAuthInvalidRequestError("model must be a string")
    reasoning_effort, reasoning_mode, reasoning_context = _reasoning_fields(
        body.get("reasoning_effort"),
        body.get("reasoning"),
    )
    resolved_model = await run_in_threadpool(
        provider.resolve_model,
        requested_model,
    )
    nested_request_effort = body.get("reasoning", {}).get("effort") if isinstance(body.get("reasoning"), dict) else None
    _validate_configured_reasoning_effort(
        resolved_model[1],
        reasoning_effort,
        request_has_effort=body.get("reasoning_effort") is not None or nested_request_effort is not None,
    )
    result = await run_in_threadpool(
        provider.inspect_images,
        prompt,
        model=requested_model,
        images=images,
        reasoning_effort=reasoning_effort,
        reasoning_mode=reasoning_mode,
        reasoning_context=reasoning_context,
        responses_lite=body.get("responses_lite"),
        safety_identifier=body.get("safety_identifier"),
        prompt_cache_options=body.get("prompt_cache_options"),
        verbosity=body.get("verbosity"),
        _resolved_model=resolved_model,
    )
    return JSONResponse(
        content={"content": result},
    )


@app.post("/v1/compact")
@app.post("/v1/messages/compact")
async def compact(request: Request) -> JSONResponse:
    """Compact a conversation into a checkpoint for continuation.

    Body: {"messages": [{"role": "system|user|assistant|tool", "content": str, ...}], "reasoning_effort": str?}
    Also accepts Anthropic Messages fields at /v1/messages/compact.
    """
    provider = _get_provider()
    body = await _request_json_object(request)
    _reject_explicit_null_responses_lite(body)
    is_anthropic_compact = request.url.path == "/v1/messages/compact"
    claude_session_id = _claude_code_session_id(request.headers) if is_anthropic_compact else None
    _reject_unknown_fields(
        body,
        _ANTHROPIC_COMPACT_FIELDS if is_anthropic_compact else _OPENAI_COMPACT_FIELDS,
    )
    if is_anthropic_compact:
        _reject_explicit_null_anthropic_fields(body)
    _reject_unsupported_generation_features(body, anthropic=is_anthropic_compact)
    for field in ("safety_identifier", "include", "prompt_cache_retention"):
        if body.get(field) is not None:
            raise ChatGPTOAuthInvalidRequestError(f"{field} is not supported by the compact facade")
    messages, tools, reasoning_effort, anthropic_text = _messages_from_compact_body(
        body,
        anthropic=is_anthropic_compact,
        claude_session_id=claude_session_id,
    )
    if is_anthropic_compact:
        compact_parallel_tool_calls = anthropic_parallel_tool_calls(
            cast(dict[str, Any] | None, body.get("tool_choice"))
        )
        if compact_parallel_tool_calls is True:
            raise ChatGPTOAuthInvalidRequestError("messages compact cannot preserve disable_parallel_tool_use=false")
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
    requested_model = body.get("model")
    if requested_model is not None and not isinstance(requested_model, str):
        raise ChatGPTOAuthInvalidRequestError("model must be a string")
    if is_anthropic_compact:
        resolved_model = await run_in_threadpool(
            _anthropic_backend_model,
            provider,
            requested_model,
        )
    else:
        resolved_model = await run_in_threadpool(
            provider.resolve_model,
            requested_model,
        )
    selected_effort = _request_reasoning_effort(explicit_effort)
    _validate_configured_reasoning_effort(
        resolved_model[1],
        selected_effort,
        request_has_effort=explicit_effort is not None,
    )
    compact_options: dict[str, Any] = {
        "model": resolved_model[1].slug,
        "tools": tools,
        "reasoning_effort": selected_effort,
        "responses_lite": body.get("responses_lite"),
        "_resolved_model": resolved_model,
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
    checkpoint = await run_in_threadpool(
        provider.compact_messages,
        messages,
        **compact_options,
    )
    return JSONResponse(
        content={"checkpoint": checkpoint},
    )


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main() -> None:
    import uvicorn

    uvicorn.run("codex_as_api.server:app", host=HOST, port=PORT, log_level="info")
