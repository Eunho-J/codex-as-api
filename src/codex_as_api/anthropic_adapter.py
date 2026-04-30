from __future__ import annotations

import json
import uuid
from typing import Any, Iterator, Sequence

from .messages import Message, MessageRole, ToolCall, ToolSchema, Usage, AssistantResponse


# ---------------------------------------------------------------------------
# Request conversion: Anthropic → internal
# ---------------------------------------------------------------------------

def anthropic_request_to_internal(
    *,
    model: str,
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    stop_sequences: list[str] | None = None,
    thinking: dict[str, Any] | None = None,
) -> tuple[list[Message], list[ToolSchema] | None, str | dict | None, list[str] | None, str | None]:
    """Convert Anthropic Messages request fields to internal types.

    Returns (messages, tools, tool_choice, stop, reasoning_effort).
    """
    internal_messages: list[Message] = []

    # System prompt → SYSTEM message
    if system is not None:
        sys_text = _extract_system_text(system)
        if sys_text:
            internal_messages.append(Message(role=MessageRole.SYSTEM, content=sys_text))

    # Convert messages
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            _convert_user_message(content, internal_messages)
        elif role == "assistant":
            _convert_assistant_message(content, internal_messages)

    # Convert tools
    internal_tools = _convert_tools(tools) if tools else None

    # Convert tool_choice
    internal_tool_choice = _convert_tool_choice(tool_choice)

    # Convert thinking → reasoning_effort
    reasoning_effort = _convert_thinking(thinking)

    return internal_messages, internal_tools, internal_tool_choice, stop_sequences, reasoning_effort


def _extract_system_text(system: str | list[dict[str, Any]]) -> str:
    if isinstance(system, str):
        return system
    parts: list[str] = []
    for block in system:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n\n".join(parts)


def _convert_user_message(content: str | list[dict[str, Any]], out: list[Message]) -> None:
    if isinstance(content, str):
        out.append(Message(role=MessageRole.USER, content=content))
        return
    text_parts: list[str] = []
    image_urls: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        elif block_type == "tool_result":
            if text_parts or image_urls:
                out.append(Message(
                    role=MessageRole.USER,
                    content="".join(text_parts),
                    images=tuple(image_urls),
                ))
                text_parts = []
                image_urls = []
            tool_use_id = block.get("tool_use_id") or "tool-call"
            result_content = block.get("content", "")
            tool_result_images: list[str] = []
            if isinstance(result_content, list):
                text_pieces: list[str] = []
                for p in result_content:
                    if not isinstance(p, dict):
                        continue
                    if p.get("type") == "text":
                        text_pieces.append(p.get("text", ""))
                    elif p.get("type") == "image":
                        source = p.get("source", {})
                        if isinstance(source, dict) and source.get("type") == "base64":
                            media_type = source.get("media_type", "image/png")
                            data = source.get("data", "")
                            tool_result_images.append(f"data:{media_type};base64,{data}")
                result_content = "".join(text_pieces)
            elif not isinstance(result_content, str):
                result_content = str(result_content) if result_content else ""
            out.append(Message(
                role=MessageRole.TOOL,
                content=result_content,
                tool_call_id=tool_use_id,
                name=tool_use_id,
            ))
            if tool_result_images:
                out.append(Message(
                    role=MessageRole.USER,
                    content="",
                    images=tuple(tool_result_images),
                ))
        elif block_type == "image":
            source = block.get("source", {})
            if isinstance(source, dict) and source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                image_urls.append(f"data:{media_type};base64,{data}")
    if text_parts or image_urls:
        out.append(Message(
            role=MessageRole.USER,
            content="".join(text_parts),
            images=tuple(image_urls),
        ))


def _convert_assistant_message(content: str | list[dict[str, Any]], out: list[Message]) -> None:
    if isinstance(content, str):
        out.append(Message(role=MessageRole.ASSISTANT, content=content))
        return
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    reasoning_content: str | None = None
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(ToolCall(
                id=block.get("id") or uuid.uuid4().hex,
                name=block.get("name") or "",
                arguments=block.get("input") or {},
            ))
        elif block_type == "thinking":
            thinking_text = block.get("thinking")
            if isinstance(thinking_text, str) and thinking_text:
                reasoning_content = thinking_text
    out.append(Message(
        role=MessageRole.ASSISTANT,
        content="".join(text_parts),
        tool_calls=tuple(tool_calls) if tool_calls else (),
        reasoning_content=reasoning_content,
    ))


def _convert_tools(tools: list[dict[str, Any]]) -> list[ToolSchema]:
    result: list[ToolSchema] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        result.append(ToolSchema(
            name=str(name),
            description=str(tool.get("description") or ""),
            parameters=tool.get("input_schema") or {},
        ))
    return result


def _convert_tool_choice(tc: dict[str, Any] | None) -> str | dict | None:
    if tc is None:
        return None
    tc_type = tc.get("type")
    if tc_type == "auto":
        return "auto"
    if tc_type == "any":
        return "required"
    if tc_type == "tool":
        return {"type": "function", "function": {"name": tc.get("name")}}
    if tc_type == "none":
        return "none"
    return "auto"


def _convert_thinking(thinking: dict[str, Any] | None) -> str | None:
    if thinking is None:
        return None
    if thinking.get("type") == "enabled":
        return "high"
    if thinking.get("type") == "adaptive":
        return "medium"
    return None


# ---------------------------------------------------------------------------
# Non-streaming response: internal → Anthropic
# ---------------------------------------------------------------------------

def internal_response_to_anthropic(
    response: AssistantResponse,
    model: str,
    request_id: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []

    if response.reasoning_content:
        content.append({
            "type": "thinking",
            "thinking": response.reasoning_content,
            "signature": "sig-placeholder",
        })

    if response.content:
        content.append({"type": "text", "text": response.content})

    for tc in response.tool_calls:
        content.append({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.name,
            "input": tc.arguments,
        })

    stop_reason = _map_stop_reason(response.finish_reason, bool(response.tool_calls))

    usage_dict: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
    if response.usage:
        usage_dict = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": response.usage.cached_tokens,
        }

    if not content:
        content.append({"type": "text", "text": ""})

    return {
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage_dict,
    }


def _map_stop_reason(finish_reason: str, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_use"
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "max_tokens": "max_tokens",
        "tool_calls": "tool_use",
        "tool_use": "tool_use",
        "stop_sequence": "stop_sequence",
    }
    return mapping.get(finish_reason, "end_turn")


# ---------------------------------------------------------------------------
# Streaming adapter: provider events → Anthropic SSE
# ---------------------------------------------------------------------------

def anthropic_stream_adapter(
    event_stream: Iterator[dict[str, Any]],
    model: str,
    request_id: str,
) -> Iterator[str]:
    """Convert provider chat_stream events into Anthropic SSE strings."""
    # Emit message_start
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    block_index = 0
    current_block: str | None = None  # "thinking", "text", "tool_use"
    has_any_content = False
    output_tokens = 0

    for event in event_stream:
        typ = event.get("type")

        if typ in ("reasoning_delta", "reasoning_raw_delta"):
            has_any_content = True
            text = str(event.get("text", ""))
            if current_block != "thinking":
                if current_block is not None:
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                    block_index += 1
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                })
                current_block = "thinking"
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "thinking_delta", "thinking": text},
            })

        elif typ == "content":
            has_any_content = True
            text = str(event.get("text", ""))
            if current_block == "thinking":
                # Close thinking block, emit signature
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "signature_delta", "signature": "sig-placeholder"},
                })
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                block_index += 1
                current_block = None
            if current_block != "text":
                if current_block is not None:
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                    block_index += 1
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                })
                current_block = "text"
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "text_delta", "text": text},
            })

        elif typ == "tool_call":
            has_any_content = True
            if current_block is not None:
                if current_block == "thinking":
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "signature_delta", "signature": "sig-placeholder"},
                    })
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                block_index += 1
            tool_id = str(event.get("id", ""))
            tool_name = str(event.get("name", ""))
            tool_args = event.get("arguments") or {}
            yield _sse("content_block_start", {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}},
            })
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_args, ensure_ascii=False)},
            })
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
            block_index += 1
            current_block = None

        elif typ == "finish":
            if current_block is not None:
                if current_block == "thinking":
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "signature_delta", "signature": "sig-placeholder"},
                    })
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                current_block = None

            if not has_any_content:
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                })
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})

            finish_reason = str(event.get("finish_reason") or "stop")
            stop_reason = _map_stop_reason(finish_reason, False)
            usage = event.get("usage")
            if isinstance(usage, dict):
                output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))

            yield _sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            })
            yield _sse("message_stop", {"type": "message_stop"})


def _sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------

def format_anthropic_error(status: int, message: str) -> dict[str, Any]:
    type_map = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
        500: "api_error",
        529: "overloaded_error",
    }
    return {
        "type": "error",
        "error": {
            "type": type_map.get(status, "api_error"),
            "message": message,
        },
    }
