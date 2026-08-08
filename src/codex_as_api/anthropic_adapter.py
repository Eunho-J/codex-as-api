from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

from .messages import AssistantResponse, Message, MessageRole, ToolCall, ToolSchema

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
    output_format: dict[str, Any] | None = None,
    output_config: object = None,
) -> tuple[
    list[Message], list[ToolSchema] | None, str | dict | None, list[str] | None, str | None, dict[str, Any] | None
]:
    """Convert Anthropic Messages request fields to internal types.

    Returns (messages, tools, tool_choice, stop, reasoning_effort, text).
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

    reasoning_effort = _convert_reasoning_effort(thinking, output_config)
    text = _convert_output_format(output_format, output_config)

    return internal_messages, internal_tools, internal_tool_choice, stop_sequences, reasoning_effort, text


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
                out.append(
                    Message(
                        role=MessageRole.USER,
                        content="".join(text_parts),
                        images=tuple(image_urls),
                    )
                )
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
                        image_url = _anthropic_image_url(source)
                        tool_result_images.append(image_url)
                    else:
                        rendered = _render_anthropic_content_block(p)
                        if rendered:
                            text_pieces.append(rendered)
                result_content = "".join(text_pieces)
            elif not isinstance(result_content, str):
                result_content = str(result_content) if result_content else ""
            if block.get("is_error") is True:
                result_content = f"[tool_error]\n{result_content}"
            out.append(
                Message(
                    role=MessageRole.TOOL,
                    content=result_content,
                    tool_call_id=tool_use_id,
                    name=tool_use_id,
                )
            )
            if tool_result_images:
                out.append(
                    Message(
                        role=MessageRole.USER,
                        content="",
                        images=tuple(tool_result_images),
                    )
                )
        elif block_type == "image":
            source = block.get("source", {})
            image_url = _anthropic_image_url(source)
            image_urls.append(image_url)
        else:
            rendered = _render_anthropic_content_block(block)
            if rendered:
                text_parts.append(rendered)
    if text_parts or image_urls:
        out.append(
            Message(
                role=MessageRole.USER,
                content="".join(text_parts),
                images=tuple(image_urls),
            )
        )


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
            tool_calls.append(
                ToolCall(
                    id=block.get("id") or uuid.uuid4().hex,
                    name=block.get("name") or "",
                    arguments=block.get("input") or {},
                )
            )
        elif block_type == "thinking":
            thinking_text = block.get("thinking")
            if isinstance(thinking_text, str) and thinking_text:
                reasoning_content = thinking_text
        elif block_type == "redacted_thinking":
            if reasoning_content is None:
                reasoning_content = "[redacted_thinking omitted]"
        elif block_type == "server_tool_use":
            text_parts.append(_render_server_tool_use_block(block))
        elif block_type == "web_search_tool_result":
            text_parts.append(_render_generic_tool_result_block(block))
        else:
            rendered = _render_anthropic_content_block(block)
            if rendered:
                text_parts.append(rendered)
    out.append(
        Message(
            role=MessageRole.ASSISTANT,
            content="".join(text_parts),
            tool_calls=tuple(tool_calls) if tool_calls else (),
            reasoning_content=reasoning_content,
        )
    )


def _convert_tools(tools: list[dict[str, Any]]) -> list[ToolSchema]:
    result: list[ToolSchema] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "programmatic_tool_calling":
            raise ValueError(
                "programmatic_tool_calling requires native Responses program/caller replay "
                "and is not supported by the Anthropic facade"
            )
        if any(key in tool for key in ("allowed_callers", "output_schema")):
            raise ValueError(
                f"Anthropic tool {index} uses Programmatic Tool Calling fields that cannot be preserved"
            )
        for field in ("strict", "defer_loading", "eager_input_streaming"):
            value = tool.get(field)
            if value is not None and value is not False:
                raise ValueError(f"Anthropic tool {index} field {field} is not supported by the Codex OAuth backend")
        name = tool.get("name")
        if not name:
            continue
        if _is_anthropic_web_search_tool(tool):
            result.append(
                ToolSchema(
                    name="web_search",
                    description="Anthropic hosted web search",
                    parameters=_anthropic_web_search_parameters(tool),
                )
            )
            continue
        result.append(
            ToolSchema(
                name=str(name),
                description=str(tool.get("description") or ""),
                parameters=tool.get("input_schema") or {},
            )
        )
    return result


def _anthropic_image_url(source: object) -> str:
    if not isinstance(source, dict):
        raise ValueError("Anthropic image source must be an object")
    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data")
        if isinstance(media_type, str) and isinstance(data, str) and data:
            return f"data:{media_type};base64,{data}"
        raise ValueError("Anthropic base64 image source requires non-empty data")
    if source_type == "url":
        url = source.get("url")
        if isinstance(url, str) and url:
            return url
        raise ValueError("Anthropic URL image source requires a non-empty url")
    raise ValueError("Anthropic image source type must be base64 or url")


def _is_anthropic_web_search_tool(tool: dict[str, Any]) -> bool:
    return (
        tool.get("name") == "web_search"
        and isinstance(tool.get("type"), str)
        and (tool["type"] == "web_search" or tool["type"].startswith("web_search_"))
    )


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    out = [v for v in value if isinstance(v, str) and v]
    return out or None


def _anthropic_web_search_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    if _string_list(tool.get("blocked_domains")):
        raise ValueError(
            "Anthropic web_search blocked_domains is not supported by OpenAI Responses web_search; "
            "use allowed_domains instead"
        )

    openai_tool: dict[str, Any] = {"type": "web_search", "external_web_access": True}
    allowed_domains = _string_list(tool.get("allowed_domains"))
    if allowed_domains:
        openai_tool["filters"] = {"allowed_domains": allowed_domains}
    user_location = tool.get("user_location")
    if isinstance(user_location, dict):
        openai_tool["user_location"] = user_location

    return {
        "__codex_as_api_tool_type": "web_search",
        "openai_tool": openai_tool,
        "anthropic": {
            "type": tool.get("type"),
            "max_uses": tool.get("max_uses"),
        },
    }


def _convert_tool_choice(tc: dict[str, Any] | None) -> str | dict | None:
    if tc is None:
        return None
    tc_type = tc.get("type")
    if tc_type == "auto":
        return "auto"
    if tc_type == "any":
        return "required"
    if tc_type == "tool":
        if tc.get("name") == "web_search":
            return {"type": "web_search"}
        return {"type": "function", "name": tc.get("name")}
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
    if thinking.get("type") == "disabled":
        return "none"
    return None


def _convert_reasoning_effort(thinking: dict[str, Any] | None, output_config: object) -> str | None:
    output_effort: str | None = None
    if output_config is not None:
        if not isinstance(output_config, dict):
            raise ValueError("output_config must be an object")
        unknown = sorted(set(output_config) - {"effort", "format", "task_budget"})
        if unknown:
            raise ValueError("output_config contains unsupported fields: " + ", ".join(unknown))
        task_budget = output_config.get("task_budget")
        if task_budget is not None:
            raise ValueError("output_config.task_budget is not supported by the Codex OAuth backend")
        raw_effort = output_config.get("effort")
        if raw_effort is not None:
            if not isinstance(raw_effort, str) or raw_effort not in {
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
            }:
                raise ValueError("output_config.effort must be one of: low, medium, high, xhigh, max")
            output_effort = raw_effort

    thinking_effort = _convert_thinking(thinking)
    if thinking_effort == "none":
        return "none"
    return output_effort or thinking_effort


def _convert_output_format(
    output_format: object,
    output_config: object,
) -> dict[str, Any] | None:
    config_format: object = None
    if isinstance(output_config, dict):
        config_format = output_config.get("format")

    converted: list[tuple[object, dict[str, Any]]] = []
    for field, value in (("output_format", output_format), ("output_config.format", config_format)):
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be an object")
        _validate_anthropic_output_format(value, field)
        text = anthropic_output_format_to_openai_text(value)
        if text is None:  # pragma: no cover - validation above guarantees conversion
            raise AssertionError("validated Anthropic output format did not convert")
        converted.append((value, text))

    if len(converted) == 2 and converted[0][0] != converted[1][0]:
        raise ValueError("output_format conflicts with output_config.format")
    return converted[0][1] if converted else None


def anthropic_output_format_to_openai_text(output_format: dict[str, Any] | None) -> dict[str, Any] | None:
    if output_format is None:
        return None
    _validate_anthropic_output_format(output_format, "output_format")
    typ = output_format.get("type")
    if typ == "json_schema":
        schema = output_format.get("schema")
        name = _sanitize_json_schema_name(str(output_format.get("name") or "structured_output"))
        fmt: dict[str, Any] = {
            "type": "json_schema",
            "name": name,
            "schema": schema,
        }
        if isinstance(output_format.get("description"), str):
            fmt["description"] = output_format["description"]
        if isinstance(output_format.get("strict"), bool):
            fmt["strict"] = output_format["strict"]
        return {"format": fmt}
    if typ == "json_object":
        return {"format": {"type": "json_object"}}
    raise AssertionError("validated Anthropic output format has an unknown type")


def _validate_anthropic_output_format(output_format: dict[str, Any], field: str) -> None:
    typ = output_format.get("type")
    if not isinstance(typ, str):
        raise ValueError(f"{field}.type must be a string")
    if typ == "json_object":
        unknown = sorted(set(output_format) - {"type"})
        if unknown:
            raise ValueError(f"{field}.{unknown[0]} is not supported")
        return
    if typ != "json_schema":
        raise ValueError(f"{field}.type must be one of: json_object, json_schema")

    unknown = sorted(set(output_format) - {"type", "name", "description", "schema", "strict"})
    if unknown:
        raise ValueError(f"{field}.{unknown[0]} is not supported")
    if not isinstance(output_format.get("schema"), dict):
        raise ValueError(f"{field}.schema must be an object")
    name = output_format.get("name")
    if name is not None and (not isinstance(name, str) or not name):
        raise ValueError(f"{field}.name must be a non-empty string when provided")
    description = output_format.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError(f"{field}.description must be a string when provided")
    strict = output_format.get("strict")
    if strict is not None and not isinstance(strict, bool):
        raise ValueError(f"{field}.strict must be a boolean when provided")


def _sanitize_json_schema_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)[:64]
    return cleaned or "structured_output"


def _render_anthropic_content_block(block: dict[str, Any]) -> str:
    typ = block.get("type") if isinstance(block.get("type"), str) else "unknown"
    if typ == "document":
        return _render_document_block(block)
    if typ == "search_result":
        return _render_search_result_block(block)
    if isinstance(typ, str) and typ.endswith("_tool_result"):
        return _render_generic_tool_result_block(block)
    return f"\n\n[{typ}] {_safe_json(block)}\n"


def _render_document_block(block: dict[str, Any]) -> str:
    title = (
        block.get("title")
        if isinstance(block.get("title"), str)
        else block.get("name")
        if isinstance(block.get("name"), str)
        else "document"
    )
    source = block.get("source")
    body = ""
    if isinstance(source, dict):
        if source.get("type") == "text" and isinstance(source.get("data"), str):
            body = source["data"]
        elif source.get("type") == "url" and isinstance(source.get("url"), str):
            body = source["url"]
        elif isinstance(source.get("media_type"), str):
            body = f"[{source['media_type']}]"
    return f"\n\n[document: {title}]" + (f"\n{body}" if body else "") + "\n"


def _render_search_result_block(block: dict[str, Any]) -> str:
    title = block.get("title") if isinstance(block.get("title"), str) else "search result"
    url = block.get("url") if isinstance(block.get("url"), str) else ""
    content = block.get("content") if isinstance(block.get("content"), str) else ""
    return f"\n\n[search_result] {title}" + (f" ({url})" if url else "") + (f"\n{content}" if content else "") + "\n"


def _render_server_tool_use_block(block: dict[str, Any]) -> str:
    name = block.get("name") if isinstance(block.get("name"), str) else "server_tool"
    return f"\n\n[server_tool_use: {name}] {_safe_json(block.get('input') or {})}\n"


def _render_generic_tool_result_block(block: dict[str, Any]) -> str:
    typ = block.get("type") if isinstance(block.get("type"), str) else "tool_result"
    content = block.get("content")
    if isinstance(content, list):
        lines: list[str] = []
        for item in content:
            if isinstance(item, dict):
                title = item.get("title") if isinstance(item.get("title"), str) else None
                url = item.get("url") if isinstance(item.get("url"), str) else ""
                text = (
                    item.get("text")
                    if isinstance(item.get("text"), str)
                    else item.get("content")
                    if isinstance(item.get("content"), str)
                    else ""
                )
                if title or url or text:
                    lines.append(
                        f"- {title or 'result'}" + (f" ({url})" if url else "") + (f": {text}" if text else "")
                    )
                else:
                    lines.append(_safe_json(item))
            else:
                lines.append(str(item))
        return f"\n\n[{typ}]" + (f"\n{chr(10).join(lines)}" if lines else "") + "\n"
    if isinstance(content, dict):
        return f"\n\n[{typ}] {_safe_json(content)}\n"
    if isinstance(content, str):
        return f"\n\n[{typ}]\n{content}\n"
    return f"\n\n[{typ}]\n"


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


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
        content.append(
            {
                "type": "thinking",
                "thinking": response.reasoning_content,
                "signature": "sig-placeholder",
            }
        )

    web_search_blocks = _web_search_blocks_from_raw(response.raw)
    content.extend(web_search_blocks)

    if response.content:
        content.append({"type": "text", "text": response.content})

    for tc in response.tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            }
        )

    stop_reason = _map_stop_reason(response.finish_reason, bool(response.tool_calls))

    usage_dict: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
    if response.usage:
        usage_dict = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "cache_creation_input_tokens": response.usage.cache_write_tokens,
            "cache_read_input_tokens": response.usage.cached_tokens,
        }
    usage_dict = _merge_server_tool_usage(usage_dict, response.raw, len(web_search_blocks) // 2)

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


def _web_search_blocks_from_raw(raw: dict | None) -> list[dict[str, Any]]:
    events = raw.get("events") if isinstance(raw, dict) else None
    if not isinstance(events, list):
        return []
    blocks: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "web_search_call":
            continue
        tool_id = str(event.get("id") or f"srvtoolu_{len(blocks) // 2}")
        input_obj = event.get("input") if isinstance(event.get("input"), dict) else {"query": ""}
        result_content = event.get("content") if isinstance(event.get("content"), list) else []
        blocks.append({"type": "server_tool_use", "id": tool_id, "name": "web_search", "input": input_obj})
        blocks.append({"type": "web_search_tool_result", "tool_use_id": tool_id, "content": result_content})
    return blocks


def _merge_server_tool_usage(
    usage: dict[str, Any],
    raw: dict | None,
    web_search_requests: int,
) -> dict[str, Any]:
    events = raw.get("events") if isinstance(raw, dict) else None
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict) or event.get("type") != "finish":
                continue
            raw_usage = event.get("usage")
            if isinstance(raw_usage, dict) and "server_tool_use" in raw_usage:
                usage["server_tool_use"] = raw_usage["server_tool_use"]
                return usage
    if web_search_requests > 0 and "server_tool_use" not in usage:
        usage["server_tool_use"] = {"web_search_requests": web_search_requests}
    return usage


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
        "pause_turn": "pause_turn",
        "refusal": "refusal",
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
    yield _message_start_sse(model, request_id, {"input_tokens": 0, "output_tokens": 0})
    yield from _render_anthropic_stream_events(event_stream)


def _message_start_sse(model: str, request_id: str, usage: dict[str, Any]) -> str:
    return _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": request_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage,
            },
        },
    )


def _num(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _anthropic_usage_from_provider(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0}
    token_details = usage.get("input_tokens_details", usage.get("prompt_tokens_details"))
    cache_read = _num(usage.get("cache_read_input_tokens", usage.get("cached_input_tokens")))
    cache_write = _num(
        usage.get(
            "cache_creation_input_tokens",
            usage.get("cache_write_tokens", usage.get("cache_write_input_tokens")),
        )
    )
    if cache_read == 0 and isinstance(token_details, dict):
        cache_read = _num(token_details.get("cached_tokens"))
    if cache_write == 0 and isinstance(token_details, dict):
        cache_write = _num(token_details.get("cache_write_tokens"))
    out: dict[str, Any] = {
        "input_tokens": _num(usage.get("input_tokens", usage.get("prompt_tokens"))),
        "output_tokens": _num(usage.get("output_tokens", usage.get("completion_tokens"))),
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
    }
    for key in ("cache_creation", "server_tool_use", "service_tier"):
        if key in usage:
            out[key] = usage[key]
    return out


def _render_anthropic_stream_events(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    block_index = 0
    current_block: str | None = None  # "thinking", "text", "tool_use"
    has_any_content = False
    web_search_requests = 0

    for event in events:
        typ = event.get("type")

        if typ in ("reasoning_delta", "reasoning_raw_delta"):
            has_any_content = True
            text = str(event.get("text", ""))
            if current_block != "thinking":
                if current_block is not None:
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                    block_index += 1
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                    },
                )
                current_block = "thinking"
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "thinking_delta", "thinking": text},
                },
            )

        elif typ == "content":
            has_any_content = True
            text = str(event.get("text", ""))
            if current_block == "thinking":
                # Close thinking block, emit signature
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "signature_delta", "signature": "sig-placeholder"},
                    },
                )
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                block_index += 1
                current_block = None
            if current_block != "text":
                if current_block is not None:
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                    block_index += 1
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                current_block = "text"
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )

        elif typ == "tool_call":
            has_any_content = True
            if current_block is not None:
                if current_block == "thinking":
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "signature_delta", "signature": "sig-placeholder"},
                        },
                    )
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                block_index += 1
            tool_id = str(event.get("id", ""))
            tool_name = str(event.get("name", ""))
            tool_args = event.get("arguments") or {}
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}},
                },
            )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_args, ensure_ascii=False)},
                },
            )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
            block_index += 1
            current_block = None

        elif typ == "web_search_call":
            has_any_content = True
            web_search_requests += 1
            if current_block is not None:
                if current_block == "thinking":
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "signature_delta", "signature": "sig-placeholder"},
                        },
                    )
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                block_index += 1
            tool_id = str(event.get("id") or "")
            tool_input = event.get("input") if isinstance(event.get("input"), dict) else {"query": ""}
            result_content = event.get("content") if isinstance(event.get("content"), list) else []
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "server_tool_use", "id": tool_id, "name": "web_search", "input": {}},
                },
            )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_input, ensure_ascii=False)},
                },
            )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
            block_index += 1
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "web_search_tool_result",
                        "tool_use_id": tool_id,
                        "content": result_content,
                    },
                },
            )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
            block_index += 1
            current_block = None

        elif typ == "finish":
            if current_block is not None:
                if current_block == "thinking":
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "signature_delta", "signature": "sig-placeholder"},
                        },
                    )
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                current_block = None

            if not has_any_content:
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})

            finish_reason = str(event.get("finish_reason") or "stop")
            stop_reason = _map_stop_reason(finish_reason, False)

            yield _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": _usage_with_synthesized_web_search(
                        _anthropic_usage_from_provider(event.get("usage")),
                        web_search_requests,
                    ),
                },
            )
            yield _sse("message_stop", {"type": "message_stop"})


def _usage_with_synthesized_web_search(usage: dict[str, Any], web_search_requests: int) -> dict[str, Any]:
    if web_search_requests > 0 and "server_tool_use" not in usage:
        usage["server_tool_use"] = {"web_search_requests": web_search_requests}
    return usage


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
