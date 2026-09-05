from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from .auth import ChatGPTOAuthProtocolError
from .messages import AssistantResponse, Message, MessageRole, ToolCall, ToolSchema, Usage
from .strict_json import as_js_safe_integer, strict_json_loads

# ---------------------------------------------------------------------------
# Request conversion: Anthropic → internal
# ---------------------------------------------------------------------------


def anthropic_request_to_internal(
    *,
    model: object,
    messages: object,
    system: str | list[dict[str, Any]] | None = None,
    tools: object = None,
    tool_choice: dict[str, Any] | None = None,
    stop_sequences: list[str] | None = None,
    thinking: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    output_format: dict[str, Any] | None = None,
    output_config: object = None,
) -> tuple[
    list[Message], list[ToolSchema] | None, str | dict | None, list[str] | None, str | None, dict[str, Any] | None
]:
    """Convert Anthropic Messages request fields to internal types.

    Returns (messages, tools, tool_choice, stop, reasoning_effort, text).
    """
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    if stop_sequences is not None and (
        not isinstance(stop_sequences, list) or any(not isinstance(value, str) or not value for value in stop_sequences)
    ):
        raise ValueError("stop_sequences must be an array of non-empty strings")
    internal_messages: list[Message] = []

    # System prompt → SYSTEM message
    if system is not None:
        sys_text = _extract_system_text(system)
        if sys_text:
            internal_messages.append(Message(role=MessageRole.SYSTEM, content=sys_text))

    # Convert messages
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{index}] must be an object")
        _reject_unknown_fields(msg, {"role", "content"}, f"messages[{index}]")
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            _convert_user_message(content, internal_messages)
        elif role == "assistant":
            _convert_assistant_message(content, internal_messages)
        else:
            raise ValueError(f"messages[{index}].role must be user or assistant")

    # Convert tools
    internal_tools = _convert_tools(tools) if tools is not None else None

    # Convert tool_choice
    internal_tool_choice = _convert_tool_choice(tool_choice)

    reasoning_effort = _convert_reasoning_effort(thinking, output_config, max_tokens)
    text = _convert_output_format(output_format, output_config)

    return internal_messages, internal_tools, internal_tool_choice, stop_sequences, reasoning_effort, text


def _extract_system_text(system: str | list[dict[str, Any]]) -> str:
    if isinstance(system, str):
        return system
    if not isinstance(system, list):
        raise ValueError("system must be a string or array")
    parts: list[str] = []
    for index, block in enumerate(system):
        if not isinstance(block, dict) or block.get("type") != "text":
            raise ValueError(f"system[{index}] must be a text block")
        _validate_nullable_unrepresentable_field(block, "citations", f"system[{index}]")
        _reject_unknown_fields(block, {"type", "text", "citations"}, f"system[{index}]")
        text = block.get("text")
        if not isinstance(text, str):
            raise ValueError(f"system[{index}].text must be a string")
        parts.append(text)
    return "\n\n".join(parts)


def _convert_user_message(content: object, out: list[Message]) -> None:
    if isinstance(content, str):
        out.append(Message(role=MessageRole.USER, content=content))
        return
    if not isinstance(content, list):
        raise ValueError("Anthropic user message content must be a string or array")
    if not content:
        raise ValueError("Anthropic user message content blocks must be non-empty")
    text_parts: list[str] = []
    image_urls: list[str] = []
    for index, block in enumerate(content):
        if not isinstance(block, dict):
            raise ValueError(f"Anthropic user content block {index} must be an object")
        block_type = block.get("type")
        if block_type == "text":
            _validate_nullable_unrepresentable_field(
                block,
                "citations",
                f"Anthropic user content block {index}",
            )
            _reject_unknown_fields(
                block,
                {"type", "text", "citations"},
                f"Anthropic user content block {index}",
            )
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Anthropic user text block {index} requires string text")
            text_parts.append(text)
        elif block_type == "tool_result":
            _reject_unknown_fields(
                block,
                {"type", "tool_use_id", "content", "is_error"},
                f"Anthropic tool_result block {index}",
            )
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
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str):
                raise ValueError(f"Anthropic tool_result block {index} requires string tool_use_id")
            result_content = block.get("content", "")
            tool_result_images: list[str] = []
            if isinstance(result_content, list):
                text_pieces: list[str] = []
                for part_index, p in enumerate(result_content):
                    if not isinstance(p, dict):
                        raise ValueError(f"Anthropic tool_result block {index} content {part_index} must be an object")
                    if p.get("type") == "text":
                        _validate_nullable_unrepresentable_field(
                            p,
                            "citations",
                            f"Anthropic tool_result block {index} content {part_index}",
                        )
                        _reject_unknown_fields(
                            p,
                            {"type", "text", "citations"},
                            f"Anthropic tool_result block {index} content {part_index}",
                        )
                        text = p.get("text")
                        if not isinstance(text, str):
                            raise ValueError(f"Anthropic tool_result block {index} text content requires string text")
                        text_pieces.append(text)
                    elif p.get("type") == "image":
                        _reject_unknown_fields(
                            p,
                            {"type", "source"},
                            f"Anthropic tool_result block {index} content {part_index}",
                        )
                        source = p.get("source")
                        image_url = _anthropic_image_url(source)
                        tool_result_images.append(image_url)
                    else:
                        text_pieces.append(_render_anthropic_content_block(p))
                result_content = "".join(text_pieces)
            elif not isinstance(result_content, str):
                raise ValueError(f"Anthropic tool_result block {index} content must be a string or array")
            if "is_error" in block and not isinstance(block["is_error"], bool):
                raise ValueError(f"Anthropic tool_result block {index} is_error must be a boolean")
            if block.get("is_error") is True:
                result_content = f"[tool_error]\n{result_content}"
            out.append(
                Message(
                    role=MessageRole.TOOL,
                    content=result_content,
                    tool_call_id=tool_use_id,
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
            _reject_unknown_fields(
                block,
                {"type", "source"},
                f"Anthropic user content block {index}",
            )
            source = block.get("source")
            image_url = _anthropic_image_url(source)
            image_urls.append(image_url)
        else:
            text_parts.append(_render_anthropic_content_block(block))
    if text_parts or image_urls:
        out.append(
            Message(
                role=MessageRole.USER,
                content="".join(text_parts),
                images=tuple(image_urls),
            )
        )


def _convert_assistant_message(content: object, out: list[Message]) -> None:
    if isinstance(content, str):
        out.append(Message(role=MessageRole.ASSISTANT, content=content))
        return
    if not isinstance(content, list):
        raise ValueError("Anthropic assistant message content must be a string or array")
    if not content:
        out.append(Message(role=MessageRole.ASSISTANT, content=""))
        return
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    tool_call_ids: set[str] = set()
    for index, block in enumerate(content):
        if not isinstance(block, dict):
            raise ValueError(f"Anthropic assistant content block {index} must be an object")
        block_type = block.get("type")
        if block_type == "text":
            _validate_nullable_unrepresentable_field(
                block,
                "citations",
                f"Anthropic assistant content block {index}",
            )
            _reject_unknown_fields(
                block,
                {"type", "text", "citations"},
                f"Anthropic assistant content block {index}",
            )
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Anthropic assistant text block {index} requires string text")
            text_parts.append(text)
        elif block_type == "tool_use":
            _reject_unknown_fields(
                block,
                {"type", "id", "name", "input", "caller"},
                f"Anthropic tool_use block {index}",
            )
            _validate_direct_tool_caller(block, f"Anthropic tool_use block {index}")
            call_id = block.get("id")
            name = block.get("name")
            arguments = block.get("input")
            if not isinstance(call_id, str):
                raise ValueError(f"Anthropic tool_use block {index} requires string id")
            if not isinstance(name, str):
                raise ValueError(f"Anthropic tool_use block {index} requires string name")
            if not isinstance(arguments, dict):
                raise ValueError(f"Anthropic tool_use block {index} input must be an object")
            if call_id in tool_call_ids:
                raise ValueError(f"Anthropic assistant tool_use blocks contain duplicate id {call_id!r}")
            tool_call_ids.add(call_id)
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                )
            )
        elif block_type in {
            "thinking",
            "redacted_thinking",
            "server_tool_use",
            "web_search_tool_result",
        }:
            raise ValueError(
                f"Anthropic assistant content block type {block_type!r} cannot be preserved by this facade"
            )
        else:
            text_parts.append(_render_anthropic_content_block(block))
    out.append(
        Message(
            role=MessageRole.ASSISTANT,
            content="".join(text_parts),
            tool_calls=tuple(tool_calls) if tool_calls else (),
        )
    )


def _convert_tools(tools: object) -> list[ToolSchema]:
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")
    result: list[ToolSchema] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"Anthropic tool {index} must be an object")
        if tool.get("type") == "programmatic_tool_calling":
            raise ValueError(
                "programmatic_tool_calling requires native Responses program/caller replay "
                "and is not supported by the Anthropic facade"
            )
        if "allowed_callers" in tool or "output_schema" in tool:
            raise ValueError(f"Anthropic tool {index} uses Programmatic Tool Calling fields that cannot be preserved")
        if "defer_loading" in tool:
            raise ValueError(f"Anthropic tool {index} field defer_loading is not supported by the Codex OAuth backend")
        for field in ("eager_input_streaming",):
            value = tool.get(field)
            if value is not None:
                raise ValueError(f"Anthropic tool {index} field {field} is not supported by the Codex OAuth backend")
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Anthropic tool {index} requires a non-empty name")
        if _is_anthropic_web_search_tool(tool):
            raise ValueError(
                "Anthropic hosted web_search cannot be represented losslessly by this facade"
            )
        tool_type = tool.get("type", "custom")
        if tool_type is not None and tool_type != "custom":
            raise ValueError(f"Anthropic tool {index} type must be 'custom' or null")
        _reject_unknown_fields(
            tool,
            {
                "type",
                "name",
                "description",
                "input_schema",
                "strict",
                "defer_loading",
                "eager_input_streaming",
                "allowed_callers",
                "output_schema",
            },
            f"Anthropic tool {index}",
        )
        result.append(
            ToolSchema(
                name=name,
                description=_optional_tool_description(tool, index),
                parameters=_required_tool_schema(tool, index),
                strict=_tool_strict(tool, index),
            )
        )
    return result


def _anthropic_image_url(source: object) -> str:
    if not isinstance(source, dict):
        raise ValueError("Anthropic image source must be an object")
    source_type = source.get("type")
    if source_type == "base64":
        _reject_unknown_fields(source, {"type", "media_type", "data"}, "Anthropic image source")
        media_type = source.get("media_type")
        data = source.get("data")
        if media_type in {"image/jpeg", "image/png", "image/gif", "image/webp"} and isinstance(data, str) and data:
            return f"data:{media_type};base64,{data}"
        raise ValueError(
            "Anthropic base64 image source media_type must be one of: "
            "image/jpeg, image/png, image/gif, image/webp, and data must be non-empty"
        )
    if source_type == "url":
        _reject_unknown_fields(source, {"type", "url"}, "Anthropic image source")
        url = source.get("url")
        if isinstance(url, str) and url:
            return url
        raise ValueError("Anthropic URL image source requires a non-empty url")
    raise ValueError("Anthropic image source type must be base64 or url")


def _optional_tool_description(tool: dict[str, Any], index: int) -> str | None:
    if "description" not in tool:
        return None
    description = tool["description"]
    if not isinstance(description, str):
        raise ValueError(f"Anthropic tool {index} description must be a string")
    return description


def _tool_strict(tool: dict[str, Any], index: int) -> bool:
    if "strict" not in tool:
        return False
    strict = tool["strict"]
    if not isinstance(strict, bool):
        raise ValueError(f"Anthropic tool {index} strict must be a boolean")
    return strict


def _required_tool_schema(tool: dict[str, Any], index: int) -> dict[str, Any]:
    schema = tool.get("input_schema")
    if not isinstance(schema, dict):
        raise ValueError(f"Anthropic tool {index} input_schema must be an object")
    return schema


def _is_anthropic_web_search_tool(tool: dict[str, Any]) -> bool:
    return (
        tool.get("name") == "web_search"
        and isinstance(tool.get("type"), str)
        and tool["type"] in {"web_search", "web_search_20250305", "web_search_20260209"}
    )


def _convert_tool_choice(tc: dict[str, Any] | None) -> str | dict | None:
    if tc is None:
        return None
    if not isinstance(tc, dict):
        raise ValueError("tool_choice must be an object")
    tc_type = tc.get("type")
    if tc_type == "tool":
        allowed = {"type", "name", "disable_parallel_tool_use"}
    elif tc_type == "none":
        allowed = {"type"}
    else:
        allowed = {"type", "disable_parallel_tool_use"}
    _reject_unknown_fields(tc, allowed, "tool_choice")
    anthropic_parallel_tool_calls(tc)
    if tc_type == "auto":
        return "auto"
    if tc_type == "any":
        return "required"
    if tc_type == "tool":
        name = tc.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_choice.name must be a non-empty string")
        if name == "web_search":
            raise ValueError(
                "Anthropic hosted web_search cannot be represented losslessly by this facade"
            )
        return {"type": "function", "name": name}
    if tc_type == "none":
        return "none"
    raise ValueError("tool_choice.type must be one of: auto, any, tool, none")


def anthropic_parallel_tool_calls(
    tool_choice: dict[str, Any] | None,
) -> bool | None:
    if tool_choice is None or "disable_parallel_tool_use" not in tool_choice:
        return None
    value = tool_choice["disable_parallel_tool_use"]
    if not isinstance(value, bool):
        raise ValueError("tool_choice.disable_parallel_tool_use must be a boolean")
    return not value


def _convert_thinking(thinking: dict[str, Any] | None, max_tokens: int | None) -> str | None:
    if thinking is None:
        return None
    if not isinstance(thinking, dict):
        raise ValueError("thinking must be an object")
    thinking_type = thinking.get("type")
    if thinking_type == "enabled":
        _reject_unknown_fields(thinking, {"type", "budget_tokens", "display"}, "thinking")
        display = thinking.get("display")
        if display is not None and display != "omitted":
            raise ValueError("thinking.display must be 'omitted' when provided")
        budget_tokens = thinking.get("budget_tokens")
        parsed_budget_tokens = as_js_safe_integer(budget_tokens)
        if parsed_budget_tokens is None or parsed_budget_tokens < 1024:
            raise ValueError("thinking.budget_tokens must be an integer greater than or equal to 1024")
        if max_tokens is not None and parsed_budget_tokens >= max_tokens:
            raise ValueError("thinking.budget_tokens must be less than max_tokens")
        return "high"
    if thinking_type == "adaptive":
        _reject_unknown_fields(thinking, {"type", "display"}, "thinking")
        display = thinking.get("display")
        if display is not None and display != "omitted":
            raise ValueError("thinking.display must be 'omitted' when provided")
        return "medium"
    if thinking_type == "disabled":
        _reject_unknown_fields(thinking, {"type"}, "thinking")
        return "none"
    raise ValueError("thinking.type must be one of: enabled, adaptive, disabled")


def _convert_reasoning_effort(
    thinking: dict[str, Any] | None,
    output_config: object,
    max_tokens: int | None,
) -> str | None:
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

    thinking_effort = _convert_thinking(thinking, max_tokens)
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
        name = output_format.get("name", "codex_output_schema")
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
    name = output_format.get("name", "codex_output_schema")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError(
            f"{field}.name must contain only ASCII letters, digits, underscores, or hyphens "
            "and be at most 64 characters"
        )
    description = output_format.get("description")
    if "description" in output_format and not isinstance(description, str):
        raise ValueError(f"{field}.description must be a string when provided")
    strict = output_format.get("strict")
    if strict is not None and not isinstance(strict, bool):
        raise ValueError(f"{field}.strict must be a boolean when provided")


def _render_anthropic_content_block(block: dict[str, Any]) -> str:
    typ = block.get("type")
    if not isinstance(typ, str) or not typ:
        raise ValueError("Anthropic content block requires a non-empty type")
    if typ in {"document", "search_result"}:
        raise ValueError(f"Anthropic content block type {typ!r} cannot be preserved by this facade")
    raise ValueError(f"Anthropic content block type {typ!r} is not supported by this facade")


def _validate_nullable_unrepresentable_field(
    value: dict[str, Any],
    field: str,
    location: str,
) -> None:
    if field in value and value[field] is not None:
        raise ValueError(f"{location}.{field} cannot be preserved by this facade")


def _validate_direct_tool_caller(block: dict[str, Any], location: str) -> None:
    if "caller" not in block:
        return
    caller = block["caller"]
    if not isinstance(caller, dict) or caller != {"type": "direct"}:
        raise ValueError(f"{location}.caller must be exactly {{'type': 'direct'}}")


def _reject_unknown_fields(
    value: dict[str, Any],
    allowed: set[str],
    location: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{location} contains unsupported fields: {', '.join(unknown)}")


def _parse_anthropic_tool_arguments(value: object) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ChatGPTOAuthProtocolError("tool call arguments must be a JSON object string")
    try:
        parsed = strict_json_loads(value)
    except ValueError as exc:
        raise ChatGPTOAuthProtocolError("tool call arguments must contain valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ChatGPTOAuthProtocolError("tool call arguments JSON must be an object")
    return parsed


# ---------------------------------------------------------------------------
# Non-streaming response: internal → Anthropic
# ---------------------------------------------------------------------------


def internal_response_to_anthropic(
    response: AssistantResponse,
    model: str,
    request_id: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []

    _reject_unrepresentable_response_events(response.raw)

    if response.content:
        content.append({"type": "text", "text": response.content, "citations": None})

    for tc in response.tool_calls:
        arguments = _parse_anthropic_tool_arguments(tc.arguments)
        content.append(
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": arguments,
                "caller": {"type": "direct"},
            }
        )

    stop_reason = _map_stop_reason(response.finish_reason, bool(response.tool_calls))
    if response.usage is None:
        raise ChatGPTOAuthProtocolError("provider response requires authoritative usage")
    usage_dict = _merge_usage_extensions(
        _anthropic_usage_from_internal(response.usage),
        response.raw,
    )

    result: dict[str, Any] = {
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "container": None,
        "content": content,
        "context_management": None,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage_dict,
    }
    if response.reasoning_content:
        result["codex_reasoning"] = response.reasoning_content
    return result


def _raw_response_events(raw: dict | None) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise ChatGPTOAuthProtocolError("provider raw response must be an object")
    events = raw.get("events")
    if not isinstance(events, list):
        raise ChatGPTOAuthProtocolError("provider raw response requires an events array")
    validated: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ChatGPTOAuthProtocolError(f"provider raw event {index} must be an object")
        if not isinstance(event.get("type"), str) or not event["type"]:
            raise ChatGPTOAuthProtocolError(f"provider raw event {index} requires a non-empty string type")
        validated.append(event)
    return validated


def _reject_unrepresentable_response_events(raw: dict | None) -> None:
    if any(event.get("type") == "web_search_call" for event in _raw_response_events(raw)):
        raise ChatGPTOAuthProtocolError(
            "provider web_search_call output cannot be represented losslessly by the Anthropic facade"
        )


def _anthropic_usage_from_internal(usage: Usage) -> dict[str, Any]:
    return {
        "cache_creation": None,
        "cache_creation_input_tokens": usage.cache_write_tokens,
        "cache_read_input_tokens": usage.cached_tokens,
        "inference_geo": None,
        "input_tokens": usage.prompt_tokens,
        "iterations": None,
        "output_tokens": usage.completion_tokens,
        "server_tool_use": None,
        "service_tier": None,
        "speed": None,
    }


def _merge_usage_extensions(
    usage: dict[str, Any],
    raw: dict | None,
) -> dict[str, Any]:
    for event in _raw_response_events(raw):
        if event.get("type") != "finish":
            continue
        raw_usage = event.get("usage")
        if raw_usage is None:
            continue
        if not isinstance(raw_usage, dict):
            raise ChatGPTOAuthProtocolError("provider finish event usage must be an object")
        for key, required_fields in (
            ("cache_creation", {"ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"}),
            ("server_tool_use", {"web_search_requests", "web_fetch_requests"}),
        ):
            value = raw_usage.get(key)
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ChatGPTOAuthProtocolError(f"provider usage {key} must be an object or null")
            _validate_usage_counter_object(value, key, required_fields)
            usage[key] = value
        service_tier = raw_usage.get("service_tier")
        if service_tier is not None:
            if service_tier not in {"standard", "priority", "batch"}:
                raise ChatGPTOAuthProtocolError(
                    "provider usage service_tier must be standard, priority, batch, or null"
                )
            usage["service_tier"] = service_tier
        return usage
    return usage


def _map_stop_reason(finish_reason: str | None, has_tool_calls: bool) -> str:
    if finish_reason not in {None, "stop", "tool_calls"}:
        raise ChatGPTOAuthProtocolError(f"unsupported provider finish_reason {finish_reason!r}")
    if finish_reason is None:
        raise ChatGPTOAuthProtocolError("provider response requires a non-null finish_reason")
    if finish_reason == "tool_calls":
        if not has_tool_calls:
            raise ChatGPTOAuthProtocolError("provider finish_reason tool_calls requires at least one tool call")
        return "tool_use"
    if has_tool_calls:
        raise ChatGPTOAuthProtocolError("provider finish_reason stop conflicts with emitted tool calls")
    return "end_turn"


# ---------------------------------------------------------------------------
# Streaming adapter: provider events → Anthropic SSE
# ---------------------------------------------------------------------------


def anthropic_stream_adapter(
    event_stream: Iterator[dict[str, Any]],
    model: str,
    request_id: str,
) -> Iterator[str]:
    """Convert provider chat_stream events into Anthropic SSE strings."""
    yield _message_start_sse(model, request_id)
    yield from _render_anthropic_stream_events(event_stream)


def _message_start_sse(model: str, request_id: str) -> str:
    return _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": request_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "container": None,
                "content": [],
                "context_management": None,
                "stop_reason": None,
                "stop_sequence": None,
            },
        },
    )


def _usage_int(value: Any, field: str, *, required: bool = False) -> int | None:
    if value is None and not required:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        requirement = "a non-negative integer"
        raise ChatGPTOAuthProtocolError(f"provider usage {field} must be {requirement}")
    return value


def _anthropic_usage_from_provider(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        raise ChatGPTOAuthProtocolError("provider finish usage must be an object")
    unsupported_aliases = sorted(
        set(usage)
        & {
            "prompt_tokens",
            "completion_tokens",
            "prompt_tokens_details",
            "cached_input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        }
    )
    if unsupported_aliases:
        raise ChatGPTOAuthProtocolError(
            "provider finish usage contains unsupported public aliases: " + ", ".join(unsupported_aliases)
        )
    token_details = usage.get("input_tokens_details")
    if token_details is not None and not isinstance(token_details, dict):
        raise ChatGPTOAuthProtocolError("provider usage input token details must be an object or null")
    cache_read: int | None = None
    cache_write: int | None = None
    if isinstance(token_details, dict):
        has_detail_cache_write = "cache_write_tokens" in token_details
        detail_cache_read = _usage_int(
            token_details.get("cached_tokens"),
            "cached_tokens",
            required=True,
        )
        detail_cache_write = _usage_int(
            token_details.get("cache_write_tokens"),
            "cache_write_tokens",
            required=has_detail_cache_write,
        )
        cache_read = detail_cache_read
        if has_detail_cache_write:
            cache_write = detail_cache_write
    input_tokens = _usage_int(
        usage.get("input_tokens"),
        "input_tokens",
        required=True,
    )
    output_tokens = _usage_int(
        usage.get("output_tokens"),
        "output_tokens",
        required=True,
    )
    total_tokens = _usage_int(
        usage.get("total_tokens"),
        "total_tokens",
        required=True,
    )
    if input_tokens is None or output_tokens is None or total_tokens is None:
        raise ChatGPTOAuthProtocolError("provider usage must include input_tokens, output_tokens, and total_tokens")
    if total_tokens != input_tokens + output_tokens:
        raise ChatGPTOAuthProtocolError("provider usage total_tokens must equal input_tokens plus output_tokens")
    server_tool_use = usage.get("server_tool_use")
    if server_tool_use is not None:
        if not isinstance(server_tool_use, dict):
            raise ChatGPTOAuthProtocolError("provider usage server_tool_use must be an object or null")
        _validate_usage_counter_object(
            server_tool_use,
            "server_tool_use",
            {"web_search_requests", "web_fetch_requests"},
        )
    cache_creation = usage.get("cache_creation")
    if cache_creation is not None:
        if not isinstance(cache_creation, dict):
            raise ChatGPTOAuthProtocolError("provider usage cache_creation must be an object or null")
        _validate_usage_counter_object(
            cache_creation,
            "cache_creation",
            {"ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"},
        )
    service_tier = usage.get("service_tier")
    if service_tier is not None and (
        not isinstance(service_tier, str) or service_tier not in {"standard", "priority", "batch"}
    ):
        raise ChatGPTOAuthProtocolError(
            "provider usage service_tier must be standard, priority, batch, or null"
        )
    out: dict[str, Any] = {
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
        "input_tokens": input_tokens,
        "iterations": None,
        "output_tokens": output_tokens,
        "server_tool_use": server_tool_use,
    }
    return out


def _validate_usage_counter_object(
    value: dict[str, Any],
    field: str,
    required_fields: set[str],
) -> None:
    unknown = sorted(set(value) - required_fields)
    if unknown:
        raise ChatGPTOAuthProtocolError(f"provider usage {field} contains unsupported fields: " + ", ".join(unknown))
    missing = sorted(required_fields - set(value))
    if missing:
        raise ChatGPTOAuthProtocolError(f"provider usage {field} is missing required fields: " + ", ".join(missing))
    for name, counter in value.items():
        if not isinstance(counter, int) or isinstance(counter, bool) or counter < 0:
            raise ChatGPTOAuthProtocolError(f"provider usage {field}.{name} must be a non-negative integer")


def _render_anthropic_stream_events(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    block_index = 0
    current_block: str | None = None
    has_tool_calls = False

    for event in events:
        if not isinstance(event, dict):
            raise ChatGPTOAuthProtocolError("provider stream event must be an object")
        typ = event.get("type")

        if typ in ("reasoning_delta", "reasoning_raw_delta"):
            text = event.get("text")
            if not isinstance(text, str):
                raise ChatGPTOAuthProtocolError(f"provider {typ} event requires string text")
            yield _sse(
                "codex_reasoning_delta",
                {
                    "type": "codex_reasoning_delta",
                    "delta": text,
                },
            )

        elif typ == "content":
            text = event.get("text")
            if not isinstance(text, str):
                raise ChatGPTOAuthProtocolError("provider content event requires string text")
            if current_block != "text":
                if current_block is not None:
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                    block_index += 1
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "text", "text": "", "citations": None},
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
            has_tool_calls = True
            if current_block is not None:
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                block_index += 1
            tool_id = event.get("id")
            tool_name = event.get("name")
            tool_args = event.get("arguments")
            if not isinstance(tool_id, str):
                raise ChatGPTOAuthProtocolError("provider tool_call event requires a string id")
            if not isinstance(tool_name, str):
                raise ChatGPTOAuthProtocolError("provider tool_call event requires a string name")
            _parse_anthropic_tool_arguments(tool_args)
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": {},
                        "caller": {"type": "direct"},
                    },
                },
            )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "input_json_delta", "partial_json": tool_args},
                },
            )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
            block_index += 1
            current_block = None

        elif typ == "web_search_call":
            raise ChatGPTOAuthProtocolError(
                "provider web_search_call output cannot be represented losslessly by the Anthropic facade"
            )

        elif typ == "reasoning_section_break":
            continue

        elif typ == "finish":
            if current_block is not None:
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                current_block = None

            finish_reason = event.get("finish_reason")
            if finish_reason is not None and (not isinstance(finish_reason, str) or not finish_reason):
                raise ChatGPTOAuthProtocolError("provider finish event finish_reason must be non-empty or null")
            stop_reason = _map_stop_reason(finish_reason, has_tool_calls)
            usage = _anthropic_usage_from_provider(event.get("usage"))

            message_delta: dict[str, Any] = {
                "type": "message_delta",
                "context_management": None,
                "delta": {"container": None, "stop_reason": stop_reason, "stop_sequence": None},
                "usage": usage,
            }
            yield _sse("message_delta", message_delta)
            yield _sse("message_stop", {"type": "message_stop"})
            return

        else:
            raise ChatGPTOAuthProtocolError(f"unsupported normalized response event type {typ!r}")

    raise ChatGPTOAuthProtocolError("provider stream ended without a finish event")


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
