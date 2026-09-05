from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .auth import ChatGPTOAuthProtocolError

_MISSING = object()


def get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def normalize_stream_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for index, item in enumerate(content):
            text = get_value(item, "text", _MISSING)
            if text is _MISSING:
                raise ChatGPTOAuthProtocolError(f"stream content item {index} requires text")
            if not isinstance(text, str):
                raise ChatGPTOAuthProtocolError(f"stream content item {index} text must be a string")
            parts.append(text)
        return "".join(parts)
    raise ChatGPTOAuthProtocolError("stream content must be a string, array, or null")


def normalize_openai_chat_completion_chunk(chunk: Any) -> list[dict[str, Any]]:
    """Convert OpenAI-compatible chat-completion stream chunks into internal stream events."""
    events: list[dict[str, Any]] = []
    choices = get_value(chunk, "choices", _MISSING)
    if choices is _MISSING:
        raise ChatGPTOAuthProtocolError("chat completion stream chunk requires choices")
    if not isinstance(choices, (list, tuple)):
        raise ChatGPTOAuthProtocolError("chat completion stream chunk choices must be an array")
    if not choices:
        return events
    choice = choices[0]
    delta = get_value(choice, "delta", _MISSING)
    finish_reason = get_value(choice, "finish_reason", _MISSING)
    if delta is _MISSING and finish_reason is _MISSING:
        raise ChatGPTOAuthProtocolError("chat completion stream choice must be an object")
    if delta is not _MISSING and delta is not None:
        if not isinstance(delta, dict) and not any(
            hasattr(delta, key)
            for key in (
                "content",
                "reasoning_content",
                "reasoning",
                "reasoning_summary",
                "reasoning_text",
                "tool_calls",
            )
        ):
            raise ChatGPTOAuthProtocolError("chat completion stream choice delta must be an object or null")
        content = normalize_stream_content(get_value(delta, "content"))
        if content:
            events.append({"type": "content", "text": content})
        for key in ("reasoning_content", "reasoning", "reasoning_summary"):
            reasoning = normalize_stream_content(get_value(delta, key))
            if reasoning:
                events.append({"type": "reasoning_delta", "text": reasoning, "source_key": key})
        raw_reasoning = normalize_stream_content(get_value(delta, "reasoning_text"))
        if raw_reasoning:
            events.append({"type": "reasoning_raw_delta", "text": raw_reasoning, "source_key": "reasoning_text"})
        tool_calls = get_value(delta, "tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise ChatGPTOAuthProtocolError("chat completion stream tool_calls must be an array or null")
            if tool_calls:
                events.append({"type": "tool_call_delta", "delta": tool_calls})
    if finish_reason is not _MISSING and finish_reason is not None and not isinstance(finish_reason, str):
        raise ChatGPTOAuthProtocolError("chat completion stream finish_reason must be a string or null")
    if isinstance(finish_reason, str) and finish_reason:
        events.append({"type": "finish", "finish_reason": finish_reason})
    return events


def response_failure_message(event: dict[str, Any], status: str) -> str:
    response = event.get("response")
    error: Any = event.get("error")
    incomplete_details: Any = event.get("incomplete_details")
    if isinstance(response, dict):
        error = response.get("error", error)
        incomplete_details = response.get("incomplete_details", incomplete_details)
    detail_parts: list[str] = []
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or error.get("type")
        if isinstance(message, str) and message:
            detail_parts.append(message)
    elif isinstance(error, str) and error:
        detail_parts.append(error)
    if isinstance(incomplete_details, dict):
        reason = incomplete_details.get("reason") or incomplete_details.get("message")
        if isinstance(reason, str) and reason:
            detail_parts.append(reason)
    elif isinstance(incomplete_details, str) and incomplete_details:
        detail_parts.append(incomplete_details)
    detail = "; ".join(detail_parts) if detail_parts else json.dumps(event, ensure_ascii=False)[:500]
    return f"OpenAI protocol response {status}: {detail}"


def reasoning_parts_from_response_items(items: Iterable[dict[str, Any]]) -> tuple[str, str]:
    summary_parts: list[str] = []
    content_parts: list[str] = []
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ChatGPTOAuthProtocolError(f"response item {item_index} must be an object")
        if item.get("type") != "reasoning":
            continue
        summary = item.get("summary", _MISSING)
        if not isinstance(summary, list):
            raise ChatGPTOAuthProtocolError("reasoning item summary must be an array")
        content = item.get("content", _MISSING)
        fields: list[tuple[str, list[Any], frozenset[str], list[str]]] = [
            ("summary", summary, frozenset({"summary_text"}), summary_parts)
        ]
        if content is not _MISSING and content is not None:
            if not isinstance(content, list):
                raise ChatGPTOAuthProtocolError("reasoning item content must be an array or null")
            fields.append(("content", content, frozenset({"reasoning_text", "text"}), content_parts))
        for field, value, expected_types, target in fields:
            for index, part in enumerate(value):
                if not isinstance(part, dict) or part.get("type") not in expected_types:
                    expected = " or ".join(sorted(expected_types))
                    raise ChatGPTOAuthProtocolError(f"reasoning item {field}[{index}] must be a {expected} object")
                text = part.get("text")
                if not isinstance(text, str):
                    raise ChatGPTOAuthProtocolError(f"reasoning item {field}[{index}] requires string text")
                target.append(text)
    return "".join(summary_parts), "".join(content_parts)


def reasoning_from_response_items(items: Iterable[dict[str, Any]]) -> str:
    summary, content = reasoning_parts_from_response_items(items)
    return summary + content
