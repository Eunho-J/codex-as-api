from __future__ import annotations

import json
from typing import Any, Iterable


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
        for item in content:
            text = get_value(item, "text")
            if text:
                parts.append(str(text))
        return "".join(parts)
    return str(content)


def normalize_openai_chat_completion_chunk(chunk: Any) -> list[dict[str, Any]]:
    """Convert OpenAI-compatible chat-completion stream chunks into internal stream events."""
    events: list[dict[str, Any]] = []
    choices = get_value(chunk, "choices") or ()
    if not choices:
        return events
    choice = choices[0]
    delta = get_value(choice, "delta")
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
    if tool_calls:
        events.append({"type": "tool_call_delta", "delta": tool_calls})
    finish_reason = get_value(choice, "finish_reason")
    if finish_reason:
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
    detail = "; ".join(detail_parts) if detail_parts else json.dumps(event, ensure_ascii=False, default=str)[:500]
    return f"OpenAI protocol response {status}: {detail}"


def reasoning_from_response_items(items: Iterable[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in items:
        if item.get("type") != "reasoning":
            continue
        for field in ("summary", "content"):
            value = item.get(field)
            if isinstance(value, str) and value:
                parts.append(value)
                continue
            if not isinstance(value, list):
                continue
            for part in value:
                if isinstance(part, str) and part:
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
    return "".join(parts)
