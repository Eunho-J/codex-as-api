from __future__ import annotations

import pytest

from codex_as_api.auth import ChatGPTOAuthProtocolError
from codex_as_api.protocol import (
    get_value,
    normalize_openai_chat_completion_chunk,
    normalize_stream_content,
    reasoning_from_response_items,
    response_failure_message,
)

# ---------------------------------------------------------------------------
# get_value
# ---------------------------------------------------------------------------


def test_get_value_dict():
    assert get_value({"key": "val"}, "key") == "val"


def test_get_value_dict_missing_default():
    assert get_value({"key": "val"}, "other", "default") == "default"


def test_get_value_object():
    class Obj:
        key = "attr_val"

    assert get_value(Obj(), "key") == "attr_val"


def test_get_value_object_missing_default():
    class Obj:
        pass

    assert get_value(Obj(), "missing", 99) == 99


# ---------------------------------------------------------------------------
# normalize_stream_content
# ---------------------------------------------------------------------------


def test_normalize_stream_content_none():
    assert normalize_stream_content(None) == ""


def test_normalize_stream_content_str():
    assert normalize_stream_content("hello") == "hello"


def test_normalize_stream_content_list_of_dicts():
    items = [{"text": "foo"}, {"text": "bar"}]
    assert normalize_stream_content(items) == "foobar"


def test_normalize_stream_content_rejects_missing_text():
    with pytest.raises(ChatGPTOAuthProtocolError, match="requires text"):
        normalize_stream_content([{"other": "x"}, {"text": "ok"}])


def test_normalize_stream_content_rejects_other_type():
    with pytest.raises(ChatGPTOAuthProtocolError, match="must be a string"):
        normalize_stream_content(42)


def test_normalize_stream_content_rejects_non_string_item_text():
    with pytest.raises(ChatGPTOAuthProtocolError, match="text must be a string"):
        normalize_stream_content([{"text": 42}])


# ---------------------------------------------------------------------------
# normalize_openai_chat_completion_chunk
# ---------------------------------------------------------------------------


def _make_chunk(delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "choices": [
            {
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ]
    }


def test_normalize_chunk_content_delta():
    chunk = _make_chunk({"content": "Hello"})
    events = normalize_openai_chat_completion_chunk(chunk)
    assert any(e["type"] == "content" and e["text"] == "Hello" for e in events)


def test_normalize_chunk_no_choices_empty():
    events = normalize_openai_chat_completion_chunk({"choices": []})
    assert events == []


def test_normalize_chunk_rejects_missing_choices():
    with pytest.raises(ChatGPTOAuthProtocolError, match="requires choices"):
        normalize_openai_chat_completion_chunk({})


def test_normalize_chunk_rejects_malformed_choice():
    with pytest.raises(ChatGPTOAuthProtocolError, match="choice must be an object"):
        normalize_openai_chat_completion_chunk({"choices": [42]})


def test_normalize_chunk_rejects_malformed_delta():
    with pytest.raises(ChatGPTOAuthProtocolError, match="delta must be an object"):
        normalize_openai_chat_completion_chunk({"choices": [{"delta": 42}]})


def test_normalize_chunk_rejects_non_array_tool_calls():
    chunk = _make_chunk({"tool_calls": {"id": "tc1"}})
    with pytest.raises(ChatGPTOAuthProtocolError, match="tool_calls must be an array"):
        normalize_openai_chat_completion_chunk(chunk)


def test_normalize_chunk_rejects_non_string_finish_reason():
    chunk = _make_chunk({}, finish_reason=42)  # type: ignore[arg-type]
    with pytest.raises(ChatGPTOAuthProtocolError, match="finish_reason must be a string"):
        normalize_openai_chat_completion_chunk(chunk)


def test_normalize_chunk_reasoning_content():
    chunk = _make_chunk({"reasoning_content": "thinking..."})
    events = normalize_openai_chat_completion_chunk(chunk)
    assert any(e["type"] == "reasoning_delta" and e["text"] == "thinking..." for e in events)


def test_normalize_chunk_reasoning_key():
    chunk = _make_chunk({"reasoning": "thought"})
    events = normalize_openai_chat_completion_chunk(chunk)
    assert any(e["type"] == "reasoning_delta" and e["source_key"] == "reasoning" for e in events)


def test_normalize_chunk_reasoning_text_raw():
    chunk = _make_chunk({"reasoning_text": "raw"})
    events = normalize_openai_chat_completion_chunk(chunk)
    assert any(e["type"] == "reasoning_raw_delta" and e["text"] == "raw" for e in events)


def test_normalize_chunk_tool_calls():
    tc = [{"id": "tc1", "function": {"name": "fn"}}]
    chunk = _make_chunk({"tool_calls": tc})
    events = normalize_openai_chat_completion_chunk(chunk)
    assert any(e["type"] == "tool_call_delta" for e in events)


def test_normalize_chunk_finish_reason():
    chunk = _make_chunk({}, finish_reason="stop")
    events = normalize_openai_chat_completion_chunk(chunk)
    assert any(e["type"] == "finish" and e["finish_reason"] == "stop" for e in events)


def test_normalize_chunk_no_content_no_event():
    chunk = _make_chunk({"content": ""})
    events = normalize_openai_chat_completion_chunk(chunk)
    assert not any(e["type"] == "content" for e in events)


# ---------------------------------------------------------------------------
# response_failure_message
# ---------------------------------------------------------------------------


def test_response_failure_error_dict():
    event = {"error": {"message": "Rate limit exceeded"}}
    msg = response_failure_message(event, "failed")
    assert "Rate limit exceeded" in msg
    assert "failed" in msg


def test_response_failure_error_string():
    event = {"error": "something went wrong"}
    msg = response_failure_message(event, "incomplete")
    assert "something went wrong" in msg


def test_response_failure_incomplete_details_dict():
    event = {"incomplete_details": {"reason": "timeout"}}
    msg = response_failure_message(event, "incomplete")
    assert "timeout" in msg


def test_response_failure_nested_response():
    event = {"response": {"error": {"message": "nested error"}}}
    msg = response_failure_message(event, "failed")
    assert "nested error" in msg


def test_response_failure_fallback_json_dump():
    event = {"other_field": "data"}
    msg = response_failure_message(event, "failed")
    assert "failed" in msg
    assert "other_field" in msg


# ---------------------------------------------------------------------------
# reasoning_from_response_items
# ---------------------------------------------------------------------------


def test_reasoning_from_items_reads_official_summary_shape():
    items = [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "dict text"}],
        }
    ]
    assert reasoning_from_response_items(items) == "dict text"


@pytest.mark.parametrize("part_type", ["reasoning_text", "text"])
def test_reasoning_from_items_reads_official_content_shape(part_type):
    items = [
        {
            "type": "reasoning",
            "summary": [],
            "content": [{"type": part_type, "text": "detail"}],
        }
    ]
    assert reasoning_from_response_items(items) == "detail"


@pytest.mark.parametrize(
    "reasoning",
    [
        {"summary": "direct"},
        {"summary": ["bare"]},
        {"summary": [{"text": "untyped"}]},
        {"summary": [], "content": "direct"},
        {"summary": [], "content": ["bare"]},
        {"summary": [], "content": [{"text": "untyped"}]},
    ],
)
def test_reasoning_from_items_rejects_stale_wire_fallbacks(reasoning):
    with pytest.raises(ChatGPTOAuthProtocolError):
        reasoning_from_response_items([{"type": "reasoning", **reasoning}])


def test_reasoning_from_items_skips_non_reasoning():
    items = [
        {"type": "message", "content": "ignored"},
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "kept"}],
        },
    ]
    assert reasoning_from_response_items(items) == "kept"


def test_reasoning_from_items_empty_list():
    assert reasoning_from_response_items([]) == ""


def test_reasoning_from_items_multiple_reasoning_items():
    items = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "A"}]},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "B"}]},
    ]
    assert reasoning_from_response_items(items) == "AB"


def test_reasoning_from_items_rejects_non_object_item():
    with pytest.raises(ChatGPTOAuthProtocolError, match="response item 0"):
        reasoning_from_response_items([42])  # type: ignore[list-item]
