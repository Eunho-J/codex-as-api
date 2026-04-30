from __future__ import annotations

import json

import pytest

from codex_as_api.auth import ChatGPTOAuthError
from codex_as_api.messages import Message, MessageRole, ToolCall, ToolSchema
from codex_as_api.provider import (
    REMOTE_COMPACTION_MARKER,
    _decode_sse_block,
    _image_generation_from_item,
    _message_item,
    _messages_to_response_items,
    _set_reasoning_payload,
    _split_instructions_and_input,
    _text_from_response_items,
    _tool_call_from_response_item,
    _tool_schema_to_response_dict,
    _usage_from_response,
    _validate_image_content_items,
)


# ---------------------------------------------------------------------------
# _set_reasoning_payload
# ---------------------------------------------------------------------------


def test_set_reasoning_payload_valid_effort():
    payload: dict = {}
    _set_reasoning_payload(payload, "high")
    assert payload["reasoning"] == {"effort": "high"}


def test_set_reasoning_payload_none_is_noop():
    payload: dict = {}
    _set_reasoning_payload(payload, None)
    assert "reasoning" not in payload


def test_set_reasoning_payload_lowercases():
    payload: dict = {}
    _set_reasoning_payload(payload, "HIGH")
    assert payload["reasoning"]["effort"] == "high"


def test_set_reasoning_payload_invalid_raises():
    with pytest.raises(ChatGPTOAuthError, match="reasoning_effort must be one of"):
        _set_reasoning_payload({}, "ultra")


def test_set_reasoning_payload_empty_string_raises():
    with pytest.raises(ChatGPTOAuthError):
        _set_reasoning_payload({}, "")


def test_set_reasoning_payload_all_valid_efforts():
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh"):
        payload: dict = {}
        _set_reasoning_payload(payload, effort)
        assert payload["reasoning"]["effort"] == effort


# ---------------------------------------------------------------------------
# _tool_call_from_response_item
# ---------------------------------------------------------------------------


def test_tool_call_from_function_call():
    item = {
        "type": "function_call",
        "name": "search",
        "call_id": "cid-1",
        "arguments": '{"q": "hello"}',
    }
    tc = _tool_call_from_response_item(item)
    assert tc is not None
    assert tc.name == "search"
    assert tc.arguments == {"q": "hello"}
    assert tc.id == "cid-1"


def test_tool_call_from_custom_tool_call():
    item = {
        "type": "custom_tool_call",
        "name": "my_tool",
        "call_id": "cid-2",
        "arguments": "{}",
    }
    tc = _tool_call_from_response_item(item)
    assert tc is not None
    assert tc.name == "my_tool"


def test_tool_call_non_tool_type_returns_none():
    item = {"type": "message", "content": "hi"}
    assert _tool_call_from_response_item(item) is None


def test_tool_call_missing_name_returns_none():
    item = {"type": "function_call", "call_id": "cid-3", "arguments": "{}"}
    assert _tool_call_from_response_item(item) is None


def test_tool_call_dict_arguments():
    item = {
        "type": "function_call",
        "name": "fn",
        "call_id": "cid-4",
        "arguments": {"key": "value"},
    }
    tc = _tool_call_from_response_item(item)
    assert tc.arguments == {"key": "value"}


def test_tool_call_invalid_json_arguments_stored_as_input():
    item = {
        "type": "function_call",
        "name": "fn",
        "call_id": "cid-5",
        "arguments": "not json {{{",
    }
    tc = _tool_call_from_response_item(item)
    assert "input" in tc.arguments


# ---------------------------------------------------------------------------
# _text_from_response_items
# ---------------------------------------------------------------------------


def test_text_from_output_text_item():
    items = [{"type": "output_text", "text": "hello"}]
    assert _text_from_response_items(items) == "hello"


def test_text_from_text_item():
    items = [{"type": "text", "text": "world"}]
    assert _text_from_response_items(items) == "world"


def test_text_from_message_with_content_list():
    items = [{
        "type": "message",
        "content": [{"type": "output_text", "text": "msg text"}],
    }]
    assert _text_from_response_items(items) == "msg text"


def test_text_from_message_with_string_content_parts():
    items = [{
        "type": "message",
        "content": ["part one", "part two"],
    }]
    assert _text_from_response_items(items) == "part onepart two"


def test_text_from_items_ignores_non_text_types():
    items = [{"type": "function_call", "name": "fn"}, {"type": "output_text", "text": "ok"}]
    assert _text_from_response_items(items) == "ok"


def test_text_from_empty_items():
    assert _text_from_response_items([]) == ""


# ---------------------------------------------------------------------------
# _usage_from_response
# ---------------------------------------------------------------------------


def test_usage_from_response_input_output_tokens():
    value = {"input_tokens": 10, "output_tokens": 5}
    u = _usage_from_response(value)
    assert u is not None
    assert u.prompt_tokens == 10
    assert u.completion_tokens == 5
    assert u.total_tokens == 15


def test_usage_from_response_prompt_completion_tokens():
    value = {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
    u = _usage_from_response(value)
    assert u.prompt_tokens == 20
    assert u.completion_tokens == 8
    assert u.total_tokens == 28


def test_usage_from_response_cached_tokens_from_details():
    value = {
        "input_tokens": 100,
        "output_tokens": 50,
        "input_tokens_details": {"cached_tokens": 30},
    }
    u = _usage_from_response(value)
    assert u.cached_tokens == 30


def test_usage_from_response_cached_input_tokens_fallback():
    value = {"input_tokens": 100, "output_tokens": 50, "cached_input_tokens": 25}
    u = _usage_from_response(value)
    assert u.cached_tokens == 25


def test_usage_from_response_cache_read_input_tokens_fallback():
    value = {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 15}
    u = _usage_from_response(value)
    assert u.cached_tokens == 15


def test_usage_from_response_non_dict_returns_none():
    assert _usage_from_response(None) is None
    assert _usage_from_response("text") is None
    assert _usage_from_response(42) is None


def test_usage_from_response_missing_tokens_returns_none():
    assert _usage_from_response({"total_tokens": 10}) is None


# ---------------------------------------------------------------------------
# _split_instructions_and_input
# ---------------------------------------------------------------------------


def test_split_instructions_system_becomes_instructions():
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are helpful."),
        Message(role=MessageRole.USER, content="Hello"),
    ]
    instructions, input_items = _split_instructions_and_input(messages)
    assert instructions == "You are helpful."
    assert any(item.get("role") == "user" for item in input_items)


def test_split_instructions_multiple_system_joined():
    messages = [
        Message(role=MessageRole.SYSTEM, content="Part one."),
        Message(role=MessageRole.SYSTEM, content="Part two."),
        Message(role=MessageRole.USER, content="Hi"),
    ]
    instructions, _ = _split_instructions_and_input(messages)
    assert instructions == "Part one.\n\nPart two."


def test_split_instructions_compaction_marker_goes_to_input():
    compacted_content = REMOTE_COMPACTION_MARKER + "\n" + json.dumps([{"type": "message"}])
    messages = [
        Message(role=MessageRole.SYSTEM, content="System prompt."),
        Message(role=MessageRole.SYSTEM, content=compacted_content),
        Message(role=MessageRole.USER, content="Hi"),
    ]
    instructions, input_items = _split_instructions_and_input(messages)
    assert instructions == "System prompt."
    assert len(input_items) >= 2


# ---------------------------------------------------------------------------
# _messages_to_response_items
# ---------------------------------------------------------------------------


def test_messages_to_response_items_user():
    messages = [Message(role=MessageRole.USER, content="Hello")]
    items = _messages_to_response_items(messages)
    assert items[0]["role"] == "user"
    assert items[0]["type"] == "message"


def test_messages_to_response_items_assistant():
    messages = [Message(role=MessageRole.ASSISTANT, content="Hi")]
    items = _messages_to_response_items(messages)
    assert items[0]["role"] == "assistant"


def test_messages_to_response_items_tool():
    messages = [Message(role=MessageRole.TOOL, content="result", tool_call_id="tc-1", name="fn")]
    items = _messages_to_response_items(messages)
    assert items[0]["type"] == "function_call_output"
    assert items[0]["output"] == "result"
    assert items[0]["call_id"] == "tc-1"


def test_messages_to_response_items_assistant_with_tool_calls():
    tc = ToolCall(id="tid", name="search", arguments={"q": "test"})
    messages = [Message(role=MessageRole.ASSISTANT, content="", tool_calls=(tc,))]
    items = _messages_to_response_items(messages)
    assert any(i["type"] == "function_call" and i["name"] == "search" for i in items)


def test_messages_to_response_items_compacted():
    inner = [{"type": "message", "role": "user", "content": []}]
    compacted_content = REMOTE_COMPACTION_MARKER + "\n" + json.dumps(inner)
    messages = [Message(role=MessageRole.SYSTEM, content=compacted_content)]
    items = _messages_to_response_items(messages)
    assert items == inner


# ---------------------------------------------------------------------------
# _message_item
# ---------------------------------------------------------------------------


def test_message_item_assistant_type():
    item = _message_item("assistant", "hello")
    assert item["role"] == "assistant"
    assert item["content"][0]["type"] == "output_text"
    assert item["content"][0]["text"] == "hello"


def test_message_item_user_type():
    item = _message_item("user", "question")
    assert item["role"] == "user"
    assert item["content"][0]["type"] == "input_text"


def test_message_item_empty_content():
    item = _message_item("user", "")
    assert item["content"][0]["text"] == ""


# ---------------------------------------------------------------------------
# _tool_schema_to_response_dict
# ---------------------------------------------------------------------------


def test_tool_schema_to_response_dict():
    tool = ToolSchema(name="search", description="Search the web", parameters={"type": "object"})
    d = _tool_schema_to_response_dict(tool)
    assert d["type"] == "function"
    assert d["name"] == "search"
    assert d["description"] == "Search the web"
    assert d["parameters"] == {"type": "object"}
    assert d["strict"] is False


# ---------------------------------------------------------------------------
# _validate_image_content_items
# ---------------------------------------------------------------------------


def test_validate_image_content_items_valid():
    images = [{"image_url": "data:image/png;base64,abc123"}]
    items = _validate_image_content_items(images)
    assert items[0]["type"] == "input_image"
    assert items[0]["image_url"] == "data:image/png;base64,abc123"


def test_validate_image_content_items_non_dict_raises():
    with pytest.raises(ChatGPTOAuthError, match="must be an object"):
        _validate_image_content_items(["not a dict"])


def test_validate_image_content_items_missing_url_raises():
    with pytest.raises(ChatGPTOAuthError, match="requires image_url"):
        _validate_image_content_items([{"other": "field"}])


def test_validate_image_content_items_non_data_url_raises():
    with pytest.raises(ChatGPTOAuthError, match="data:image"):
        _validate_image_content_items([{"image_url": "https://example.com/img.png"}])


def test_validate_image_content_items_empty_list():
    assert _validate_image_content_items([]) == []


# ---------------------------------------------------------------------------
# _image_generation_from_item
# ---------------------------------------------------------------------------


def test_image_generation_from_item_correct_type():
    item = {
        "type": "image_generation_call",
        "id": "img-1",
        "status": "completed",
        "result": "data:image/png;base64,ABC",
        "revised_prompt": "a cat",
    }
    result = _image_generation_from_item(item)
    assert result is not None
    assert result["id"] == "img-1"
    assert result["status"] == "completed"
    assert result["result"] == "data:image/png;base64,ABC"
    assert result["revised_prompt"] == "a cat"


def test_image_generation_from_item_wrong_type_returns_none():
    item = {"type": "message", "content": "hi"}
    assert _image_generation_from_item(item) is None


def test_image_generation_from_item_empty_result_raises():
    item = {"type": "image_generation_call", "id": "img-2", "result": ""}
    with pytest.raises(ChatGPTOAuthError, match="empty result"):
        _image_generation_from_item(item)


def test_image_generation_from_item_none_result_raises():
    item = {"type": "image_generation_call", "id": "img-3", "result": None}
    with pytest.raises(ChatGPTOAuthError, match="empty result"):
        _image_generation_from_item(item)


# ---------------------------------------------------------------------------
# _decode_sse_block
# ---------------------------------------------------------------------------


def test_decode_sse_block_valid_json():
    lines = ['data: {"type": "ping"}']
    event = _decode_sse_block(lines)
    assert event == {"type": "ping"}


def test_decode_sse_block_done_returns_none():
    lines = ["data: [DONE]"]
    assert _decode_sse_block(lines) is None


def test_decode_sse_block_no_data_lines_returns_none():
    lines = ["event: ping", "id: 1"]
    assert _decode_sse_block(lines) is None


def test_decode_sse_block_invalid_json_raises():
    lines = ["data: {invalid json"]
    with pytest.raises(ChatGPTOAuthError, match="invalid SSE event JSON"):
        _decode_sse_block(lines)


def test_decode_sse_block_strips_data_prefix():
    lines = ["data:   {\"k\": \"v\"}"]
    event = _decode_sse_block(lines)
    assert event == {"k": "v"}
