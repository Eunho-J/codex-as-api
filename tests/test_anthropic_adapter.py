"""Tests for the Anthropic Messages API adapter."""
from __future__ import annotations

import json

from codex_as_api.anthropic_adapter import (
    anthropic_request_to_internal,
    anthropic_stream_adapter,
    format_anthropic_error,
    internal_response_to_anthropic,
)
from codex_as_api.messages import (
    AssistantResponse,
    Message,
    MessageRole,
    ToolCall,
    Usage,
)


# ---------------------------------------------------------------------------
# Request conversion tests
# ---------------------------------------------------------------------------

class TestAnthropicRequestToInternal:
    def test_system_string(self):
        messages, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            system="You are helpful.",
        )
        assert messages[0].role is MessageRole.SYSTEM
        assert messages[0].content == "You are helpful."
        assert messages[1].role is MessageRole.USER

    def test_system_content_blocks(self):
        messages, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            system=[
                {"type": "text", "text": "Rule 1"},
                {"type": "text", "text": "Rule 2"},
            ],
        )
        assert messages[0].content == "Rule 1\n\nRule 2"

    def test_no_system(self):
        messages, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert messages[0].role is MessageRole.USER

    def test_user_text_message(self):
        messages, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert len(messages) == 1
        assert messages[0].role is MessageRole.USER
        assert messages[0].content == "Hello"

    def test_user_content_blocks(self):
        messages, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the result:"},
                    {"type": "tool_result", "tool_use_id": "call-1", "content": "42"},
                ],
            }],
        )
        assert len(messages) == 2
        assert messages[0].role is MessageRole.USER
        assert messages[0].content == "Here is the result:"
        assert messages[1].role is MessageRole.TOOL
        assert messages[1].content == "42"
        assert messages[1].tool_call_id == "call-1"

    def test_user_tool_result_only(self):
        messages, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call-1", "content": "result1"},
                    {"type": "tool_result", "tool_use_id": "call-2", "content": "result2"},
                ],
            }],
        )
        assert len(messages) == 2
        assert all(m.role is MessageRole.TOOL for m in messages)

    def test_assistant_text(self):
        messages, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "assistant", "content": "Hello!"}],
        )
        assert messages[0].role is MessageRole.ASSISTANT
        assert messages[0].content == "Hello!"

    def test_assistant_tool_use_blocks(self):
        messages, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "tc-1", "name": "get_weather", "input": {"city": "Seoul"}},
                ],
            }],
        )
        assert messages[0].content == "Let me check."
        assert len(messages[0].tool_calls) == 1
        assert messages[0].tool_calls[0].name == "get_weather"
        assert messages[0].tool_calls[0].arguments == {"city": "Seoul"}

    def test_assistant_thinking_block(self):
        messages, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think...", "signature": "sig-abc"},
                    {"type": "text", "text": "The answer is 42."},
                ],
            }],
        )
        assert messages[0].reasoning_content == "Let me think..."
        assert messages[0].content == "The answer is 42."

    def test_tools_conversion(self):
        _, tools, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
        )
        assert tools is not None
        assert len(tools) == 1
        assert tools[0].name == "get_weather"
        assert tools[0].parameters == {"type": "object", "properties": {"city": {"type": "string"}}}

    def test_tool_choice_auto(self):
        _, _, tc, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice={"type": "auto"},
        )
        assert tc == "auto"

    def test_tool_choice_any(self):
        _, _, tc, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice={"type": "any"},
        )
        assert tc == "required"

    def test_tool_choice_specific(self):
        _, _, tc, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice={"type": "tool", "name": "get_weather"},
        )
        assert tc == {"type": "function", "function": {"name": "get_weather"}}

    def test_tool_choice_none(self):
        _, _, tc, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice={"type": "none"},
        )
        assert tc == "none"

    def test_thinking_enabled(self):
        _, _, _, _, effort = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "enabled", "budget_tokens": 4096},
        )
        assert effort == "high"

    def test_thinking_adaptive(self):
        _, _, _, _, effort = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive"},
        )
        assert effort == "medium"

    def test_thinking_disabled(self):
        _, _, _, _, effort = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "disabled"},
        )
        assert effort is None

    def test_stop_sequences(self):
        _, _, _, stop, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stop_sequences=["STOP", "END"],
        )
        assert stop == ["STOP", "END"]

    def test_user_image_block(self):
        messages, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANS",
                    }},
                ],
            }],
        )
        assert len(messages) == 1
        assert messages[0].role is MessageRole.USER
        assert messages[0].content == "What is in this image?"
        assert len(messages[0].images) == 1
        assert messages[0].images[0] == "data:image/png;base64,iVBORw0KGgoAAAANS"

    def test_tool_result_with_content_blocks(self):
        messages, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call-1", "content": [
                        {"type": "text", "text": "result line 1"},
                        {"type": "text", "text": "result line 2"},
                    ]},
                ],
            }],
        )
        assert messages[0].role is MessageRole.TOOL
        assert messages[0].content == "result line 1result line 2"

    def test_tool_result_with_image(self):
        messages, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call-img", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo"}},
                    ]},
                ],
            }],
        )
        assert len(messages) == 2
        assert messages[0].role is MessageRole.TOOL
        assert messages[0].tool_call_id == "call-img"
        assert messages[0].content == ""
        assert messages[1].role is MessageRole.USER
        assert len(messages[1].images) == 1
        assert messages[1].images[0] == "data:image/png;base64,iVBORw0KGgo"

    def test_tool_result_with_text_and_image(self):
        messages, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call-mix", "content": [
                        {"type": "text", "text": "file contents"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQ"}},
                    ]},
                ],
            }],
        )
        assert len(messages) == 2
        assert messages[0].role is MessageRole.TOOL
        assert messages[0].content == "file contents"
        assert messages[1].role is MessageRole.USER
        assert messages[1].images[0] == "data:image/jpeg;base64,/9j/4AAQ"


# ---------------------------------------------------------------------------
# Non-streaming response conversion
# ---------------------------------------------------------------------------

class TestInternalResponseToAnthropic:
    def test_text_response(self):
        resp = AssistantResponse(
            content="Hello!",
            tool_calls=(),
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            reasoning_content=None,
            raw=None,
        )
        result = internal_response_to_anthropic(resp, "test-model", "msg_123")
        assert result["id"] == "msg_123"
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_use_response(self):
        resp = AssistantResponse(
            content="",
            tool_calls=(ToolCall(id="tc-1", name="get_weather", arguments={"city": "Seoul"}),),
            finish_reason="stop",
            usage=Usage(prompt_tokens=20, completion_tokens=10),
            reasoning_content=None,
            raw=None,
        )
        result = internal_response_to_anthropic(resp, "test-model", "msg_123")
        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["name"] == "get_weather"

    def test_reasoning_response(self):
        resp = AssistantResponse(
            content="42",
            tool_calls=(),
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            reasoning_content="Let me think about this...",
            raw=None,
        )
        result = internal_response_to_anthropic(resp, "test-model", "msg_123")
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "thinking"
        assert result["content"][0]["thinking"] == "Let me think about this..."
        assert result["content"][1]["type"] == "text"

    def test_empty_response(self):
        resp = AssistantResponse(
            content="",
            tool_calls=(),
            finish_reason="stop",
            usage=None,
            reasoning_content=None,
            raw=None,
        )
        result = internal_response_to_anthropic(resp, "test-model", "msg_123")
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == ""

    def test_cached_tokens_in_usage(self):
        resp = AssistantResponse(
            content="hi",
            tool_calls=(),
            finish_reason="stop",
            usage=Usage(prompt_tokens=100, completion_tokens=10, cached_tokens=50),
            reasoning_content=None,
            raw=None,
        )
        result = internal_response_to_anthropic(resp, "m", "msg_1")
        assert result["usage"]["cache_read_input_tokens"] == 50


# ---------------------------------------------------------------------------
# Streaming adapter
# ---------------------------------------------------------------------------

class TestAnthropicStreamAdapter:
    def _collect_events(self, events: list[dict]) -> list[dict]:
        """Run stream adapter and parse the SSE output back into dicts."""
        result = []
        for sse_str in anthropic_stream_adapter(iter(events), "test-model", "msg_test"):
            for line in sse_str.strip().split("\n"):
                if line.startswith("data: "):
                    result.append(json.loads(line[6:]))
        return result

    def test_text_only_stream(self):
        events = [
            {"type": "content", "text": "Hello"},
            {"type": "content", "text": " world"},
            {"type": "finish", "finish_reason": "stop", "usage": {"output_tokens": 5}},
        ]
        result = self._collect_events(events)
        types = [e["type"] for e in result]
        assert types[0] == "message_start"
        assert "content_block_start" in types
        assert "content_block_delta" in types
        assert "content_block_stop" in types
        assert "message_delta" in types
        assert types[-1] == "message_stop"

        # Check text deltas
        text_deltas = [e for e in result if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta"]
        assert len(text_deltas) == 2
        assert text_deltas[0]["delta"]["text"] == "Hello"
        assert text_deltas[1]["delta"]["text"] == " world"

    def test_thinking_then_text(self):
        events = [
            {"type": "reasoning_delta", "text": "thinking..."},
            {"type": "content", "text": "result"},
            {"type": "finish", "finish_reason": "stop"},
        ]
        result = self._collect_events(events)
        types = [e["type"] for e in result]

        # Should have: message_start, thinking block start, thinking delta, signature delta, thinking block stop, text block start, text delta, text block stop, message_delta, message_stop
        block_starts = [e for e in result if e["type"] == "content_block_start"]
        assert len(block_starts) == 2
        assert block_starts[0]["content_block"]["type"] == "thinking"
        assert block_starts[1]["content_block"]["type"] == "text"

    def test_tool_call_stream(self):
        events = [
            {"type": "tool_call", "id": "tc-1", "name": "get_weather", "arguments": {"city": "Seoul"}},
            {"type": "finish", "finish_reason": "stop"},
        ]
        result = self._collect_events(events)

        block_starts = [e for e in result if e["type"] == "content_block_start"]
        assert len(block_starts) == 1
        assert block_starts[0]["content_block"]["type"] == "tool_use"
        assert block_starts[0]["content_block"]["name"] == "get_weather"

        json_deltas = [e for e in result if e.get("delta", {}).get("type") == "input_json_delta"]
        assert len(json_deltas) == 1
        assert json.loads(json_deltas[0]["delta"]["partial_json"]) == {"city": "Seoul"}

    def test_text_then_tool_call(self):
        events = [
            {"type": "content", "text": "Let me check."},
            {"type": "tool_call", "id": "tc-1", "name": "search", "arguments": {"q": "test"}},
            {"type": "finish", "finish_reason": "stop"},
        ]
        result = self._collect_events(events)
        block_starts = [e for e in result if e["type"] == "content_block_start"]
        assert len(block_starts) == 2
        assert block_starts[0]["content_block"]["type"] == "text"
        assert block_starts[1]["content_block"]["type"] == "tool_use"

    def test_empty_stream(self):
        events = [
            {"type": "finish", "finish_reason": "stop"},
        ]
        result = self._collect_events(events)
        # Should have at least an empty text block
        block_starts = [e for e in result if e["type"] == "content_block_start"]
        assert len(block_starts) == 1
        assert block_starts[0]["content_block"]["type"] == "text"

    def test_message_delta_stop_reason(self):
        events = [
            {"type": "content", "text": "hi"},
            {"type": "finish", "finish_reason": "stop", "usage": {"output_tokens": 3}},
        ]
        result = self._collect_events(events)
        msg_delta = [e for e in result if e["type"] == "message_delta"][0]
        assert msg_delta["delta"]["stop_reason"] == "end_turn"
        assert msg_delta["usage"]["output_tokens"] == 3

    def test_multiple_tool_calls(self):
        events = [
            {"type": "tool_call", "id": "tc-1", "name": "tool_a", "arguments": {"a": 1}},
            {"type": "tool_call", "id": "tc-2", "name": "tool_b", "arguments": {"b": 2}},
            {"type": "finish", "finish_reason": "stop"},
        ]
        result = self._collect_events(events)
        block_starts = [e for e in result if e["type"] == "content_block_start"]
        assert len(block_starts) == 2
        assert block_starts[0]["content_block"]["name"] == "tool_a"
        assert block_starts[1]["content_block"]["name"] == "tool_b"
        # Verify indices are sequential
        assert block_starts[0]["index"] == 0
        assert block_starts[1]["index"] == 1


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------

class TestFormatAnthropicError:
    def test_auth_error(self):
        result = format_anthropic_error(401, "bad key")
        assert result["type"] == "error"
        assert result["error"]["type"] == "authentication_error"
        assert result["error"]["message"] == "bad key"

    def test_server_error(self):
        result = format_anthropic_error(500, "internal")
        assert result["error"]["type"] == "api_error"

    def test_rate_limit(self):
        result = format_anthropic_error(429, "slow down")
        assert result["error"]["type"] == "rate_limit_error"
