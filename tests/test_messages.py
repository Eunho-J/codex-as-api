from __future__ import annotations

import pytest

from codex_as_api.messages import (
    AssistantResponse,
    Message,
    MessageRole,
    TerminatorCall,
    ToolCall,
    ToolResult,
    ToolSchema,
    Usage,
)


# ---------------------------------------------------------------------------
# Message creation
# ---------------------------------------------------------------------------


def test_message_system_role():
    m = Message(role=MessageRole.SYSTEM, content="You are helpful.")
    assert m.role is MessageRole.SYSTEM
    assert m.content == "You are helpful."


def test_message_user_role():
    m = Message(role=MessageRole.USER, content="Hello")
    assert m.role is MessageRole.USER


def test_message_assistant_role():
    m = Message(role=MessageRole.ASSISTANT, content="Hi there")
    assert m.role is MessageRole.ASSISTANT


def test_message_tool_role_valid():
    m = Message(role=MessageRole.TOOL, content="result", tool_call_id="tc-1", name="my_tool")
    assert m.tool_call_id == "tc-1"
    assert m.name == "my_tool"


# ---------------------------------------------------------------------------
# Message validation
# ---------------------------------------------------------------------------


def test_message_tool_missing_tool_call_id_raises():
    with pytest.raises(ValueError, match="tool messages require"):
        Message(role=MessageRole.TOOL, content="x", name="fn")


def test_message_tool_missing_name_raises():
    with pytest.raises(ValueError, match="tool messages require"):
        Message(role=MessageRole.TOOL, content="x", tool_call_id="tc-1")


def test_message_non_tool_with_tool_call_id_raises():
    with pytest.raises(ValueError, match="only allowed on tool messages"):
        Message(role=MessageRole.USER, content="x", tool_call_id="tc-1")


def test_message_non_tool_with_name_raises():
    with pytest.raises(ValueError, match="only allowed on tool messages"):
        Message(role=MessageRole.SYSTEM, content="x", name="fn")


def test_message_tool_calls_only_on_assistant():
    tc = ToolCall(id="id1", name="fn", arguments={})
    with pytest.raises(ValueError, match="only allowed on assistant messages"):
        Message(role=MessageRole.USER, content="x", tool_calls=(tc,))


def test_message_assistant_with_tool_calls():
    tc = ToolCall(id="id1", name="fn", arguments={"key": "val"})
    m = Message(role=MessageRole.ASSISTANT, content="", tool_calls=(tc,))
    assert len(m.tool_calls) == 1
    assert m.tool_calls[0].name == "fn"


def test_message_tool_calls_list_coerced_to_tuple():
    tc = ToolCall(id="id1", name="fn", arguments={})
    m = Message(role=MessageRole.ASSISTANT, content="", tool_calls=[tc])
    assert isinstance(m.tool_calls, tuple)


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


def test_usage_auto_calculates_total():
    u = Usage(prompt_tokens=10, completion_tokens=5)
    assert u.total_tokens == 15


def test_usage_explicit_total_preserved():
    u = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=20)
    assert u.total_tokens == 20


def test_usage_cache_hit_rate_zero_when_no_cache():
    u = Usage(prompt_tokens=100, completion_tokens=50)
    assert u.cache_hit_rate == 0.0


def test_usage_cache_hit_rate_correct():
    u = Usage(prompt_tokens=100, completion_tokens=50, cached_tokens=40)
    assert u.cache_hit_rate == pytest.approx(0.4)


def test_usage_cache_hit_rate_zero_when_no_prompt_tokens():
    u = Usage(prompt_tokens=0, completion_tokens=0)
    assert u.cache_hit_rate == 0.0


# ---------------------------------------------------------------------------
# AssistantResponse
# ---------------------------------------------------------------------------


def test_assistant_response_defaults():
    r = AssistantResponse(content="hello")
    assert r.finish_reason == "stop"
    assert r.tool_calls == ()
    assert r.usage is None
    assert r.reasoning_content is None


def test_assistant_response_list_tool_calls_coerced_to_tuple():
    tc = ToolCall(id="x", name="fn", arguments={})
    r = AssistantResponse(content="", tool_calls=[tc])
    assert isinstance(r.tool_calls, tuple)
    assert r.tool_calls[0].name == "fn"


def test_assistant_response_with_usage():
    u = Usage(prompt_tokens=5, completion_tokens=3)
    r = AssistantResponse(content="text", usage=u)
    assert r.usage.total_tokens == 8


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


def test_tool_call_construction():
    tc = ToolCall(id="abc", name="search", arguments={"q": "hello"})
    assert tc.id == "abc"
    assert tc.name == "search"
    assert tc.arguments == {"q": "hello"}


# ---------------------------------------------------------------------------
# ToolSchema
# ---------------------------------------------------------------------------


def test_tool_schema_construction():
    ts = ToolSchema(name="my_fn", description="does stuff", parameters={"type": "object"})
    assert ts.name == "my_fn"
    assert ts.description == "does stuff"
    assert ts.parameters == {"type": "object"}


# ---------------------------------------------------------------------------
# TerminatorCall
# ---------------------------------------------------------------------------


def test_terminator_call_valid():
    t = TerminatorCall(name="finish", arguments={"reason": "done"})
    assert t.name == "finish"
    assert t.arguments == {"reason": "done"}


def test_terminator_call_empty_name_raises():
    with pytest.raises(ValueError, match="non-empty"):
        TerminatorCall(name="", arguments={})


def test_terminator_call_non_dict_arguments_raises():
    with pytest.raises(TypeError, match="must be a dict"):
        TerminatorCall(name="fn", arguments="not a dict")


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


def test_tool_result_ok():
    r = ToolResult(ok=True, content="output")
    assert r.ok is True
    assert r.content == "output"
    assert r.payload is None


def test_tool_result_with_payload():
    r = ToolResult(ok=False, content="error", payload={"code": 404})
    assert r.payload == {"code": 404}
