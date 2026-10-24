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
    with pytest.raises(TypeError) as caught:
        Message(role=MessageRole.TOOL, content="x", name="fn")
    assert "tool messages require" in str(caught.value)


def test_message_tool_name_is_optional():
    message = Message(role=MessageRole.TOOL, content="x", tool_call_id="tc-1")

    assert message.name is None


def test_message_non_tool_with_tool_call_id_raises():
    with pytest.raises(ValueError) as caught:
        Message(role=MessageRole.USER, content="x", tool_call_id="tc-1")
    assert "only allowed on tool messages" in str(caught.value)


def test_message_non_tool_with_name_raises():
    with pytest.raises(ValueError) as caught:
        Message(role=MessageRole.SYSTEM, content="x", name="fn")
    assert "only allowed on tool messages" in str(caught.value)


def test_message_tool_calls_only_on_assistant():
    tc = ToolCall(id="id1", name="fn", arguments="{}")
    with pytest.raises(ValueError) as caught:
        Message(role=MessageRole.USER, content="x", tool_calls=(tc,))
    assert "only allowed on assistant messages" in str(caught.value)


def test_message_assistant_with_tool_calls():
    tc = ToolCall(id="id1", name="fn", arguments='{"key":"val"}')
    m = Message(role=MessageRole.ASSISTANT, content="", tool_calls=(tc,))
    assert len(m.tool_calls) == 1
    assert m.tool_calls[0].name == "fn"


def test_message_tool_calls_list_coerced_to_tuple():
    tc = ToolCall(id="id1", name="fn", arguments="{}")
    m = Message(role=MessageRole.ASSISTANT, content="", tool_calls=[tc])
    assert isinstance(m.tool_calls, tuple)


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


def test_usage_requires_actual_consistent_total():
    u = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert u.total_tokens == 15

    with pytest.raises(TypeError):
        Usage(prompt_tokens=10, completion_tokens=5)


def test_usage_rejects_inconsistent_explicit_total():
    with pytest.raises(ValueError) as caught:
        Usage(prompt_tokens=10, completion_tokens=5, total_tokens=20)
    assert "must equal" in str(caught.value)


# ---------------------------------------------------------------------------
# AssistantResponse
# ---------------------------------------------------------------------------


def test_assistant_response_requires_actual_finish_reason():
    r = AssistantResponse(content="hello", finish_reason="stop")
    assert r.finish_reason == "stop"
    assert r.tool_calls == ()
    assert r.usage is None
    assert r.reasoning_content is None


def test_assistant_response_preserves_absent_finish_reason():
    assert AssistantResponse(content="hello", finish_reason=None).finish_reason is None


def test_assistant_response_list_tool_calls_coerced_to_tuple():
    tc = ToolCall(id="x", name="fn", arguments="{}")
    r = AssistantResponse(content="", finish_reason="tool_calls", tool_calls=[tc])
    assert isinstance(r.tool_calls, tuple)
    assert r.tool_calls[0].name == "fn"


def test_assistant_response_with_usage():
    u = Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    r = AssistantResponse(content="text", finish_reason="stop", usage=u)
    assert r.usage.total_tokens == 8


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


def test_tool_call_construction():
    tc = ToolCall(id="abc", name="search", arguments='{"q":"hello"}')
    assert tc.id == "abc"
    assert tc.name == "search"
    assert tc.arguments == '{"q":"hello"}'


# ---------------------------------------------------------------------------
# ToolSchema
# ---------------------------------------------------------------------------


def test_tool_schema_construction():
    ts = ToolSchema(name="my_fn", description="does stuff", parameters={"type": "object"})
    assert ts.name == "my_fn"
    assert ts.description == "does stuff"
    assert ts.parameters == {"type": "object"}
    assert ts.strict is False


def test_tool_schema_accepts_missing_description_and_explicit_strict():
    ts = ToolSchema(
        name="my_fn",
        description=None,
        parameters={"type": "object"},
        strict=True,
    )

    assert ts.description is None
    assert ts.strict is True


# ---------------------------------------------------------------------------
# TerminatorCall
# ---------------------------------------------------------------------------


def test_terminator_call_valid():
    t = TerminatorCall(name="finish", arguments={"reason": "done"})
    assert t.name == "finish"
    assert t.arguments == {"reason": "done"}


def test_terminator_call_empty_name_raises():
    with pytest.raises(ValueError) as caught:
        TerminatorCall(name="", arguments={})
    assert "non-empty" in str(caught.value)


def test_terminator_call_non_dict_arguments_raises():
    with pytest.raises(TypeError) as caught:
        TerminatorCall(name="fn", arguments="not a dict")
    assert "must be a dict" in str(caught.value)


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


def test_message_and_assistant_response_reasoning_items_validation():
    item = {
        "v": 1,
        "model": "gpt-5.6-sol",
        "item": {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "enc_1",
            "summary": [{"type": "summary_text", "text": "step 1"}],
        },
    }
    msg = Message(role=MessageRole.ASSISTANT, content="answer", reasoning_items=[item])
    assert msg.reasoning_items == (item,)

    resp = AssistantResponse(content="answer", finish_reason="stop", reasoning_items=[item])
    assert resp.reasoning_items == (item,)

    with pytest.raises(ValueError) as role_err:
        Message(role=MessageRole.USER, content="hi", reasoning_items=[item])
    assert "only allowed on assistant messages" in str(role_err.value)

    with pytest.raises(TypeError) as type_err:
        Message(role=MessageRole.ASSISTANT, content="hi", reasoning_items=["bad"])  # type: ignore[list-item]
    assert "must contain dict values" in str(type_err.value)

    with pytest.raises(TypeError) as resp_type_err:
        AssistantResponse(content="hi", finish_reason="stop", reasoning_items=["bad"])  # type: ignore[list-item]
    assert "must contain dict values" in str(resp_type_err.value)
