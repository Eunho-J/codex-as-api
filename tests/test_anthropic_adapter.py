"""Tests for the Anthropic Messages API adapter."""

from __future__ import annotations

import json

import pytest

from codex_as_api.anthropic_adapter import (
    anthropic_parallel_tool_calls,
    anthropic_request_to_internal,
    anthropic_stream_adapter,
    format_anthropic_error,
    internal_response_to_anthropic,
)
from codex_as_api.auth import ChatGPTOAuthProtocolError
from codex_as_api.messages import (
    AssistantResponse,
    MessageRole,
    ToolCall,
    Usage,
)

# ---------------------------------------------------------------------------
# Request conversion tests
# ---------------------------------------------------------------------------


class TestAnthropicRequestToInternal:
    def test_system_string(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            system="You are helpful.",
        )
        assert messages[0].role is MessageRole.SYSTEM
        assert messages[0].content == "You are helpful."
        assert messages[1].role is MessageRole.USER

    @pytest.mark.parametrize(
        "tool",
        [
            {"type": "programmatic_tool_calling"},
            {
                "name": "lookup",
                "input_schema": {"type": "object"},
                "allowed_callers": ["programmatic"],
            },
            {
                "name": "lookup",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
        ],
    )
    def test_rejects_programmatic_tool_calling_fields(self, tool):
        with pytest.raises(ValueError, match="Programmatic|programmatic"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                tools=[tool],
            )

    def test_system_content_blocks(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            system=[
                {"type": "text", "text": "Rule 1"},
                {"type": "text", "text": "Rule 2"},
            ],
        )
        assert messages[0].content == "Rule 1\n\nRule 2"

    def test_no_system(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert messages[0].role is MessageRole.USER

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("messages", [{"role": "user", "content": []}]),
            ("tools", {}),
            ("tools", ""),
        ],
    )
    def test_rejects_empty_content_and_non_array_tools(self, field, value):
        kwargs = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            field: value,
        }

        with pytest.raises(ValueError):
            anthropic_request_to_internal(**kwargs)

    def test_empty_system_blocks_are_omitted(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            system=[],
        )
        assert [message.role for message in messages] == [MessageRole.USER]

    def test_user_text_message(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert len(messages) == 1
        assert messages[0].role is MessageRole.USER
        assert messages[0].content == "Hello"

    def test_user_content_blocks(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is the result:"},
                        {"type": "tool_result", "tool_use_id": "call-1", "content": "42"},
                    ],
                }
            ],
        )
        assert len(messages) == 2
        assert messages[0].role is MessageRole.USER
        assert messages[0].content == "Here is the result:"
        assert messages[1].role is MessageRole.TOOL
        assert messages[1].content == "42"
        assert messages[1].tool_call_id == "call-1"

    def test_user_tool_result_only(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-1", "content": "result1"},
                        {"type": "tool_result", "tool_use_id": "call-2", "content": "result2"},
                    ],
                }
            ],
        )
        assert len(messages) == 2
        assert all(m.role is MessageRole.TOOL for m in messages)

    @pytest.mark.parametrize(
        "block",
        [
            {"type": "tool_result", "content": "result"},
            {"type": "tool_result", "tool_use_id": None, "content": "result"},
            {"type": "tool_result", "tool_use_id": 1, "content": "result"},
        ],
    )
    def test_user_tool_result_requires_string_id(self, block):
        with pytest.raises(ValueError, match="string tool_use_id"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": [block]}],
            )

    def test_assistant_text(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "assistant", "content": "Hello!"}],
        )
        assert messages[0].role is MessageRole.ASSISTANT
        assert messages[0].content == "Hello!"

    def test_assistant_tool_use_blocks(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "tc-1",
                            "name": "get_weather",
                            "input": {"city": "Seoul"},
                            "caller": {"type": "direct"},
                        },
                    ],
                }
            ],
        )
        assert messages[0].content == "Let me check."
        assert len(messages[0].tool_calls) == 1
        assert messages[0].tool_calls[0].name == "get_weather"
        assert messages[0].tool_calls[0].arguments == '{"city":"Seoul"}'

    @pytest.mark.parametrize(
        "caller",
        [
            None,
            "direct",
            {},
            {"type": "code_execution_20260120", "tool_id": "srv_1"},
            {"type": "direct", "extra": True},
        ],
    )
    def test_assistant_tool_use_rejects_non_direct_or_malformed_caller(self, caller):
        with pytest.raises(ValueError, match="caller"):
            anthropic_request_to_internal(
                model="test",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tc-1",
                                "name": "get_weather",
                                "input": {"city": "Seoul"},
                                "caller": caller,
                            }
                        ],
                    }
                ],
            )

    def test_assistant_thinking_block(self):
        with pytest.raises(ValueError, match="cannot be preserved"):
            anthropic_request_to_internal(
                model="test",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Let me think...", "signature": "sig-abc"},
                            {"type": "text", "text": "The answer is 42."},
                        ],
                    }
                ],
            )

    def test_rejects_assistant_server_web_search_history(self):
        with pytest.raises(ValueError, match="cannot be preserved"):
            anthropic_request_to_internal(
                model="test",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "server_tool_use",
                                "id": "srv_1",
                                "name": "web_search",
                                "input": {"query": "codex"},
                            },
                            {
                                "type": "web_search_tool_result",
                                "tool_use_id": "srv_1",
                                "content": [
                                    {"title": "Codex", "url": "https://example.com", "page_age": "1d"},
                                ],
                            },
                            {"type": "text", "text": "Summary"},
                        ],
                    }
                ],
            )

    @pytest.mark.parametrize(
        "messages",
        [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": [
                                {
                                    "type": "search_result",
                                    "title": "Docs",
                                    "url": "https://docs.example",
                                    "content": "body",
                                }
                            ],
                        }
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "title": "Spec",
                            "source": {"type": "text", "data": "document body"},
                        }
                    ],
                }
            ],
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "search_result",
                            "title": "Docs",
                            "url": "https://docs.example",
                            "content": "body",
                        }
                    ],
                }
            ],
        ],
    )
    def test_rejects_unrepresentable_document_and_search_result_blocks(self, messages):
        with pytest.raises(ValueError, match="cannot be preserved"):
            anthropic_request_to_internal(model="test", messages=messages)

    def test_tools_conversion(self):
        _, tools, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                    "strict": True,
                }
            ],
        )
        assert tools is not None
        assert len(tools) == 1
        assert tools[0].name == "get_weather"
        assert tools[0].parameters == {"type": "object", "properties": {"city": {"type": "string"}}}
        assert tools[0].strict is True

    @pytest.mark.parametrize("field", ["defer_loading", "eager_input_streaming"])
    @pytest.mark.parametrize("value", [False, True])
    def test_non_null_beta_tool_fields_fail_loudly(self, field, value):
        with pytest.raises(ValueError, match=field):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                tools=[
                    {
                        "name": "lookup",
                        "input_schema": {"type": "object"},
                        field: value,
                    }
                ],
            )

    def test_nullable_beta_tool_fields_are_omitted(self):
        _, tools, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "name": "lookup",
                    "input_schema": {"type": "object"},
                    "strict": False,
                    "eager_input_streaming": None,
                }
            ],
        )
        assert tools is not None
        assert tools[0].name == "lookup"
        assert tools[0].strict is False

    @pytest.mark.parametrize("strict", [None, "true", 1])
    def test_non_boolean_tool_strict_fails_loudly(self, strict):
        with pytest.raises(ValueError, match="strict"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                tools=[
                    {
                        "name": "lookup",
                        "input_schema": {"type": "object"},
                        "strict": strict,
                    }
                ],
            )

    @pytest.mark.parametrize("tool_type", [pytest.param("__omitted__", id="omitted"), "custom", None])
    def test_custom_tool_type_variants_are_equivalent(self, tool_type):
        tool = {"name": "lookup", "input_schema": {"type": "object"}}
        if tool_type != "__omitted__":
            tool["type"] = tool_type
        _, tools, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tools=[tool],
        )
        assert tools is not None
        assert tools[0].name == "lookup"

    @pytest.mark.parametrize("tool_type", ["future", 1, False, {}])
    def test_invalid_custom_tool_type_is_rejected(self, tool_type):
        with pytest.raises(ValueError, match="type"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": tool_type, "name": "lookup", "input_schema": {}}],
            )

    def test_tool_choice_auto(self):
        _, _, tc, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice={"type": "auto"},
        )
        assert tc == "auto"

    def test_tool_choice_any(self):
        _, _, tc, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice={"type": "any"},
        )
        assert tc == "required"

    def test_tool_choice_specific(self):
        _, _, tc, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice={"type": "tool", "name": "get_weather"},
        )
        assert tc == {"type": "function", "name": "get_weather"}

    @pytest.mark.parametrize(
        "tool",
        [
            {"type": "web_search", "name": "web_search"},
            {"type": "web_search_20250305", "name": "web_search", "blocked_domains": ["example.com"]},
            {
                "type": "web_search_20260209",
                "name": "web_search",
                "allowed_domains": ["example.com"],
                "max_uses": 8,
                "strict": False,
                "user_location": {"type": "approximate", "country": "US"},
            },
        ],
    )
    def test_hosted_web_search_tools_are_rejected(self, tool):
        with pytest.raises(ValueError, match="cannot be represented losslessly"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                tools=[tool],
            )

    @pytest.mark.parametrize("tool_type", ["web_search_20240101", "web_search_future"])
    def test_unknown_web_search_tool_versions_fail_loudly(self, tool_type):
        with pytest.raises(ValueError, match="type"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": tool_type, "name": "web_search"}],
            )

    def test_hosted_web_search_tool_choice_is_rejected(self):
        with pytest.raises(ValueError, match="cannot be represented losslessly"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                tool_choice={"type": "tool", "name": "web_search"},
            )

    def test_tool_choice_none(self):
        _, _, tc, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice={"type": "none"},
        )
        assert tc == "none"

    @pytest.mark.parametrize(
        ("disable_parallel_tool_use", "expected"),
        [(True, False), (False, True)],
    )
    def test_tool_choice_maps_disable_parallel_tool_use(
        self,
        disable_parallel_tool_use,
        expected,
    ):
        choice = {
            "type": "auto",
            "disable_parallel_tool_use": disable_parallel_tool_use,
        }

        _, _, converted, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice=choice,
        )

        assert converted == "auto"
        assert anthropic_parallel_tool_calls(choice) is expected

    def test_tool_choice_absent_parallel_control_stays_unspecified(self):
        assert anthropic_parallel_tool_calls(None) is None
        assert anthropic_parallel_tool_calls({"type": "auto"}) is None

    @pytest.mark.parametrize("value", [None, 0, 1, "false", []])
    def test_tool_choice_parallel_control_requires_boolean(self, value):
        with pytest.raises(ValueError, match="disable_parallel_tool_use"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                tool_choice={
                    "type": "auto",
                    "disable_parallel_tool_use": value,
                },
            )

    def test_thinking_enabled(self):
        _, _, _, _, effort, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "enabled", "budget_tokens": 4096},
        )
        assert effort == "high"

    def test_thinking_adaptive(self):
        _, _, _, _, effort, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive"},
        )
        assert effort == "medium"

    def test_thinking_disabled(self):
        _, _, _, _, effort, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "disabled"},
        )
        assert effort == "none"

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_output_config_effort_overrides_adaptive_thinking(self, effort):
        _, _, _, _, converted, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive", "display": "omitted"},
            output_config={"effort": effort},
        )
        assert converted == effort

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_disabled_thinking_overrides_output_config_effort(self, effort):
        _, _, _, _, converted, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "disabled"},
            output_config={"effort": effort},
        )
        assert converted == "none"

    def test_disabled_thinking_preserves_output_config_format(self):
        _, _, _, _, converted, text = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "disabled"},
            output_config={"effort": "high", "format": {"type": "json_object"}},
        )
        assert converted == "none"
        assert text == {"format": {"type": "json_object"}}

    @pytest.mark.parametrize("effort", ["", 1, [], {}, "ultra"])
    def test_disabled_thinking_does_not_bypass_output_config_effort_validation(self, effort):
        with pytest.raises(ValueError, match="output_config.effort"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                thinking={"type": "disabled"},
                output_config={"effort": effort},
            )

    def test_output_config_task_budget_fails_loudly(self):
        with pytest.raises(ValueError, match="task_budget"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                output_config={"task_budget": {"type": "tokens", "total": 20_000}},
            )

    @pytest.mark.parametrize(
        "override",
        [
            {"messages": [{"role": "user", "content": "hi", "future": True}]},
            {"system": [{"type": "text", "text": "rules", "future": True}]},
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hi", "future": True}],
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": "https://example.com/image.png",
                                    "future": True,
                                },
                            }
                        ],
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call-1",
                                "name": "lookup",
                                "input": {},
                                "future": True,
                            }
                        ],
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call-1",
                                "content": "ok",
                                "future": True,
                            }
                        ],
                    }
                ]
            },
            {
                "tools": [
                    {
                        "name": "lookup",
                        "input_schema": {"type": "object"},
                        "future": True,
                    }
                ]
            },
            {"tool_choice": {"type": "auto", "future": True}},
            {"thinking": {"type": "adaptive", "future": True}},
        ],
    )
    def test_unknown_nested_request_fields_fail_loudly(self, override):
        request = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            **override,
        }

        with pytest.raises(ValueError, match="unsupported fields"):
            anthropic_request_to_internal(**request)

    @pytest.mark.parametrize("budget_tokens", [None, 0, 1023, True, 1.5, "4096"])
    def test_enabled_thinking_requires_positive_integer_budget(self, budget_tokens):
        thinking = {"type": "enabled"}
        if budget_tokens is not None:
            thinking["budget_tokens"] = budget_tokens

        with pytest.raises(ValueError, match="thinking.budget_tokens"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                thinking=thinking,
            )

    @pytest.mark.parametrize("budget_tokens", [1024.0, 2e3, 9_007_199_254_740_991.0])
    def test_enabled_thinking_accepts_safe_integral_json_numbers(self, budget_tokens):
        _, _, _, _, effort, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "enabled", "budget_tokens": budget_tokens},
        )
        assert effort == "high"

    def test_enabled_thinking_budget_must_be_less_than_max_tokens(self):
        for budget_tokens in (2048, 4096):
            with pytest.raises(ValueError, match="less than max_tokens"):
                anthropic_request_to_internal(
                    model="test",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=2048,
                    thinking={"type": "enabled", "budget_tokens": budget_tokens},
                )
        _, _, _, _, effort, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024, "display": "omitted"},
        )
        assert effort == "high"

    def test_enabled_thinking_rejects_summarized_display(self):
        with pytest.raises(ValueError, match="display"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                thinking={"type": "enabled", "budget_tokens": 1024, "display": "summarized"},
            )

    def test_tool_choice_none_rejects_parallel_control(self):
        with pytest.raises(ValueError, match="unsupported fields"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                tool_choice={"type": "none", "disable_parallel_tool_use": False},
            )

    @pytest.mark.parametrize("output_config", [{"effort": "ultra"}, {"future_control": True}])
    def test_unknown_output_config_values_fail_loudly(self, output_config):
        with pytest.raises(ValueError, match="output_config"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                output_config=output_config,
            )

    def test_output_format_json_schema_maps_to_text_format(self):
        _, _, _, _, _, text = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            output_format={
                "type": "json_schema",
                "name": "my_schema",
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
                "strict": False,
            },
        )
        assert text == {
            "format": {
                "type": "json_schema",
                "name": "my_schema",
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
                "strict": False,
            },
        }

    def test_output_format_json_schema_uses_pinned_default_name_when_omitted(self):
        _, _, _, _, _, text = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            output_format={"type": "json_schema", "schema": {"type": "object"}},
        )

        assert text == {
            "format": {
                "type": "json_schema",
                "name": "codex_output_schema",
                "schema": {"type": "object"},
            }
        }

    def test_output_format_json_schema_omits_nullable_strict(self):
        _, _, _, _, _, text = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            output_format={
                "type": "json_schema",
                "schema": {"type": "object"},
                "strict": None,
            },
        )

        assert text == {
            "format": {
                "type": "json_schema",
                "name": "codex_output_schema",
                "schema": {"type": "object"},
            }
        }

    def test_output_config_format_maps_without_top_level_alias(self):
        _, _, _, _, _, text = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            output_config={"format": {"type": "json_object"}},
        )
        assert text == {"format": {"type": "json_object"}}

    @pytest.mark.parametrize(
        ("output_format", "output_config"),
        [
            ("json", None),
            (None, {"format": "json"}),
            ({"type": "future"}, None),
            ({"type": "json_object", "extra": True}, None),
            ({"type": "json_schema", "schema": {}, "strict": "yes"}, None),
            ({"type": "json_schema", "schema": {}, "name": None}, None),
            ({"type": "json_schema", "schema": {}, "description": None}, None),
            ({"type": "json_schema", "name": "", "schema": {}}, None),
            ({"type": "json_schema", "name": "my schema", "schema": {}}, None),
            ({"type": "json_schema", "name": "schéma", "schema": {}}, None),
            ({"type": "json_schema", "name": "x" * 65, "schema": {}}, None),
            (
                {"type": "json_object"},
                {"format": {"type": "json_schema", "name": "nested", "schema": {"type": "object"}}},
            ),
        ],
    )
    def test_invalid_or_conflicting_output_formats_fail_loudly(self, output_format, output_config):
        with pytest.raises(ValueError, match=r"output_(?:format|config\.format)"):
            anthropic_request_to_internal(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                output_format=output_format,
                output_config=output_config,
            )

    def test_stop_sequences(self):
        _, _, _, stop, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stop_sequences=["STOP", "END"],
        )
        assert stop == ["STOP", "END"]

    def test_user_image_block(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgoAAAANS",
                            },
                        },
                    ],
                }
            ],
        )
        assert len(messages) == 1
        assert messages[0].role is MessageRole.USER
        assert messages[0].content == "What is in this image?"
        assert len(messages[0].images) == 1
        assert messages[0].images[0] == "data:image/png;base64,iVBORw0KGgoAAAANS"

    def test_user_url_image_block(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.com/image.png"},
                        }
                    ],
                }
            ],
        )
        assert messages[0].images == ("https://example.com/image.png",)

    @pytest.mark.parametrize(
        "source",
        [
            None,
            "image.png",
            {},
            {"type": "file", "file_id": "file-1"},
            {"type": "base64", "media_type": "image/svg+xml", "data": "PHN2Zz4="},
            {"type": "base64", "media_type": "text/plain", "data": "dGV4dA=="},
        ],
    )
    def test_missing_or_unsupported_image_sources_fail_loudly(self, source):
        direct = {"type": "image", "source": source}
        tool_result = {
            "type": "tool_result",
            "tool_use_id": "call-image",
            "content": [direct],
        }
        for block in (direct, tool_result):
            with pytest.raises(ValueError, match="image source"):
                anthropic_request_to_internal(
                    model="test",
                    messages=[{"role": "user", "content": [block]}],
                )

    def test_tool_result_with_content_blocks(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": [
                                {"type": "text", "text": "result line 1"},
                                {"type": "text", "text": "result line 2"},
                            ],
                        },
                    ],
                }
            ],
        )
        assert messages[0].role is MessageRole.TOOL
        assert messages[0].content == "result line 1result line 2"

    def test_tool_result_with_image(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-img",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo"},
                                },
                            ],
                        },
                    ],
                }
            ],
        )
        assert len(messages) == 2
        assert messages[0].role is MessageRole.TOOL
        assert messages[0].tool_call_id == "call-img"
        assert messages[0].content == ""
        assert messages[1].role is MessageRole.USER
        assert len(messages[1].images) == 1
        assert messages[1].images[0] == "data:image/png;base64,iVBORw0KGgo"

    def test_tool_result_with_text_and_image(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-mix",
                            "content": [
                                {"type": "text", "text": "file contents"},
                                {
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQ"},
                                },
                            ],
                        },
                    ],
                }
            ],
        )
        assert len(messages) == 2
        assert messages[0].role is MessageRole.TOOL
        assert messages[0].content == "file contents"
        assert messages[1].role is MessageRole.USER
        assert messages[1].images[0] == "data:image/jpeg;base64,/9j/4AAQ"

    def test_tool_result_url_image_and_error_state_are_preserved(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-error",
                            "is_error": True,
                            "content": [
                                {"type": "text", "text": "failed"},
                                {
                                    "type": "image",
                                    "source": {"type": "url", "url": "https://example.com/failure.png"},
                                },
                            ],
                        }
                    ],
                }
            ],
        )
        assert len(messages) == 2
        assert messages[0].role is MessageRole.TOOL
        assert messages[0].content == "[tool_error]\nfailed"
        assert messages[1].role is MessageRole.USER
        assert messages[1].images == ("https://example.com/failure.png",)

    def test_tool_result_missing_content_becomes_empty_string(self):
        messages, _, _, _, _, _ = anthropic_request_to_internal(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "call-empty"}],
                }
            ],
        )
        assert messages[0].role is MessageRole.TOOL
        assert messages[0].content == ""

    def test_duplicate_assistant_tool_use_ids_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate id"):
            anthropic_request_to_internal(
                model="test",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "call-1", "name": "a", "input": {}},
                            {"type": "tool_use", "id": "call-1", "name": "b", "input": {}},
                        ],
                    }
                ],
            )

    def test_null_tool_result_error_state_is_rejected(self):
        with pytest.raises(ValueError, match="is_error"):
            anthropic_request_to_internal(
                model="test",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call-ok",
                                "is_error": None,
                                "content": "ok",
                            }
                        ],
                    }
                ],
            )


# ---------------------------------------------------------------------------
# Non-streaming response conversion
# ---------------------------------------------------------------------------


class TestInternalResponseToAnthropic:
    def test_absent_finish_reason_is_rejected(self):
        resp = AssistantResponse(
            content="Hello!",
            finish_reason=None,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        with pytest.raises(ChatGPTOAuthProtocolError, match="non-null finish_reason"):
            internal_response_to_anthropic(resp, "test-model", "msg_123")

    def test_text_response(self):
        resp = AssistantResponse(
            content="Hello!",
            tool_calls=(),
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
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
        assert result["content"][0]["citations"] is None
        assert result["container"] is None
        assert result["context_management"] is None
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5
        assert result["usage"] == {
            "cache_creation": None,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
            "inference_geo": None,
            "input_tokens": 10,
            "iterations": None,
            "output_tokens": 5,
            "server_tool_use": None,
            "service_tier": None,
            "speed": None,
        }

    def test_tool_use_response(self):
        resp = AssistantResponse(
            content="",
            tool_calls=(ToolCall(id="tc-1", name="get_weather", arguments='{"city":"Seoul"}'),),
            finish_reason="tool_calls",
            usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            reasoning_content=None,
            raw=None,
        )
        result = internal_response_to_anthropic(resp, "test-model", "msg_123")
        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["name"] == "get_weather"
        assert result["content"][0]["caller"] == {"type": "direct"}

        replayed, _, _, _, _, _ = anthropic_request_to_internal(
            model="test-model",
            messages=[{"role": "assistant", "content": result["content"]}],
        )
        assert replayed[0].tool_calls == resp.tool_calls

    def test_empty_tool_use_id_roundtrips_through_tool_result(self):
        resp = AssistantResponse(
            content="",
            tool_calls=(ToolCall(id="", name="lookup", arguments="{}"),),
            finish_reason="tool_calls",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        result = internal_response_to_anthropic(resp, "test-model", "msg_empty_call")

        replayed, _, _, _, _, _ = anthropic_request_to_internal(
            model="test-model",
            messages=[
                {"role": "assistant", "content": result["content"]},
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "", "content": "done"}],
                },
            ],
        )

        assert replayed[0].tool_calls == resp.tool_calls
        assert replayed[1].role is MessageRole.TOOL
        assert replayed[1].tool_call_id == ""
        assert replayed[1].content == "done"

    def test_reasoning_response(self):
        resp = AssistantResponse(
            content="42",
            tool_calls=(),
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            reasoning_content="Let me think about this...",
            raw=None,
        )
        result = internal_response_to_anthropic(resp, "test-model", "msg_123")
        assert result["content"] == [{"type": "text", "text": "42", "citations": None}]
        assert result["codex_reasoning"] == "Let me think about this..."
        assert all(block["type"] != "thinking" for block in result["content"])

    def test_missing_usage_is_rejected(self):
        resp = AssistantResponse(
            content="",
            tool_calls=(),
            finish_reason="stop",
            usage=None,
            reasoning_content=None,
            raw=None,
        )
        with pytest.raises(ChatGPTOAuthProtocolError, match="authoritative usage"):
            internal_response_to_anthropic(resp, "test-model", "msg_123")

    def test_cached_tokens_in_usage(self):
        resp = AssistantResponse(
            content="hi",
            tool_calls=(),
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=10,
                total_tokens=110,
                cached_tokens=50,
                cache_write_tokens=25,
            ),
            reasoning_content=None,
            raw=None,
        )
        result = internal_response_to_anthropic(resp, "m", "msg_1")
        assert result["usage"]["cache_read_input_tokens"] == 50
        assert result["usage"]["cache_creation_input_tokens"] == 25

    def test_web_search_provider_output_is_rejected(self):
        resp = AssistantResponse(
            content="Final answer",
            tool_calls=(),
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            reasoning_content=None,
            raw={
                "events": [
                    {
                        "type": "web_search_call",
                        "id": "srvtoolu_ws1",
                        "input": {"query": "latest news"},
                        "content": [{"type": "web_search_result", "url": "https://example.com", "title": "Example"}],
                    }
                ]
            },
        )
        with pytest.raises(ChatGPTOAuthProtocolError, match="cannot be represented losslessly"):
            internal_response_to_anthropic(resp, "m", "msg_1")

    @pytest.mark.parametrize(
        "server_tool_use",
        [
            {"web_search_requests": "1"},
            {"web_search_requests": 1, "web_fetch_requests": 0, "future": 2},
            {"web_search_requests": 1},
        ],
    )
    def test_rejects_malformed_actual_server_tool_usage(self, server_tool_use):
        response = AssistantResponse(
            content="answer",
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            raw={
                "events": [
                    {
                        "type": "finish",
                        "usage": {"server_tool_use": server_tool_use},
                    }
                ]
            },
        )

        with pytest.raises(ChatGPTOAuthProtocolError, match="server_tool_use"):
            internal_response_to_anthropic(response, "m", "msg_1")

    def test_reasoning_only_response_uses_codex_extension_with_empty_content(self):
        resp = AssistantResponse(
            content="",
            tool_calls=(),
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            reasoning_content="private reasoning",
            raw=None,
        )

        result = internal_response_to_anthropic(resp, "m", "msg_1")

        assert result["content"] == []
        assert result["codex_reasoning"] == "private reasoning"
        replayed, _, _, _, _, _ = anthropic_request_to_internal(
            model="m",
            messages=[{"role": "assistant", "content": result["content"]}],
        )
        assert len(replayed) == 1
        assert replayed[0].role is MessageRole.ASSISTANT
        assert replayed[0].content == ""


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
            {
                "type": "finish",
                "finish_reason": "stop",
                "usage": {"input_tokens": 2, "output_tokens": 5, "total_tokens": 7},
            },
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
        text_deltas = [
            e
            for e in result
            if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta"
        ]
        assert len(text_deltas) == 2
        assert text_deltas[0]["delta"]["text"] == "Hello"
        assert text_deltas[1]["delta"]["text"] == " world"

    def test_empty_text_and_reasoning_deltas_are_preserved(self):
        result = self._collect_events(
            [
                {"type": "reasoning_delta", "text": ""},
                {"type": "reasoning_raw_delta", "text": ""},
                {"type": "content", "text": ""},
                {
                    "type": "finish",
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                },
            ]
        )

        reasoning = [event for event in result if event["type"] == "codex_reasoning_delta"]
        text = [
            event
            for event in result
            if event["type"] == "content_block_delta" and event["delta"]["type"] == "text_delta"
        ]
        assert [event["delta"] for event in reasoning] == ["", ""]
        assert [event["delta"]["text"] for event in text] == [""]

    def test_thinking_then_text(self):
        events = [
            {"type": "reasoning_delta", "text": "thinking..."},
            {"type": "content", "text": "result"},
            {
                "type": "finish",
                "finish_reason": "stop",
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            },
        ]
        result = self._collect_events(events)

        block_starts = [e for e in result if e["type"] == "content_block_start"]
        assert len(block_starts) == 1
        assert block_starts[0]["content_block"]["type"] == "text"
        reasoning = [e for e in result if e["type"] == "codex_reasoning_delta"]
        assert reasoning == [{"type": "codex_reasoning_delta", "delta": "thinking..."}]
        assert all(e.get("type") != "thinking" for e in result)

    def test_tool_call_stream(self):
        events = [
            {"type": "tool_call", "id": "tc-1", "name": "get_weather", "arguments": '{"city":"Seoul"}'},
            {
                "type": "finish",
                "finish_reason": "tool_calls",
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            },
        ]
        result = self._collect_events(events)

        block_starts = [e for e in result if e["type"] == "content_block_start"]
        assert len(block_starts) == 1
        assert block_starts[0]["content_block"]["type"] == "tool_use"
        assert block_starts[0]["content_block"]["name"] == "get_weather"
        assert block_starts[0]["content_block"]["caller"] == {"type": "direct"}

        json_deltas = [e for e in result if e.get("delta", {}).get("type") == "input_json_delta"]
        assert len(json_deltas) == 1
        assert json.loads(json_deltas[0]["delta"]["partial_json"]) == {"city": "Seoul"}
        message_delta = [e for e in result if e["type"] == "message_delta"]
        assert message_delta[0]["delta"]["stop_reason"] == "tool_use"

    def test_text_then_tool_call(self):
        events = [
            {"type": "content", "text": "Let me check."},
            {"type": "tool_call", "id": "tc-1", "name": "search", "arguments": '{"q":"test"}'},
            {
                "type": "finish",
                "finish_reason": "tool_calls",
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            },
        ]
        result = self._collect_events(events)
        block_starts = [e for e in result if e["type"] == "content_block_start"]
        assert len(block_starts) == 2
        assert block_starts[0]["content_block"]["type"] == "text"
        assert block_starts[1]["content_block"]["type"] == "tool_use"

    def test_empty_stream(self):
        events = [
            {
                "type": "finish",
                "finish_reason": "stop",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
        ]
        result = self._collect_events(events)
        block_starts = [e for e in result if e["type"] == "content_block_start"]
        assert block_starts == []

    def test_message_delta_stop_reason(self):
        events = [
            {"type": "content", "text": "hi"},
            {
                "type": "finish",
                "finish_reason": "stop",
                "usage": {"input_tokens": 1, "output_tokens": 3, "total_tokens": 4},
            },
        ]
        result = self._collect_events(events)
        msg_delta = [e for e in result if e["type"] == "message_delta"][0]
        assert msg_delta["delta"]["stop_reason"] == "end_turn"
        assert msg_delta["usage"]["output_tokens"] == 3
        assert msg_delta["usage"]["cache_read_input_tokens"] is None
        assert msg_delta["usage"]["cache_creation_input_tokens"] is None
        assert msg_delta["usage"]["iterations"] is None
        assert msg_delta["usage"]["server_tool_use"] is None
        assert msg_delta["context_management"] is None
        assert msg_delta["delta"]["container"] is None

    def test_routes_real_cumulative_usage_into_message_delta(self):
        events = [
            {"type": "content", "text": "hi"},
            {
                "type": "finish",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 123,
                    "output_tokens": 7,
                    "total_tokens": 130,
                    "input_tokens_details": {
                        "cached_tokens": 13,
                        "cache_write_tokens": 11,
                    },
                    "cache_creation": {"ephemeral_5m_input_tokens": 11, "ephemeral_1h_input_tokens": 0},
                    "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1},
                },
            },
        ]
        result = self._collect_events(events)
        msg_start = [e for e in result if e["type"] == "message_start"][0]
        assert "usage" not in msg_start["message"]
        msg_delta = [e for e in result if e["type"] == "message_delta"][0]
        assert msg_delta["usage"] == {
            "cache_creation_input_tokens": 11,
            "cache_read_input_tokens": 13,
            "input_tokens": 123,
            "iterations": None,
            "output_tokens": 7,
            "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1},
        }

    def test_missing_message_delta_usage_is_rejected(self):
        events = [
            {"type": "content", "text": "hi"},
            {"type": "finish", "finish_reason": "stop"},
        ]

        with pytest.raises(ChatGPTOAuthProtocolError, match="finish usage"):
            self._collect_events(events)

    @pytest.mark.parametrize(
        "usage",
        [
            None,
            "invalid",
            {"input_tokens": "1", "output_tokens": 2, "total_tokens": 3},
            {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "cache_read_input_tokens": None,
            },
            {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "server_tool_use": {"web_search_requests": "2", "web_fetch_requests": 0},
            },
            {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "cache_creation": {"future": 1},
            },
            {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "service_tier": "",
            },
        ],
    )
    def test_rejects_malformed_provider_finish_usage(self, usage):
        events = [
            {"type": "content", "text": "hi"},
            {"type": "finish", "finish_reason": "stop", "usage": usage},
        ]

        with pytest.raises(ChatGPTOAuthProtocolError, match="provider (finish )?usage"):
            self._collect_events(events)

    def test_maps_responses_cache_write_details_into_message_delta(self):
        result = self._collect_events(
            [
                {
                    "type": "finish",
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 5,
                        "total_tokens": 105,
                        "input_tokens_details": {
                            "cached_tokens": 20,
                            "cache_write_tokens": 30,
                        },
                    },
                },
            ]
        )

        msg_delta = [event for event in result if event["type"] == "message_delta"][0]
        assert msg_delta["usage"]["cache_read_input_tokens"] == 20
        assert msg_delta["usage"]["cache_creation_input_tokens"] == 30

    def test_multiple_tool_calls(self):
        events = [
            {"type": "tool_call", "id": "tc-1", "name": "tool_a", "arguments": '{"a":1}'},
            {"type": "tool_call", "id": "tc-2", "name": "tool_b", "arguments": '{"b":2}'},
            {
                "type": "finish",
                "finish_reason": "tool_calls",
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            },
        ]
        result = self._collect_events(events)
        block_starts = [e for e in result if e["type"] == "content_block_start"]
        assert len(block_starts) == 2
        assert block_starts[0]["content_block"]["name"] == "tool_a"
        assert block_starts[1]["content_block"]["name"] == "tool_b"
        # Verify indices are sequential
        assert block_starts[0]["index"] == 0
        assert block_starts[1]["index"] == 1

    def test_web_search_call_stream_is_rejected(self):
        events = [
            {
                "type": "web_search_call",
                "id": "srvtoolu_ws1",
                "input": {"query": "current time"},
                "content": [{"type": "web_search_result", "url": "https://example.com", "title": "Example"}],
            },
            {"type": "content", "text": "It is noon."},
            {
                "type": "finish",
                "finish_reason": "stop",
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            },
        ]
        with pytest.raises(ChatGPTOAuthProtocolError, match="cannot be represented losslessly"):
            self._collect_events(events)

    def test_only_diagnostic_reasoning_section_break_is_ignored(self):
        result = self._collect_events(
            [
                {"type": "reasoning_section_break", "summary_index": 1},
                {
                    "type": "finish",
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            ]
        )
        assert [event["type"] for event in result] == ["message_start", "message_delta", "message_stop"]

    def test_unknown_normalized_stream_event_is_rejected(self):
        with pytest.raises(ChatGPTOAuthProtocolError, match="unsupported normalized response event type"):
            self._collect_events([{"type": "computer_call"}])

    def test_reasoning_only_stream_uses_codex_extension(self):
        events = [
            {"type": "reasoning_delta", "text": "thinking..."},
            {
                "type": "finish",
                "finish_reason": "stop",
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            },
        ]

        result = self._collect_events(events)

        assert result[1] == {"type": "codex_reasoning_delta", "delta": "thinking..."}
        assert result[-2]["type"] == "message_delta"
        assert result[-1] == {"type": "message_stop"}


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
