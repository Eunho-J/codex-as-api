from __future__ import annotations

import json
from copy import deepcopy

import pytest

from codex_as_api.auth import ChatGPTOAuthError, ChatGPTOAuthInvalidRequestError
from codex_as_api.messages import Message, MessageRole, ToolCall, ToolSchema
from codex_as_api.model_capabilities import RESPONSES_LITE_ENV
from codex_as_api.provider import (
    REMOTE_COMPACTION_MARKER,
    RESPONSE_CHAIN_CAPACITY,
    ChatGPTOAuthProvider,
    _decode_sse_block,
    _filter_compacted_history_items,
    _image_generation_from_item,
    _message_item,
    _messages_to_response_items,
    _ResponseChainStore,
    _set_reasoning_payload,
    _split_instructions_and_input,
    _text_from_response_items,
    _tool_call_from_response_item,
    _tool_schema_to_response_dict,
    _usage_from_response,
    _validate_image_content_items,
    _web_search_event_from_response_item,
    codex_cli_headers_for_version,
)


@pytest.fixture(autouse=True)
def _isolate_responses_lite_mode(monkeypatch):
    monkeypatch.setenv(RESPONSES_LITE_ENV, "auto")


def _provider_messages() -> list[Message]:
    return [
        Message(role=MessageRole.SYSTEM, content="You are helpful."),
        Message(role=MessageRole.USER, content="Hello"),
    ]


def _provider_messages_with_image_detail(detail: str) -> list[Message]:
    return [
        Message(role=MessageRole.SYSTEM, content="You are helpful."),
        Message(
            role=MessageRole.USER,
            content="Inspect this image.",
            content_parts=(
                {"type": "input_text", "text": "Inspect this image."},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": detail,
                },
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# ChatGPTOAuthProvider payload
# ---------------------------------------------------------------------------


def test_responses_payload_omits_max_output_tokens_when_max_tokens_is_set():
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001 - regression test for backend payload
        _provider_messages(),
        model="gpt-5.5",
        tools=None,
        temperature=None,
        reasoning_effort=None,
        stop=None,
        prompt_cache_key=None,
        max_tokens=1024,
    )

    assert "max_output_tokens" not in payload


@pytest.mark.parametrize("stop", [None, "", [], [""], ["", ""]])
def test_chat_stream_omits_empty_stop_without_forwarding(stop, monkeypatch):
    provider = ChatGPTOAuthProvider()
    captured: list[dict] = []

    def completed_sse(_path, payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        captured.append(deepcopy(payload))
        yield {
            "type": "response.completed",
            "response": {"id": "resp-empty-stop", "output": [], "usage": {}},
        }

    monkeypatch.setattr(provider, "_post_sse", completed_sse)

    list(provider.chat_stream(_provider_messages(), model="gpt-5.5", stop=stop))

    assert len(captured) == 1
    assert "stop" not in captured[0]


def test_chat_stream_rejects_non_empty_stop_before_transport(monkeypatch):
    provider = ChatGPTOAuthProvider()
    transport_calls = 0

    def forbidden_sse(*_args, **_kwargs):
        nonlocal transport_calls
        transport_calls += 1
        raise AssertionError("unsupported stop must fail before upstream")

    monkeypatch.setattr(provider, "_post_sse", forbidden_sse)

    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="stop is not supported"):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.5", stop=["END"]))

    assert transport_calls == 0


@pytest.mark.parametrize(
    "model",
    ["gpt-5.2", "gpt-5.3-codex", "gpt-5.3-codex-spark", "future-model"],
)
def test_responses_payload_rejects_original_image_detail_for_unsupported_model(model: str):
    provider = ChatGPTOAuthProvider()

    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages_with_image_detail("original"),
            model=model,
            responses_lite=False,
        )


@pytest.mark.parametrize("detail", ["auto", "low", "high"])
def test_responses_payload_preserves_non_original_image_detail_for_gpt_5_2(detail: str):
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages_with_image_detail(detail),
        model="gpt-5.2",
        responses_lite=False,
    )

    assert payload["input"][0]["content"][1]["detail"] == detail


@pytest.mark.parametrize("operation", ["generate", "inspect"])
def test_image_reference_paths_reject_original_detail_for_gpt_5_2(operation: str):
    provider = ChatGPTOAuthProvider()
    provider._collect_response_output_items = lambda _payload: pytest.fail(  # type: ignore[method-assign]  # noqa: SLF001
        "private transport must not start"
    )
    images = [{"image_url": "data:image/png;base64,AAAA", "detail": "original"}]

    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        if operation == "generate":
            provider.generate_image("Draw this", model="gpt-5.2", reference_images=images)
        else:
            provider.inspect_images("Inspect this", model="gpt-5.2", images=images)


def test_responses_payload_includes_web_search_sources():
    provider = ChatGPTOAuthProvider()
    web_search_tool = ToolSchema(
        name="web_search",
        description="Web search",
        parameters={
            "__codex_as_api_tool_type": "web_search",
            "openai_tool": {"type": "web_search", "external_web_access": True},
        },
    )

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        tools=[web_search_tool],
        tool_choice={"type": "web_search"},
        temperature=None,
        reasoning_effort="low",
        stop=None,
        prompt_cache_key=None,
    )

    assert payload["tools"] == [{"type": "web_search", "external_web_access": True}]
    assert payload["tool_choice"] == {"type": "web_search"}
    assert payload["include"] == ["web_search_call.action.sources", "reasoning.encrypted_content"]


def test_responses_payload_reasoning_includes_encrypted_content():
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        reasoning_effort="high",
    )

    assert payload["reasoning"] == {"effort": "high"}
    assert payload["include"] == ["reasoning.encrypted_content"]


def test_responses_payload_forces_responses_lite_shape():
    provider = ChatGPTOAuthProvider()
    tool = ToolSchema(name="lookup", description="Lookup", parameters={"type": "object"})

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        tools=[tool],
        reasoning_effort="low",
        responses_lite=True,
    )

    assert "tools" not in payload
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is False
    assert payload["reasoning"]["context"] == "all_turns"
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["input"][0] == {
        "type": "additional_tools",
        "role": "developer",
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Lookup",
                "parameters": {"type": "object"},
                "strict": False,
            }
        ],
    }
    assert payload["input"][1] == {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "You are helpful."}],
    }


def test_responses_payload_lite_always_starts_with_additional_tools_array():
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.6-sol",
    )

    assert payload["input"][0] == {
        "type": "additional_tools",
        "role": "developer",
        "tools": [],
    }
    assert payload["input"][1] == {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "You are helpful."}],
    }
    assert payload["reasoning"] == {"effort": "low", "context": "all_turns"}


def test_forced_lite_unknown_model_does_not_invent_reasoning():
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="unknown-model",
        responses_lite=True,
    )

    assert "reasoning" not in payload
    assert payload["include"] == []


@pytest.mark.parametrize(
    ("model", "default_effort"),
    [
        ("gpt-5.6-sol", "low"),
        ("gpt-5.6-terra", "medium"),
        ("gpt-5.6-luna", "medium"),
    ],
)
def test_gpt_5_6_payloads_apply_catalog_reasoning_defaults(
    model: str,
    default_effort: str,
):
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model=model,
    )

    assert payload["reasoning"] == {
        "effort": default_effort,
        "context": "all_turns",
    }


def test_standard_reasoning_mode_uses_public_medium_default_without_mode_field():
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.6-sol",
        reasoning_mode="standard",
        responses_lite=False,
    )

    assert payload["reasoning"] == {"effort": "medium"}


def test_responses_payload_responses_lite_auto_uses_capability_table():
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        service_tier="priority",
        responses_lite="auto",
    )
    fast_payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        service_tier="fast",
    )

    assert "tools" in payload
    assert payload["text"] == {"verbosity": "low"}
    assert payload["service_tier"] == "priority"
    assert fast_payload["service_tier"] == "priority"
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages(),
            model="unknown-model",
            service_tier="priority",
        )


def test_default_service_tier_is_omitted():
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.6-sol",
        service_tier="default",
    )

    assert "service_tier" not in payload


def test_safety_identifier_is_rejected_before_transport():
    provider = ChatGPTOAuthProvider()

    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages(),
            model="gpt-5.5",
            safety_identifier="stable-user-id",
        )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"mode": "implicit"},
        {"mode": "implicit", "ttl": "30m"},
        {"mode": "explicit", "ttl": "30m"},
    ],
)
def test_prompt_cache_options_are_rejected_before_transport(options):
    provider = ChatGPTOAuthProvider()

    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages(),
            model="gpt-5.6-sol",
            prompt_cache_options=options,
        )


def test_image_prompt_cache_breakpoint_is_rejected():
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        _validate_image_content_items(
            [
                {
                    "image_url": "data:image/png;base64,AAAA",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ]
        )


def test_null_prompt_cache_breakpoint_is_omitted_from_structured_content():
    item = _message_item(
        "user",
        "hello",
        content_parts=(
            {
                "type": "input_text",
                "text": "hello",
                "prompt_cache_breakpoint": None,
            },
        ),
    )

    assert item["content"] == [{"type": "input_text", "text": "hello"}]


def test_tool_schema_property_named_prompt_cache_breakpoint_is_preserved():
    provider = ChatGPTOAuthProvider()
    tool = ToolSchema(
        name="lookup",
        description="Lookup",
        parameters={
            "type": "object",
            "properties": {"prompt_cache_breakpoint": {"type": "string"}},
        },
    )

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.6-sol",
        tools=[tool],
        responses_lite=False,
    )

    assert payload["tools"][0]["parameters"] == tool.parameters


def test_nested_text_verbosity_is_validated_without_top_level_verbosity():
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        text={"verbosity": "high"},
    )

    assert payload["text"] == {"verbosity": "high"}
    with pytest.raises(ChatGPTOAuthError, match="text.verbosity must be one of"):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages(),
            model="gpt-5.5",
            text={"verbosity": "verbose"},
        )


def test_responses_payload_parallel_tool_calls_uses_capability_table():
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        parallel_tool_calls=True,
    )
    spark_payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.3-codex-spark",
        parallel_tool_calls=True,
    )
    lite_payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        parallel_tool_calls=True,
        responses_lite=True,
    )

    assert payload["parallel_tool_calls"] is True
    assert spark_payload["parallel_tool_calls"] is False
    assert lite_payload["parallel_tool_calls"] is False


def test_responses_payload_client_metadata_mode_overlays_reserved_keys(auth_json_factory):
    path = auth_json_factory()
    provider = ChatGPTOAuthProvider(auth_json_path=str(path))

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        client_metadata={"app": "kept", "session_id": "session-root", "turn_id": "user-value"},
        codex_metadata=True,
    )

    metadata = payload["client_metadata"]
    assert metadata["app"] == "kept"
    assert metadata["session_id"] == "session-root"
    assert metadata["thread_id"] == "session-root"
    assert metadata["turn_id"] != "user-value"
    assert metadata["x-codex-installation-id"]
    turn_metadata = json.loads(metadata["x-codex-turn-metadata"])
    assert turn_metadata["source"] == "codex-as-api"
    assert turn_metadata["session_id"] == "session-root"
    assert turn_metadata["thread_id"] == "session-root"
    assert payload["prompt_cache_key"] == "session-root"


def test_responses_payload_codex_metadata_preserves_thread_and_refreshes_only_turn_identity(auth_json_factory):
    provider = ChatGPTOAuthProvider(auth_json_path=str(auth_json_factory()))
    options = {
        "model": "gpt-5.5",
        "client_metadata": {"session_id": "session-root", "thread_id": "thread-child"},
        "codex_metadata": True,
    }

    first = provider._responses_payload(_provider_messages(), **options)  # noqa: SLF001
    second = provider._responses_payload(_provider_messages(), **options)  # noqa: SLF001
    first_metadata = first["client_metadata"]
    second_metadata = second["client_metadata"]

    assert first_metadata["session_id"] == second_metadata["session_id"] == "session-root"
    assert first_metadata["thread_id"] == second_metadata["thread_id"] == "thread-child"
    assert first_metadata["x-codex-installation-id"] == second_metadata["x-codex-installation-id"]
    assert first_metadata["x-codex-window-id"] == second_metadata["x-codex-window-id"]
    assert first_metadata["turn_id"] != second_metadata["turn_id"]


def test_responses_payload_codex_metadata_requires_explicit_session_identity():
    provider = ChatGPTOAuthProvider()

    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages(),
            model="gpt-5.5",
            codex_metadata=True,
        )


def test_responses_payload_prompt_cache_key_precedence_and_session_fallback():
    provider = ChatGPTOAuthProvider()

    fallback = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        client_metadata={"session_id": "session-cache"},
    )
    explicit = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        prompt_cache_key="explicit-cache",
        client_metadata={"session_id": "session-cache"},
    )
    absent = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
    )
    blank_session = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        client_metadata={"session_id": "   "},
    )
    explicit_with_blank_session = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        prompt_cache_key="explicit-cache",
        client_metadata={"session_id": "   "},
    )

    assert fallback["prompt_cache_key"] == "session-cache"
    assert explicit["prompt_cache_key"] == "explicit-cache"
    assert "prompt_cache_key" not in absent
    assert "prompt_cache_key" not in blank_session
    assert explicit_with_blank_session["prompt_cache_key"] == "explicit-cache"


def test_chat_stream_adds_responses_lite_header(monkeypatch):
    provider = ChatGPTOAuthProvider()
    captured: dict[str, object] = {}

    def fake_post_sse(path, payload, extra_headers=None):  # noqa: ANN001
        captured["path"] = path
        captured["payload"] = payload
        captured["headers"] = extra_headers
        return iter(
            [
                {
                    "type": "response.completed",
                    "response": {"id": "response-1", "output": []},
                }
            ]
        )

    monkeypatch.setattr(provider, "_post_sse", fake_post_sse)
    list(provider.chat_stream(_provider_messages(), model="gpt-5.5", responses_lite=True))

    assert captured["headers"]["x-openai-internal-codex-responses-lite"] == "true"


def test_chat_stream_marks_output_item_tool_calls_as_terminal_tool_calls(monkeypatch):
    provider = ChatGPTOAuthProvider()
    tool_call = {
        "type": "function_call",
        "id": "item-1",
        "call_id": "call-1",
        "name": "lookup",
        "arguments": '{"query":"one"}',
    }

    def fake_post_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.output_item.done", "item": tool_call}
        yield {
            "type": "response.completed",
            "response": {"id": "response-1", "output": [tool_call], "usage": {}},
        }

    monkeypatch.setattr(provider, "_post_sse", fake_post_sse)

    events = list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))
    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    finish = [event for event in events if event.get("type") == "finish"]

    assert [(event["id"], event["arguments"]) for event in tool_calls] == [("call-1", {"query": "one"})]
    assert [event["finish_reason"] for event in finish] == ["tool_calls"]


def test_chat_stream_rejects_eof_before_response_completed(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def truncated_post_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.output_text.delta", "delta": "partial"}

    monkeypatch.setattr(provider, "_post_sse", truncated_post_sse)

    with pytest.raises(ChatGPTOAuthError, match=r"ended before response\.completed"):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))


def test_chat_stream_stops_reading_after_response_completed(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def post_sse_with_trailing_failure(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {
            "type": "response.completed",
            "response": {"id": "response-1", "output": [], "usage": {}},
        }
        raise AssertionError("events after response.completed must not be consumed")

    monkeypatch.setattr(provider, "_post_sse", post_sse_with_trailing_failure)

    events = list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))

    assert events[-1]["type"] == "finish"


def test_previous_response_id_replays_exact_output_item_done_without_private_wire(monkeypatch):
    provider = ChatGPTOAuthProvider()
    requests: list[dict] = []
    response_ids = iter(["resp-root", "resp-branch-a", "resp-branch-b"])
    root_output = [
        {
            "type": "reasoning",
            "id": "reasoning-1",
            "encrypted_content": "encrypted-root",
            "summary": [{"type": "summary_text", "text": "root summary"}],
        },
        {
            "type": "function_call",
            "id": "item-call-1",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"query":"root"}',
        },
        {
            "type": "message",
            "id": "message-1",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "root answer"}],
        },
    ]

    def completed_sse(_path, payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        requests.append(deepcopy(payload))
        response_id = next(response_ids)
        output = root_output if response_id == "resp-root" else []
        for item in output:
            yield {"type": "response.output_item.done", "item": item}
        yield {
            "type": "response.completed",
            "response": {"id": response_id, "output": [], "usage": {}},
        }

    monkeypatch.setattr(provider, "_post_sse", completed_sse)
    root = provider.chat(_provider_messages(), model="gpt-5.5")
    root_output[0]["encrypted_content"] = "mutated-after-completion"

    branch_a_messages = [
        Message(role=MessageRole.SYSTEM, content="You are helpful."),
        Message(role=MessageRole.USER, content="branch a"),
    ]
    branch_b_messages = [
        Message(role=MessageRole.SYSTEM, content="You are helpful."),
        Message(role=MessageRole.USER, content="branch b"),
    ]
    list(
        provider.chat_stream(
            branch_a_messages,
            model="gpt-5.5",
            previous_response_id=root.response_id,
        )
    )
    list(
        provider.chat_stream(
            branch_b_messages,
            model="gpt-5.5",
            previous_response_id=root.response_id,
        )
    )

    root_input = requests[0]["input"]
    stored_root_output = deepcopy(root_output)
    stored_root_output[0]["encrypted_content"] = "encrypted-root"
    branch_a_current = requests[1]["input"][-1]
    branch_b_current = requests[2]["input"][-1]
    assert root.response_id == "resp-root"
    assert "previous_response_id" not in requests[1]
    assert "previous_response_id" not in requests[2]
    assert requests[1]["input"] == [*root_input, *stored_root_output, branch_a_current]
    assert requests[2]["input"] == [*root_input, *stored_root_output, branch_b_current]
    assert branch_a_current["content"][0]["text"] == "branch a"
    assert branch_b_current["content"][0]["text"] == "branch b"


def test_unknown_previous_response_id_fails_before_upstream(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def forbidden_sse(*_args, **_kwargs):
        raise AssertionError("unknown previous_response_id must fail before upstream")

    monkeypatch.setattr(provider, "_post_sse", forbidden_sse)

    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        list(
            provider.chat_stream(
                _provider_messages(),
                model="gpt-5.5",
                previous_response_id="resp-missing",
            )
        )


def test_lite_replay_injects_one_current_developer_prefix(monkeypatch):
    provider = ChatGPTOAuthProvider()
    requests: list[dict] = []
    response_ids = iter(["resp-lite-root", "resp-lite-next"])

    def completed_sse(_path, payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        requests.append(deepcopy(payload))
        response_id = next(response_ids)
        yield {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": response_id}],
                    }
                ],
            },
        }

    monkeypatch.setattr(provider, "_post_sse", completed_sse)
    first = provider.chat(_provider_messages(), model="gpt-5.6-sol")
    provider.chat(
        [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="next"),
        ],
        model="gpt-5.6-sol",
        previous_response_id=first.response_id,
    )

    next_input = requests[1]["input"]
    assert sum(item.get("type") == "additional_tools" for item in next_input) == 1
    assert sum(item.get("role") == "developer" for item in next_input) == 2
    assert [
        part["text"]
        for item in next_input
        if item.get("type") == "message" and item.get("role") == "user"
        for part in item["content"]
    ] == ["Hello", "next"]


def test_incomplete_response_is_not_committed_to_previous_response_store(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def failed_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {
            "type": "response.output_item.done",
            "item": {"type": "message", "role": "assistant", "content": []},
        }
        yield {"type": "response.failed", "response": {"id": "resp-failed"}}

    monkeypatch.setattr(provider, "_post_sse", failed_sse)

    with pytest.raises(ChatGPTOAuthError):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.5"))
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages(),
            model="gpt-5.5",
            previous_response_id="resp-failed",
        )


def test_response_chain_store_is_deep_copied_and_lru_bounded():
    store = _ResponseChainStore()
    request_item = {"type": "message", "role": "user", "content": []}
    output_item = {"type": "reasoning", "encrypted_content": "original"}
    for index in range(RESPONSE_CHAIN_CAPACITY):
        store.commit(f"resp-{index}", [request_item], [output_item])

    resolved = store.resolve("resp-0")
    resolved[1]["encrypted_content"] = "mutated"
    store.commit("resp-overflow", [request_item], [output_item])

    assert store.resolve("resp-0")[1]["encrypted_content"] == "original"
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        store.resolve("resp-1")


def test_output_item_collector_rejects_eof_before_response_completed(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def truncated_post_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {
            "type": "response.output_item.done",
            "item": {"type": "message", "role": "assistant", "content": []},
        }

    monkeypatch.setattr(provider, "_post_sse", truncated_post_sse)

    with pytest.raises(ChatGPTOAuthError, match=r"ended before response\.completed"):
        provider._collect_response_output_items({})  # noqa: SLF001


def test_output_item_collector_ignores_response_completed_output(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def completed_post_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {
            "type": "response.completed",
            "response": {
                "id": "response-1",
                "output": [{"type": "message", "role": "assistant", "content": []}],
            },
        }

    monkeypatch.setattr(provider, "_post_sse", completed_post_sse)

    assert provider._collect_response_output_items({}) == []  # noqa: SLF001


def test_output_item_collector_stops_reading_after_response_completed(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def post_sse_with_trailing_failure(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.completed", "response": {"id": "response-1"}}
        raise AssertionError("events after response.completed must not be consumed")

    monkeypatch.setattr(provider, "_post_sse", post_sse_with_trailing_failure)

    assert provider._collect_response_output_items({}) == []  # noqa: SLF001


@pytest.mark.parametrize("response", [None, [], {}, {"id": ""}, {"id": 42}])
def test_chat_stream_rejects_malformed_response_completed(response, monkeypatch):
    provider = ChatGPTOAuthProvider()

    def malformed_post_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.completed", "response": response}

    monkeypatch.setattr(provider, "_post_sse", malformed_post_sse)

    with pytest.raises(ChatGPTOAuthError, match="response with a non-empty id"):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))


@pytest.mark.parametrize("item", [None, [], "not-an-object"])
def test_chat_stream_rejects_malformed_output_item_done_without_committing(item, monkeypatch):
    provider = ChatGPTOAuthProvider()

    def malformed_post_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.output_item.done", "item": item}
        yield {"type": "response.completed", "response": {"id": "response-1"}}

    monkeypatch.setattr(provider, "_post_sse", malformed_post_sse)

    with pytest.raises(ChatGPTOAuthError):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages(),
            model="gpt-5.6-sol",
            previous_response_id="response-1",
        )


def test_output_item_collector_rejects_malformed_response_completed(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def malformed_post_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.completed", "response": {"id": ""}}

    monkeypatch.setattr(provider, "_post_sse", malformed_post_sse)

    with pytest.raises(ChatGPTOAuthError, match="response with a non-empty id"):
        provider._collect_response_output_items({})  # noqa: SLF001


def test_codex_cli_headers_include_official_originator_and_versioned_user_agent():
    headers = codex_cli_headers_for_version("1.2.3\n")

    assert headers["originator"] == "codex_cli_rs"
    assert headers["User-Agent"].startswith("codex_cli_rs/1.2.3 (")
    assert headers["User-Agent"].endswith(") codex-as-api")


def test_codex_cli_headers_omit_user_agent_for_invalid_version():
    headers = codex_cli_headers_for_version("not-a-version")

    assert headers == {"originator": "codex_cli_rs"}


def test_provider_headers_include_codex_cli_headers(auth_json_factory, monkeypatch):
    monkeypatch.setenv("CODEX_AS_API_CODEX_CLI_VERSION", "9.8.7")
    provider = ChatGPTOAuthProvider(auth_json_path=str(auth_json_factory()))

    headers = provider._headers()  # noqa: SLF001 - regression test for backend request headers

    assert headers["originator"] == "codex_cli_rs"
    assert headers["User-Agent"].startswith("codex_cli_rs/9.8.7 (")
    assert headers["User-Agent"].endswith(") codex-as-api")
    assert headers["Authorization"].startswith("Bearer ")


# ---------------------------------------------------------------------------
# _set_reasoning_payload
# ---------------------------------------------------------------------------


def test_set_reasoning_payload_valid_effort():
    payload: dict = {}
    _set_reasoning_payload(payload, "high")
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["include"] == ["reasoning.encrypted_content"]


def test_set_reasoning_payload_none_is_noop():
    payload: dict = {}
    _set_reasoning_payload(payload, None)
    assert "reasoning" not in payload


def test_set_reasoning_payload_standard_mode_is_omitted_from_private_wire():
    payload: dict = {}

    _set_reasoning_payload(payload, None, reasoning_mode="standard", model="gpt-5.6-sol")

    assert "reasoning" not in payload
    assert "include" not in payload


def test_set_reasoning_payload_pro_mode_is_rejected():
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        _set_reasoning_payload({}, None, reasoning_mode="pro", model="gpt-5.6-sol")


def test_set_reasoning_payload_preserves_case_for_custom_values():
    payload: dict = {}
    _set_reasoning_payload(payload, "HIGH")
    assert payload["reasoning"]["effort"] == "HIGH"


def test_set_reasoning_payload_ultra_uses_max_on_wire():
    payload: dict = {}
    _set_reasoning_payload(payload, "ultra")
    assert payload["reasoning"] == {"effort": "max"}

    custom_case_payload: dict = {}
    _set_reasoning_payload(custom_case_payload, "ULTRA")
    assert custom_case_payload["reasoning"] == {"effort": "ULTRA"}


def test_set_reasoning_payload_custom_effort_is_preserved():
    payload: dict = {}
    _set_reasoning_payload(payload, "future-Effort")
    assert payload["reasoning"] == {"effort": "future-Effort"}


def test_set_reasoning_payload_empty_string_raises():
    with pytest.raises(ChatGPTOAuthError):
        _set_reasoning_payload({}, "")


def test_set_reasoning_payload_all_known_wire_efforts():
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        payload: dict = {}
        _set_reasoning_payload(payload, effort)
        assert payload["reasoning"]["effort"] == effort


def test_responses_lite_hosted_web_search_fails_loudly():
    provider = ChatGPTOAuthProvider()
    web_search_tool = ToolSchema(
        name="web_search",
        description="Web search",
        parameters={
            "__codex_as_api_tool_type": "web_search",
            "openai_tool": {"type": "web_search", "external_web_access": True},
        },
    )

    with pytest.raises(ChatGPTOAuthError):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages(),
            model="gpt-5.6-sol",
            tools=[web_search_tool],
        )


def test_responses_lite_hosted_image_generation_fails_loudly():
    provider = ChatGPTOAuthProvider()

    with pytest.raises(ChatGPTOAuthError):
        provider.generate_image("draw a cat", model="gpt-5.6-sol")


def test_classic_compact_omits_instructions_without_base_system_message(monkeypatch):
    provider = ChatGPTOAuthProvider()
    captured: dict[str, object] = {}

    def fake_post_json(path, payload, extra_headers=None):  # noqa: ANN001
        captured["path"] = path
        captured["payload"] = payload
        captured["headers"] = extra_headers
        return {
            "output": [{"type": "message", "role": "assistant", "content": []}],
        }

    monkeypatch.setattr(provider, "_post_json", fake_post_json)

    provider.compact_messages(
        [Message(role=MessageRole.USER, content="hello")],
        model="gpt-5.6-sol",
        responses_lite=False,
    )

    assert captured["path"] == "/responses/compact"
    assert "instructions" not in captured["payload"]


@pytest.mark.parametrize("tool_choice", ["required", {"type": "function", "name": "lookup"}, ""])
def test_responses_lite_rejects_non_auto_tool_choice(tool_choice):
    provider = ChatGPTOAuthProvider()

    with pytest.raises(ChatGPTOAuthError, match="exact string 'auto'"):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages(),
            model="gpt-5.6-sol",
            tool_choice=tool_choice,
        )


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
    items = [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "msg text"}],
        }
    ]
    assert _text_from_response_items(items) == "msg text"


def test_text_from_message_with_string_content_parts():
    items = [
        {
            "type": "message",
            "content": ["part one", "part two"],
        }
    ]
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


def test_usage_from_response_cache_write_tokens_from_details():
    u = _usage_from_response(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": {
                "cached_tokens": 30,
                "cache_write_tokens": 40,
            },
        }
    )

    assert u is not None
    assert u.cached_tokens == 30
    assert u.cache_write_tokens == 40


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
    compacted_content = (
        REMOTE_COMPACTION_MARKER
        + "\n"
        + json.dumps(
            [
                {"type": "message", "role": "assistant", "content": []},
            ]
        )
    )
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


def test_messages_to_response_items_tool_without_optional_name():
    messages = [Message(role=MessageRole.TOOL, content="result", tool_call_id="tc-1")]
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


def test_compacted_history_filter_matches_installed_replacement_history():
    real_user = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "continue from here"}],
    }
    assistant = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "checkpoint"}],
    }
    retained = [
        real_user,
        assistant,
        {"type": "agent_message", "author": "/root", "recipient": "/root/worker", "content": []},
        {"type": "compaction", "encrypted_content": "current"},
        {"type": "compaction_summary", "encrypted_content": "legacy"},
        {"type": "context_compaction", "encrypted_content": "context"},
    ]
    dropped = [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "stale instructions"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "<environment_context>stale</environment_context>"}],
        },
        {"type": "additional_tools", "role": "developer", "tools": []},
        {"type": "reasoning", "summary": []},
        {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}"},
    ]

    assert _filter_compacted_history_items([*dropped, *retained]) == retained


def test_compacted_history_context_wrappers_require_complete_canonical_shape():
    uppercase_context = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "<ENVIRONMENT_CONTEXT>stale</ENVIRONMENT_CONTEXT>"}],
    }
    dynamic_context = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "<external_design>stale</external_design>"}],
    }
    malformed_opening = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "<environment_context>not a complete wrapper"}],
    }
    user_discussion = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Why does <environment_context> exist?"}],
    }
    hook_prompt = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": '<hook_prompt hook_run_id="run-1">retry</hook_prompt>'}],
    }
    hook_with_image = {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": '<hook_prompt hook_run_id="run-1">retry</hook_prompt>'},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ],
    }

    assert _filter_compacted_history_items(
        [
            uppercase_context,
            dynamic_context,
            malformed_opening,
            user_discussion,
            hook_prompt,
            hook_with_image,
        ]
    ) == [malformed_opening, user_discussion, hook_prompt]


def test_compaction_marker_replay_revalidates_installed_history():
    kept = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "checkpoint"}],
    }
    crafted = [
        {"type": "message", "role": "developer", "content": []},
        {"type": "reasoning", "summary": []},
        kept,
    ]
    marker = REMOTE_COMPACTION_MARKER + "\n" + json.dumps(crafted)

    assert _messages_to_response_items([Message(role=MessageRole.SYSTEM, content=marker)]) == [kept]


def test_compaction_marker_rejects_non_object_items():
    marker = (
        REMOTE_COMPACTION_MARKER
        + "\n"
        + json.dumps(
            [
                {"type": "agent_message", "author": "/root", "recipient": "/worker", "content": []},
                "invalid",
            ]
        )
    )

    with pytest.raises(ChatGPTOAuthError, match="marker item 1 must be an object"):
        _messages_to_response_items([Message(role=MessageRole.SYSTEM, content=marker)])


def test_compaction_marker_rejects_invalid_json():
    marker = REMOTE_COMPACTION_MARKER + "\nnot-json"

    with pytest.raises(ChatGPTOAuthError, match="marker contains invalid JSON"):
        _messages_to_response_items([Message(role=MessageRole.SYSTEM, content=marker)])


@pytest.mark.parametrize(
    "malformed",
    [
        {"role": "assistant", "content": []},
        {"type": "message", "role": "assistant"},
        {"type": "message", "role": "assistant", "content": "text"},
        {"type": "message", "role": "assistant", "content": ["text"]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text"}]},
    ],
)
def test_compacted_history_rejects_malformed_message_items(malformed):
    with pytest.raises(ChatGPTOAuthError, match="remote compact output (?:item|message item) 0"):
        _filter_compacted_history_items([malformed])


@pytest.mark.parametrize(
    "malformed",
    [
        {"type": "agent_message", "author": "/root", "content": []},
        {"type": "agent_message", "author": "/root", "recipient": "/worker", "content": [42]},
        {
            "type": "agent_message",
            "author": "/root",
            "recipient": "/worker",
            "content": [{"type": "encrypted_content", "encrypted_content": 42}],
        },
        {"type": "compaction"},
        {"type": "compaction_summary", "encrypted_content": None},
        {"type": "context_compaction", "encrypted_content": 42},
    ],
)
def test_compacted_history_rejects_malformed_retained_variants(malformed):
    with pytest.raises(ChatGPTOAuthError):
        _filter_compacted_history_items([malformed])


@pytest.mark.parametrize("detail", ["auto", "low", "high", "original", None])
def test_compacted_history_accepts_supported_image_detail(detail):
    image = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": detail}],
    }

    assert _filter_compacted_history_items([image]) == [image]


def test_compacted_history_rejects_invalid_image_detail():
    image = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "full"}],
    }

    with pytest.raises(ChatGPTOAuthError, match="content part 0 is invalid"):
        _filter_compacted_history_items([image])


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


def test_tool_schema_to_response_dict_web_search():
    tool = ToolSchema(
        name="web_search",
        description="Web search",
        parameters={
            "__codex_as_api_tool_type": "web_search",
            "openai_tool": {
                "type": "web_search",
                "external_web_access": True,
                "filters": {"allowed_domains": ["example.com"]},
            },
        },
    )
    assert _tool_schema_to_response_dict(tool) == {
        "type": "web_search",
        "external_web_access": True,
        "filters": {"allowed_domains": ["example.com"]},
    }


def test_web_search_event_from_response_item_action_sources():
    result = _web_search_event_from_response_item(
        {
            "type": "web_search_call",
            "id": "ws_1",
            "action": {
                "type": "search",
                "query": "hello",
                "sources": [{"url": "https://example.com", "title": "Example", "page_age": "today"}],
            },
        }
    )
    assert result is not None
    assert result["id"] == "srvtoolu_ws_1"
    assert result["input"] == {"query": "hello"}
    assert result["content"] == [
        {
            "type": "web_search_result",
            "url": "https://example.com",
            "title": "Example",
            "page_age": "today",
        }
    ]


def test_web_search_event_from_response_item_annotation_fallback():
    result = _web_search_event_from_response_item(
        {"type": "web_search_call", "id": "ws_1", "action": {"type": "search", "queries": ["q"]}},
        [
            {"type": "web_search_call", "id": "ws_1"},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "answer",
                        "annotations": [{"type": "url_citation", "url": "https://a.test", "title": "A"}],
                    }
                ],
            },
        ],
    )
    assert result is not None
    assert result["content"] == [
        {
            "type": "web_search_result",
            "url": "https://a.test",
            "title": "A",
        }
    ]


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


@pytest.mark.parametrize("value", ["[]", '"event"', "42", "null"])
def test_decode_sse_block_rejects_non_object_json(value):
    with pytest.raises(ChatGPTOAuthError, match="SSE event JSON must be an object"):
        _decode_sse_block([f"data: {value}"])


def test_decode_sse_block_strips_data_prefix():
    lines = ['data:   {"k": "v"}']
    event = _decode_sse_block(lines)
    assert event == {"k": "v"}
