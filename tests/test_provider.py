from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
from copy import deepcopy
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import codex_as_api.provider as provider_module
from codex_as_api.auth import (
    ChatGPTOAuthCatalogUnavailableError,
    ChatGPTOAuthError,
    ChatGPTOAuthInvalidRequestError,
    ChatGPTOAuthMissingError,
    ChatGPTOAuthModelNotFoundError,
    ChatGPTOAuthProtocolError,
    ChatGPTOAuthRefreshError,
    ChatGPTOAuthUpstreamError,
)
from codex_as_api.messages import Message, MessageRole, ToolCall, ToolSchema
from codex_as_api.model_capabilities import (
    RESPONSES_LITE_ENV,
    CatalogLoadResult,
    ModelReasoningLevel,
)
from codex_as_api.provider import (
    REMOTE_COMPACTION_MARKER,
    RESPONSE_CHAIN_CAPACITY,
    ChatGPTOAuthProvider,
    _apply_responses_lite_payload,
    _decode_sse_block,
    _filter_compacted_history_items,
    _has_any_image_detail,
    _has_input_images,
    _has_original_image_detail,
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
    _validate_response_event,
    _web_search_event_from_response_item,
    codex_cli_headers_for_version,
    resolve_codex_cli_version,
    resolve_model_reasoning_effort,
)


@pytest.fixture(autouse=True)
def _isolate_responses_lite_mode(monkeypatch, model_catalog_snapshot):
    monkeypatch.setenv(RESPONSES_LITE_ENV, "auto")
    monkeypatch.setattr(
        ChatGPTOAuthProvider,
        "get_model_catalog",
        lambda _self: model_catalog_snapshot,
    )


def _provider_messages() -> list[Message]:
    return [
        Message(role=MessageRole.SYSTEM, content="You are helpful."),
        Message(role=MessageRole.USER, content="Hello"),
    ]


def _provider_usage(input_tokens: int = 1, output_tokens: int = 1) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": 0},
    }


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


def test_image_capability_scans_only_responses_message_content():
    opaque_image_shape = {"type": "input_image", "detail": "original"}
    payload = {
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        "tools": [{"type": "function", "parameters": {"example": opaque_image_shape}}],
        "client_metadata": {"example": opaque_image_shape},
    }
    assert not _has_input_images(payload)
    assert not _has_any_image_detail(payload)
    assert not _has_original_image_detail(payload)

    payload["input"][0]["content"].append(
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "original"}
    )
    assert _has_input_images(payload)
    assert _has_any_image_detail(payload)
    assert _has_original_image_detail(payload)


# ---------------------------------------------------------------------------
# ChatGPTOAuthProvider payload
# ---------------------------------------------------------------------------


def test_responses_payload_rejects_unmapped_max_tokens():
    provider = ChatGPTOAuthProvider()

    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="max_tokens"):
        provider._responses_payload(  # noqa: SLF001 - regression test for backend payload
            _provider_messages(),
            model="gpt-5.5",
            max_tokens=1024,
        )


@pytest.mark.parametrize("stop", ["", [], [""], ["", ""]])
def test_chat_stream_rejects_explicit_empty_stop_without_forwarding(stop, monkeypatch):
    provider = ChatGPTOAuthProvider()

    def forbidden_sse(*_args, **_kwargs):
        raise AssertionError("explicit stop must fail before upstream")

    monkeypatch.setattr(provider, "_post_sse", forbidden_sse)

    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="stop is not supported"):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.5", stop=stop))


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
    ["gpt-5.2", "gpt-5.3-codex", "gpt-5.3-codex-spark"],
)
def test_responses_payload_rejects_original_image_detail_for_unsupported_model(model: str):
    provider = ChatGPTOAuthProvider()

    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="image detail"):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages_with_image_detail("original"),
            model=model,
            responses_lite=False,
        )


def test_responses_payload_rejects_unknown_model_before_capability_checks():
    provider = ChatGPTOAuthProvider()

    with pytest.raises(ChatGPTOAuthModelNotFoundError):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages_with_image_detail("original"),
            model="future-model",
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

    assert payload["reasoning"] == {"effort": "high", "summary": "auto"}
    assert payload["include"] == ["reasoning.encrypted_content"]


def test_responses_payload_forces_responses_lite_shape():
    provider = ChatGPTOAuthProvider()
    tool = ToolSchema(name="lookup", description="Lookup", parameters={"type": "object"})

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.6-sol",
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


def test_explicit_responses_lite_on_does_not_require_live_auto_flag():
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        responses_lite=True,
    )

    assert "tools" not in payload
    assert payload["input"][0] == {
        "type": "additional_tools",
        "role": "developer",
        "tools": [],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"instructions": 1, "input": [], "tools": []},
        {"instructions": "system", "input": None, "tools": []},
        {"instructions": "system", "input": [], "tools": None},
        {"instructions": "system", "input": [], "tools": [], "reasoning": None},
    ],
)
def test_responses_lite_rejects_malformed_internal_shape(payload):
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        _apply_responses_lite_payload(payload, [])


def test_responses_lite_absent_reasoning_gets_official_context():
    payload = {"instructions": "system", "input": [], "tools": []}

    _apply_responses_lite_payload(payload, [])

    assert payload["reasoning"] == {"context": "all_turns"}


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
    assert payload["reasoning"] == {
        "effort": "low",
        "summary": "auto",
        "context": "all_turns",
    }


def test_forced_lite_unknown_model_is_rejected():
    provider = ChatGPTOAuthProvider()

    with pytest.raises(ChatGPTOAuthModelNotFoundError):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages(),
            model="unknown-model",
            responses_lite=True,
        )


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
        "summary": "auto",
        "context": "all_turns",
    }


def test_standard_reasoning_mode_is_rejected_instead_of_silently_omitted():
    provider = ChatGPTOAuthProvider()

    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="not supported"):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages(),
            model="gpt-5.6-sol",
            reasoning_mode="standard",
            responses_lite=False,
        )


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
    assert payload["text"] == {"verbosity": "medium"}
    assert payload["service_tier"] == "priority"
    assert fast_payload["service_tier"] == "priority"
    with pytest.raises(ChatGPTOAuthModelNotFoundError):
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


@pytest.mark.parametrize(
    "text",
    [
        {"future": True},
        {"format": "json"},
        {"format": {"type": "json_object", "future": True}},
        {"format": {"type": "json_schema", "schema": []}},
    ],
)
def test_text_options_reject_malformed_or_unknown_fields(text):
    provider = ChatGPTOAuthProvider()

    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages(),
            model="gpt-5.5",
            text=text,
        )


def test_null_text_verbosity_is_omitted_then_live_default_is_applied():
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        text={"verbosity": None},
    )

    assert payload["text"] == {"verbosity": "medium"}


def test_text_format_is_forwarded_when_model_does_not_support_verbosity(
    monkeypatch,
    model_catalog_snapshot,
):
    provider = ChatGPTOAuthProvider()
    capability = replace(
        model_catalog_snapshot.model("gpt-5.5"),
        support_verbosity=False,
        default_verbosity=None,
    )
    monkeypatch.setattr(
        provider,
        "resolve_model",
        lambda _requested=None: (model_catalog_snapshot, capability),
    )

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        text={"format": {"type": "json_object"}, "verbosity": None},
    )

    assert payload["text"] == {"format": {"type": "json_object"}}


def test_verbosity_requires_live_model_support(
    monkeypatch,
    model_catalog_snapshot,
):
    provider = ChatGPTOAuthProvider()
    capability = replace(
        model_catalog_snapshot.model("gpt-5.5"),
        support_verbosity=False,
        default_verbosity=None,
    )
    monkeypatch.setattr(
        provider,
        "resolve_model",
        lambda _requested=None: (model_catalog_snapshot, capability),
    )

    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="verbosity"):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages(),
            model="gpt-5.5",
            verbosity="high",
        )


def test_responses_payload_parallel_tool_calls_uses_transport_mode(model_catalog_snapshot):
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        parallel_tool_calls=True,
    )
    lite_payload = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.6-sol",
        parallel_tool_calls=False,
        responses_lite=True,
    )

    assert payload["parallel_tool_calls"] is True
    assert lite_payload["parallel_tool_calls"] is False
    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="Responses Lite"):
        provider._responses_payload(  # noqa: SLF001
            _provider_messages(),
            model="gpt-5.6-sol",
            parallel_tool_calls=True,
            responses_lite=True,
        )


def test_live_input_modalities_gate_only_present_content(
    monkeypatch,
    model_catalog_snapshot,
):
    provider = ChatGPTOAuthProvider()
    base = model_catalog_snapshot.model("gpt-5.5")
    image_only = replace(base, input_modalities=("image",))
    monkeypatch.setattr(
        provider,
        "resolve_model",
        lambda _requested=None: (model_catalog_snapshot, image_only),
    )
    image_messages = [
        Message(
            role=MessageRole.USER,
            content="",
            content_parts=(
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            ),
        )
    ]

    payload = provider._responses_payload(image_messages, model="image-only")  # noqa: SLF001
    assert payload["input"][0]["content"][0]["type"] == "input_image"
    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="text input"):
        provider._responses_payload(  # noqa: SLF001
            [Message(role=MessageRole.USER, content="text")],
            model="image-only",
        )

    audio_only = replace(base, input_modalities=("audio",))
    monkeypatch.setattr(
        provider,
        "resolve_model",
        lambda _requested=None: (model_catalog_snapshot, audio_only),
    )
    audio_messages = [
        Message(
            role=MessageRole.USER,
            content="",
            content_parts=(
                {"type": "input_audio", "audio_url": "data:audio/wav;base64,AAAA"},
            ),
        )
    ]
    payload = provider._responses_payload(audio_messages, model="audio-only")  # noqa: SLF001
    assert payload["input"][0]["content"][0]["type"] == "input_audio"


def test_image_generation_requires_live_image_capability(
    monkeypatch,
    model_catalog_snapshot,
):
    provider = ChatGPTOAuthProvider()
    text_only = replace(
        model_catalog_snapshot.model("gpt-5.5"),
        input_modalities=("text",),
    )
    monkeypatch.setattr(
        provider,
        "resolve_model",
        lambda _requested=None: (model_catalog_snapshot, text_only),
    )

    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="image generation"):
        provider.generate_image("Draw", model="text-only")


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


def test_codex_metadata_installation_id_uses_codex_home_auth_path(monkeypatch, tmp_path):
    codex_home = tmp_path / "custom-codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    options = {
        "model": "gpt-5.5",
        "client_metadata": {"session_id": "session-root"},
        "codex_metadata": True,
    }

    implicit = ChatGPTOAuthProvider()._responses_payload(_provider_messages(), **options)  # noqa: SLF001
    explicit = ChatGPTOAuthProvider(auth_json_path=str(codex_home / "auth.json"))._responses_payload(  # noqa: SLF001
        _provider_messages(),
        **options,
    )

    assert (
        implicit["client_metadata"]["x-codex-installation-id"]
        == explicit["client_metadata"]["x-codex-installation-id"]
    )


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
    assert fallback["prompt_cache_key"] == "session-cache"
    assert explicit["prompt_cache_key"] == "explicit-cache"
    assert "prompt_cache_key" not in absent
    for reserved_key in ("session_id", "thread_id"):
        with pytest.raises(ChatGPTOAuthInvalidRequestError, match=reserved_key):
            provider._responses_payload(  # noqa: SLF001
                _provider_messages(),
                model="gpt-5.5",
                prompt_cache_key="explicit-cache",
                client_metadata={reserved_key: "   ", "opaque": ""},
            )

    opaque_empty = provider._responses_payload(  # noqa: SLF001
        _provider_messages(),
        model="gpt-5.5",
        client_metadata={"opaque": ""},
    )
    assert opaque_empty["client_metadata"]["opaque"] == ""


def test_chat_stream_adds_responses_lite_header(monkeypatch):
    provider = ChatGPTOAuthProvider()
    captured: dict[str, object] = {}
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok"}],
        }
    ]

    def fake_post_sse(path, payload, extra_headers=None):  # noqa: ANN001
        captured["path"] = path
        captured["payload"] = payload
        captured["headers"] = extra_headers
        return iter(
            [
                {"type": "response.output_item.done", "item": output[0]},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "response-1",
                        "output": [],
                        "usage": _provider_usage(),
                    },
                },
            ]
        )

    monkeypatch.setattr(provider, "_post_sse", fake_post_sse)
    list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol", responses_lite=True))

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
            "response": {
                "id": "response-1",
                "output": [tool_call],
                "usage": _provider_usage(),
            },
        }

    monkeypatch.setattr(provider, "_post_sse", fake_post_sse)

    events = list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))
    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    finish = [event for event in events if event.get("type") == "finish"]

    assert [(event["id"], event["arguments"]) for event in tool_calls] == [
        ("call-1", '{"query":"one"}')
    ]
    assert [event["finish_reason"] for event in finish] == ["tool_calls"]


def _duplicate_tool_call_sse():
    yield {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "call_id": "duplicate-call",
            "name": "first",
            "arguments": "{}",
        },
    }
    yield {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "call_id": "duplicate-call",
            "name": "second",
            "arguments": "{}",
        },
    }


def test_chat_stream_rejects_duplicate_tool_call_ids(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def duplicate_post_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        return _duplicate_tool_call_sse()

    monkeypatch.setattr(provider, "_post_sse", duplicate_post_sse)

    with pytest.raises(ChatGPTOAuthProtocolError, match="duplicate call_id"):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))


def test_chat_rejects_duplicate_tool_call_ids(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def duplicate_post_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        return _duplicate_tool_call_sse()

    monkeypatch.setattr(provider, "_post_sse", duplicate_post_sse)

    with pytest.raises(ChatGPTOAuthProtocolError, match="duplicate call_id"):
        provider.chat(_provider_messages(), model="gpt-5.6-sol")


def _custom_tool_call_sse():
    yield {
        "type": "response.output_item.done",
        "item": {
            "type": "custom_tool_call",
            "call_id": "custom-call",
            "name": "shell",
            "input": '{"command":"pwd"}',
        },
    }


def test_chat_stream_rejects_custom_tool_call_output(monkeypatch):
    provider = ChatGPTOAuthProvider()
    monkeypatch.setattr(provider, "_post_sse", lambda *_args, **_kwargs: _custom_tool_call_sse())

    with pytest.raises(ChatGPTOAuthProtocolError, match="custom_tool_call"):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))


def test_chat_rejects_custom_tool_call_output(monkeypatch):
    provider = ChatGPTOAuthProvider()
    monkeypatch.setattr(provider, "_post_sse", lambda *_args, **_kwargs: _custom_tool_call_sse())

    with pytest.raises(ChatGPTOAuthProtocolError, match="custom_tool_call"):
        provider.chat(_provider_messages(), model="gpt-5.6-sol")


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
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok"}],
        }
    ]

    def post_sse_with_trailing_failure(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.output_item.done", "item": output[0]}
        yield {
            "type": "response.completed",
            "response": {
                "id": "response-1",
                "output": [],
                "usage": _provider_usage(),
            },
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
        output = (
            root_output
            if response_id == "resp-root"
            else [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "branch answer"}],
                }
            ]
        )
        for item in output:
            yield {"type": "response.output_item.done", "item": item}
        yield {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "output": output,
                "usage": _provider_usage(),
            },
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

    secret = "access-token-sentinel"
    with pytest.raises(ChatGPTOAuthInvalidRequestError) as caught:
        list(
            provider.chat_stream(
                _provider_messages(),
                model="gpt-5.5",
                previous_response_id=secret,
            )
        )
    assert secret not in str(caught.value)


def test_lite_replay_injects_one_current_developer_prefix(monkeypatch):
    provider = ChatGPTOAuthProvider()
    requests: list[dict] = []
    response_ids = iter(["resp-lite-root", "resp-lite-next"])

    def completed_sse(_path, payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        requests.append(deepcopy(payload))
        response_id = next(response_ids)
        item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": response_id}],
        }
        yield {"type": "response.output_item.done", "item": item}
        yield {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "output": [],
                "usage": _provider_usage(),
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
        store.commit(
            f"resp-{index}",
            [request_item],
            [output_item],
            account_id="account-a",
        )

    resolved = store.resolve("resp-0", account_id="account-a")
    resolved[1]["encrypted_content"] = "mutated"
    store.commit(
        "resp-overflow",
        [request_item],
        [output_item],
        account_id="account-a",
    )

    assert store.resolve("resp-0", account_id="account-a")[1]["encrypted_content"] == "original"
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        store.resolve("resp-1", account_id="account-a")


def test_response_chain_store_is_namespaced_by_account():
    store = _ResponseChainStore()
    store.commit(
        "resp-shared",
        [{"type": "message", "role": "user", "content": []}],
        [],
        account_id="account-a",
    )

    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        store.resolve("resp-shared", account_id="account-b")


def test_response_chain_store_requires_compaction_only_for_known_hash_mismatches():
    store = _ResponseChainStore()
    store.commit(
        "resp-hashed",
        [{"type": "message", "role": "user", "content": []}],
        [],
        account_id="account-a",
        comp_hash="family-a",
    )

    assert store.resolve(
        "resp-hashed",
        account_id="account-a",
        current_comp_hash="family-a",
    )
    assert store.resolve(
        "resp-hashed",
        account_id="account-a",
        current_comp_hash=None,
    )
    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="requires compaction"):
        store.resolve(
            "resp-hashed",
            account_id="account-a",
            current_comp_hash="family-b",
        )


def test_previous_response_comp_hash_is_wired_through_prepare_and_commit(
    monkeypatch,
    model_catalog_snapshot,
):
    provider = ChatGPTOAuthProvider()
    transport_calls = 0

    def completed_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        nonlocal transport_calls
        del extra_headers
        transport_calls += 1
        item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok"}],
        }
        yield {"type": "response.output_item.done", "item": item}
        yield {
            "type": "response.completed",
            "response": {"id": "resp-wired", "output": [], "end_turn": True},
        }

    monkeypatch.setattr(provider, "_post_sse", completed_sse)
    source = replace(model_catalog_snapshot.model("gpt-5.5"), comp_hash="family-a")
    prepared = provider.preflight_chat(
        _provider_messages(),
        responses_lite=False,
        _resolved_model=(model_catalog_snapshot, source),
    )
    provider.chat(_provider_messages(), _prepared_request=prepared)

    compatible = replace(model_catalog_snapshot.model("gpt-5.4"), comp_hash="family-a")
    compatible_request = provider.preflight_chat(
        _provider_messages(),
        previous_response_id="resp-wired",
        responses_lite=False,
        _resolved_model=(model_catalog_snapshot, compatible),
    )
    assert compatible_request.payload["model"] == "gpt-5.4"

    incompatible = replace(model_catalog_snapshot.model("gpt-5.4"), comp_hash="family-b")
    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="requires compaction"):
        provider.preflight_chat(
            _provider_messages(),
            previous_response_id="resp-wired",
            responses_lite=False,
            _resolved_model=(model_catalog_snapshot, incompatible),
        )
    assert transport_calls == 1


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


def test_output_item_collector_uses_done_items_and_ignores_completed_output(monkeypatch):
    provider = ChatGPTOAuthProvider()
    item = {"type": "message", "role": "assistant", "content": []}

    def completed_post_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.output_item.done", "item": item}
        yield {
            "type": "response.completed",
            "response": {
                "id": "response-1",
                "output": "not-authoritative",
                "usage": _provider_usage(),
            },
        }

    monkeypatch.setattr(provider, "_post_sse", completed_post_sse)

    assert provider._collect_response_output_items({}) == [  # noqa: SLF001
        item
    ]


@pytest.mark.parametrize("output", [None, []])
def test_output_item_collector_accepts_empty_completion_without_done_items(
    monkeypatch,
    output,
):
    provider = ChatGPTOAuthProvider()

    def completed_post_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        response = {"id": "response-1"}
        if output is not None:
            response["output"] = output
        yield {"type": "response.completed", "response": response}

    monkeypatch.setattr(provider, "_post_sse", completed_post_sse)

    assert provider._collect_response_output_items({}) == []  # noqa: SLF001


def test_output_item_collector_stops_reading_after_response_completed(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def post_sse_with_trailing_failure(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        item = {"type": "message", "role": "assistant", "content": []}
        yield {"type": "response.output_item.done", "item": item}
        yield {
            "type": "response.completed",
            "response": {
                "id": "response-1",
                "output": [],
                "usage": _provider_usage(output_tokens=0),
            },
        }
        raise AssertionError("events after response.completed must not be consumed")

    monkeypatch.setattr(provider, "_post_sse", post_sse_with_trailing_failure)

    assert provider._collect_response_output_items({}) == [  # noqa: SLF001
        {"type": "message", "role": "assistant", "content": []}
    ]


def test_output_item_collector_preserves_duplicate_done_items(monkeypatch):
    provider = ChatGPTOAuthProvider()
    item = {"type": "message", "role": "assistant", "content": []}

    def duplicate_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.output_item.done", "item": item}
        yield {"type": "response.output_item.done", "item": item}
        yield {"type": "response.completed", "response": {"id": "response-1"}}

    monkeypatch.setattr(provider, "_post_sse", duplicate_sse)

    assert provider._collect_response_output_items({}) == [item, item]  # noqa: SLF001


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


@pytest.mark.parametrize(
    "item",
    [
        {"type": "unknown"},
        {"type": "function_call", "call_id": "call-1", "name": "tool"},
        {"type": "image_generation_call", "id": "image-1", "status": "completed"},
        {"type": "web_search_call", "id": "search-1", "action": None},
        {"type": "message", "role": "user", "content": []},
    ],
)
def test_chat_stream_rejects_malformed_output_item_done_items(item, monkeypatch):
    provider = ChatGPTOAuthProvider()

    def malformed_post_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.output_item.done", "item": item}
        yield {
            "type": "response.completed",
            "response": {
                "id": "response-1",
                "output": [],
                "usage": _provider_usage(),
            },
        }

    monkeypatch.setattr(provider, "_post_sse", malformed_post_sse)

    with pytest.raises(ChatGPTOAuthProtocolError):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))


def test_chat_stream_rejects_valid_image_generation_items_in_normal_chat(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def image_post_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {
            "type": "response.output_item.done",
            "item": {
                "type": "image_generation_call",
                "id": "image-1",
                "status": "completed",
                "result": "data:image/png;base64,AAAA",
                "revised_prompt": None,
            },
        }

    monkeypatch.setattr(provider, "_post_sse", image_post_sse)

    with pytest.raises(ChatGPTOAuthProtocolError, match="cannot be represented by normal chat"):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))


@pytest.mark.parametrize(
    "event",
    [
        {"type": "response.output_text.delta", "delta": None},
        {"type": "response.reasoning_summary_text.delta", "delta": None},
        {"type": "response.reasoning_text.delta", "delta": 1},
    ],
)
def test_chat_stream_rejects_malformed_text_deltas(event, monkeypatch):
    provider = ChatGPTOAuthProvider()

    monkeypatch.setattr(
        provider,
        "_post_sse",
        lambda *_args, **_kwargs: iter([event]),
    )

    with pytest.raises(ChatGPTOAuthProtocolError, match="delta"):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))


def test_empty_upstream_deltas_are_valid_strings():
    for event in [
        {"type": "response.output_text.delta", "delta": ""},
        {
            "type": "response.reasoning_summary_text.delta",
            "delta": "",
            "summary_index": 0,
        },
        {"type": "response.reasoning_text.delta", "delta": "", "content_index": 0},
        {
            "type": "response.reasoning_summary_text.done",
            "item_id": "",
            "text": "",
            "summary_index": 0,
        },
    ]:
        assert _validate_response_event(event) is event


@pytest.mark.parametrize("optional_fields", [{}, {"encrypted_content": None}, {"encrypted_content": ""}])
def test_done_reasoning_accepts_optional_string_encrypted_content(optional_fields):
    event = {
        "type": "response.output_item.done",
        "item": {
            "type": "reasoning",
            "summary": [],
            **optional_fields,
        },
    }

    assert _validate_response_event(event) is event


@pytest.mark.parametrize("encrypted_content", [{}, 42, []])
def test_done_reasoning_rejects_non_string_encrypted_content(encrypted_content):
    event = {
        "type": "response.output_item.done",
        "item": {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": encrypted_content,
        },
    }

    with pytest.raises(ChatGPTOAuthProtocolError, match="encrypted_content"):
        _validate_response_event(event)


@pytest.mark.parametrize(
    "item",
    [
        {
            "type": "message",
            "id": "",
            "role": "assistant",
            "content": [],
            "phase": "final_answer",
            "internal_chat_message_metadata_passthrough": {
                "turn_id": None,
                "create_time": 1.5,
                "content_item_kinds": {"future": True},
            },
            "future": True,
        },
        {
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "text", "text": "raw", "future": True}],
            "encrypted_content": None,
            "future": True,
        },
        {
            "type": "function_call",
            "name": "",
            "call_id": "",
            "arguments": "not-json",
            "namespace": None,
            "encrypted_function_args": ["a", ""],
            "future": True,
        },
        {
            "type": "web_search_call",
            "id": "",
            "status": None,
            "action": {"type": "search", "query": "q", "sources": []},
            "future": True,
        },
        {
            "type": "image_generation_call",
            "id": "",
            "status": "",
            "result": "",
            "revised_prompt": None,
            "future": True,
        },
    ],
)
def test_response_item_optional_fields_accept_pinned_nullable_values_and_additive_fields(item):
    event = {"type": "response.output_item.done", "item": item}
    assert _validate_response_event(event) is event


@pytest.mark.parametrize(
    ("item", "field"),
    [
        ({"type": "message", "id": 1, "role": "assistant", "content": []}, "id"),
        (
            {
                "type": "message",
                "role": "assistant",
                "content": [],
                "internal_chat_message_metadata_passthrough": [],
            },
            "metadata_passthrough",
        ),
        (
            {
                "type": "message",
                "role": "assistant",
                "content": [],
                "internal_chat_message_metadata_passthrough": {"turn_id": 1},
            },
            "turn_id",
        ),
        (
            {
                "type": "message",
                "role": "assistant",
                "content": [],
                "internal_chat_message_metadata_passthrough": {"create_time": "now"},
            },
            "create_time",
        ),
        ({"type": "message", "role": "assistant", "content": [], "phase": "future"}, "phase"),
        (
            {
                "type": "function_call",
                "name": "f",
                "call_id": "c",
                "arguments": "{}",
                "namespace": 1,
            },
            "namespace",
        ),
        (
            {
                "type": "function_call",
                "name": "f",
                "call_id": "c",
                "arguments": "{}",
                "encrypted_function_args": [1],
            },
            "encrypted_function_args",
        ),
        (
            {
                "type": "web_search_call",
                "id": "search",
                "status": 1,
                "action": {"type": "search", "query": "q", "sources": []},
            },
            "status",
        ),
        (
            {"type": "image_generation_call", "status": "", "result": "", "revised_prompt": 1},
            "revised_prompt",
        ),
    ],
)
def test_response_item_optional_fields_reject_malformed_known_values(item, field):
    with pytest.raises(ChatGPTOAuthProtocolError, match=field):
        _validate_response_event({"type": "response.output_item.done", "item": item})


def test_added_response_item_applies_common_optional_field_validation():
    with pytest.raises(ChatGPTOAuthProtocolError, match="metadata_passthrough"):
        _validate_response_event(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "internal_chat_message_metadata_passthrough": "bad",
                },
            }
        )


@pytest.mark.parametrize(
    "event_type",
    [
        "response.file_search_call.in_progress",
        "response.file_search_call.searching",
        "response.file_search_call.completed",
        "response.code_interpreter_call.in_progress",
        "response.code_interpreter_call.interpreting",
        "response.code_interpreter_call_code.delta",
        "response.code_interpreter_call_code.done",
        "response.code_interpreter_call.completed",
        "response.mcp_call.in_progress",
        "response.mcp_call_arguments.delta",
        "response.mcp_call_arguments.done",
        "response.mcp_call.completed",
        "response.mcp_call.failed",
        "response.mcp_list_tools.in_progress",
        "response.mcp_list_tools.completed",
        "response.mcp_list_tools.failed",
        "response.shell_call_command.added",
        "response.shell_call_command.delta",
        "response.shell_call_command.done",
        "response.shell_call_output_content.delta",
        "response.shell_call_output_content.done",
        "response.audio.delta",
        "response.audio.done",
        "response.audio.transcript.delta",
        "response.audio.transcript.done",
        "response.refusal.delta",
        "response.refusal.done",
        "response.output_text.annotation.added",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
    ],
)
def test_known_unsupported_semantic_event_types_fail_immediately(event_type):
    with pytest.raises(ChatGPTOAuthProtocolError, match="unsupported semantic type"):
        _validate_response_event({"type": event_type})


@pytest.mark.parametrize(
    "item_type",
    [
        "tool_search_call",
        "computer_call",
        "file_search_call",
        "code_interpreter_call",
        "mcp_call",
        "local_shell_call",
    ],
)
def test_unknown_added_semantic_items_fail_immediately(item_type):
    with pytest.raises(ChatGPTOAuthProtocolError, match="unsupported type"):
        _validate_response_event(
            {
                "type": "response.output_item.added",
                "item": {"type": item_type},
            }
        )


@pytest.mark.parametrize("event_type", ["response.content_part.added", "response.content_part.done"])
def test_content_part_events_accept_output_text_and_reject_unrepresentable_parts(event_type):
    assert _validate_response_event(
        {
            "type": event_type,
            "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
        }
    )["type"] == event_type

    for part in (
        None,
        {"type": "refusal", "refusal": "blocked"},
        {"type": "future_content", "value": "opaque"},
        {"type": "output_text"},
        {"type": "output_text", "text": "ok", "annotations": {}},
        {"type": "output_text", "text": "ok", "logprobs": {}},
    ):
        with pytest.raises(ChatGPTOAuthProtocolError):
            _validate_response_event({"type": event_type, "part": part})


def test_chat_stream_ignores_unknown_events_before_a_valid_completion(monkeypatch):
    provider = ChatGPTOAuthProvider()
    events = [
        {"type": "", "opaque": "ignored"},
        {"type": "response.future_telemetry", "opaque": True},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
        },
        {"type": "response.completed", "response": {"id": "response-1"}},
    ]
    monkeypatch.setattr(provider, "_post_sse", lambda *_args, **_kwargs: iter(events))

    emitted = list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))
    assert [event["type"] for event in emitted] == ["content", "finish"]
    assert emitted[0]["text"] == "done"


def test_nonstream_chat_accepts_empty_normalized_text_and_reasoning_deltas(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def normalized_events(*_args, **_kwargs):
        yield {"type": "content", "text": ""}
        yield {"type": "reasoning_delta", "text": ""}
        yield {"type": "reasoning_raw_delta", "text": ""}
        yield {
            "type": "finish",
            "finish_reason": "stop",
            "reasoning_content": None,
            "response_id": "response-1",
        }

    monkeypatch.setattr(provider, "chat_stream", normalized_events)
    response = provider.chat(_provider_messages(), model="gpt-5.6-sol")
    assert response.content == ""
    assert response.reasoning_content is None


def test_streamed_text_must_match_done_item_text(monkeypatch):
    provider = ChatGPTOAuthProvider()
    events = [
        {"type": "response.output_text.delta", "delta": "streamed"},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "final"}],
            },
        },
        {
            "type": "response.completed",
            "response": {"id": "response-1", "usage": _provider_usage()},
        },
    ]
    monkeypatch.setattr(provider, "_post_sse", lambda *_args, **_kwargs: iter(events))

    with pytest.raises(ChatGPTOAuthProtocolError, match="does not match"):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))


@pytest.mark.parametrize(
    ("delta_event", "reasoning_item", "message"),
    [
        (
            {"type": "response.reasoning_summary_text.delta", "delta": "streamed", "summary_index": 0},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "done"}]},
            "reasoning summary",
        ),
        (
            {"type": "response.reasoning_text.delta", "delta": "streamed", "content_index": 0},
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "done"}],
            },
            "reasoning content",
        ),
    ],
)
def test_streamed_reasoning_families_must_match_done_items(
    delta_event,
    reasoning_item,
    message,
    monkeypatch,
):
    provider = ChatGPTOAuthProvider()
    events = [
        delta_event,
        {"type": "response.output_item.done", "item": reasoning_item},
        {"type": "response.completed", "response": {"id": "response-reasoning-mismatch"}},
    ]
    monkeypatch.setattr(provider, "_post_sse", lambda *_args, **_kwargs: iter(events))

    with pytest.raises(ChatGPTOAuthProtocolError, match=message):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        provider._response_chains.resolve("response-reasoning-mismatch", account_id="test-account")


@pytest.mark.parametrize(
    "event",
    [
        {"type": "response.created"},
        {"type": "response.output_item.added", "item": None},
        {"type": "response.custom_tool_call_input.delta", "delta": ""},
        {
            "type": "response.reasoning_summary_text.delta",
            "delta": "summary",
        },
        {
            "type": "response.reasoning_summary_text.done",
            "item_id": "reasoning-1",
            "text": "summary",
        },
        {
            "type": "response.reasoning_text.delta",
            "delta": "raw",
            "summary_index": 0,
        },
        {"type": "response.reasoning_summary_part.added", "part_index": 0},
    ],
)
def test_chat_stream_rejects_malformed_consumed_lifecycle_events(event, monkeypatch):
    provider = ChatGPTOAuthProvider()
    monkeypatch.setattr(
        provider,
        "_post_sse",
        lambda *_args, **_kwargs: iter([event]),
    )

    with pytest.raises(ChatGPTOAuthProtocolError):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))


def test_chat_stream_uses_official_reasoning_event_indexes(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def reasoning_events(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.metadata", "metadata": {"trace": "ignored"}}
        yield {"type": "codex.response.metadata", "headers": {}}
        yield {"type": "responsesapi.websocket_timing", "elapsed_ms": 1}
        yield {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "call_id": "call-1",
                "name": "apply_patch",
                "arguments": "",
            },
        }
        yield {
            "type": "response.output_item.added",
            "item": {
                "type": "web_search_call",
                "id": "search-1",
                "status": "searching",
            },
        }
        yield {
            "type": "response.reasoning_summary_part.added",
            "summary_index": 2,
        }
        yield {
            "type": "response.reasoning_text.delta",
            "delta": "raw",
            "content_index": 3,
        }
        yield {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "text", "text": "raw"}],
            },
        }
        yield {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}],
            },
        }
        yield {
            "type": "response.completed",
            "response": {"id": "response-1", "usage": _provider_usage()},
        }

    monkeypatch.setattr(provider, "_post_sse", reasoning_events)

    events = list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))

    assert {
        "type": "reasoning_section_break",
        "summary_index": 2,
    } in events
    assert {
        "type": "reasoning_raw_delta",
        "text": "raw",
        "content_index": 3,
    } in events


def test_chat_stream_accepts_empty_assistant_response(
    monkeypatch,
):
    provider = ChatGPTOAuthProvider()

    def completed(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {
            "type": "response.output_item.done",
            "item": {"type": "message", "role": "assistant", "content": []},
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "response-1",
                "output": [],
                "usage": _provider_usage(output_tokens=0),
            },
        }

    monkeypatch.setattr(provider, "_post_sse", completed)

    events = list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))

    assert events == [
        {
            "type": "finish",
            "finish_reason": "stop",
            "reasoning_content": None,
            "response_id": "response-1",
            "usage": _provider_usage(output_tokens=0),
        }
    ]


@pytest.mark.parametrize(("end_turn", "expected"), [(None, "stop"), (False, None), (True, "stop")])
def test_completed_end_turn_controls_text_finish_reason(monkeypatch, end_turn, expected):
    provider = ChatGPTOAuthProvider()
    response = {"id": "response-1", "usage": _provider_usage()}
    if end_turn is not None:
        response["end_turn"] = end_turn
    events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
        },
        {"type": "response.completed", "response": response},
    ]
    monkeypatch.setattr(provider, "_post_sse", lambda *_args, **_kwargs: iter(events))

    normalized = list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))
    assert normalized[-1]["finish_reason"] == expected


@pytest.mark.parametrize("end_turn", [0, "true", [], {}])
def test_completed_rejects_malformed_end_turn(monkeypatch, end_turn):
    provider = ChatGPTOAuthProvider()
    monkeypatch.setattr(
        provider,
        "_post_sse",
        lambda *_args, **_kwargs: iter(
            [{"type": "response.completed", "response": {"id": "response-1", "end_turn": end_turn}}]
        ),
    )

    with pytest.raises(ChatGPTOAuthProtocolError, match="end_turn"):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))


def test_chat_stream_rejects_streamed_text_without_matching_done_item(
    monkeypatch,
):
    provider = ChatGPTOAuthProvider()

    def completed(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.output_text.delta", "delta": "partial"}
        yield {"type": "response.completed", "response": {"id": "response-1"}}

    monkeypatch.setattr(provider, "_post_sse", completed)

    with pytest.raises(ChatGPTOAuthProtocolError, match="does not match"):
        list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))


@pytest.mark.parametrize("output", [None, []])
def test_chat_stream_accepts_completed_without_done_items_for_empty_output(
    monkeypatch,
    output,
):
    provider = ChatGPTOAuthProvider()

    def completed(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        response = {"id": "response-1", "usage": _provider_usage(output_tokens=0)}
        if output is not None:
            response["output"] = output
        yield {"type": "response.completed", "response": response}

    monkeypatch.setattr(provider, "_post_sse", completed)

    events = list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))
    assert events[-1]["type"] == "finish"


def test_chat_stream_ignores_additive_completed_output_without_done_items(
    monkeypatch,
):
    provider = ChatGPTOAuthProvider()

    def completed(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {
            "type": "response.completed",
            "response": {
                "id": "response-1",
                "output": [{"type": "message", "role": "assistant", "content": []}],
            },
        }

    monkeypatch.setattr(provider, "_post_sse", completed)

    events = list(provider.chat_stream(_provider_messages(), model="gpt-5.6-sol"))
    assert events[-1]["type"] == "finish"


def test_output_item_collector_rejects_malformed_response_completed(monkeypatch):
    provider = ChatGPTOAuthProvider()

    def malformed_post_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.completed", "response": {"id": ""}}

    monkeypatch.setattr(provider, "_post_sse", malformed_post_sse)

    with pytest.raises(ChatGPTOAuthError, match="response with a non-empty id"):
        provider._collect_response_output_items({})  # noqa: SLF001


def test_codex_cli_headers_include_official_originator_and_versioned_user_agent():
    headers = codex_cli_headers_for_version("0.153.3")

    assert headers["originator"] == "codex_cli_rs"
    assert headers["User-Agent"].startswith("codex_cli_rs/0.153.3 (")
    assert headers["User-Agent"].endswith(") codex-as-api/0.7.0")


def test_codex_cli_headers_reject_unpinned_version():
    with pytest.raises(ValueError, match="pinned"):
        codex_cli_headers_for_version("not-a-version")


def test_codex_cli_version_defaults_to_pinned_upstream_contract(monkeypatch):
    monkeypatch.delenv("CODEX_AS_API_CODEX_CLI_VERSION", raising=False)

    assert resolve_codex_cli_version() == "0.153.3"


def test_packaged_upstream_contract_takes_precedence_over_source_checkout(
    monkeypatch,
    tmp_path,
):
    packaged = tmp_path / "package-contract.json"
    source = tmp_path / "source-contract.json"
    packaged.write_text(json.dumps({"upstream": {"version": "1.2.3"}}), encoding="utf-8")
    source.write_text(json.dumps({"upstream": {"version": "9.9.9"}}), encoding="utf-8")
    monkeypatch.setattr(provider_module, "_PACKAGE_UPSTREAM_CONTRACT_PATH", packaged)
    monkeypatch.setattr(provider_module, "_UPSTREAM_CONTRACT_PATH", source)

    assert provider_module._load_codex_compatibility_version() == "1.2.3"


def test_codex_cli_version_override_fails_loudly(monkeypatch):
    monkeypatch.setenv("CODEX_AS_API_CODEX_CLI_VERSION", "9.8.7")

    with pytest.raises(ValueError, match="not supported"):
        resolve_codex_cli_version()

    with pytest.raises(ValueError, match="not supported"):
        ChatGPTOAuthProvider()


@pytest.mark.parametrize("value", ["", "   "])
def test_provider_rejects_empty_explicit_auth_path(value):
    with pytest.raises(ValueError, match="auth_json_path"):
        ChatGPTOAuthProvider(auth_json_path=value)


@pytest.mark.parametrize("value", ["", "   "])
def test_provider_rejects_empty_explicit_model(value):
    with pytest.raises(ValueError, match="model"):
        ChatGPTOAuthProvider(model=value)


@pytest.mark.parametrize(
    "catalog_failure",
    [
        ChatGPTOAuthMissingError("missing auth"),
        ChatGPTOAuthCatalogUnavailableError("catalog unavailable"),
    ],
)
def test_resolve_model_rejects_invalid_and_unconfigured_claude_models_before_catalog_fetch(
    monkeypatch,
    catalog_failure,
):
    provider = ChatGPTOAuthProvider()
    fetch_count = 0

    def fail_catalog():
        nonlocal fetch_count
        fetch_count += 1
        raise catalog_failure

    monkeypatch.setattr(provider, "get_model_catalog", fail_catalog)

    for requested in ["", " x ", 123]:
        with pytest.raises(ChatGPTOAuthInvalidRequestError):
            provider.resolve_model(requested)  # type: ignore[arg-type]
    with pytest.raises(ChatGPTOAuthInvalidRequestError, match=r"claude-\*"):
        provider.resolve_model("claude-sonnet", anthropic_facade=True)
    assert fetch_count == 0


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-a-url",
        "ftp://example.com/codex",
        "https://user:secret@example.com/codex",
        "https://example.com/codex?mode=test",
        "https://example.com/codex#fragment",
        " https://example.com/codex",
        "https://example.com/codex ",
        "http://example.com\n.evil/codex",
        "https://example.com/\u0080codex",
        "https://example.com/\u009fcodex",
        "https://example.com/a path",
        "https://example.com/bad%escape",
    ],
)
def test_provider_rejects_unsafe_or_invalid_base_url(value):
    with pytest.raises(ValueError, match="base_url"):
        ChatGPTOAuthProvider(base_url=value)


def test_provider_allows_percent_encoded_base_url_path():
    provider = ChatGPTOAuthProvider(base_url="https://example.com/codex%20api")
    assert provider.base_url == "https://example.com/codex%20api"


def test_provider_rejects_unsafe_refresh_endpoint_at_initialization(monkeypatch):
    monkeypatch.setenv(
        "CODEX_REFRESH_TOKEN_URL_OVERRIDE",
        "https://user:secret@auth.openai.com/oauth/token",
    )
    with pytest.raises(ChatGPTOAuthRefreshError, match="CODEX_REFRESH_TOKEN_URL_OVERRIDE"):
        ChatGPTOAuthProvider()


@pytest.mark.parametrize("subagent", ["has space", "line\nbreak", "tab\tvalue", "한글"])
def test_provider_rejects_header_unsafe_subagent_before_transport(subagent):
    provider = ChatGPTOAuthProvider()

    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="visible ASCII") as caught:
        list(
            provider.chat_stream(
                _provider_messages(),
                model="gpt-5.6-sol",
                subagent=subagent,
            )
        )
    assert subagent not in str(caught.value)


def test_cancel_current_requests_attempts_all_closes_and_reports_failures():
    provider = ChatGPTOAuthProvider()
    close_calls: list[str] = []

    class Response:
        def __init__(self, name: str, failure: Exception | None = None) -> None:
            self.name = name
            self.failure = failure

        def close(self) -> None:
            close_calls.append(self.name)
            if self.failure is not None:
                raise self.failure

    first = OSError("first close failure")
    second = RuntimeError("second close failure")
    provider._active_responses.update(  # noqa: SLF001 - deterministic cancellation boundary test
        [Response("first", first), Response("ok"), Response("second", second)]
    )

    with pytest.raises(RuntimeError) as caught:
        provider.cancel_current_requests()
    assert set(close_calls) == {"first", "ok", "second"}
    assert "2 active response(s)" in str(caught.value)
    assert caught.value.__cause__ in {first, second}


def test_provider_uses_fixed_default_timeouts_and_catalog_ttl():
    provider = ChatGPTOAuthProvider()

    assert provider.timeout == 300
    assert provider.catalog_timeout == 5
    assert provider.catalog_ttl == 300


def test_provider_catalog_cache_is_instance_scoped_across_different_ttls(model_catalog_document):
    long_lived = ChatGPTOAuthProvider(catalog_ttl=300)
    short_lived = ChatGPTOAuthProvider(catalog_ttl=1)
    key = ("account", long_lived.base_url, "0.153.3")
    loads = {"long": 0, "short": 0}

    def load_long():
        loads["long"] += 1
        return CatalogLoadResult(model_catalog_document, '"long"')

    def load_short():
        loads["short"] += 1
        return CatalogLoadResult(model_catalog_document, '"short"')

    long_snapshot = long_lived._model_catalog_cache.get(  # noqa: SLF001
        key,
        load_long,
        ttl_seconds=long_lived.catalog_ttl,
    )
    short_snapshot = short_lived._model_catalog_cache.get(  # noqa: SLF001
        key,
        load_short,
        ttl_seconds=short_lived.catalog_ttl,
    )

    assert long_lived._model_catalog_cache is not short_lived._model_catalog_cache  # noqa: SLF001
    assert loads == {"long": 1, "short": 1}
    assert long_snapshot.etag == '"long"'
    assert short_snapshot.etag == '"short"'


def _test_token():
    return SimpleNamespace(
        access_token="access",
        refresh_token="refresh",
        id_token="id",
        account_id="account",
        fedramp=False,
    )


def test_transport_boundaries_do_not_reclassify_programming_errors(monkeypatch):
    provider = ChatGPTOAuthProvider()
    token = _test_token()
    monkeypatch.setattr(provider_module, "token_for_request", lambda _path: token)
    monkeypatch.setattr(
        provider_module,
        "_urlopen_no_redirect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bug")),
    )

    with pytest.raises(RuntimeError, match="bug"):
        provider._fetch_model_catalog(token, "0.153.3")  # noqa: SLF001
    with pytest.raises(RuntimeError, match="bug"):
        provider._request_json("/responses", {})  # noqa: SLF001
    with pytest.raises(RuntimeError, match="bug"):
        list(provider._post_sse("/responses", {}))  # noqa: SLF001


def test_transport_boundaries_classify_url_errors(monkeypatch):
    provider = ChatGPTOAuthProvider()
    token = _test_token()
    monkeypatch.setattr(provider_module, "token_for_request", lambda _path: token)
    monkeypatch.setattr(
        provider_module,
        "_urlopen_no_redirect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )

    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        provider._fetch_model_catalog(token, "0.153.3")  # noqa: SLF001
    with pytest.raises(ChatGPTOAuthError) as json_error:
        provider._request_json("/responses", {})  # noqa: SLF001
    assert json_error.value.status == 502
    with pytest.raises(ChatGPTOAuthError) as sse_error:
        list(provider._post_sse("/responses", {}))  # noqa: SLF001
    assert sse_error.value.status == 502


def test_catalog_timeout_is_a_total_deadline_for_drip_response(monkeypatch):
    body = b'{"models": []}'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            for byte in body:
                try:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                except OSError:
                    return
                time.sleep(0.04)

        def log_message(self, _format, *args):  # noqa: ANN001
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        token = _test_token()
        provider = ChatGPTOAuthProvider(
            base_url=f"http://127.0.0.1:{server.server_port}",
            catalog_timeout=0.12,
        )
        monkeypatch.setattr(provider_module, "token_for_request", lambda _path: token)

        started = time.monotonic()
        with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
            provider._fetch_model_catalog(token, "0.153.3")  # noqa: SLF001
        assert time.monotonic() - started < 0.4
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_json_response_timeout_is_a_total_deadline_for_drip_response(monkeypatch):
    body = b'{"output":[]}'

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            for byte in body:
                try:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                except OSError:
                    return
                time.sleep(0.04)

        def log_message(self, _format, *args):  # noqa: ANN001
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        token = _test_token()
        provider = ChatGPTOAuthProvider(
            base_url=f"http://127.0.0.1:{server.server_port}",
            timeout=0.12,
        )
        monkeypatch.setattr(provider_module, "token_for_request", lambda _path: token)

        started = time.monotonic()
        with pytest.raises(ChatGPTOAuthUpstreamError) as caught:
            provider._request_json("/responses/compact", {})  # noqa: SLF001
        assert caught.value.status == 502
        assert time.monotonic() - started < 0.4
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_sse_http_error_body_has_a_total_deadline(monkeypatch):
    body = b'{"error":"rate limited"}'

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            for byte in body:
                try:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                except OSError:
                    return
                time.sleep(0.04)

        def log_message(self, _format, *args):  # noqa: ANN001
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        token = _test_token()
        provider = ChatGPTOAuthProvider(
            base_url=f"http://127.0.0.1:{server.server_port}",
            timeout=0.12,
        )
        monkeypatch.setattr(provider_module, "token_for_request", lambda _path: token)

        started = time.monotonic()
        with pytest.raises(ChatGPTOAuthUpstreamError) as caught:
            list(provider._post_sse("/responses", {}))  # noqa: SLF001
        assert caught.value.status == 429
        assert time.monotonic() - started < 0.4
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _UnreadUnauthorizedBody:
    def __init__(self) -> None:
        self.reads = 0
        self.closes = 0

    def read(self, *_args) -> bytes:
        self.reads += 1
        raise AssertionError("the first 401 body must not be read")

    def close(self) -> None:
        self.closes += 1


class _BufferedUpstreamResponse(io.BytesIO):
    def __init__(self, content: bytes, headers: dict[str, str] | None = None) -> None:
        super().__init__(content)
        self.headers = headers or {}


def _unread_unauthorized(body: _UnreadUnauthorizedBody) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.invalid",
        401,
        "unauthorized",
        {},
        body,
    )


def test_catalog_first_401_is_closed_and_refreshed_without_reading_its_body(monkeypatch):
    provider = ChatGPTOAuthProvider()
    token = _test_token()
    unauthorized_body = _UnreadUnauthorizedBody()
    calls = 0
    refreshes = 0

    def open_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _unread_unauthorized(unauthorized_body)
        return _BufferedUpstreamResponse(b'{"models":[]}', {"ETag": '"catalog"'})

    def refresh(observed):
        nonlocal refreshes
        refreshes += 1
        return observed

    monkeypatch.setattr(provider_module, "_urlopen_no_redirect", open_request)
    monkeypatch.setattr(provider_module, "refresh_after_unauthorized", refresh)

    result = provider._fetch_model_catalog(token, "0.153.3")  # noqa: SLF001

    assert result.document == {"models": []}
    assert calls == 2
    assert refreshes == 1
    assert unauthorized_body.reads == 0
    assert unauthorized_body.closes == 1


def test_json_first_401_is_closed_and_refreshed_without_reading_its_body(monkeypatch):
    provider = ChatGPTOAuthProvider()
    token = _test_token()
    unauthorized_body = _UnreadUnauthorizedBody()
    calls = 0
    refreshes = 0

    def open_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _unread_unauthorized(unauthorized_body)
        return _BufferedUpstreamResponse(b'{"output":[]}')

    def refresh(_observed):
        nonlocal refreshes
        refreshes += 1

    monkeypatch.setattr(provider_module, "token_for_request", lambda _path: token)
    monkeypatch.setattr(provider_module, "_urlopen_no_redirect", open_request)
    monkeypatch.setattr(provider_module, "refresh_after_unauthorized", refresh)

    result = provider._request_json("/responses/compact", {})  # noqa: SLF001

    assert result == b'{"output":[]}'
    assert calls == 2
    assert refreshes == 1
    assert unauthorized_body.reads == 0
    assert unauthorized_body.closes == 1


def test_sse_first_401_is_closed_and_refreshed_without_reading_its_body(monkeypatch):
    provider = ChatGPTOAuthProvider()
    token = _test_token()
    unauthorized_body = _UnreadUnauthorizedBody()
    calls = 0
    refreshes = 0

    def open_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _unread_unauthorized(unauthorized_body)
        return _BufferedUpstreamResponse(b'data: {"type":"response.completed","response":{}}\n\n')

    def refresh(_observed):
        nonlocal refreshes
        refreshes += 1

    monkeypatch.setattr(provider_module, "token_for_request", lambda _path: token)
    monkeypatch.setattr(provider_module, "_urlopen_no_redirect", open_request)
    monkeypatch.setattr(provider_module, "refresh_after_unauthorized", refresh)

    events = list(provider._post_sse("/responses", {}))  # noqa: SLF001

    assert events == [{"type": "response.completed", "response": {}}]
    assert calls == 2
    assert refreshes == 1
    assert unauthorized_body.reads == 0
    assert unauthorized_body.closes == 1


def test_json_response_rejects_nonstandard_json_numbers(monkeypatch):
    provider = ChatGPTOAuthProvider()
    monkeypatch.setattr(provider, "_request_json", lambda *_args, **_kwargs: b'{"value":NaN}')

    with pytest.raises(ChatGPTOAuthProtocolError, match="invalid JSON"):
        provider._post_json("/responses", {})  # noqa: SLF001


def test_catalog_refresh_rejects_an_authenticated_account_switch(monkeypatch):
    provider = ChatGPTOAuthProvider()
    initial = _test_token()
    refreshed = SimpleNamespace(
        access_token="new-access",
        refresh_token="new-refresh",
        id_token="new-id",
        account_id="other-account",
        fedramp=False,
    )
    requests = 0

    def unauthorized(*_args, **_kwargs):
        nonlocal requests
        requests += 1
        raise urllib.error.HTTPError(
            "https://example.invalid/models",
            401,
            "unauthorized",
            {},
            io.BytesIO(b"unauthorized"),
        )

    monkeypatch.setattr(provider_module, "_urlopen_no_redirect", unauthorized)
    monkeypatch.setattr(
        provider_module,
        "refresh_after_unauthorized",
        lambda _token: refreshed,
    )

    with pytest.raises(ChatGPTOAuthRefreshError, match="account changed"):
        provider._fetch_model_catalog(initial, "0.153.3")  # noqa: SLF001
    assert requests == 1


def test_response_request_rejects_account_switch_as_auth_refresh_failure(monkeypatch):
    provider = ChatGPTOAuthProvider()
    switched = SimpleNamespace(
        access_token="new-access",
        refresh_token="new-refresh",
        id_token="new-id",
        account_id="other-account",
        fedramp=False,
    )
    monkeypatch.setattr(provider_module, "token_for_request", lambda _path: switched)

    with pytest.raises(ChatGPTOAuthRefreshError, match="account changed after model catalog preflight"):
        provider._request_json(  # noqa: SLF001
            "/responses",
            {},
            catalog_key=("account", provider.base_url, "0.153.3"),
        )


def test_authenticated_catalog_and_response_requests_do_not_follow_redirects(
    monkeypatch,
):
    redirected_requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/models"):
                self.send_response(307)
                self.send_header("Location", "/redirect-target")
                self.end_headers()
                return
            redirected_requests.append(self.path)
            self.send_response(200)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/responses":
                self.send_response(307)
                self.send_header("Location", "/redirect-target")
                self.end_headers()
                return
            redirected_requests.append(self.path)
            self.send_response(200)
            self.end_headers()

        def log_message(self, _format, *args):  # noqa: ANN001
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        token = _test_token()
        provider = ChatGPTOAuthProvider(base_url=f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setattr(provider_module, "token_for_request", lambda _path: token)

        with pytest.raises(ChatGPTOAuthUpstreamError) as catalog_error:
            provider._fetch_model_catalog(token, "0.153.3")  # noqa: SLF001
        assert catalog_error.value.status == 307
        with pytest.raises(ChatGPTOAuthUpstreamError) as json_error:
            provider._request_json("/responses", {})  # noqa: SLF001
        assert json_error.value.status == 307
        with pytest.raises(ChatGPTOAuthUpstreamError) as sse_error:
            list(provider._post_sse("/responses", {}))  # noqa: SLF001
        assert sse_error.value.status == 307
        assert redirected_requests == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_authenticated_sse_failure_events_redact_request_credentials_before_aggregation(
    monkeypatch,
):
    token = _test_token()
    secrets = (
        token.access_token,
        token.refresh_token,
        token.id_token,
        token.account_id,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            message = " ".join(secrets)
            body = (
                "data: "
                + json.dumps(
                    {
                        "type": "response.failed",
                        "response": {
                            "error": {
                                "message": message,
                                "metadata": {"credential_echo": message},
                            }
                        },
                    }
                )
                + "\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *args):  # noqa: ANN001
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = ChatGPTOAuthProvider(base_url=f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setattr(provider_module, "token_for_request", lambda _path: token)

        with pytest.raises(ChatGPTOAuthUpstreamError) as caught:
            provider._collect_response_output_items({})  # noqa: SLF001
        message = str(caught.value)
        assert "***" in message
        for secret in secrets:
            assert secret not in message
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_error_body_read_failures_preserve_upstream_status(monkeypatch):
    provider = ChatGPTOAuthProvider()
    token = _test_token()
    monkeypatch.setattr(provider_module, "token_for_request", lambda _path: token)

    class BrokenBody:
        def read(self):
            raise OSError("secret local detail")

        def close(self):
            pass

    def fail(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://example.invalid",
            429,
            "rate limited",
            {},
            BrokenBody(),
        )

    monkeypatch.setattr(provider_module, "_urlopen_no_redirect", fail)
    for invoke in [
        lambda: provider._fetch_model_catalog(token, "0.153.3"),  # noqa: SLF001
        lambda: provider._request_json("/responses", {}),  # noqa: SLF001
        lambda: list(provider._post_sse("/responses", {})),  # noqa: SLF001
    ]:
        with pytest.raises(ChatGPTOAuthError) as caught:
            invoke()
        assert caught.value.status == 429
        assert "could not read upstream error body" in str(caught.value)
        assert "secret local detail" not in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeout", 0),
        ("timeout", True),
        ("timeout", float("inf")),
        ("catalog_timeout", float("nan")),
        ("catalog_ttl", "300"),
    ],
)
def test_provider_rejects_non_positive_or_non_finite_timeouts(field, value):
    with pytest.raises(ValueError, match="positive finite"):
        ChatGPTOAuthProvider(**{field: value})


def test_provider_headers_include_codex_cli_headers(auth_json_factory):
    provider = ChatGPTOAuthProvider(auth_json_path=str(auth_json_factory()))

    headers = provider._headers()  # noqa: SLF001 - regression test for backend request headers

    assert headers["originator"] == "codex_cli_rs"
    assert headers["User-Agent"].startswith("codex_cli_rs/0.153.3 (")
    assert headers["User-Agent"].endswith(") codex-as-api/0.7.0")
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


@pytest.mark.parametrize("mode", ["standard", "pro"])
def test_set_reasoning_payload_modes_are_rejected(mode):
    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="not supported"):
        _set_reasoning_payload({}, None, reasoning_mode=mode)


def test_set_reasoning_payload_preserves_case_for_custom_values():
    payload: dict = {}
    _set_reasoning_payload(payload, "HIGH")
    assert payload["reasoning"]["effort"] == "HIGH"


def test_set_reasoning_payload_preserves_pre_resolved_ultra():
    payload: dict = {}
    _set_reasoning_payload(payload, "ultra")
    assert payload["reasoning"] == {"effort": "ultra"}


def test_persistent_reasoning_maps_to_disabled_for_explicit_and_catalog_default(model_catalog_snapshot):
    base = model_catalog_snapshot.models[0]
    capability = replace(
        base,
        supported_reasoning_levels=(ModelReasoningLevel("persistent", "persistent"),),
        default_reasoning_level="persistent",
        use_responses_lite=False,
    )

    assert resolve_model_reasoning_effort(capability, "persistent") == "disabled"

    payload: dict[str, object] = {}
    provider_module._finalize_responses_payload(
        payload,
        capability=capability,
        reasoning_effort=None,
    )
    assert payload["reasoning"] == {"effort": "disabled", "summary": "auto"}

    custom_case_payload: dict = {}
    _set_reasoning_payload(custom_case_payload, "ULTRA")
    assert custom_case_payload["reasoning"] == {"effort": "ULTRA"}


@pytest.mark.parametrize(
    ("supported", "default_summary", "expected_summary"),
    [
        (True, "detailed", "detailed"),
        (True, "none", None),
        (False, "concise", None),
    ],
)
def test_live_reasoning_summary_controls_determine_private_wire_payload(
    model_catalog_snapshot,
    supported,
    default_summary,
    expected_summary,
):
    capability = replace(
        model_catalog_snapshot.models[0],
        default_reasoning_level=None,
        supported_reasoning_levels=(),
        use_responses_lite=False,
        supports_reasoning_summary_parameter=supported,
        default_reasoning_summary=default_summary,
    )
    payload: dict[str, object] = {}

    provider_module._finalize_responses_payload(
        payload,
        capability=capability,
        reasoning_effort=None,
    )

    if expected_summary is None:
        assert payload.get("reasoning") in (None, {})
    else:
        assert payload["reasoning"] == {"summary": expected_summary}
    assert payload["include"] == ["reasoning.encrypted_content"]


def test_set_reasoning_payload_custom_effort_is_preserved():
    payload: dict = {}
    _set_reasoning_payload(payload, "future-Effort")
    assert payload["reasoning"] == {"effort": "future-Effort"}


def test_set_reasoning_payload_empty_string_raises():
    with pytest.raises(ChatGPTOAuthError):
        _set_reasoning_payload({}, "")


@pytest.mark.parametrize("reasoning", [None, "medium", [], 1])
def test_set_reasoning_payload_rejects_existing_non_object(reasoning):
    payload = {"reasoning": reasoning}
    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="reasoning must be an object"):
        _set_reasoning_payload(payload, "high")
    assert payload["reasoning"] == reasoning


def test_set_reasoning_payload_preserves_existing_fields():
    payload = {"reasoning": {"summary": "auto"}}
    _set_reasoning_payload(payload, "high")
    assert payload["reasoning"] == {"summary": "auto", "effort": "high"}


def test_set_reasoning_payload_rejects_existing_mode():
    payload = {"reasoning": {"mode": "standard"}}
    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="not supported"):
        _set_reasoning_payload(payload, "high")


def test_set_reasoning_payload_rejects_non_array_include():
    payload = {"include": "reasoning.encrypted_content"}
    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="include must be an array"):
        _set_reasoning_payload(payload, "high")


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


def test_image_routes_do_not_invent_prompt_cache_keys(monkeypatch):
    provider = ChatGPTOAuthProvider()
    captured: list[dict] = []

    def collect(payload, *, catalog_key):  # noqa: ANN001
        del catalog_key
        captured.append(payload)
        if payload["tools"]:
            return [
                {
                    "type": "image_generation_call",
                    "id": "image-1",
                    "status": "completed",
                    "result": "encoded-image",
                }
            ]
        return [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "visible"}],
            }
        ]

    monkeypatch.setattr(provider, "_collect_response_output_items", collect)
    provider.generate_image("draw", model="gpt-5.5", responses_lite=False)
    provider.inspect_images(
        "inspect",
        model="gpt-5.5",
        images=[{"image_url": "data:image/png;base64,AAAA"}],
        responses_lite=False,
    )

    assert len(captured) == 2
    assert all("prompt_cache_key" not in payload for payload in captured)


@pytest.mark.parametrize(
    "unexpected",
    [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ignored"}]},
        {"type": "function_call", "name": "lookup", "call_id": "call-1", "arguments": "{}"},
        {"type": "web_search_call", "id": "search-1", "action": {"type": "search", "query": "q", "sources": []}},
    ],
)
def test_image_generation_rejects_endpoint_incompatible_output_items(monkeypatch, unexpected):
    provider = ChatGPTOAuthProvider()
    image = {
        "type": "image_generation_call",
        "status": "completed",
        "result": "encoded-image",
    }
    monkeypatch.setattr(provider, "_collect_response_output_items", lambda *_args, **_kwargs: [image, unexpected])

    with pytest.raises(ChatGPTOAuthProtocolError, match="unsupported output item"):
        provider.generate_image("draw", model="gpt-5.5", responses_lite=False)


@pytest.mark.parametrize(
    "unexpected",
    [
        {"type": "function_call", "name": "lookup", "call_id": "call-1", "arguments": "{}"},
        {"type": "image_generation_call", "status": "completed", "result": "encoded-image"},
        {"type": "web_search_call", "id": "search-1", "action": {"type": "search", "query": "q", "sources": []}},
    ],
)
def test_image_inspection_rejects_endpoint_incompatible_output_items(monkeypatch, unexpected):
    provider = ChatGPTOAuthProvider()
    message = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "visible"}]}
    monkeypatch.setattr(provider, "_collect_response_output_items", lambda *_args, **_kwargs: [message, unexpected])

    with pytest.raises(ChatGPTOAuthProtocolError, match="unsupported output item"):
        provider.inspect_images(
            "inspect",
            model="gpt-5.5",
            images=[{"image_url": "data:image/png;base64,AAAA"}],
            responses_lite=False,
        )


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
    assert tc.arguments == '{"q": "hello"}'
    assert tc.id == "cid-1"


def test_tool_call_rejects_custom_tool_call():
    item = {
        "type": "custom_tool_call",
        "name": "my_tool",
        "call_id": "cid-2",
        "input": "{}",
    }
    with pytest.raises(ChatGPTOAuthProtocolError, match="custom_tool_call"):
        _tool_call_from_response_item(item)


def test_tool_call_non_tool_type_returns_none():
    item = {"type": "message", "content": "hi"}
    assert _tool_call_from_response_item(item) is None


def test_tool_call_missing_name_is_protocol_error():
    item = {"type": "function_call", "call_id": "cid-3", "arguments": "{}"}
    with pytest.raises(ChatGPTOAuthProtocolError, match="name"):
        _tool_call_from_response_item(item)


@pytest.mark.parametrize(
    "item",
    [
        {
            "type": "function_call",
            "name": "fn",
            "call_id": "cid-4",
            "arguments": {"key": "value"},
        },
        {
            "type": "function_call",
            "name": "fn",
            "call_id": "cid-4",
            "input": "{}",
        },
        {
            "type": "custom_tool_call",
            "name": "fn",
            "call_id": "cid-4",
            "arguments": "{}",
        },
        {
            "type": "function_call",
            "name": "fn",
            "id": "item-id-is-not-call-id",
            "arguments": "{}",
        },
    ],
)
def test_tool_call_rejects_field_and_identifier_fallbacks(item):
    with pytest.raises(ChatGPTOAuthProtocolError):
        _tool_call_from_response_item(item)


@pytest.mark.parametrize("arguments", ["not json {{{", "[]", '  {"b":2, "a":1}  '])
def test_tool_call_preserves_raw_arguments(arguments):
    item = {
        "type": "function_call",
        "name": "fn",
        "call_id": "cid-5",
        "arguments": arguments,
    }
    assert _tool_call_from_response_item(item).arguments == arguments


# ---------------------------------------------------------------------------
# _text_from_response_items
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("item_type", ["output_text", "text"])
def test_text_from_items_rejects_stale_top_level_text_aliases(item_type):
    with pytest.raises(ChatGPTOAuthProtocolError, match="unsupported"):
        _text_from_response_items([{"type": item_type, "text": "hello"}])


def test_text_from_message_with_content_list():
    items = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "msg text"}],
        }
    ]
    assert _text_from_response_items(items) == "msg text"


def test_text_from_message_rejects_string_content_parts():
    items = [
        {
            "type": "message",
            "role": "assistant",
            "content": ["part one", "part two"],
        }
    ]
    with pytest.raises(ChatGPTOAuthProtocolError, match="must be an object"):
        _text_from_response_items(items)


def test_text_from_items_ignores_non_text_types():
    items = [
        {"type": "function_call", "name": "fn"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok"}],
        },
    ]
    assert _text_from_response_items(items) == "ok"


def test_text_from_empty_items():
    assert _text_from_response_items([]) == ""


# ---------------------------------------------------------------------------
# _usage_from_response
# ---------------------------------------------------------------------------


def test_usage_from_response_input_output_tokens():
    value = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 0},
    }
    u = _usage_from_response(value)
    assert u is not None
    assert u.prompt_tokens == 10
    assert u.completion_tokens == 5
    assert u.total_tokens == 15


def test_usage_from_response_rejects_chat_completion_token_aliases():
    value = {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}

    with pytest.raises(ChatGPTOAuthProtocolError, match="token aliases"):
        _usage_from_response(value)


def test_usage_from_response_cached_tokens_from_details():
    value = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "input_tokens_details": {"cached_tokens": 30},
    }
    u = _usage_from_response(value)
    assert u.cached_tokens == 30


def test_usage_from_response_cache_write_tokens_from_details():
    u = _usage_from_response(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_tokens_details": {
                "cached_tokens": 30,
                "cache_write_tokens": 40,
            },
        }
    )

    assert u is not None
    assert u.cached_tokens == 30
    assert u.cache_write_tokens == 40


def test_usage_from_response_rejects_cached_input_tokens_alias():
    value = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "input_tokens_details": {"cached_tokens": 0},
        "cached_input_tokens": 25,
    }
    with pytest.raises(ChatGPTOAuthProtocolError, match="token aliases"):
        _usage_from_response(value)


def test_usage_from_response_rejects_cache_read_input_tokens_alias():
    value = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "input_tokens_details": {"cached_tokens": 0},
        "cache_read_input_tokens": 15,
    }
    with pytest.raises(ChatGPTOAuthProtocolError, match="token aliases"):
        _usage_from_response(value)


@pytest.mark.parametrize(
    "alias",
    [
        "prompt_tokens",
        "completion_tokens",
        "prompt_tokens_details",
        "cache_creation_input_tokens",
    ],
)
def test_usage_from_response_rejects_public_token_aliases(alias):
    value = {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
        alias: 0,
    }

    with pytest.raises(ChatGPTOAuthProtocolError, match="token aliases"):
        _usage_from_response(value)


def test_usage_from_response_accepts_absent_usage():
    assert _usage_from_response(None) is None


@pytest.mark.parametrize("value", ["text", 42])
def test_usage_from_response_non_dict_is_protocol_error(value):
    with pytest.raises(ChatGPTOAuthProtocolError, match="must be an object"):
        _usage_from_response(value)


def test_usage_from_response_missing_tokens_is_protocol_error():
    with pytest.raises(ChatGPTOAuthProtocolError, match="requires"):
        _usage_from_response({"total_tokens": 10})


@pytest.mark.parametrize(
    "value",
    [
        {"input_tokens": 1, "output_tokens": 2},
        {"input_tokens": 1, "output_tokens": 2, "total_tokens": 4},
        {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
            "input_tokens_details": {},
        },
    ],
)
def test_usage_from_response_rejects_missing_or_inconsistent_actual_fields(value):
    with pytest.raises(ChatGPTOAuthProtocolError):
        _usage_from_response(value)


def test_usage_from_response_preserves_null_token_details():
    usage = _usage_from_response(
        {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
            "input_tokens_details": None,
        }
    )

    assert usage is not None
    assert usage.cached_tokens is None
    assert usage.cache_write_tokens is None


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


@pytest.mark.parametrize("responses_lite", [False, True])
def test_responses_payload_accepts_user_input_without_system_instructions(responses_lite):
    provider = ChatGPTOAuthProvider()

    payload = provider._responses_payload(  # noqa: SLF001
        [Message(role=MessageRole.USER, content="Hello")],
        model="gpt-5.6-sol",
        responses_lite=responses_lite,
    )

    assert "instructions" not in payload
    assert any(item.get("role") == "user" for item in payload["input"])


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


def test_messages_to_response_items_tool():
    messages = [Message(role=MessageRole.TOOL, content="result", tool_call_id="tc-1", name="fn")]
    items = _messages_to_response_items(messages)
    assert items[0]["type"] == "function_call_output"
    assert items[0]["output"] == "result"
    assert items[0]["call_id"] == "tc-1"


def test_messages_to_response_items_assistant_with_tool_calls():
    tc = ToolCall(id="tid", name="search", arguments='{"q":"test"}')
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


@pytest.mark.parametrize(
    "malformed",
    [
        {"type": "additional_tools", "role": "user", "tools": []},
        {"type": "additional_tools", "role": "developer", "tools": [42]},
        {"type": "reasoning", "summary": "bad"},
        {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": 42},
        {"type": "future_item"},
        {"type": "message", "role": "system", "content": []},
    ],
)
def test_compacted_history_validates_dropped_items_and_source_error_classification(malformed):
    with pytest.raises(ChatGPTOAuthProtocolError):
        _filter_compacted_history_items([malformed], source="output")

    marker = REMOTE_COMPACTION_MARKER + "\n" + json.dumps([malformed])
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        _messages_to_response_items([Message(role=MessageRole.SYSTEM, content=marker)])


def test_compacted_history_validates_common_response_item_fields_for_all_accepted_variants():
    valid = [
        {
            "type": "message",
            "id": "",
            "role": "assistant",
            "content": [],
            "phase": "commentary",
            "internal_chat_message_metadata_passthrough": {
                "turn_id": None,
                "create_time": 1.5,
                "content_item_kinds": "default-on-error",
            },
            "future": True,
        },
        {
            "type": "agent_message",
            "id": None,
            "author": "agent",
            "recipient": "parent",
            "content": [],
        },
        {"type": "compaction", "id": "", "encrypted_content": "opaque"},
        {"type": "context_compaction", "id": None},
    ]
    assert _filter_compacted_history_items(valid) == valid

    malformed = [
        {"type": "message", "role": "assistant", "content": [], "phase": "future"},
        {"type": "agent_message", "id": 42, "author": "agent", "recipient": "parent", "content": []},
        {"type": "additional_tools", "role": "developer", "tools": [], "id": 42},
        {"type": "compaction", "encrypted_content": "opaque", "id": 42},
        {"type": "context_compaction", "internal_chat_message_metadata_passthrough": "bad"},
    ]
    for item in malformed:
        with pytest.raises(ChatGPTOAuthProtocolError):
            _filter_compacted_history_items([item], source="output")
        marker = REMOTE_COMPACTION_MARKER + "\n" + json.dumps([item])
        with pytest.raises(ChatGPTOAuthInvalidRequestError):
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


def test_responses_lite_treats_null_retained_image_detail_as_absent():
    image = {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,AAAA",
                "detail": None,
            }
        ],
    }
    marker = REMOTE_COMPACTION_MARKER + "\n" + json.dumps([image])

    payload = ChatGPTOAuthProvider()._responses_payload(  # noqa: SLF001
        [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.SYSTEM, content=marker),
        ],
        model="gpt-5.6-sol",
        responses_lite=True,
    )

    assert image in payload["input"]


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


def test_tool_schema_to_response_dict_preserves_strict_and_omits_missing_description():
    tool = ToolSchema(
        name="search",
        description=None,
        parameters={"type": "object"},
        strict=True,
    )

    assert _tool_schema_to_response_dict(tool) == {
        "type": "function",
        "name": "search",
        "parameters": {"type": "object"},
        "strict": True,
    }


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
                "sources": [
                    {"url": "https://example.com", "title": "Example", "page_age": "today"},
                    {"url": "https://example.com", "title": "", "page_age": ""},
                    {"url": "https://other.example"},
                ],
            },
        }
    )
    assert result is not None
    assert result["id"] == "ws_1"
    assert result["input"] == {"query": "hello"}
    assert result["content"] == [
        {
            "type": "web_search_result",
            "url": "https://example.com",
            "title": "Example",
            "page_age": "today",
        },
        {
            "type": "web_search_result",
            "url": "https://example.com",
            "title": "",
            "page_age": "",
        },
        {
            "type": "web_search_result",
            "url": "https://other.example",
        },
    ]


@pytest.mark.parametrize(
    "item",
    [
        {"type": "web_search_call", "id": 1, "action": {"type": "search", "query": "q", "sources": []}},
        {"type": "web_search_call", "action": {"type": "search", "query": "q", "sources": []}},
        {"type": "web_search_call", "id": None, "action": {"type": "search", "query": "q", "sources": []}},
        {"type": "web_search_call", "id": "ws_1", "action": {"type": "search", "query": "q"}},
        {"type": "web_search_call", "id": "ws_1", "action": {"query": "q", "sources": []}},
        {
            "type": "web_search_call",
            "id": "ws_1",
            "action": {"type": "search", "query": "q", "queries": "bad", "sources": []},
        },
        {"type": "web_search_call", "id": "ws_1", "action": {"type": "search", "queries": ["a", "b"], "sources": []}},
        {
            "type": "web_search_call",
            "id": "ws_1",
            "action": {"type": "search", "query": "a", "queries": ["b"], "sources": []},
        },
        {"type": "web_search_call", "id": "ws_1", "action": {"type": "search", "sources": []}},
        {"type": "web_search_call", "id": "ws_1", "action": {"type": "search", "queries": [], "sources": []}},
        {
            "type": "web_search_call",
            "id": "ws_1",
            "action": {"type": "open_page", "url": "https://example.test", "sources": []},
        },
        {
            "type": "web_search_call",
            "id": "ws_1",
            "action": {"type": "find_in_page", "pattern": "needle", "sources": []},
        },
        {"type": "web_search_call", "id": "ws_1", "action": {"type": "future", "sources": []}},
        {
            "type": "web_search_call",
            "id": "ws_1",
            "action": {"type": "search", "query": "q", "sources": [{"url": "https://a.test", "title": 1}]},
        },
        {
            "type": "web_search_call",
            "id": "ws_1",
            "action": {"type": "search", "query": "q", "sources": [{"url": "https://a.test", "page_age": 1}]},
        },
    ],
)
def test_web_search_event_rejects_fallback_or_malformed_fields(item):
    with pytest.raises(ChatGPTOAuthProtocolError):
        _web_search_event_from_response_item(item)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ({"query": "direct", "queries": None, "sources": []}, "direct"),
        ({"query": None, "queries": ["fallback"], "sources": []}, "fallback"),
        ({"query": "", "sources": []}, ""),
        ({"query": "same", "queries": ["same"], "sources": []}, "same"),
    ],
)
def test_web_search_event_treats_nullable_query_fields_as_omitted(action, expected):
    event = _web_search_event_from_response_item(
        {"type": "web_search_call", "id": "ws_1", "action": {"type": "search", **action}}
    )
    assert event is not None
    assert event["input"] == {"query": expected}


def test_web_search_event_preserves_empty_string_id():
    event = _web_search_event_from_response_item(
        {"type": "web_search_call", "id": "", "action": {"type": "search", "query": "q", "sources": []}}
    )
    assert event is not None
    assert event["id"] == ""


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


def test_validate_image_content_items_rejects_unknown_fields():
    with pytest.raises(ChatGPTOAuthError, match="unsupported fields"):
        _validate_image_content_items(
            [
                {
                    "image_url": "data:image/png;base64,abc123",
                    "future": True,
                }
            ]
        )


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


@pytest.mark.parametrize(
    "optional_value", [pytest.param({}, id="missing"), pytest.param({"id": None, "revised_prompt": None}, id="null")]
)
def test_image_generation_from_item_omits_absent_optional_fields(optional_value):
    result = _image_generation_from_item(
        {
            "type": "image_generation_call",
            "status": "completed",
            "result": "data:image/png;base64,ABC",
            **optional_value,
        }
    )

    assert result == {
        "status": "completed",
        "result": "data:image/png;base64,ABC",
    }


@pytest.mark.parametrize("item_id", [123, False])
def test_image_generation_from_item_rejects_invalid_present_id(item_id):
    with pytest.raises(ChatGPTOAuthError, match="id must be a string or null"):
        _image_generation_from_item(
            {
                "type": "image_generation_call",
                "id": item_id,
                "status": "completed",
                "result": "data:image/png;base64,ABC",
            }
        )


def test_image_generation_from_item_rejects_non_string_revised_prompt():
    with pytest.raises(ChatGPTOAuthError, match="revised_prompt must be a string or null"):
        _image_generation_from_item(
            {
                "type": "image_generation_call",
                "status": "completed",
                "result": "data:image/png;base64,ABC",
                "revised_prompt": 123,
            }
        )


def test_image_generation_from_item_wrong_type_returns_none():
    item = {"type": "message", "content": "hi"}
    assert _image_generation_from_item(item) is None


def test_image_generation_from_item_preserves_empty_plain_string_fields():
    item = {"type": "image_generation_call", "id": "", "status": "", "result": ""}
    assert _image_generation_from_item(item) == {"id": "", "status": "", "result": ""}


def test_image_generation_from_item_none_result_raises():
    item = {"type": "image_generation_call", "id": "img-3", "status": "", "result": None}
    with pytest.raises(ChatGPTOAuthError, match="string result"):
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
    with pytest.raises(ChatGPTOAuthError, match="must contain valid JSON"):
        _decode_sse_block(lines)


@pytest.mark.parametrize("value", ["NaN", "1e400", '"\\ud800"'])
def test_decode_sse_block_rejects_nonstandard_json_values(value):
    with pytest.raises(ChatGPTOAuthProtocolError, match="must contain valid JSON"):
        _decode_sse_block([f'data: {{"type":"ping","value":{value}}}'])


@pytest.mark.parametrize("value", ["[]", '"event"', "42", "null"])
def test_decode_sse_block_rejects_non_object_json(value):
    with pytest.raises(ChatGPTOAuthError, match="SSE event JSON must be an object"):
        _decode_sse_block([f"data: {value}"])


def test_decode_sse_block_strips_data_prefix():
    lines = ['data:   {"k": "v"}']
    event = _decode_sse_block(lines)
    assert event == {"k": "v"}


def test_upstream_protocol_diagnostics_do_not_reflect_unknown_values():
    secret = "access-token-sentinel"
    invocations = [
        lambda: _decode_sse_block([f"data: {secret}"]),
        lambda: _validate_response_event(
            {
                "type": "response.output_item.done",
                "item": {"type": secret},
            }
        ),
        lambda: _validate_response_event(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": secret, "text": "ignored"}],
                },
            }
        ),
        lambda: _text_from_response_items([{"type": secret}]),
    ]
    for invoke in invocations:
        with pytest.raises(ChatGPTOAuthError) as caught:
            invoke()
        assert secret not in str(caught.value)
