from __future__ import annotations

import hashlib
import json
import queue
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, NamedTuple

import pytest

from codex_as_api.codex_config import CodexConfig
from codex_as_api.model_capabilities import LITE_HEADER_NAME, LITE_HEADER_VALUE, RESPONSES_LITE_ENV
from codex_as_api.provider import REMOTE_COMPACTION_MARKER, ChatGPTOAuthProvider


def _has_nested_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_has_nested_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_has_nested_key(child, key) for child in value)
    return False


class RecordingBackend(NamedTuple):
    base_url: str
    requests: queue.Queue[dict[str, Any]]
    compact_output: list[dict[str, Any]]


@pytest.fixture()
def recording_backend() -> Iterator[RecordingBackend]:
    requests: queue.Queue[dict[str, Any]] = queue.Queue()
    compact_output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "compact-checkpoint"}],
        }
    ]
    raw_compact_output = [
        {"type": "additional_tools", "role": "developer", "tools": []},
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "compact-only instructions"}],
        },
        *compact_output,
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            content_length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(content_length))
            requests.put(
                {
                    "path": self.path,
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                    "body": body,
                }
            )
            if self.path == "/responses/compact":
                encoded = json.dumps({"output": raw_compact_output}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return

            tools = body.get("tools") if isinstance(body, dict) else None
            has_image_generation = isinstance(tools, list) and any(
                isinstance(tool, dict) and tool.get("type") == "image_generation" for tool in tools
            )
            output = (
                [
                    {
                        "type": "image_generation_call",
                        "id": "img-1",
                        "status": "completed",
                        "result": "data:image/png;base64,AAAA",
                    }
                ]
                if has_image_generation
                else [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "backend-ok"}],
                    }
                ]
            )
            events = [
                *({"type": "response.output_item.done", "item": item} for item in output),
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp-local",
                        "output": [],
                        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                    },
                },
            ]
            encoded = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format: str, *args: object) -> None:
            del args

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield RecordingBackend(
            base_url=f"http://{host}:{port}",
            requests=requests,
            compact_output=compact_output,
        )
    finally:
        httpd.shutdown()
        thread.join(timeout=2)
        httpd.server_close()


@pytest.fixture()
def client(monkeypatch, auth_json_factory, recording_backend: RecordingBackend):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod

    monkeypatch.setattr(
        server_mod,
        "_provider",
        ChatGPTOAuthProvider(
            base_url=recording_backend.base_url,
            auth_json_path=str(auth_json_factory()),
            timeout=2,
        ),
    )
    return TestClient(server_mod.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _isolate_responses_lite_mode(monkeypatch):
    monkeypatch.setenv(RESPONSES_LITE_ENV, "auto")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "auth_available" in body
    assert "model" in body
    assert "codex_config_path" in body
    assert "context_window" in body
    assert "auto_compact_token_limit" in body
    assert "reasoning_effort" in body


def test_model_environment_value_is_trimmed_and_whitespace_uses_default(monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setenv("CODEX_AS_API_MODEL", "  gpt-5.6-sol  ")
    assert server_mod._env_str("CODEX_AS_API_MODEL", "gpt-5.5") == "gpt-5.6-sol"

    monkeypatch.setenv("CODEX_AS_API_MODEL", "   ")
    assert server_mod._env_str("CODEX_AS_API_MODEL", "gpt-5.5") == "gpt-5.5"


def test_health_uses_sol_catalog_context_compaction_and_effort(client, monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(server_mod, "MODEL", "gpt-5.6-sol")
    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(codex_home="/tmp/codex", config_path="/tmp/codex/config.toml"),
    )

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["context_window"] == 372_000
    assert body["auto_compact_token_limit"] == 334_800
    assert body["reasoning_effort"] == "low"


def test_health_clamps_config_overrides_to_catalog_limits_without_wire_normalization(client, monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(server_mod, "MODEL", "gpt-5.6-sol")
    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(
            codex_home="/tmp/codex",
            config_path="/tmp/codex/config.toml",
            model_reasoning_effort="ultra",
            model_context_window=400_000,
            model_auto_compact_token_limit=390_000,
        ),
    )

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["context_window"] == 372_000
    assert body["auto_compact_token_limit"] == 334_800
    assert body["reasoning_effort"] == "ultra"


def test_health_unknown_model_keeps_legacy_context_fallback(client, monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(server_mod, "MODEL", "unknown-model")
    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(codex_home="/tmp/codex", config_path="/tmp/codex/config.toml"),
    )

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["context_window"] == 200_000
    assert body["auto_compact_token_limit"] == 160_000
    assert body["reasoning_effort"] is None


def test_health_clamps_unknown_model_compact_override_to_fallback_context(client, monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(server_mod, "MODEL", "unknown-model")
    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(
            codex_home="/tmp/codex",
            config_path="/tmp/codex/config.toml",
            model_auto_compact_token_limit=190_000,
        ),
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["auto_compact_token_limit"] == 180_000


def test_health_rejects_empty_config_effort(client, monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(
            codex_home="/tmp/codex",
            config_path="/tmp/codex/config.toml",
            model_reasoning_effort="",
        ),
    )

    response = client.get("/health")

    assert response.status_code == 500
    assert response.json()["error"] == {
        "message": "reasoning_effort must be a non-empty string when provided",
        "type": "chatgpt_oauth_error",
    }


def test_empty_config_effort_fails_before_opening_stream(client, monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(
            codex_home="/tmp/codex",
            config_path="/tmp/codex/config.toml",
            model_reasoning_effort="",
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [{"role": "system", "content": "system"}],
            "stream": True,
        },
    )

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["type"] == "chatgpt_oauth_error"


def test_empty_request_effort_returns_invalid_request(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [{"role": "system", "content": "system"}],
            "reasoning_effort": "",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "reasoning_effort must be a non-empty string when provided",
        "type": "chatgpt_oauth_error",
    }


@pytest.mark.parametrize("stop", ["END", ["END"]])
def test_non_empty_stop_returns_400_before_upstream(
    client,
    recording_backend: RecordingBackend,
    stop,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
            "stop": stop,
            "stream": True,
        },
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"] == {
        "message": "stop is not supported by the private Codex OAuth HTTP transport",
        "type": "chatgpt_oauth_error",
    }
    assert recording_backend.requests.empty()


def test_invalid_responses_lite_mode_returns_structured_error_before_stream(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [{"role": "system", "content": "system"}],
            "responses_lite": "bogus",
            "stream": True,
        },
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"] == {
        "message": "responses_lite must be one of: off, on, auto",
        "type": "chatgpt_oauth_error",
    }
    assert recording_backend.requests.empty()


def test_chat_handler_reaches_real_provider_with_sol_ultra_lite_contract(
    client,
    auth_json_factory,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    import codex_as_api.server as server_mod

    monkeypatch.setenv("CODEX_AS_API_CODEX_CLI_VERSION", "1.2.3")
    monkeypatch.setattr(
        server_mod,
        "_provider",
        ChatGPTOAuthProvider(
            base_url=recording_backend.base_url,
            auth_json_path=str(auth_json_factory()),
            timeout=2,
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Lookup",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "reasoning_effort": "ultra",
            "parallel_tool_calls": True,
        },
    )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["choices"][0]["message"]["content"] == "backend-ok"
    assert response_body["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        },
    }
    recorded = recording_backend.requests.get(timeout=1)
    assert recorded["path"] == "/responses"
    assert recorded["headers"][LITE_HEADER_NAME] == LITE_HEADER_VALUE
    request_body = recorded["body"]
    assert request_body["model"] == "gpt-5.6-sol"
    assert request_body["reasoning"] == {"effort": "max", "context": "all_turns"}
    assert request_body["parallel_tool_calls"] is False
    assert request_body["include"] == ["reasoning.encrypted_content"]
    assert request_body["tool_choice"] == "auto"
    assert "instructions" not in request_body
    assert "tools" not in request_body
    assert request_body["input"][0] == {
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
    assert request_body["input"][1] == {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "system"}],
    }


def test_chat_handler_accepts_tool_result_without_optional_name(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "look it up"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "result"},
            ],
        },
    )

    assert response.status_code == 200, response.text
    recorded = recording_backend.requests.get(timeout=1)
    assert {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "result",
    } in recorded["body"]["input"]


def test_chat_handler_omits_standard_reasoning_mode_and_preserves_supported_fields(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "stable prefix",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,AAAA",
                                "detail": "original",
                            },
                        },
                    ],
                },
            ],
            "reasoning": {
                "effort": "max",
                "mode": "standard",
                "context": "current_turn",
            },
            "verbosity": "high",
            "responses_lite": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["response_id"] == "resp-local"
    recorded = recording_backend.requests.get(timeout=1)
    body = recorded["body"]
    assert body["reasoning"] == {"effort": "max", "context": "current_turn"}
    assert "prompt_cache_options" not in body
    assert "safety_identifier" not in body
    assert body["text"]["verbosity"] == "high"
    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "stable prefix",
                },
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": "original",
                },
            ],
        }
    ]


def test_chat_handler_codex_metadata_requires_client_session_before_upstream(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
            "codex_metadata": True,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "chatgpt_oauth_error"
    assert recording_backend.requests.empty()


def test_chat_handler_rejects_pro_mode_before_upstream(
    client,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(codex_home="/tmp/codex", config_path="/tmp/codex/config.toml"),
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
            "reasoning": {"mode": "pro"},
            "responses_lite": False,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "chatgpt_oauth_error"
    assert recording_backend.requests.empty()


def test_response_id_replays_full_history_without_private_previous_response_id(
    client,
    recording_backend: RecordingBackend,
):
    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "first"},
            ],
            "responses_lite": False,
        },
    )
    response_id = first.json()["response_id"]
    first_request = recording_backend.requests.get(timeout=1)["body"]

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "second"},
            ],
            "reasoning": {"context": "all_turns"},
            "previous_response_id": response_id,
            "responses_lite": False,
        },
    )

    assert second.status_code == 200
    second_request = recording_backend.requests.get(timeout=1)["body"]
    assert "previous_response_id" not in second_request
    assert second_request["input"] == [
        *first_request["input"],
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "backend-ok"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "second"}],
        },
    ]


def test_streaming_previous_response_is_resolved_once_before_headers(
    client,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    import codex_as_api.server as server_mod

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "first"},
            ],
            "responses_lite": False,
        },
    )
    response_id = first.json()["response_id"]
    recording_backend.requests.get(timeout=1)

    provider = server_mod._provider
    original_resolve = provider._response_chains.resolve  # noqa: SLF001
    resolved_ids: list[str] = []

    def resolve_once(value: str):
        resolved_ids.append(value)
        return original_resolve(value)

    monkeypatch.setattr(provider._response_chains, "resolve", resolve_once)  # noqa: SLF001
    streamed = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "second"},
            ],
            "stream": True,
            "previous_response_id": response_id,
            "responses_lite": False,
        },
    )

    assert streamed.status_code == 200
    assert resolved_ids == [response_id]
    recorded = recording_backend.requests.get(timeout=1)["body"]
    assert "previous_response_id" not in recorded


def test_chat_handler_rejects_cache_breakpoint_before_upstream(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,AAAA",
                                "detail": "original",
                            },
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ],
                },
            ],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "chatgpt_oauth_error"
    assert recording_backend.requests.empty()


@pytest.mark.parametrize(
    "extra",
    [
        {"reasoning_effort": "low", "reasoning": {"effort": "high"}},
        {"reasoning": {"mode": "turbo"}},
        {"reasoning": {"mode": "pro"}},
        {"reasoning": {"context": "forever"}},
        {"reasoning": {"context": "current_turn"}},
        {"prompt_cache_options": {"mode": "implicit", "ttl": "30m"}},
        {"prompt_cache_options": {"mode": "explicit", "ttl": "30m"}},
        {"prompt_cache_options": {"ttl": "24h"}},
        {"prompt_cache_key": ""},
        {"verbosity": "low", "text": {"verbosity": "high"}},
        {"text": {"verbosity": "verbose"}},
        {"safety_identifier": "   "},
        {"safety_identifier": "x" * 65},
        {"safety_identifier": "stable-user"},
        {"service_tier": "flex"},
        {"multi_agent": {"enabled": True}},
        {"programmatic_tool_calling": {"enabled": True}},
        {"tools": [{"type": "programmatic_tool_calling"}]},
        {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "system",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ],
                },
                {"role": "user", "content": "hello"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "prior answer",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ],
                },
                {"role": "user", "content": "hello"},
            ],
        },
        {
            "model": "gpt-5.5",
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ],
                },
            ],
        },
    ],
)
def test_chat_handler_rejects_incomplete_or_invalid_gpt56_wires_before_upstream(
    client,
    recording_backend: RecordingBackend,
    extra,
):
    payload = {
        "model": "gpt-5.6-sol",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ],
        **extra,
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "chatgpt_oauth_error"
    assert recording_backend.requests.empty()


@pytest.mark.parametrize(
    "extra",
    [
        {"multi_agent": {"enabled": True}},
        {"programmatic_tool_calling": {"enabled": True}},
        {"tools": [{"type": "programmatic_tool_calling"}]},
    ],
)
def test_anthropic_handler_rejects_native_responses_lifecycles_before_upstream(
    client,
    recording_backend: RecordingBackend,
    extra,
):
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
            **extra,
        },
    )

    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


def test_chat_reasoning_request_overrides_config_and_config_overrides_catalog(
    client,
    auth_json_factory,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    import codex_as_api.server as server_mod

    monkeypatch.setenv("CODEX_AS_API_CODEX_CLI_VERSION", "1.2.3")
    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(
            codex_home="/tmp/codex",
            config_path="/tmp/codex/config.toml",
            model_reasoning_effort="ultra",
        ),
    )
    monkeypatch.setattr(
        server_mod,
        "_provider",
        ChatGPTOAuthProvider(
            base_url=recording_backend.base_url,
            auth_json_path=str(auth_json_factory()),
            timeout=2,
        ),
    )
    base_request = {
        "model": "gpt-5.6-sol",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ],
    }

    config_response = client.post("/v1/chat/completions", json=base_request)
    request_response = client.post(
        "/v1/chat/completions",
        json={**base_request, "reasoning_effort": "HIGH"},
    )

    assert config_response.status_code == 200
    assert request_response.status_code == 200
    config_request = recording_backend.requests.get(timeout=1)
    explicit_request = recording_backend.requests.get(timeout=1)
    assert config_request["body"]["reasoning"] == {
        "effort": "max",
        "context": "all_turns",
    }
    assert explicit_request["body"]["reasoning"] == {
        "effort": "HIGH",
        "context": "all_turns",
    }


def test_compact_handler_reaches_real_provider_with_lite_json_transport(
    client,
    auth_json_factory,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    import codex_as_api.server as server_mod
    from codex_as_api.messages import Message, MessageRole

    monkeypatch.setenv("CODEX_AS_API_CODEX_CLI_VERSION", "1.2.3")
    monkeypatch.setattr(server_mod, "MODEL", "gpt-5.6-sol")
    provider = ChatGPTOAuthProvider(
        base_url=recording_backend.base_url,
        auth_json_path=str(auth_json_factory()),
        timeout=2,
    )
    monkeypatch.setattr(
        server_mod,
        "_provider",
        provider,
    )

    response = client.post(
        "/v1/compact",
        json={
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Lookup",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "reasoning_effort": "ultra",
        },
    )

    assert response.status_code == 200
    marker, encoded_output = response.json()["checkpoint"].split("\n", 1)
    assert marker == REMOTE_COMPACTION_MARKER
    assert json.loads(encoded_output) == recording_backend.compact_output
    recorded = recording_backend.requests.get(timeout=1)
    assert recorded["path"] == "/responses/compact"
    assert recorded["headers"][LITE_HEADER_NAME] == LITE_HEADER_VALUE
    request_body = recorded["body"]
    assert request_body["model"] == "gpt-5.6-sol"
    assert request_body["reasoning"] == {"effort": "max", "context": "all_turns"}
    assert request_body["parallel_tool_calls"] is False
    assert "include" not in request_body
    assert "instructions" not in request_body
    assert "tools" not in request_body
    assert "tool_choice" not in request_body
    assert request_body["input"][0] == {
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
    assert request_body["input"][1] == {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "system"}],
    }
    user_texts = [
        part.get("text")
        for item in request_body["input"]
        if item.get("type") == "message" and item.get("role") == "user"
        for part in item.get("content", [])
        if isinstance(part, dict)
    ]
    assert user_texts == ["hello"]

    checkpoint = response.json()["checkpoint"]
    continuation = provider._responses_payload(  # noqa: SLF001 - compact-to-continuation contract
        [
            Message(role=MessageRole.SYSTEM, content="fresh system"),
            Message(role=MessageRole.SYSTEM, content=checkpoint),
            Message(role=MessageRole.USER, content="next turn"),
        ],
        model="gpt-5.6-sol",
    )
    continuation_user_texts = [
        part.get("text")
        for item in continuation["input"]
        if item.get("type") == "message" and item.get("role") == "user"
        for part in item.get("content", [])
        if isinstance(part, dict)
    ]
    assert continuation_user_texts == ["next turn"]
    assert continuation["input"][1] == {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "fresh system"}],
    }


def test_inspect_images_uses_lite_defaults_and_real_transport(
    auth_json_factory,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    monkeypatch.setenv("CODEX_AS_API_CODEX_CLI_VERSION", "1.2.3")
    provider = ChatGPTOAuthProvider(
        base_url=recording_backend.base_url,
        auth_json_path=str(auth_json_factory()),
        timeout=2,
    )

    result = provider.inspect_images(
        "inspect",
        model="gpt-5.6-sol",
        images=[{"image_url": "data:image/png;base64,AAAA"}],
    )

    assert result == "backend-ok"
    recorded = recording_backend.requests.get(timeout=1)
    assert recorded["headers"][LITE_HEADER_NAME] == LITE_HEADER_VALUE
    request_body = recorded["body"]
    assert request_body["reasoning"] == {"effort": "low", "context": "all_turns"}
    assert request_body["tool_choice"] == "auto"
    assert request_body["input"][0]["tools"] == []
    assert request_body["input"][2]["content"] == [
        {"type": "input_text", "text": "inspect"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]


def test_responses_lite_off_allows_classic_hosted_image_generation(
    auth_json_factory,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    monkeypatch.setenv("CODEX_AS_API_CODEX_CLI_VERSION", "1.2.3")
    monkeypatch.setenv(RESPONSES_LITE_ENV, "off")
    provider = ChatGPTOAuthProvider(
        base_url=recording_backend.base_url,
        auth_json_path=str(auth_json_factory()),
        timeout=2,
    )

    images = provider.generate_image("draw", model="gpt-5.6-sol")

    assert images == [
        {
            "id": "img-1",
            "status": "completed",
            "revised_prompt": None,
            "result": "data:image/png;base64,AAAA",
        }
    ]
    recorded = recording_backend.requests.get(timeout=1)
    assert LITE_HEADER_NAME not in recorded["headers"]
    request_body = recorded["body"]
    assert request_body["tools"] == [{"type": "image_generation", "output_format": "png"}]
    assert request_body["reasoning"] == {"effort": "low"}


def test_route_responses_lite_false_reaches_anthropic_inspect_compact_and_image(
    client,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(server_mod, "MODEL", "gpt-5.6-sol")
    requests = [
        (
            "/v1/messages",
            {
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "system": "system",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "responses_lite": False,
            },
            "/responses",
        ),
        (
            "/v1/inspect",
            {
                "prompt": "inspect",
                "images": [{"image_url": "data:image/png;base64,AAAA"}],
                "responses_lite": False,
            },
            "/responses",
        ),
        (
            "/v1/compact",
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "hello"},
                ],
                "responses_lite": False,
            },
            "/responses/compact",
        ),
        (
            "/v1/images/generations",
            {"model": "gpt-5.6-sol", "prompt": "draw", "responses_lite": False},
            "/responses",
        ),
    ]

    for path, body, upstream_path in requests:
        response = client.post(path, json=body)
        assert response.status_code == 200
        recorded = recording_backend.requests.get(timeout=1)
        assert recorded["path"] == upstream_path
        assert LITE_HEADER_NAME not in recorded["headers"]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/v1/messages",
            {
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "system": "system",
                "messages": [{"role": "user", "content": "hello"}],
                "responses_lite": "bogus",
            },
        ),
        (
            "/v1/inspect",
            {"prompt": "inspect", "images": [], "responses_lite": "bogus"},
        ),
        (
            "/v1/compact",
            {
                "messages": [{"role": "system", "content": "system"}],
                "responses_lite": "bogus",
            },
        ),
        (
            "/v1/images/generations",
            {"model": "gpt-5.6-sol", "prompt": "draw", "responses_lite": "bogus"},
        ),
    ],
)
def test_invalid_responses_lite_mode_is_structured_400_on_all_routes(
    client,
    recording_backend: RecordingBackend,
    monkeypatch,
    path,
    body,
):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(server_mod, "MODEL", "gpt-5.6-sol")

    response = client.post(path, json=body)

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert "responses_lite must be one of: off, on, auto" in json.dumps(response.json())
    assert recording_backend.requests.empty()


@pytest.mark.parametrize(
    ("path", "base_body"),
    [
        (
            "/v1/images/generations",
            {"model": "gpt-5.6-sol", "prompt": "draw"},
        ),
        (
            "/v1/inspect",
            {"prompt": "inspect", "images": []},
        ),
        (
            "/v1/compact",
            {
                "messages": [{"role": "system", "content": "system"}],
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "unsupported",
    [
        {"multi_agent": {"enabled": True}},
        {"programmatic_tool_calling": {"enabled": True}},
        {"tools": [{"type": "programmatic_tool_calling"}]},
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "allowed_callers": ["programmatic"],
                    },
                }
            ]
        },
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "output_schema": {"type": "object"},
                    },
                }
            ]
        },
    ],
)
def test_non_chat_routes_reject_native_responses_lifecycles_before_upstream(
    client,
    recording_backend: RecordingBackend,
    path,
    base_body,
    unsupported,
):
    response = client.post(path, json={**base_body, **unsupported})

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert recording_backend.requests.empty()


@pytest.mark.parametrize(
    "unsupported",
    [
        {"safety_identifier": "stable-user"},
        {"include": ["reasoning.encrypted_content"]},
        {"prompt_cache_retention": "24h"},
        {"prompt_cache_options": {"mode": "implicit", "ttl": "30m"}},
    ],
)
def test_compact_rejects_unsupported_response_compact_params_before_upstream(
    client,
    recording_backend: RecordingBackend,
    unsupported,
):
    response = client.post(
        "/v1/compact",
        json={
            "messages": [{"role": "system", "content": "system"}],
            **unsupported,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "chatgpt_oauth_error"
    assert recording_backend.requests.empty()


def test_compact_treats_null_unsupported_params_as_omitted(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/compact",
        json={
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
            "safety_identifier": None,
            "prompt_cache_options": None,
            "include": None,
            "prompt_cache_retention": None,
        },
    )

    assert response.status_code == 200
    outbound = recording_backend.requests.get(timeout=1)["body"]
    assert "safety_identifier" not in outbound
    assert "prompt_cache_options" not in outbound
    assert "include" not in outbound
    assert "prompt_cache_retention" not in outbound


def test_chat_treats_null_cache_safety_and_breakpoint_as_omitted(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello",
                            "prompt_cache_breakpoint": None,
                        }
                    ],
                },
            ],
            "safety_identifier": None,
            "prompt_cache_options": None,
            "responses_lite": False,
        },
    )

    assert response.status_code == 200
    outbound = recording_backend.requests.get(timeout=1)["body"]
    assert "safety_identifier" not in outbound
    assert "prompt_cache_options" not in outbound
    assert "prompt_cache_breakpoint" not in outbound["input"][0]["content"][0]


@pytest.mark.parametrize("previous_response_id", ["", "resp-unknown"])
def test_compact_rejects_invalid_previous_response_id_before_upstream(
    client,
    recording_backend: RecordingBackend,
    previous_response_id,
):
    response = client.post(
        "/v1/compact",
        json={
            "messages": [{"role": "system", "content": "system"}],
            "previous_response_id": previous_response_id,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "chatgpt_oauth_error"
    assert recording_backend.requests.empty()


@pytest.mark.parametrize("failure", ["response_failed", "truncated_eof"])
def test_openai_runtime_stream_failure_is_reported_in_band(client, monkeypatch, failure):
    import codex_as_api.server as server_mod

    def failing_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.output_text.delta", "delta": "partial"}
        if failure == "response_failed":
            yield {"type": "response.failed", "response": {"error": {"message": "upstream failed"}}}

    monkeypatch.setattr(server_mod._provider, "_post_sse", failing_sse)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "stream": True,
            "messages": [{"role": "system", "content": "system"}],
        },
    )

    assert response.status_code == 200
    assert "partial" in response.text
    assert '"error"' in response.text
    assert "upstream failed" in response.text if failure == "response_failed" else "response.completed" in response.text
    assert "data: [DONE]" in response.text


@pytest.mark.parametrize("failure", ["response_failed", "truncated_eof"])
def test_anthropic_runtime_stream_failure_is_reported_in_band(client, monkeypatch, failure):
    import codex_as_api.server as server_mod

    def failing_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.output_text.delta", "delta": "partial"}
        if failure == "response_failed":
            yield {"type": "response.failed", "response": {"error": {"message": "upstream failed"}}}

    monkeypatch.setattr(server_mod._provider, "_post_sse", failing_sse)
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "stream": True,
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert "event: message_start" in response.text
    assert "partial" in response.text
    assert "event: error" in response.text
    assert "upstream failed" in response.text if failure == "response_failed" else "response.completed" in response.text


def test_openai_stream_tool_delta_has_index_and_responses_usage_keys(client, monkeypatch):
    import codex_as_api.server as server_mod

    tool_call = {
        "type": "function_call",
        "call_id": "call-1",
        "name": "lookup",
        "arguments": '{"query":"one"}',
    }

    def tool_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {
            "type": "response.output_item.done",
            "item": tool_call,
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "response-1",
                "output": [tool_call],
                "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
            },
        }

    monkeypatch.setattr(server_mod._provider, "_post_sse", tool_sse)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "stream": True,
            "messages": [{"role": "system", "content": "system"}],
        },
    )

    chunks = [
        json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: {")
    ]
    tool_chunk = next(chunk for chunk in chunks if chunk["choices"] and "tool_calls" in chunk["choices"][0]["delta"])
    assert tool_chunk["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    usage_chunk = next(chunk for chunk in chunks if chunk.get("usage"))
    assert usage_chunk["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        },
    }


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — schema validation
# ---------------------------------------------------------------------------


def test_chat_completions_invalid_body_returns_422(client):
    resp = client.post("/v1/chat/completions", json={})
    assert resp.status_code == 422


def test_chat_completions_valid_schema_reaches_provider(client):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200


def test_chat_completions_auth_error_not_422(client):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200


def test_chat_completions_subagent_field_accepted(client):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        "subagent": "my-subagent",
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200


def test_chat_completions_memgen_request_field_accepted(client):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        "memgen_request": True,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200


def test_chat_completions_unknown_previous_response_id_is_rejected_before_upstream(
    client,
    recording_backend: RecordingBackend,
):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        "previous_response_id": "resp-abc123",
        "stream": True,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "chatgpt_oauth_error"
    assert recording_backend.requests.empty()


def test_chat_completions_supported_extended_fields_are_accepted(client):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        "subagent": "agent-1",
        "memgen_request": False,
        "reasoning_effort": "high",
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200


def test_chat_completions_missing_auth_returns_auth_error(
    client,
    recording_backend: RecordingBackend,
    tmp_path,
    monkeypatch,
):
    import codex_as_api.server as server_mod

    missing_auth_path = str(tmp_path / "nonexistent.json")
    monkeypatch.setattr(server_mod, "AUTH_PATH", missing_auth_path)
    monkeypatch.setattr(
        server_mod,
        "_provider",
        ChatGPTOAuthProvider(
            base_url=recording_backend.base_url,
            auth_json_path=missing_auth_path,
            timeout=2,
        ),
    )
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 401
    body = resp.json()
    assert "error" in body
    assert body["error"]["type"] == "chatgpt_oauth_error"


def test_messages_count_tokens_counts_normalized_tools_without_provider_call(client, monkeypatch):
    import codex_as_api.server as server_mod

    def fail_provider_call():
        raise AssertionError("count_tokens must not call the Codex backend")

    monkeypatch.setattr(server_mod, "_get_provider", fail_provider_call)
    tools = [
        {
            "name": "lookup",
            "description": "Search docs",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]
    resp = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "tools": tools,
            "messages": [{"role": "user", "content": "hello"}],
            "multi_agent": None,
            "programmatic_tool_calling": None,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["input_tokens"] == 48
    assert body["context_window"] >= body["auto_compact_token_limit"]


def test_messages_count_tokens_accepts_disabled_thinking_with_ambient_effort(client, monkeypatch):
    import codex_as_api.server as server_mod

    def fail_provider_call():
        raise AssertionError("count_tokens must not call the Codex backend")

    monkeypatch.setattr(server_mod, "_get_provider", fail_provider_call)
    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "stream": True,
            "messages": [{"role": "user", "content": "search the web"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "high"},
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["input_tokens"] > 0


def test_messages_count_tokens_large_ascii_payload_is_not_double_counted(client):
    resp = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "x" * 4000}],
        },
    )

    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 512


def test_messages_count_tokens_ignores_non_model_control_fields(client):
    base_payload = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "same"}],
    }
    control_payload = {
        **base_payload,
        "max_tokens": 8192,
        "stream": True,
        "temperature": 0.2,
        "top_p": 0.8,
        "metadata": {"opaque": "not model visible" * 100},
        "stop_sequences": ["done"],
    }

    base = client.post("/v1/messages/count_tokens", json=base_payload)
    with_controls = client.post("/v1/messages/count_tokens", json=control_payload)

    assert base.status_code == 200
    assert with_controls.status_code == 200
    assert base.json()["input_tokens"] == 13
    assert with_controls.json()["input_tokens"] == 13


@pytest.mark.parametrize(
    "unsupported",
    [
        {"multi_agent": {"enabled": True}},
        {"programmatic_tool_calling": {"enabled": True}},
    ],
)
def test_messages_count_tokens_rejects_native_responses_lifecycles_without_provider_call(
    client,
    recording_backend: RecordingBackend,
    monkeypatch,
    unsupported,
):
    import codex_as_api.server as server_mod

    class DummyProvider:
        def count_tokens(self, *args, **kwargs):
            raise AssertionError("count_tokens must not call the Codex backend")

    monkeypatch.setattr(server_mod, "_provider", DummyProvider())
    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            **unsupported,
        },
    )

    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


def test_messages_count_tokens_uses_o200k_for_multilingual_text(client):
    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "안녕 👋"}],
    }
    resp = client.post("/v1/messages/count_tokens", json=payload)
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 16


def test_messages_count_tokens_counts_each_image_once(client):
    payload = {
        "model": "claude-sonnet-4-5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://example.com/image.png"},
                    },
                ],
            }
        ],
    }

    resp = client.post("/v1/messages/count_tokens", json=payload)

    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 8513


def test_messages_count_tokens_includes_tool_and_reasoning_metadata_once(client):
    payload = {
        "model": "claude-sonnet-4-5",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "plan"},
                    {"type": "text", "text": "checking"},
                    {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "lookup",
                        "input": {"query": "docs"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "result",
                    }
                ],
            },
        ],
    }

    resp = client.post("/v1/messages/count_tokens", json=payload)

    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 44


def test_messages_count_tokens_reports_effective_request_model_context(client, monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(server_mod, "MODEL", "gpt-5.5")
    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(codex_home="/tmp/codex", config_path="/tmp/codex/config.toml"),
    )

    known = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    fallback = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert known.status_code == 200
    assert known.json()["context_window"] == 372_000
    assert known.json()["auto_compact_token_limit"] == 334_800
    assert fallback.status_code == 200
    assert fallback.json()["context_window"] == 272_000
    assert fallback.json()["auto_compact_token_limit"] == 244_800


def test_messages_compact_accepts_anthropic_body(client, monkeypatch):
    import codex_as_api.server as server_mod

    class DummyProvider:
        def compact_messages(self, messages, *, model=None, tools=None, reasoning_effort=None, responses_lite=None):
            assert model == server_mod.MODEL
            assert reasoning_effort == "high"
            assert responses_lite is False
            assert [m.content for m in messages] == ["sys", "hello"]
            assert [tool.name for tool in tools] == ["lookup"]
            return "checkpoint"

    monkeypatch.setattr(server_mod, "_provider", DummyProvider())
    resp = client.post(
        "/v1/messages/compact",
        json={
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "system": "sys",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "responses_lite": False,
            "tools": [
                {
                    "name": "lookup",
                    "description": "Lookup",
                    "input_schema": {"type": "object"},
                }
            ],
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"checkpoint": "checkpoint"}


def test_messages_compact_routes_known_gpt_model_to_backend(
    client,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(server_mod, "MODEL", "gpt-5.5")
    response = client.post(
        "/v1/messages/compact",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200, response.text
    recorded = recording_backend.requests.get(timeout=1)
    assert recorded["path"] == "/responses/compact"
    assert recorded["body"]["model"] == "gpt-5.6-sol"


def test_messages_compact_maps_fast_mode_and_rejects_invalid_speed(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/messages/compact",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            "speed": "fast",
        },
    )

    assert response.status_code == 200, response.text
    recorded = recording_backend.requests.get(timeout=1)
    assert recorded["body"]["service_tier"] == "priority"

    invalid = client.post(
        "/v1/messages/compact",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            "speed": "warp",
        },
    )
    assert invalid.status_code == 400
    assert recording_backend.requests.empty()


def test_messages_compact_wires_output_config_format_to_codex_text(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/messages/compact",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "history"}],
            "output_config": {"format": {"type": "json_object"}},
        },
    )

    assert response.status_code == 200, response.text
    recorded = recording_backend.requests.get(timeout=1)
    assert recorded["path"] == "/responses/compact"
    assert recorded["body"]["text"]["format"] == {"type": "json_object"}

    conflict = client.post(
        "/v1/messages/compact",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "history"}],
            "output_config": {"format": {"type": "json_object"}},
            "text": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
        },
    )
    assert conflict.status_code == 400
    assert recording_backend.requests.empty()


@pytest.mark.parametrize(
    "unsupported",
    [
        {"output_config": {"task_budget": {"type": "tokens", "total": 20_000}}},
        {"tools": [{"name": "lookup", "input_schema": {}, "strict": True}]},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "url", "url": ""}}],
                }
            ]
        },
    ],
)
def test_messages_compact_rejects_unrepresentable_adapter_fields_before_upstream(
    client,
    recording_backend: RecordingBackend,
    unsupported,
):
    response = client.post(
        "/v1/messages/compact",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            **unsupported,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "chatgpt_oauth_error"
    assert recording_backend.requests.empty()


def test_compact_resolves_known_previous_response_to_full_input(
    client,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(server_mod, "MODEL", "gpt-5.6-sol")
    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "first"},
            ],
            "responses_lite": False,
        },
    )
    assert first.status_code == 200
    first_request = recording_backend.requests.get(timeout=1)["body"]
    response = client.post(
        "/v1/compact",
        json={
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
            "reasoning_effort": "max",
            "previous_response_id": first.json()["response_id"],
            "prompt_cache_key": "session-1",
            "service_tier": "fast",
            "verbosity": "high",
            "responses_lite": False,
        },
    )

    assert response.status_code == 200
    recorded = recording_backend.requests.get(timeout=1)
    assert recorded["path"] == "/responses/compact"
    assert "previous_response_id" not in recorded["body"]
    assert recorded["body"]["prompt_cache_key"] == "session-1"
    assert "prompt_cache_options" not in recorded["body"]
    assert recorded["body"]["service_tier"] == "priority"
    assert recorded["body"]["text"]["verbosity"] == "high"
    assert recorded["body"]["reasoning"] == {"effort": "max"}
    assert "include" not in recorded["body"]
    assert recorded["body"]["input"] == [
        *first_request["input"],
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "backend-ok"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
    ]


def test_inspect_preserves_original_detail_in_classic_and_strips_it_in_lite(
    client,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(server_mod, "MODEL", "gpt-5.6-sol")
    base = {
        "prompt": "inspect",
        "images": [
            {
                "image_url": "data:image/png;base64,AAAA",
                "detail": "original",
            }
        ],
    }
    classic = client.post("/v1/inspect", json={**base, "responses_lite": False})
    lite = client.post("/v1/inspect", json={**base, "responses_lite": True})

    assert classic.status_code == 200
    assert lite.status_code == 200
    classic_request = recording_backend.requests.get(timeout=1)
    lite_request = recording_backend.requests.get(timeout=1)
    assert classic_request["body"]["input"][0]["content"][1]["detail"] == "original"
    assert "detail" not in lite_request["body"]["input"][2]["content"][1]


def test_messages_compact_uses_anthropic_content_block_conversion_without_system(client, monkeypatch):
    import codex_as_api.server as server_mod

    class DummyProvider:
        def compact_messages(self, messages, **kwargs):
            assert kwargs["model"] == server_mod.MODEL
            assert len(messages) == 1
            assert messages[0].content == "hello"
            assert messages[0].images == ("data:image/png;base64,AAAA",)
            return "checkpoint"

    monkeypatch.setattr(server_mod, "_provider", DummyProvider())
    response = client.post(
        "/v1/messages/compact",
        json={
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"checkpoint": "checkpoint"}


def test_anthropic_messages_uses_codex_model_for_provider_and_client_model_in_response(client, monkeypatch):
    import codex_as_api.server as server_mod
    from codex_as_api.messages import AssistantResponse

    class DummyProvider:
        def chat(self, messages, **kwargs):
            assert kwargs["model"] == server_mod.MODEL
            return AssistantResponse(content="ok")

    monkeypatch.setattr(server_mod, "_provider", DummyProvider())
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "claude-sonnet-4-5"


def test_anthropic_messages_uses_session_cache_affinity_without_codex_metadata(
    client,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    monkeypatch.setenv("CODEX_AS_API_CODEX_METADATA", "on")
    payload = {
        "model": "gpt-5.6-sol",
        "max_tokens": 1024,
        "system": "system",
        "messages": [{"role": "user", "content": "hello"}],
    }
    session_id = "claude-session-123"
    other_session_id = "claude-session-456"
    expected = hashlib.sha256(f"codex-as-api:claude-code-session:{session_id}".encode()).hexdigest()
    other_expected = hashlib.sha256(f"codex-as-api:claude-code-session:{other_session_id}".encode()).hexdigest()

    first = client.post(
        "/v1/messages",
        headers={"x-claude-code-session-id": session_id},
        json=payload,
    )
    second = client.post(
        "/v1/messages",
        headers={"x-claude-code-session-id": session_id},
        json=payload,
    )
    third = client.post(
        "/v1/messages",
        headers={"x-claude-code-session-id": other_session_id},
        json=payload,
    )
    explicit = client.post(
        "/v1/messages",
        headers={"x-claude-code-session-id": session_id},
        json={**payload, "prompt_cache_key": "explicit-cache-key"},
    )

    assert first.status_code == second.status_code == third.status_code == explicit.status_code == 200
    first_outbound = recording_backend.requests.get(timeout=1)["body"]
    second_outbound = recording_backend.requests.get(timeout=1)["body"]
    third_outbound = recording_backend.requests.get(timeout=1)["body"]
    explicit_outbound = recording_backend.requests.get(timeout=1)["body"]
    assert first_outbound["prompt_cache_key"] == expected
    assert second_outbound["prompt_cache_key"] == expected
    assert third_outbound["prompt_cache_key"] == other_expected
    assert explicit_outbound["prompt_cache_key"] == "explicit-cache-key"
    assert "client_metadata" not in first_outbound
    assert "client_metadata" not in second_outbound
    assert "client_metadata" not in third_outbound
    assert "client_metadata" not in explicit_outbound
    assert "response_id" not in first.json()
    assert "response_id" not in second.json()
    assert "response_id" not in third.json()
    assert "response_id" not in explicit.json()


def test_anthropic_messages_without_session_or_explicit_cache_key_omits_cache_key(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
            "previous_response_id": None,
        },
    )

    assert response.status_code == 200
    assert "prompt_cache_key" not in recording_backend.requests.get(timeout=1)["body"]


def test_anthropic_messages_accepts_and_strips_ephemeral_cache_control_hints(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "responses_lite": False,
            "cache_control": {"type": "ephemeral"},
            "system": [
                {
                    "type": "text",
                    "text": "system",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                    "content": [
                        {
                            "type": "text",
                            "text": "hello",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Lookup",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    outbound = recording_backend.requests.get(timeout=1)["body"]
    assert not _has_nested_key(outbound, "cache_control")
    assert outbound["instructions"] == "system"
    assert outbound["input"][0]["content"][0]["text"] == "hello"
    assert outbound["tools"][0]["name"] == "lookup"


@pytest.mark.parametrize(
    "cache_control_payload",
    [
        {"cache_control": "ephemeral"},
        {"cache_control": {"type": "persistent"}},
        {"cache_control": {"type": "ephemeral", "ttl": None}},
        {
            "system": [
                {
                    "type": "text",
                    "text": "system",
                    "cache_control": {"type": "ephemeral", "ttl": "24h"},
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello",
                            "cache_control": {"type": "ephemeral", "extra": True},
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
                    "cache_control": {"type": "ephemeral", "ttl": []},
                }
            ]
        },
    ],
)
def test_anthropic_messages_rejects_malformed_cache_control_before_upstream(
    client,
    recording_backend: RecordingBackend,
    cache_control_payload,
):
    payload = {
        "model": "gpt-5.6-sol",
        "max_tokens": 1024,
        "system": "system",
        "messages": [{"role": "user", "content": "hello"}],
        **cache_control_payload,
    }
    response = client.post("/v1/messages", json=payload)

    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


def test_anthropic_messages_rejects_non_null_previous_response_id_before_upstream(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
            "previous_response_id": "resp-prior",
        },
    )

    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


def test_anthropic_latest_claude_code_shape_routes_known_gpt_effort_and_fast_mode(
    client,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(server_mod, "MODEL", "gpt-5.5")
    response = client.post(
        "/v1/messages?beta=true",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 64_000,
            "stream": False,
            "system": [{"type": "text", "text": "system"}],
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "adaptive", "display": "omitted"},
            "context_management": {
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
            },
            "output_config": {"effort": "max"},
            "speed": "fast",
        },
    )

    assert response.status_code == 200
    assert response.json()["model"] == "gpt-5.6-sol"
    outbound = recording_backend.requests.get(timeout=1)["body"]
    assert outbound["model"] == "gpt-5.6-sol"
    assert outbound["reasoning"]["effort"] == "max"
    assert outbound["service_tier"] == "priority"
    assert "output_config" not in outbound
    assert "context_management" not in outbound
    assert "speed" not in outbound


def test_anthropic_disabled_thinking_overrides_ambient_effort_for_web_auxiliary_stream(
    client,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    monkeypatch.setenv(RESPONSES_LITE_ENV, "off")
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "stream": True,
            "system": "system",
            "messages": [{"role": "user", "content": "search the web"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "high"},
        },
    )

    assert response.status_code == 200, response.text
    outbound = recording_backend.requests.get(timeout=1)["body"]
    assert outbound["reasoning"]["effort"] == "none"
    assert "context" not in outbound["reasoning"]
    assert outbound["tools"] == [{"type": "web_search", "external_web_access": True}]
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    assert events[0]["type"] == "message_start"
    assert events[-1]["type"] == "message_stop"


def test_anthropic_output_config_format_reaches_codex_text_format(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "system": "Return structured output.",
            "messages": [{"role": "user", "content": "return json"}],
            "output_config": {"format": {"type": "json_object"}},
        },
    )

    assert response.status_code == 200, response.text
    outbound = recording_backend.requests.get(timeout=1)["body"]
    assert outbound["text"]["format"] == {"type": "json_object"}


def test_anthropic_stream_does_not_block_concurrent_claude_code_requests(monkeypatch):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod

    provider_waiting = threading.Event()
    release_provider = threading.Event()

    class GatedProvider:
        def preflight_chat(self, messages, **kwargs):
            del messages, kwargs
            return {}, []

        def chat_stream(self, messages, **kwargs):
            del messages, kwargs
            yield {"type": "content", "text": "early"}
            provider_waiting.set()
            if not release_provider.wait(timeout=5):
                raise AssertionError("test did not release the provider stream")
            yield {
                "type": "finish",
                "finish_reason": "stop",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

    monkeypatch.setattr(server_mod, "_provider", GatedProvider())
    stream_result: dict[str, Any] = {}
    health_result: dict[str, Any] = {}

    with TestClient(server_mod.app, raise_server_exceptions=False) as concurrent_client:
        def request_stream() -> None:
            stream_result["response"] = concurrent_client.post(
                "/v1/messages",
                json={
                    "model": "claude-fable-5",
                    "max_tokens": 1024,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        def request_health() -> None:
            health_result["response"] = concurrent_client.get("/health")

        stream_thread = threading.Thread(target=request_stream)
        stream_thread.start()
        assert provider_waiting.wait(timeout=2)

        health_thread = threading.Thread(target=request_health)
        health_thread.start()
        try:
            health_thread.join(timeout=2)
            assert not health_thread.is_alive(), "the provider stream blocked the ASGI event loop"
            assert health_result["response"].status_code == 200
        finally:
            release_provider.set()

        stream_thread.join(timeout=5)
        health_thread.join(timeout=5)
        assert not stream_thread.is_alive()

    response = stream_result["response"]
    assert response.status_code == 200
    events = []
    for block in response.text.split("\n\n"):
        data_lines = [line.removeprefix("data: ") for line in block.splitlines() if line.startswith("data: ")]
        if data_lines:
            events.append(json.loads(data_lines[0]))
    assert [event["type"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2]["delta"] == {"type": "text_delta", "text": "early"}


def test_anthropic_builtin_claude_model_keeps_configured_codex_fallback(
    client,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(server_mod, "MODEL", "gpt-5.5")
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-fable-5",
            "max_tokens": 1024,
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["model"] == "claude-fable-5"
    assert recording_backend.requests.get(timeout=1)["body"]["model"] == "gpt-5.5"


@pytest.mark.parametrize(
    "unsupported",
    [
        {
            "context_management": {
                "edits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "trigger": {"type": "input_tokens", "value": 30_000},
                    }
                ]
            }
        },
        {"output_config": {"task_budget": {"type": "tokens", "total": 20_000}}},
    ],
)
def test_anthropic_unrepresentable_latest_controls_fail_before_upstream(
    client,
    recording_backend: RecordingBackend,
    unsupported,
):
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            **unsupported,
        },
    )

    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


@pytest.mark.parametrize("route", ["/v1/messages", "/v1/messages/count_tokens", "/v1/messages/compact"])
@pytest.mark.parametrize(
    "unsupported",
    [
        {"thinking": {"type": "disabled"}, "output_config": {"effort": []}},
        {"output_config": {"format": "json"}},
        {"output_config": {"format": {"type": "json_object", "extra": True}}},
        {
            "output_format": {"type": "json_object"},
            "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "file", "file_id": "file-1"}}],
                }
            ]
        },
    ],
)
def test_anthropic_routes_reject_invalid_controls_and_unknown_image_sources(
    client,
    recording_backend: RecordingBackend,
    route,
    unsupported,
):
    response = client.post(
        route,
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            **unsupported,
        },
    )

    assert response.status_code == 400
    assert recording_backend.requests.empty()


def test_anthropic_stream_preflight_returns_json_error_before_upstream_request(
    client,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(server_mod, "MODEL", "gpt-5.6-sol")
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "stream": True,
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        },
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": (
                "Responses Lite cannot execute hosted tools without a standalone runtime: "
                "web_search; set CODEX_AS_API_RESPONSES_LITE=off to use classic Responses"
            ),
        },
    }
    assert recording_backend.requests.empty()
