from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest

from codex_as_api.auth import (
    ChatGPTOAuthCatalogUnavailableError,
    ChatGPTOAuthMissingError,
    ChatGPTOAuthUpstreamError,
)
from codex_as_api.codex_config import CodexConfig
from codex_as_api.messages import AssistantResponse, Message, MessageRole, Usage
from codex_as_api.model_capabilities import (
    DEFAULT_MODEL_CATALOG_TIMEOUT_SECONDS,
    DEFAULT_MODEL_CATALOG_TTL_SECONDS,
    LITE_HEADER_NAME,
    LITE_HEADER_VALUE,
    RESPONSES_LITE_ENV,
    parse_model_catalog,
)
from codex_as_api.provider import REMOTE_COMPACTION_MARKER, ChatGPTOAuthProvider

_UPSTREAM_CONTRACT = json.loads(
    (Path(__file__).resolve().parents[1] / "config" / "codex-upstream-contract.json").read_text(encoding="utf-8")
)


def _has_nested_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_has_nested_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_has_nested_key(child, key) for child in value)
    return False


class RecordingBackend(NamedTuple):
    base_url: str
    requests: queue.Queue[dict[str, Any]]
    catalog_requests: queue.Queue[dict[str, Any]]
    compact_output: list[dict[str, Any]]
    catalog_etag: list[str]
    response_etag: list[str]


@pytest.fixture()
def recording_backend(model_catalog_document) -> Iterator[RecordingBackend]:
    requests: queue.Queue[dict[str, Any]] = queue.Queue()
    catalog_requests: queue.Queue[dict[str, Any]] = queue.Queue()
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
    catalog_etag = ['"test-etag"']
    response_etag = ['"test-etag"']

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            catalog_requests.put(
                {
                    "method": self.command,
                    "path": self.path,
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                }
            )
            encoded = json.dumps(model_catalog_document).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header(_UPSTREAM_CONTRACT["models_request"]["etag_header"], catalog_etag[0])
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            content_length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(content_length))
            requests.put(
                {
                    "method": self.command,
                    "path": self.path,
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                    "body": body,
                }
            )
            if self.path == "/responses/compact":
                encoded = json.dumps({"output": raw_compact_output}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header(
                    _UPSTREAM_CONTRACT["models_request"]["responses_etag_header"],
                    response_etag[0],
                )
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
                        "end_turn": True,
                        "usage": {
                            "input_tokens": 3,
                            "output_tokens": 2,
                            "total_tokens": 5,
                            "input_tokens_details": {"cached_tokens": 0},
                        },
                    },
                },
            ]
            encoded = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header(
                _UPSTREAM_CONTRACT["models_request"]["responses_etag_header"],
                response_etag[0],
            )
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
            catalog_requests=catalog_requests,
            compact_output=compact_output,
            catalog_etag=catalog_etag,
            response_etag=response_etag,
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
        "CODEX_CONFIG",
        CodexConfig(codex_home="/tmp/codex", config_path="/tmp/codex/config.toml"),
    )
    monkeypatch.setattr(
        server_mod,
        "_provider",
        ChatGPTOAuthProvider(
            model="gpt-5.6-sol",
            base_url=recording_backend.base_url,
            auth_json_path=str(auth_json_factory()),
            timeout=2,
        ),
    )
    return TestClient(
        server_mod.app,
        raise_server_exceptions=False,
        headers={"x-claude-code-session-id": "test-claude-code-session"},
    )


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
    assert body["auth_available"] is True
    assert body["catalog_status"] == "fresh"
    assert body["model"] == "gpt-5.6-sol"
    assert body["reasoning_effort"] == "low"
    assert body["catalog_fetched_at"].endswith("Z")
    assert body["catalog_expires_at"].endswith("Z")
    assert set(body) == {
        "status",
        "auth_available",
        "catalog_status",
        "catalog_fetched_at",
        "catalog_expires_at",
        "model",
        "reasoning_effort",
        "context_window",
        "auto_compact_token_limit",
    }


def test_health_never_exposes_the_upstream_catalog_etag(
    client,
    recording_backend: RecordingBackend,
):
    secret = "access-token-sentinel"
    recording_backend.catalog_etag[0] = secret

    response = client.get("/health")

    assert response.status_code == 200
    assert "catalog_etag" not in response.json()
    assert secret not in response.text


def test_models_returns_live_safe_metadata(client):
    response = client.get("/v1/models")

    assert response.status_code == 200
    models = response.json()["data"]
    assert [model["id"] for model in models] == [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.2",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
    ]
    assert models[-1]["supported_in_api"] is False
    model = models[0]
    assert model["id"] == "gpt-5.6-sol"
    assert model["object"] == "model"
    assert model["owned_by"] == "openai"
    assert model["supported_reasoning_levels"][0] == {
        "effort": "low",
        "description": "low",
    }
    assert model["multi_agent_reasoning_effort"] == "max"
    assert model["supports_reasoning_summary_parameter"] is True
    assert model["default_reasoning_summary"] == "auto"
    assert model["comp_hash"] is None
    assert "created" not in model
    assert "base_instructions" not in model


def test_models_preserves_official_optional_catalog_semantics(
    monkeypatch,
    model_catalog_document,
):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod

    row = model_catalog_document["models"][0]
    row["comp_hash"] = " compatibility family "
    row["supported_reasoning_levels"] = []
    for field in (
        "description",
        "default_reasoning_level",
        "default_verbosity",
        "context_window",
        "max_context_window",
        "auto_compact_token_limit",
        "default_service_tier",
        "multi_agent_reasoning_effort",
        "service_tiers",
        "use_responses_lite",
        "supports_image_detail_original",
        "effective_context_window_percent",
        "input_modalities",
        "supports_reasoning_summary_parameter",
        "default_reasoning_summary",
    ):
        row.pop(field, None)
    now = time.time()
    snapshot = parse_model_catalog(
        {"models": [row]},
        key=("account", "https://example.test/codex", "0.153.3"),
        etag='"optional"',
        fetched_at=now,
        expires_at=now + 300,
    )

    class OptionalCatalogProvider:
        def get_model_catalog(self):
            return snapshot

    monkeypatch.setattr(server_mod, "_provider", OptionalCatalogProvider())
    response = TestClient(server_mod.app, raise_server_exceptions=False).get("/v1/models")

    assert response.status_code == 200
    model = response.json()["data"][0]
    assert model["description"] is None
    assert model["supported_reasoning_levels"] == []
    assert model["default_reasoning_level"] is None
    assert model["default_verbosity"] is None
    assert model["context_window"] is None
    assert model["max_context_window"] is None
    assert model["auto_compact_token_limit"] is None
    assert model["default_service_tier"] is None
    assert model["multi_agent_reasoning_effort"] is None
    assert model["service_tiers"] == []
    assert model["use_responses_lite"] is False
    assert model["supports_image_detail_original"] is False
    assert model["effective_context_window_percent"] == 95
    assert model["input_modalities"] == ["text", "image"]
    assert model["supports_reasoning_summary_parameter"] is True
    assert model["default_reasoning_summary"] == "auto"
    assert model["comp_hash"] == " compatibility family "


def test_models_exposes_an_empty_live_catalog_without_default_fallback(monkeypatch):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod

    now = time.time()
    snapshot = parse_model_catalog(
        {"models": []},
        key=("account", "https://example.test/codex", "0.153.3"),
        etag=None,
        fetched_at=now,
        expires_at=now + 300,
    )

    class EmptyCatalogProvider:
        def get_model_catalog(self):
            return snapshot

        def resolve_model(self, requested=None, *, snapshot=None):  # noqa: ANN001
            del requested, snapshot
            raise ChatGPTOAuthCatalogUnavailableError("catalog has no default model")

    monkeypatch.setattr(server_mod, "_provider", EmptyCatalogProvider())
    test_client = TestClient(server_mod.app, raise_server_exceptions=False)

    models = test_client.get("/v1/models")
    health = test_client.get("/health")

    assert models.status_code == 200
    assert models.json() == {"object": "list", "data": []}
    assert health.status_code == 503
    assert health.json()["error"]["type"] == "catalog_unavailable"


def test_health_auth_failure_is_flat_and_does_not_expose_auth_path(monkeypatch):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod

    class MissingAuthProvider:
        def get_model_catalog(self):
            raise ChatGPTOAuthMissingError("ChatGPT OAuth auth file not found: /secret/auth.json")

    monkeypatch.setattr(server_mod, "_provider", MissingAuthProvider())
    response = TestClient(server_mod.app, raise_server_exceptions=False).get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body == {
        "status": "error",
        "auth_available": False,
        "catalog_status": "unavailable",
        "catalog_fetched_at": None,
        "catalog_expires_at": None,
        "model": None,
        "reasoning_effort": None,
        "context_window": None,
        "auto_compact_token_limit": None,
        "error": {
            "message": "ChatGPT OAuth credentials are unavailable",
            "type": "authentication_error",
        },
    }
    assert "/secret/auth.json" not in response.text


def test_health_catalog_upstream_401_preserves_upstream_taxonomy(monkeypatch):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod

    class RejectedCatalogProvider:
        def get_model_catalog(self):
            raise ChatGPTOAuthUpstreamError(
                401,
                "catalog authentication rejected",
            )

    monkeypatch.setattr(server_mod, "_provider", RejectedCatalogProvider())
    test_client = TestClient(server_mod.app, raise_server_exceptions=False)
    response = test_client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["auth_available"] is True
    assert body["catalog_status"] == "unavailable"
    assert body["error"] == {
        "message": "upstream request failed",
        "type": "upstream_error",
    }
    models_response = test_client.get("/v1/models")
    assert models_response.status_code == 401
    assert models_response.json()["error"] == {
        "message": "upstream request failed",
        "type": "upstream_error",
        "code": "upstream_error",
    }


def test_health_unexpected_failure_keeps_flat_safe_schema(monkeypatch):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod

    class BrokenProvider:
        def get_model_catalog(self):
            raise RuntimeError("private implementation detail")

    monkeypatch.setattr(server_mod, "_provider", BrokenProvider())
    response = TestClient(server_mod.app, raise_server_exceptions=False).get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["catalog_status"] == "unavailable"
    assert body["error"] == {
        "message": "health preflight failed",
        "type": "server_error",
    }
    assert "private implementation detail" not in response.text


def test_health_accepts_selected_model_without_optional_default_reasoning(
    monkeypatch,
    model_catalog_document,
):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod

    row = model_catalog_document["models"][0]
    row.pop("default_reasoning_level")
    row["context_window"] = None
    row["max_context_window"] = None
    row["auto_compact_token_limit"] = None
    now = time.time()
    snapshot = parse_model_catalog(
        {"models": [row]},
        key=("account", "https://example.test/codex", "0.153.3"),
        etag='"optional"',
        fetched_at=now,
        expires_at=now + 300,
    )

    class OptionalCatalogProvider:
        def get_model_catalog(self):
            return snapshot

        def resolve_model(self, *_args, **_kwargs):
            return snapshot, snapshot.models[0]

    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(codex_home="/tmp/codex", config_path="/tmp/codex/config.toml"),
    )
    monkeypatch.setattr(server_mod, "_provider", OptionalCatalogProvider())
    test_client = TestClient(server_mod.app, raise_server_exceptions=False)
    response = test_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["catalog_status"] == "fresh"
    assert body["model"] == "gpt-5.6-sol"
    assert body["reasoning_effort"] is None
    assert body["context_window"] is None
    assert body["auto_compact_token_limit"] is None
    assert "error" not in body

    count_response = test_client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert count_response.status_code == 200
    assert count_response.json()["context_window"] is None
    assert count_response.json()["auto_compact_token_limit"] is None


def test_invalid_live_default_reasoning_is_catalog_unavailable(
    client,
    model_catalog_document,
    recording_backend: RecordingBackend,
):
    model_catalog_document["models"][0]["default_reasoning_level"] = "not-listed"

    health = client.get("/health")
    chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert health.status_code == 503
    assert health.json()["error"]["type"] == "catalog_unavailable"
    assert chat.status_code == 503
    assert chat.json()["error"]["type"] == "catalog_unavailable"
    assert recording_backend.requests.empty()


def test_live_default_ultra_without_wire_mapping_is_catalog_unavailable(
    client,
    model_catalog_document,
    recording_backend: RecordingBackend,
):
    model = model_catalog_document["models"][0]
    model["supported_reasoning_levels"] = [{"effort": "ultra", "description": "ultra"}]
    model["default_reasoning_level"] = "ultra"
    model["multi_agent_reasoning_effort"] = "ultra"

    health = client.get("/health")
    chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert health.status_code == 503
    assert health.json()["error"]["type"] == "catalog_unavailable"
    assert chat.status_code == 503
    assert chat.json()["error"]["type"] == "catalog_unavailable"
    assert recording_backend.requests.empty()


def test_zero_effective_context_window_is_catalog_unavailable(
    client,
    model_catalog_document,
    recording_backend: RecordingBackend,
):
    model = model_catalog_document["models"][0]
    model["context_window"] = 1
    model["max_context_window"] = 1
    model["auto_compact_token_limit"] = None

    health = client.get("/health")
    count = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert health.status_code == 503
    assert health.json()["error"]["type"] == "catalog_unavailable"
    assert count.status_code == 503
    assert count.json()["error"]["type"] == "api_error"
    assert recording_backend.requests.empty()


def test_direct_zero_auto_compact_limit_is_exposed(
    client,
    model_catalog_document,
    recording_backend: RecordingBackend,
):
    model_catalog_document["models"][0]["auto_compact_token_limit"] = 0

    health = client.get("/health")
    count = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert health.status_code == 200
    assert health.json()["auto_compact_token_limit"] == 0
    assert count.status_code == 200
    assert count.json()["auto_compact_token_limit"] == 0
    assert recording_backend.requests.empty()


def test_nonstandard_live_catalog_json_is_catalog_unavailable(
    client,
    model_catalog_document,
    recording_backend: RecordingBackend,
):
    model_catalog_document["models"][0]["priority"] = float("nan")

    response = client.get("/v1/models")

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "catalog_unavailable"
    assert recording_backend.requests.empty()


def test_catalog_fetch_uses_pinned_query_and_oauth_headers(
    recording_backend: RecordingBackend,
    auth_json_factory,
):
    provider = ChatGPTOAuthProvider(
        base_url=recording_backend.base_url,
        auth_json_path=str(auth_json_factory()),
    )

    snapshot = provider.get_model_catalog()
    request = recording_backend.catalog_requests.get(timeout=1)
    contract = _UPSTREAM_CONTRACT["models_request"]
    version = _UPSTREAM_CONTRACT["upstream"]["version"]
    scope_values = {
        "account_id": snapshot.account_id,
        "base_url": snapshot.base_url,
        "client_version": snapshot.client_version,
    }

    assert snapshot.etag == '"test-etag"'
    assert request["method"] == contract["method"]
    assert request["path"] == f"{contract['path']}?{contract['client_version_query']}={version}"
    assert snapshot.key == tuple(scope_values[field] for field in contract["cache_scope"])
    assert provider.catalog_ttl == DEFAULT_MODEL_CATALOG_TTL_SECONDS == contract["cache_ttl_seconds"]
    assert provider.catalog_timeout == DEFAULT_MODEL_CATALOG_TIMEOUT_SECONDS == contract["request_timeout_seconds"]
    assert contract["allow_stale_on_refresh_error"] is False
    assert request["headers"]["accept"] == "application/json"
    assert request["headers"]["authorization"].startswith("Bearer ")
    assert request["headers"]["chatgpt-account-id"] == "acc-123"
    assert request["headers"]["originator"] == "codex_cli_rs"
    assert request["headers"]["user-agent"].startswith("codex_cli_rs/0.153.3 (")


def test_model_dependent_routes_reuse_one_fresh_catalog_snapshot(
    client,
    recording_backend: RecordingBackend,
):
    requests = [
        client.get("/v1/models"),
        client.get("/health"),
        client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.5",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "hello"},
                ],
            },
        ),
        client.post(
            "/v1/images/generations",
            json={"model": "gpt-5.5", "prompt": "draw"},
        ),
        client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "gpt-5.5",
                "system": "system",
                "messages": [{"role": "user", "content": "hello"}],
            },
        ),
        client.post(
            "/v1/messages",
            json={
                "model": "gpt-5.5",
                "max_tokens": 32,
                "system": "system",
                "messages": [{"role": "user", "content": "hello"}],
            },
        ),
        client.post(
            "/v1/inspect",
            json={
                "model": "gpt-5.5",
                "prompt": "inspect",
                "images": [{"image_url": "data:image/png;base64,AAAA"}],
            },
        ),
        client.post(
            "/v1/compact",
            json={
                "model": "gpt-5.5",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "hello"},
                ],
            },
        ),
    ]

    assert [response.status_code for response in requests] == [200] * len(requests)
    assert recording_backend.catalog_requests.qsize() == 1


def test_chat_config_reasoning_uses_one_catalog_snapshot_across_preflight(
    client,
    model_catalog_document,
    recording_backend: RecordingBackend,
    monkeypatch,
):
    import codex_as_api.server as server_mod

    now = time.time()
    snapshot_a = parse_model_catalog(
        model_catalog_document,
        key=("acc-123", recording_backend.base_url, "0.153.3"),
        etag='"catalog-a"',
        fetched_at=now,
        expires_at=now,
    )
    model = model_catalog_document["models"][0]
    model["supported_reasoning_levels"] = [{"effort": "low", "description": "low"}]
    model["default_reasoning_level"] = "low"
    model["multi_agent_reasoning_effort"] = "low"
    snapshot_b = parse_model_catalog(
        model_catalog_document,
        key=("acc-123", recording_backend.base_url, "0.153.3"),
        etag='"catalog-b"',
        fetched_at=now,
        expires_at=now + 300,
    )
    catalog_calls = 0

    def rotating_catalog():
        nonlocal catalog_calls
        catalog_calls += 1
        return snapshot_a if catalog_calls == 1 else snapshot_b

    monkeypatch.setattr(server_mod._provider, "get_model_catalog", rotating_catalog)
    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(
            codex_home="/tmp/codex",
            config_path="/tmp/codex/config.toml",
            model_reasoning_effort="high",
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert catalog_calls == 1
    assert recording_backend.requests.get(timeout=1)["body"]["reasoning"]["effort"] == "high"


def test_response_models_etag_mismatch_invalidates_catalog_before_next_request(
    recording_backend: RecordingBackend,
    auth_json_factory,
):
    provider = ChatGPTOAuthProvider(
        model="gpt-5.5",
        base_url=recording_backend.base_url,
        auth_json_path=str(auth_json_factory()),
    )
    provider.get_model_catalog()
    recording_backend.catalog_requests.get(timeout=1)
    recording_backend.response_etag[0] = '"changed-etag"'

    provider.chat(
        [
            Message(role=MessageRole.SYSTEM, content="system"),
            Message(role=MessageRole.USER, content="hello"),
        ],
        model="gpt-5.5",
    )
    provider.get_model_catalog()

    refreshed = recording_backend.catalog_requests.get(timeout=1)
    contract = _UPSTREAM_CONTRACT["models_request"]
    version = _UPSTREAM_CONTRACT["upstream"]["version"]
    assert refreshed["path"] == f"{contract['path']}?{contract['client_version_query']}={version}"


@pytest.mark.parametrize("second_request_succeeds", [True, False])
def test_catalog_401_refreshes_credentials_exactly_once(
    monkeypatch,
    auth_json_factory,
    model_catalog_document,
    second_request_succeeds,
):
    import codex_as_api.provider as provider_mod

    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            requests.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                }
            )
            if len(requests) == 1 or not second_request_succeeds:
                body = b'{"error":"unauthorized"}'
                self.send_response(401)
            else:
                body = json.dumps(model_catalog_document).encode("utf-8")
                self.send_response(200)
                self.send_header("ETag", '"retry-etag"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    refreshes = 0

    def refresh(token):
        nonlocal refreshes
        refreshes += 1
        return token

    monkeypatch.setattr(provider_mod, "refresh_after_unauthorized", refresh)
    host, port = server.server_address
    provider = ChatGPTOAuthProvider(
        base_url=f"http://{host}:{port}",
        auth_json_path=str(auth_json_factory()),
    )
    try:
        if second_request_succeeds:
            assert provider.get_model_catalog().etag == '"retry-etag"'
        else:
            with pytest.raises(ChatGPTOAuthUpstreamError) as exc_info:
                provider.get_model_catalog()
            assert exc_info.value.status == 401
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert len(requests) == 2
    assert refreshes == 1
    assert all(request["authorization"] for request in requests)


@pytest.mark.parametrize("status", [429, 529])
def test_catalog_preserves_non_auth_upstream_http_status(
    auth_json_factory,
    status,
):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            body = b'{"error":"catalog unavailable"}'
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    provider = ChatGPTOAuthProvider(
        base_url=f"http://{host}:{port}",
        auth_json_path=str(auth_json_factory()),
    )
    try:
        with pytest.raises(ChatGPTOAuthUpstreamError) as exc_info:
            provider.get_model_catalog()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert exc_info.value.status == status


def test_pinned_contract_matches_recorded_lite_responses_request(recording_backend, auth_json_factory):
    provider = ChatGPTOAuthProvider(
        model="gpt-5.6-sol",
        base_url=recording_backend.base_url,
        auth_json_path=str(auth_json_factory()),
        timeout=2,
    )

    provider.chat(
        [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Hello"),
        ],
        model="gpt-5.6-sol",
        reasoning_effort="low",
        responses_lite=True,
        parallel_tool_calls=False,
    )

    recorded = recording_backend.requests.get(timeout=1)
    request_contract = _UPSTREAM_CONTRACT["responses_request"]
    lite_contract = _UPSTREAM_CONTRACT["responses_lite"]
    originator_contract = _UPSTREAM_CONTRACT["headers"]["originator"]

    assert recorded["method"] == request_contract["method"]
    assert recorded["path"] == request_contract["path"]
    assert recorded["headers"]["accept"] == request_contract["streaming_accept"]
    assert recorded["headers"][originator_contract["name"]] == originator_contract["value"]
    assert recorded["headers"][lite_contract["header"]["name"]] == lite_contract["header"]["value"]
    assert recorded["body"]["reasoning"]["context"] == lite_contract["reasoning_context"]
    assert recorded["body"]["parallel_tool_calls"] is lite_contract["parallel_tool_calls"]
    assert request_contract["reasoning_encrypted_content_include"] in recorded["body"]["include"]


@pytest.mark.parametrize("status", [401, 429, 529])
def test_openai_and_anthropic_routes_preserve_structured_upstream_status(monkeypatch, model_catalog_snapshot, status):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod

    class FailingProvider:
        model = "gpt-5.5"

        def resolve_model(self, *_args, **_kwargs):
            capability = model_catalog_snapshot.model("gpt-5.5")
            return model_catalog_snapshot, capability

        def preflight_chat(self, *_args, **_kwargs):
            return SimpleNamespace(
                snapshot=model_catalog_snapshot,
                capability=model_catalog_snapshot.model("gpt-5.5"),
                payload={},
                replay_input=(),
            )

        def chat(self, *_args, **_kwargs):
            raise ChatGPTOAuthUpstreamError(status, "upstream status without parseable digits")

    monkeypatch.setattr(server_mod, "_provider", FailingProvider())
    client = TestClient(
        server_mod.app,
        raise_server_exceptions=False,
        headers={"x-claude-code-session-id": "test-claude-code-session"},
    )
    openai = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    anthropic = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
        },
    )

    assert openai.status_code == status
    assert anthropic.status_code == status
    expected_type = {401: "authentication_error", 429: "rate_limit_error", 529: "overloaded_error"}[status]
    assert anthropic.json()["error"]["type"] == expected_type


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "gpt-5.5",
                "messages": [{"role": "system", "content": "system"}],
                "stream": True,
            },
        ),
        (
            "/v1/messages",
            {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
                "stream": True,
            },
        ),
    ],
)
def test_streaming_catalog_failure_is_json_503_before_sse_headers(
    monkeypatch,
    path,
    payload,
):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod

    class CatalogFailureProvider:
        def resolve_model(self, *_args, **_kwargs):
            raise ChatGPTOAuthCatalogUnavailableError("catalog unavailable")

        def preflight_chat(self, *_args, **_kwargs):
            raise ChatGPTOAuthCatalogUnavailableError("catalog unavailable")

    monkeypatch.setattr(server_mod, "_provider", CatalogFailureProvider())
    strict_client = TestClient(
        server_mod.app,
        raise_server_exceptions=False,
        headers={"x-claude-code-session-id": "test-claude-code-session"},
    )

    response = strict_client.post(path, json=payload)

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "gpt-5.5",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "hello"},
                ],
            },
        ),
        (
            "/v1/messages",
            {
                "model": "claude-sonnet-4-6",
                "system": "system",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            },
        ),
    ],
)
def test_malformed_successful_upstream_payload_returns_502(
    client,
    monkeypatch,
    path,
    payload,
):
    import codex_as_api.server as server_mod

    def malformed(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text"}],
            },
        }

    monkeypatch.setattr(server_mod._provider, "_post_sse", malformed)

    response = client.post(path, json=payload)

    assert response.status_code == 502
    if path == "/v1/chat/completions":
        assert response.json()["error"]["type"] == "upstream_protocol_error"
    else:
        assert response.json()["error"]["type"] == "api_error"


def test_nonstream_openai_response_requires_provider_response_id(client, monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(
        server_mod._provider,
        "chat",
        lambda *_args, **_kwargs: AssistantResponse(
            content="ok",
            tool_calls=(),
            finish_reason=None,
            usage=None,
            response_id=None,
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_protocol_error"


def test_nonstream_openai_response_requires_final_finish_reason(client, monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(
        server_mod._provider,
        "chat",
        lambda *_args, **_kwargs: AssistantResponse(
            content="ok",
            tool_calls=(),
            finish_reason=None,
            usage=None,
            response_id="response-without-finish",
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.5", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_protocol_error"


@pytest.mark.parametrize("missing", ["finish_reason", "usage"])
def test_nonstream_anthropic_response_requires_final_contract(client, monkeypatch, missing):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(
        server_mod._provider,
        "chat",
        lambda *_args, **_kwargs: AssistantResponse(
            content="ok",
            tool_calls=(),
            finish_reason=None if missing == "finish_reason" else "stop",
            usage=None
            if missing == "usage"
            else Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            response_id="response-anthropic-contract",
        ),
    )

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"


@pytest.mark.parametrize(
    "finish_event",
    [
        {"type": "finish", "finish_reason": None, "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
        {"type": "finish", "finish_reason": "stop"},
        {"type": "web_search_call", "id": "search-1", "input": {"query": "q"}, "content": []},
    ],
)
def test_streaming_anthropic_final_contract_failures_are_in_band_without_success(
    client,
    monkeypatch,
    finish_event,
):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(
        server_mod._provider,
        "chat_stream",
        lambda *_args, **_kwargs: iter([finish_event]),
    )

    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert "event: message_start" in response.text
    assert "event: error" in response.text
    assert "event: message_stop" not in response.text


def test_nonstream_openai_rejects_normalized_web_search_call(client, monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(
        server_mod._provider,
        "chat",
        lambda *_args, **_kwargs: AssistantResponse(
            content="search complete",
            tool_calls=(),
            finish_reason="stop",
            usage=None,
            response_id="response-search",
            raw={
                "events": [
                    {
                        "type": "web_search_call",
                        "id": "search-1",
                        "input": {"query": "q"},
                        "content": [],
                    }
                ]
            },
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_protocol_error"


def test_streaming_openai_rejects_normalized_web_search_call_in_band(client, monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(
        server_mod._provider,
        "chat_stream",
        lambda *_args, **_kwargs: iter(
            [
                {
                    "type": "web_search_call",
                    "id": "search-1",
                    "input": {"query": "q"},
                    "content": [],
                }
            ]
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    error_events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {") and "error" in line
    ]
    assert error_events[0]["error"]["type"] == "upstream_protocol_error"
    assert "data: [DONE]" not in response.text


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "gpt-5.5",
                "messages": [{"role": "system", "content": "system"}],
            },
        ),
        (
            "/v1/messages",
            {
                "model": "gpt-5.5",
                "max_tokens": 32,
                "system": "system",
                "messages": [{"role": "user", "content": "hello"}],
            },
        ),
    ],
)
def test_upstream_usage_aliases_return_502(client, monkeypatch, path, payload):
    import codex_as_api.server as server_mod

    item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "ok"}],
    }

    def aliased_usage(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.output_item.done", "item": item}
        yield {
            "type": "response.completed",
            "response": {
                "id": "response-1",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "prompt_tokens": 1,
                },
            },
        }

    monkeypatch.setattr(server_mod._provider, "_post_sse", aliased_usage)

    response = client.post(path, json=payload)

    assert response.status_code == 502
    expected_type = "upstream_protocol_error" if path == "/v1/chat/completions" else "api_error"
    assert response.json()["error"]["type"] == expected_type


@pytest.mark.parametrize(
    ("path", "payload", "expected_type"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "gpt-5.5",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "hello"},
                ],
            },
            "server_error",
        ),
        (
            "/v1/messages",
            {
                "model": "gpt-5.5",
                "system": "system",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            },
            "api_error",
        ),
    ],
)
def test_unexpected_provider_failure_returns_structured_500(
    client,
    monkeypatch,
    path,
    payload,
    expected_type,
):
    import codex_as_api.server as server_mod

    def fail(*_args, **_kwargs):
        raise RuntimeError("private failure detail")

    monkeypatch.setattr(server_mod._provider, "chat", fail)
    response = client.post(path, json=payload)

    assert response.status_code == 500
    assert response.json()["error"]["type"] == expected_type
    assert "private failure detail" not in response.text


def test_model_environment_value_rejects_surrounding_or_only_whitespace(monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setenv("CODEX_AS_API_MODEL", "  gpt-5.6-sol  ")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        server_mod._env_str("CODEX_AS_API_MODEL", "gpt-5.5")

    monkeypatch.setenv("CODEX_AS_API_MODEL", "   ")
    with pytest.raises(ValueError, match="non-empty"):
        server_mod._env_str("CODEX_AS_API_MODEL", "gpt-5.5")


@pytest.mark.parametrize("value", ["0", "65536", "-1", "+80", " 80", "1_000", "１２"])
def test_port_environment_value_must_be_a_valid_tcp_port(monkeypatch, value):
    import codex_as_api.server as server_mod

    monkeypatch.setenv("CODEX_AS_API_PORT", value)
    with pytest.raises(ValueError, match="ASCII decimal digits|between 1 and 65535"):
        server_mod._env_int("CODEX_AS_API_PORT", 18080)


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
    assert body["context_window"] == 258_400
    assert body["auto_compact_token_limit"] == 244_800


def test_effective_context_overflow_is_catalog_unavailable_for_health_and_count(
    client,
    model_catalog_document,
):
    model = model_catalog_document["models"][0]
    model["context_window"] = 2**53 - 1
    model["max_context_window"] = 2**53 - 1
    model["effective_context_window_percent"] = 2**53 - 1

    health = client.get("/health")
    assert health.status_code == 503
    assert health.json()["error"]["type"] == "catalog_unavailable"

    count = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": model["slug"],
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert count.status_code == 503
    assert count.json()["error"]["type"] == "api_error"


def test_context_resolution_uses_live_max_when_context_is_absent(
    monkeypatch,
    model_catalog_snapshot,
):
    import codex_as_api.server as server_mod

    capability = replace(
        model_catalog_snapshot.model("gpt-5.6-sol"),
        context_window=None,
        max_context_window=300_000,
        auto_compact_token_limit=None,
    )
    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(codex_home="/tmp/codex", config_path="/tmp/codex/config.toml"),
    )

    assert server_mod._context_window(capability) == 285_000
    assert server_mod._auto_compact_token_limit(capability) == 270_000


def test_context_config_applies_without_a_live_max_or_context(
    monkeypatch,
    model_catalog_snapshot,
):
    import codex_as_api.server as server_mod

    capability = replace(
        model_catalog_snapshot.model("gpt-5.6-sol"),
        context_window=100_000,
        max_context_window=None,
        auto_compact_token_limit=None,
    )
    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(
            codex_home="/tmp/codex",
            config_path="/tmp/codex/config.toml",
            model_context_window=200_000,
        ),
    )
    assert server_mod._context_window(capability) == 190_000
    assert server_mod._auto_compact_token_limit(capability) == 180_000

    missing = replace(capability, context_window=None)
    assert server_mod._context_window(missing) == 190_000
    assert server_mod._auto_compact_token_limit(missing) == 180_000

    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(codex_home="/tmp/codex", config_path="/tmp/codex/config.toml"),
    )
    assert server_mod._context_window(missing) is None
    assert server_mod._auto_compact_token_limit(missing) is None


def test_nonpositive_live_catalog_limits_fail_when_consumed(
    monkeypatch,
    model_catalog_snapshot,
):
    import codex_as_api.server as server_mod

    capability = model_catalog_snapshot.model("gpt-5.6-sol")
    base_config = CodexConfig(codex_home="/tmp/codex", config_path="/tmp/codex/config.toml")
    monkeypatch.setattr(server_mod, "CODEX_CONFIG", base_config)
    for invalid in [
        replace(capability, context_window=0, max_context_window=100_000),
        replace(capability, context_window=None, max_context_window=-1),
    ]:
        with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
            server_mod._context_window(invalid)

    assert server_mod._auto_compact_token_limit(replace(capability, auto_compact_token_limit=0)) == 0
    assert server_mod._auto_compact_token_limit(replace(capability, auto_compact_token_limit=-1)) == -1
    assert (
        server_mod._auto_compact_token_limit(
            replace(capability, context_window=1, max_context_window=1, auto_compact_token_limit=None)
        )
        == 0
    )

    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        replace(base_config, model_context_window=50_000),
    )
    assert (
        server_mod._context_window(  # noqa: SLF001
            replace(capability, context_window=0, max_context_window=100_000)
        )
        == 47_500
    )
    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        server_mod._context_window(replace(capability, max_context_window=0))


def test_health_clamps_config_overrides_and_reports_wire_reasoning_effort(client, monkeypatch):
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
    assert body["context_window"] == 258_400
    assert body["auto_compact_token_limit"] == 244_800
    assert body["reasoning_effort"] == "max"


def test_health_unknown_configured_model_fails_explicitly(client, monkeypatch):
    import codex_as_api.server as server_mod

    server_mod._provider.model = "unknown-model"
    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(codex_home="/tmp/codex", config_path="/tmp/codex/config.toml"),
    )

    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["catalog_status"] == "fresh"
    assert body["error"]["type"] == "catalog_unavailable"


def test_health_unknown_model_does_not_use_compact_override_as_fallback(client, monkeypatch):
    import codex_as_api.server as server_mod

    server_mod._provider.model = "unknown-model"
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

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "catalog_unavailable"


def test_health_rejects_invalid_config_effort_without_exposing_config_path(client, monkeypatch):
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

    assert response.status_code == 503
    assert "codex_config_path" not in response.json()
    assert response.json()["reasoning_effort"] is None


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
    assert response.json()["error"]["type"] == "server_error"


def test_unsupported_config_effort_is_server_error_not_caller_error(client, monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(
            codex_home="/tmp/codex",
            config_path="/tmp/codex/config.toml",
            model_reasoning_effort="not-supported",
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 500
    assert response.json()["error"] == {
        "message": "internal server error",
        "type": "server_error",
        "code": "server_error",
    }


@pytest.mark.parametrize(
    ("path", "body", "config_error_type"),
    [
        (
            "/v1/images/generations",
            {"model": "gpt-5.6-sol", "prompt": "draw"},
            "server_error",
        ),
        (
            "/v1/messages",
            {
                "model": "gpt-5.6-sol",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}],
            },
            "api_error",
        ),
        (
            "/v1/inspect",
            {
                "model": "gpt-5.6-sol",
                "prompt": "inspect",
                "images": [{"image_url": "data:image/png;base64,AAAA"}],
            },
            "server_error",
        ),
    ],
)
def test_generation_routes_distinguish_config_and_request_reasoning_errors(
    client,
    model_catalog_document,
    recording_backend: RecordingBackend,
    monkeypatch,
    path,
    body,
    config_error_type,
):
    import codex_as_api.server as server_mod

    model_catalog_document["models"][0]["supported_reasoning_levels"] = [
        {"effort": "low", "description": "low"},
        {"effort": "medium", "description": "medium"},
    ]
    monkeypatch.setattr(
        server_mod,
        "CODEX_CONFIG",
        CodexConfig(
            codex_home="/tmp/codex",
            config_path="/tmp/codex/config.toml",
            model_reasoning_effort="high",
        ),
    )

    configured = client.post(path, json=body)
    explicit = client.post(path, json={**body, "reasoning_effort": "high"})

    assert configured.status_code == 500
    assert configured.json()["error"]["type"] == config_error_type
    assert explicit.status_code == 400
    assert explicit.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


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
        "type": "invalid_request_error",
        "code": "invalid_request_error",
    }


@pytest.mark.parametrize(
    ("path", "document", "error_type"),
    [
        (
            "/v1/chat/completions",
            '{"model":"gpt-5.6-sol","messages":[{"role":"user","content":"hello"}],"reasoning_effort":NaN}',
            "invalid_request_error",
        ),
        (
            "/v1/messages",
            '{"model":"gpt-5.6-sol","max_tokens":1e400,"messages":[{"role":"user","content":"hello"}]}',
            "invalid_request_error",
        ),
    ],
)
def test_http_routes_reject_nonstandard_json_before_dispatch(
    client,
    recording_backend: RecordingBackend,
    path,
    document,
    error_type,
):
    response = client.post(
        path,
        content=document.encode("utf-8"),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == error_type
    assert response.json()["error"]["message"] == "request body must contain valid JSON"
    assert recording_backend.catalog_requests.empty()
    assert recording_backend.requests.empty()


@pytest.mark.parametrize("content_type", [None, "text/plain", "application-json"])
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_post_json_routes_reject_unsupported_content_types(path, content_type):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod

    client = TestClient(server_mod.app, raise_server_exceptions=False)
    headers = {} if content_type is None else {"content-type": content_type}
    response = client.post(path, content=b"{}", headers=headers)

    assert response.status_code == 415
    if path.startswith("/v1/messages"):
        assert response.json()["type"] == "error"
        assert "code" not in response.json().get("error", {})
    else:
        assert response.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    "content_type",
    ["application/vnd.codex+json", "Application/JSON; Charset=UTF-8"],
)
def test_post_json_routes_accept_supported_json_media_types(content_type):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod

    client = TestClient(server_mod.app, raise_server_exceptions=False)
    response = client.post(
        "/v1/chat/completions",
        content=b"{}",
        headers={"content-type": content_type},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    ("declared_size", "expected_status"),
    [
        (50 * 1024 * 1024 - 1, 400),
        (50 * 1024 * 1024, 400),
        (50 * 1024 * 1024 + 1, 413),
    ],
)
def test_post_json_body_limit_has_exact_50_mib_boundary(declared_size, expected_status):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod

    client = TestClient(server_mod.app, raise_server_exceptions=False)
    response = client.post(
        "/v1/chat/completions",
        content=b"{}",
        headers={
            "content-type": "application/json",
            "content-length": str(declared_size),
        },
    )

    assert response.status_code == expected_status


def test_chat_tool_arguments_preserve_nonstandard_inner_json_as_raw_text(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.6-sol",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"value":NaN}'},
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    outbound = recording_backend.requests.get(timeout=1)["body"]
    function_call = next(item for item in outbound["input"] if item.get("type") == "function_call")
    assert function_call == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "lookup",
        "arguments": '{"value":NaN}',
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
        "type": "invalid_request_error",
        "code": "invalid_request_error",
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
        "type": "invalid_request_error",
        "code": "invalid_request_error",
    }
    assert recording_backend.requests.empty()


def test_chat_handler_reaches_real_provider_with_sol_ultra_lite_contract(
    client,
    auth_json_factory,
    recording_backend: RecordingBackend,
    monkeypatch,
):
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
            "parallel_tool_calls": False,
        },
    )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["choices"][0]["message"]["content"] == "backend-ok"
    assert response_body["choices"][0]["message"]["refusal"] is None
    assert response_body["choices"][0]["logprobs"] is None
    assert response_body["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "prompt_tokens_details": {
            "cached_tokens": 0,
        },
    }
    recorded = recording_backend.requests.get(timeout=1)
    assert recorded["path"] == "/responses"
    assert recorded["headers"][LITE_HEADER_NAME] == LITE_HEADER_VALUE
    request_body = recorded["body"]
    assert request_body["model"] == "gpt-5.6-sol"
    assert request_body["reasoning"] == {"effort": "max", "summary": "auto", "context": "all_turns"}
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


def test_chat_handler_rejects_standard_reasoning_mode_before_upstream(
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

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


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
    assert response.json()["error"]["type"] == "invalid_request_error"
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
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
            "reasoning": {"mode": "pro"},
            "responses_lite": False,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
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

    def resolve_once(value: str, *, account_id: str, current_comp_hash: str | None = None):
        resolved_ids.append(value)
        return original_resolve(
            value,
            account_id=account_id,
            current_comp_hash=current_comp_hash,
        )

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
    assert response.json()["error"]["type"] == "invalid_request_error"
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
        {"multi_agent": None},
        {"programmatic_tool_calling": {"enabled": True}},
        {"programmatic_tool_calling": None},
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
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


@pytest.mark.parametrize(
    "extra",
    [
        {"multi_agent": {"enabled": True}},
        {"multi_agent": None},
        {"programmatic_tool_calling": {"enabled": True}},
        {"programmatic_tool_calling": None},
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
        json={**base_request, "reasoning_effort": "high"},
    )

    assert config_response.status_code == 200
    assert request_response.status_code == 200
    config_request = recording_backend.requests.get(timeout=1)
    explicit_request = recording_backend.requests.get(timeout=1)
    assert config_request["body"]["reasoning"] == {
        "effort": "max",
        "summary": "auto",
        "context": "all_turns",
    }
    assert explicit_request["body"]["reasoning"] == {
        "effort": "high",
        "summary": "auto",
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
    assert request_body["reasoning"] == {"effort": "max", "summary": "auto", "context": "all_turns"}
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
    assert request_body["reasoning"] == {"effort": "low", "summary": "auto", "context": "all_turns"}
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
            "result": "data:image/png;base64,AAAA",
        }
    ]
    recorded = recording_backend.requests.get(timeout=1)
    assert LITE_HEADER_NAME not in recorded["headers"]
    request_body = recorded["body"]
    assert request_body["tools"] == [{"type": "image_generation", "output_format": "png"}]
    assert request_body["reasoning"] == {"effort": "low", "summary": "auto"}


def test_image_generation_response_omits_absent_optional_upstream_fields(client):
    response = client.post(
        "/v1/images/generations",
        json={"model": "gpt-5.5", "prompt": "draw", "responses_lite": False},
    )

    assert response.status_code == 200
    assert response.json()["data"] == [{"url": "data:image/png;base64,AAAA"}]


def test_image_generation_reference_images_reach_input_image_wire(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/images/generations",
        json={
            "model": "gpt-5.5",
            "prompt": "draw",
            "responses_lite": False,
            "reference_images": [
                {
                    "image_url": "data:image/png;base64,BBBB",
                    "detail": "high",
                    "prompt_cache_breakpoint": None,
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    outbound = recording_backend.requests.get(timeout=1)["body"]
    assert outbound["input"][0]["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,BBBB",
        "detail": "high",
    }


@pytest.mark.parametrize(
    "reference_images",
    [
        "not-an-array",
        [{"image_url": "data:image/png;base64,BBBB", "unknown": True}],
        [{"image_url": "https://example.test/image.png"}],
        [{"image_url": "data:image/png;base64,BBBB", "detail": "full"}],
        [{"image_url": "data:image/png;base64,BBBB", "prompt_cache_breakpoint": True}],
    ],
)
def test_image_generation_rejects_malformed_reference_images_before_upstream(
    client,
    recording_backend: RecordingBackend,
    reference_images,
):
    response = client.post(
        "/v1/images/generations",
        json={
            "model": "gpt-5.5",
            "prompt": "draw",
            "reference_images": reference_images,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.catalog_requests.empty()
    assert recording_backend.requests.empty()


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
                "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
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
            {
                "prompt": "inspect",
                "images": [{"image_url": "data:image/png;base64,AAAA"}],
                "responses_lite": "bogus",
            },
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
    ("path", "body"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "gpt-5.6-sol",
                "messages": [{"role": "user", "content": "hello"}],
                "responses_lite": None,
            },
        ),
        (
            "/v1/images/generations",
            {"model": "gpt-5.6-sol", "prompt": "draw", "responses_lite": None},
        ),
        (
            "/v1/inspect",
            {
                "prompt": "inspect",
                "images": [{"image_url": "data:image/png;base64,AAAA"}],
                "responses_lite": None,
            },
        ),
    ],
)
def test_openai_routes_reject_explicit_null_responses_lite_before_upstream(
    client,
    recording_backend: RecordingBackend,
    path,
    body,
):
    response = client.post(path, json=body)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
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
            {"prompt": "inspect", "images": [{"image_url": "data:image/png;base64,AAAA"}]},
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
    ("path", "body"),
    [
        (
            "/v1/images/generations",
            {"model": "gpt-5.5", "prompt": "draw"},
        ),
        (
            "/v1/inspect",
            {
                "model": "gpt-5.5",
                "prompt": "inspect",
                "images": [{"image_url": "data:image/png;base64,AAAA"}],
            },
        ),
    ],
)
def test_image_routes_reject_unmapped_tools_before_upstream(
    client,
    recording_backend: RecordingBackend,
    path,
    body,
):
    response = client.post(
        path,
        json={
            **body,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 400
    assert recording_backend.requests.empty()


def test_inspect_rejects_empty_images_before_catalog_or_upstream(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/inspect",
        json={"model": "gpt-5.5", "prompt": "inspect", "images": []},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.catalog_requests.empty()
    assert recording_backend.requests.empty()


def test_chat_completions_translates_standard_function_tool_choice(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup"},
            },
        },
    )

    assert response.status_code == 200, response.text
    assert recording_backend.requests.get(timeout=1)["body"]["tool_choice"] == {
        "type": "function",
        "name": "lookup",
    }


def test_chat_completions_forwards_function_strict(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                        "strict": True,
                    },
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert recording_backend.requests.get(timeout=1)["body"]["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "parameters": {"type": "object"},
            "strict": True,
        }
    ]


@pytest.mark.parametrize("field", ["defer_loading", "eager_input_streaming"])
@pytest.mark.parametrize("value", [None, False, True])
def test_chat_completions_rejects_enabled_private_tool_controls_before_upstream(
    client,
    recording_backend: RecordingBackend,
    field,
    value,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [{"role": "system", "content": "system"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                        field: value,
                    },
                }
            ],
        },
    )

    assert response.status_code == 400
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
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


def test_compact_treats_nullable_unsupported_params_as_omitted(
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
            "prompt_cache_options": None,
            "include": None,
            "prompt_cache_retention": None,
        },
    )

    assert response.status_code == 200
    outbound = recording_backend.requests.get(timeout=1)["body"]
    assert "prompt_cache_options" not in outbound
    assert "include" not in outbound
    assert "prompt_cache_retention" not in outbound


@pytest.mark.parametrize(
    ("route", "body"),
    [
        (
            "/v1/images/generations",
            {"prompt": "draw", "safety_identifier": None},
        ),
        (
            "/v1/inspect",
            {
                "prompt": "inspect",
                "images": [{"image_url": "data:image/png;base64,AAAA"}],
                "safety_identifier": None,
            },
        ),
        (
            "/v1/compact",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "safety_identifier": None,
            },
        ),
        (
            "/v1/messages",
            {
                "model": "claude-sonnet-4-5",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}],
                "safety_identifier": None,
            },
        ),
    ],
)
def test_safety_identifier_null_is_rejected_before_upstream(
    client,
    recording_backend: RecordingBackend,
    route,
    body,
):
    response = client.post(route, json=body)

    assert response.status_code == 400
    assert recording_backend.requests.empty()


def test_chat_treats_null_cache_options_and_breakpoint_as_omitted(
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
            "prompt_cache_options": None,
            "responses_lite": False,
        },
    )

    assert response.status_code == 200
    outbound = recording_backend.requests.get(timeout=1)["body"]
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
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


@pytest.mark.parametrize("failure", ["response_failed", "truncated_eof"])
def test_openai_runtime_stream_failure_is_reported_in_band(client, monkeypatch, failure):
    import codex_as_api.server as server_mod

    def failing_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.output_text.delta", "delta": "partial"}
        if failure == "response_failed":
            yield {
                "type": "response.failed",
                "response": {"error": {"message": "access_token=sentinel"}},
            }

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
    expected = "upstream request failed" if failure == "response_failed" else "upstream protocol validation failed"
    assert expected in response.text
    assert "sentinel" not in response.text
    assert "data: [DONE]" not in response.text


def test_openai_stream_preserves_empty_normalized_text_deltas(client, monkeypatch):
    import codex_as_api.server as server_mod

    def empty_deltas(*_args, **_kwargs):
        yield {"type": "reasoning_delta", "text": ""}
        yield {"type": "reasoning_raw_delta", "text": ""}
        yield {"type": "content", "text": ""}
        yield {
            "type": "finish",
            "finish_reason": "stop",
            "response_id": "resp-empty-deltas",
            "usage": None,
        }

    monkeypatch.setattr(server_mod._provider, "chat_stream", empty_deltas)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    chunks = [
        json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: {")
    ]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk.get("choices")]
    assert {"content": ""} in deltas
    assert {"reasoning_content": ""} in deltas
    assert {"reasoning": ""} in deltas


def test_openai_stream_rejects_missing_final_finish_reason_in_band(client, monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(
        server_mod._provider,
        "chat_stream",
        lambda *_args, **_kwargs: iter(
            [{"type": "finish", "finish_reason": None, "response_id": "response-null-finish"}]
        ),
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert '"type": "upstream_protocol_error"' in response.text
    assert "data: [DONE]" not in response.text


@pytest.mark.parametrize("failure", ["response_failed", "truncated_eof"])
def test_anthropic_runtime_stream_failure_is_reported_in_band(client, monkeypatch, failure):
    import codex_as_api.server as server_mod

    def failing_sse(_path, _payload, extra_headers=None):  # noqa: ANN001
        del extra_headers
        yield {"type": "response.output_text.delta", "delta": "partial"}
        if failure == "response_failed":
            yield {
                "type": "response.failed",
                "response": {"error": {"message": "access_token=sentinel"}},
            }

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
    expected = "upstream request failed" if failure == "response_failed" else "upstream protocol validation failed"
    assert expected in response.text
    assert "sentinel" not in response.text


def test_request_validation_error_does_not_reflect_invalid_input(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": {"access_token": "sentinel"},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "sentinel" not in response.text


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
                "output": [],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 4,
                    "total_tokens": 15,
                    "input_tokens_details": {"cached_tokens": 0},
                },
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
        },
    }


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — schema validation
# ---------------------------------------------------------------------------


def test_chat_completions_invalid_body_returns_structured_400(client):
    resp = client.post("/v1/chat/completions", json={})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "messages": [{"role": "user", "content": "hello"}],
            "unknown_field": True,
        },
        {
            "messages": [{"role": "user", "content": "hello", "unknown_field": True}],
        },
        {
            "messages": [{"role": "USER", "content": "hello"}],
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        },
        {
            "messages": [{"role": "user", "content": "hello"}],
            "stream": "false",
        },
    ],
)
def test_chat_completions_rejects_unknown_coerced_or_noncanonical_input(
    client,
    payload,
):
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    "request_fields",
    [
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": {"query": "docs"},
                            },
                        }
                    ],
                }
            ]
        },
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {},
                        "strict": "true",
                    },
                }
            ],
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello", "future": True}],
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,AAAA",
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
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,AAAA",
                                "detail": "low",
                            },
                            "detail": "high",
                        }
                    ],
                }
            ]
        },
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup", "future": True},
            },
        },
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": {"type": "auto"},
        },
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": "",
        },
    ],
)
def test_chat_completions_rejects_malformed_content_and_tools_before_upstream(
    client,
    recording_backend: RecordingBackend,
    request_fields,
):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.5", **request_fields},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


def test_chat_completions_preserves_empty_content_arrays_and_assistant_null_semantics(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [
                {"role": "developer", "content": []},
                {"role": "user", "content": []},
                {"role": "assistant", "content": None},
                {"role": "assistant", "tool_calls": []},
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert recording_backend.requests.get(timeout=1)["body"]["input"] == [
        {"type": "message", "role": "developer", "content": []},
        {"type": "message", "role": "user", "content": []},
        {"type": "message", "role": "assistant", "content": []},
        {"type": "message", "role": "assistant", "content": []},
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "draw", "unknown_field": True},
        {"prompt": 123},
    ],
)
def test_image_generation_strict_schema_rejects_invalid_input(client, payload):
    response = client.post("/v1/images/generations", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    "field",
    [
        "audio",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "metadata",
        "modalities",
        "n",
        "prediction",
        "presence_penalty",
        "prompt_cache_retention",
        "reasoning_effort",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "top_logprobs",
        "top_p",
        "verbosity",
    ],
)
def test_chat_official_nullable_controls_accept_explicit_null(client, field):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
            field: None,
        },
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "function_call",
        "functions",
        "parallel_tool_calls",
        "prompt_cache_key",
        "response_format",
        "safety_identifier",
        "tool_choice",
        "tools",
        "user",
        "web_search_options",
    ],
)
def test_chat_official_non_nullable_controls_reject_explicit_null_before_upstream(
    client,
    recording_backend: RecordingBackend,
    field,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            field: None,
        },
    )
    assert response.status_code == 400
    assert recording_backend.requests.empty()


@pytest.mark.parametrize(
    "field",
    [
        "background",
        "model",
        "moderation",
        "n",
        "output_compression",
        "output_format",
        "partial_images",
        "quality",
        "response_format",
        "size",
        "stream",
        "style",
    ],
)
def test_image_generation_official_nullable_controls_accept_explicit_null(client, field):
    response = client.post(
        "/v1/images/generations",
        json={"prompt": "draw", field: None, "responses_lite": False},
    )
    assert response.status_code == 200, response.text


def test_image_generation_rejects_explicit_null_user_before_upstream(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/images/generations",
        json={"prompt": "draw", "user": None},
    )
    assert response.status_code == 400
    assert recording_backend.requests.empty()


@pytest.mark.parametrize("field", ["audio", "function_call", "refusal"])
def test_assistant_message_nullable_omission_fields_accept_null(client, field):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [{"role": "assistant", "content": "hello", field: None}],
        },
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": "hello", "name": None},
        {"role": "assistant", "content": "hello", "tool_calls": None},
        {"role": "user", "content": "hello", "tool_calls": None},
        {"role": "user", "content": "hello", "tool_call_id": None},
        {"role": "tool", "content": "result", "tool_call_id": "call-1", "name": None},
    ],
)
def test_chat_messages_reject_non_nullable_and_role_inapplicable_nulls_before_upstream(
    client,
    recording_backend: RecordingBackend,
    message,
):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.5", "messages": [message]},
    )
    assert response.status_code == 400
    assert recording_backend.requests.empty()


def test_openai_function_definition_defaults_missing_parameters_and_nullable_strict(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "strict": None},
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert recording_backend.requests.get(timeout=1)["body"]["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "parameters": {},
            "strict": False,
        }
    ]


@pytest.mark.parametrize("function", [{"name": "lookup", "description": None}, {"name": "lookup", "parameters": None}])
def test_openai_function_definition_rejects_non_nullable_nulls_before_upstream(
    client,
    recording_backend: RecordingBackend,
    function,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": function}],
        },
    )
    assert response.status_code == 400
    assert recording_backend.requests.empty()


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
    assert "x-codex-backend-model" not in resp.headers


@pytest.mark.parametrize("subagent", ["has space", "line\nbreak", "tab\tvalue", "한글"])
def test_chat_completions_rejects_header_unsafe_body_subagent(client, subagent):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Hi"}],
            "subagent": subagent,
        },
    )
    assert response.status_code == 400


def test_streaming_chat_rejects_header_unsafe_subagent_before_sse_headers(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
            "subagent": "has space",
            "stream": True,
        },
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["type"] == "invalid_request_error"


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


@pytest.mark.parametrize(
    ("headers", "body"),
    [
        ({"x-openai-memgen-request": "yes"}, {}),
        ({"x-openai-subagent": "   "}, {}),
        (
            {"x-openai-memgen-request": "false"},
            {"memgen_request": True},
        ),
        (
            {"x-openai-subagent": "header-agent"},
            {"subagent": "body-agent"},
        ),
    ],
)
def test_chat_completions_rejects_invalid_or_conflicting_transport_headers(
    client,
    headers,
    body,
):
    payload = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "Hi"}],
        **body,
    }

    response = client.post("/v1/chat/completions", json=payload, headers=headers)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


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
    assert resp.json()["error"]["type"] == "invalid_request_error"
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
    assert body["error"]["type"] == "authentication_error"
    assert body["error"]["message"] == ("ChatGPT OAuth credentials are unavailable; rerun codex login")
    assert missing_auth_path not in resp.text

    models = client.get("/v1/models")
    assert models.status_code == 401
    assert models.json()["error"]["message"] == ("ChatGPT OAuth credentials are unavailable; rerun codex login")
    assert missing_auth_path not in models.text


def test_messages_count_tokens_counts_normalized_tools_with_live_catalog(client):
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
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "input_tokens",
        "context_window",
        "auto_compact_token_limit",
    }
    assert body["input_tokens"] == 51
    assert body["context_window"] >= body["auto_compact_token_limit"]


def test_messages_count_tokens_accepts_disabled_thinking_with_live_catalog(client):
    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "look up docs"}],
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "high"},
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["input_tokens"] > 0


@pytest.mark.parametrize(
    "route",
    ["/v1/messages", "/v1/messages/count_tokens", "/v1/messages/compact"],
)
def test_anthropic_routes_reject_hosted_web_search_before_catalog_or_provider(
    client,
    recording_backend: RecordingBackend,
    route,
):
    response = client.post(
        route,
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "search the web"}],
            "tools": [{"type": "web_search_20260209", "name": "web_search"}],
        },
    )

    assert response.status_code == 400
    assert "cannot be represented losslessly" in response.json()["error"]["message"]
    assert recording_backend.catalog_requests.empty()
    assert recording_backend.requests.empty()


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


def test_messages_count_tokens_rejects_unknown_control_fields(client):
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
    assert with_controls.status_code == 400
    assert base.json()["input_tokens"] == 13
    assert with_controls.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/messages",
            {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
                "unknown_field": True,
            },
        ),
        (
            "/v1/messages",
            {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
                "text": {"format": {"type": "text"}},
            },
        ),
        (
            "/v1/messages/count_tokens",
            {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hello"}],
                "unknown_field": True,
            },
        ),
        (
            "/v1/inspect",
            {
                "prompt": "inspect",
                "images": [{"image_url": "data:image/png;base64,AAAA"}],
                "unknown_field": True,
            },
        ),
        (
            "/v1/compact",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "unknown_field": True,
            },
        ),
        (
            "/v1/messages/compact",
            {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
                "unknown_field": True,
            },
        ),
    ],
)
def test_raw_body_endpoints_reject_unknown_top_level_fields_before_upstream(
    client,
    recording_backend: RecordingBackend,
    path,
    payload,
):
    response = client.post(path, json=payload)

    assert response.status_code == 400
    assert recording_backend.requests.empty()


@pytest.mark.parametrize(
    ("path", "field"),
    [
        ("/v1/messages", field)
        for field in ("system", "tools", "tool_choice", "stop_sequences", "thinking", "output_config", "stream")
    ]
    + [
        (path, field)
        for path in ("/v1/messages/count_tokens", "/v1/messages/compact")
        for field in ("system", "tools", "tool_choice", "stop_sequences", "thinking", "output_config")
    ],
)
def test_anthropic_routes_reject_non_nullable_top_level_nulls_before_transport(
    client,
    recording_backend: RecordingBackend,
    path,
    field,
):
    payload = {
        "model": "gpt-5.6-sol",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
        field: None,
    }
    response = client.post(
        path,
        json=payload,
        headers={"x-claude-code-session-id": "null-validation"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.catalog_requests.empty()
    assert recording_backend.requests.empty()


def test_anthropic_cache_control_null_is_omitted_without_claude_session(
    client,
):
    messages_body = {
        "model": "gpt-5.6-sol",
        "messages": [{"role": "user", "content": "hello"}],
    }
    baseline = client.post("/v1/messages", json=messages_body)
    with_null = client.post("/v1/messages", json={**messages_body, "cache_control": None})
    assert with_null.status_code == baseline.status_code == 400
    assert with_null.json() == baseline.json()

    count = client.post(
        "/v1/messages/count_tokens",
        json={**messages_body, "cache_control": None},
    )
    compact_baseline = client.post("/v1/messages/compact", json=messages_body)
    compact_with_null = client.post(
        "/v1/messages/compact",
        json={**messages_body, "cache_control": None},
    )
    assert count.status_code == 200
    assert compact_with_null.status_code == compact_baseline.status_code == 400
    assert compact_with_null.json() == compact_baseline.json()


@pytest.mark.parametrize("tools", [{}, "", False, 0])
def test_anthropic_messages_rejects_non_array_tools_before_upstream(
    client,
    recording_backend: RecordingBackend,
    tools,
):
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": tools,
        },
    )

    assert response.status_code == 400
    assert recording_backend.requests.empty()


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/compact"])
@pytest.mark.parametrize("max_tokens", [None, 0, True, 1.5, "32"])
def test_anthropic_generation_endpoints_require_positive_integer_max_tokens(
    client,
    recording_backend: RecordingBackend,
    path,
    max_tokens,
):
    payload = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hello"}],
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    response = client.post(path, json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": "hello", "name": "user-name"},
        {
            "role": "tool",
            "content": "result",
            "tool_call_id": "call-1",
            "name": "lookup",
        },
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": " "}],
        },
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": " "}}],
        },
    ],
)
def test_chat_messages_reject_unrepresentable_names_and_blank_image_urls_before_upstream(
    client,
    recording_backend: RecordingBackend,
    message,
):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.5", "messages": [message]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.catalog_requests.empty()
    assert recording_backend.requests.empty()


def test_chat_input_audio_maps_to_private_audio_url(
    client,
    recording_backend: RecordingBackend,
    model_catalog_document,
):
    model = next(model for model in model_catalog_document["models"] if model["slug"] == "gpt-5.5")
    model["input_modalities"] = ["audio"]

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "AAAA", "format": "wav"},
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    outbound = recording_backend.requests.get(timeout=1)["body"]
    assert outbound["input"][0]["content"] == [
        {"type": "input_audio", "audio_url": "data:audio/wav;base64,AAAA"}
    ]


@pytest.mark.parametrize(
    "input_audio",
    [
        {"data": "AAAA"},
        {"data": "AAAA", "format": "flac"},
        {"data": 1, "format": "wav"},
        {"data": "AAAA", "format": "wav", "extra": True},
    ],
)
def test_chat_input_audio_rejects_invalid_shapes_before_catalog(
    client,
    recording_backend: RecordingBackend,
    input_audio,
):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "input_audio", "input_audio": input_audio}],
                }
            ],
        },
    )

    assert response.status_code == 400
    assert recording_backend.catalog_requests.empty()
    assert recording_backend.requests.empty()


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": "hello"}],
                "prompt_cache_key": " ",
            },
        ),
        (
            "/v1/compact",
            {
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": "hello"}],
                "prompt_cache_key": " ",
            },
        ),
    ],
)
def test_chat_and_compact_reject_blank_prompt_cache_key_before_upstream(
    client,
    recording_backend: RecordingBackend,
    path,
    payload,
):
    response = client.post(path, json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


@pytest.mark.parametrize(
    "messages",
    [
        [1],
        [{"role": "user", "content": [1]}],
    ],
)
def test_anthropic_cache_control_walk_does_not_hide_malformed_messages(
    client,
    recording_backend: RecordingBackend,
    messages,
):
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "messages": messages,
            "max_tokens": 32,
        },
    )

    assert response.status_code == 400
    assert recording_backend.requests.empty()


@pytest.mark.parametrize(
    "nested_payload",
    [
        {"messages": [{"role": "user", "content": "hello", "future": True}]},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello", "future": True}],
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
    ],
)
def test_anthropic_messages_rejects_unknown_nested_fields_before_upstream(
    client,
    recording_backend: RecordingBackend,
    nested_payload,
):
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
            **nested_payload,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/inspect"])
def test_raw_body_endpoints_reject_invalid_json(client, path):
    response = client.post(
        path,
        content="{",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    "unsupported",
    [
        {"multi_agent": {"enabled": True}},
        {"multi_agent": None},
        {"programmatic_tool_calling": {"enabled": True}},
        {"programmatic_tool_calling": None},
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
            "system": "system",
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


@pytest.mark.parametrize(
    ("max_tokens_literal", "budget_literal"),
    [("2048.0", "1024.0"), ("2048e0", "1024e0")],
)
def test_anthropic_max_tokens_and_thinking_budget_accept_integral_json_number_forms(
    client,
    max_tokens_literal,
    budget_literal,
):
    response = client.post(
        "/v1/messages/count_tokens",
        content=(
            '{"model":"gpt-5.5","messages":[{"role":"user","content":"hello"}],'
            f'"max_tokens":{max_tokens_literal},'
            f'"thinking":{{"type":"enabled","budget_tokens":{budget_literal}}}}}'
        ),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200, response.text


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


def test_messages_count_tokens_rejects_unrepresentable_assistant_thinking(client):
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

    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


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
    assert known.json()["context_window"] == 258_400
    assert known.json()["auto_compact_token_limit"] == 244_800
    assert fallback.status_code == 200
    assert fallback.json()["context_window"] == 258_400
    assert fallback.json()["auto_compact_token_limit"] == 244_800


def test_messages_compact_accepts_anthropic_body(
    client,
    monkeypatch,
    model_catalog_snapshot,
):
    import codex_as_api.server as server_mod

    class DummyProvider:
        model = "gpt-5.6-sol"

        def resolve_model(self, *_args, **_kwargs):
            return model_catalog_snapshot, model_catalog_snapshot.model(self.model)

        def compact_messages(
            self,
            messages,
            *,
            model=None,
            tools=None,
            reasoning_effort=None,
            responses_lite=None,
            **_kwargs,
        ):
            assert model == self.model
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
            "max_tokens": 2048,
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
            "tool_choice": {
                "type": "auto",
                "disable_parallel_tool_use": True,
            },
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"checkpoint": "checkpoint"}


def test_messages_compact_rejects_enabled_parallel_tool_use_before_upstream(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/messages/compact",
        json={
            "model": "gpt-5.5",
            "max_tokens": 1024,
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": {
                "type": "auto",
                "disable_parallel_tool_use": False,
            },
        },
    )

    assert response.status_code == 400


def test_messages_count_tokens_rejects_stop_sequences_before_catalog_or_provider(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
            "stop_sequences": ["stop"],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.catalog_requests.empty()
    assert recording_backend.requests.empty()
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


@pytest.mark.parametrize(
    "unsupported",
    [
        {"stop_sequences": ["stop"]},
        {"tool_choice": {"type": "any"}},
        {"tool_choice": {"type": "tool", "name": "lookup"}},
        {"tool_choice": {"type": "none"}},
        {"subagent": "ignored-agent"},
        {"memgen_request": False},
    ],
)
def test_messages_compact_rejects_nonforwarded_controls_before_upstream(
    client,
    recording_backend: RecordingBackend,
    unsupported,
):
    response = client.post(
        "/v1/messages/compact",
        json={
            "model": "gpt-5.5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            **unsupported,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


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
    assert invalid.json()["type"] == "error"
    assert invalid.json()["error"]["type"] == "invalid_request_error"
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
        {"tools": [{"name": "lookup", "input_schema": {}, "strict": "true"}]},
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
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
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
    assert recorded["body"]["reasoning"] == {"effort": "max", "summary": "auto"}
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


def test_inspect_preserves_original_detail_in_classic_and_rejects_lite_loss(
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
    assert lite.status_code == 400
    assert lite.json()["error"]["type"] == "invalid_request_error"
    classic_request = recording_backend.requests.get(timeout=1)
    assert classic_request["body"]["input"][0]["content"][1]["detail"] == "original"
    assert recording_backend.requests.empty()


def test_messages_compact_uses_anthropic_content_block_conversion_without_system(
    client,
    monkeypatch,
    model_catalog_snapshot,
):
    import codex_as_api.server as server_mod

    class DummyProvider:
        model = "gpt-5.6-sol"

        def resolve_model(self, *_args, **_kwargs):
            return model_catalog_snapshot, model_catalog_snapshot.model(self.model)

        def compact_messages(self, messages, **kwargs):
            assert kwargs["model"] == self.model
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


def test_anthropic_messages_uses_backend_model_and_client_model_in_response(
    client,
    recording_backend: RecordingBackend,
):
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "claude-sonnet-4-5"
    assert recording_backend.requests.get(timeout=1)["body"]["model"] == "gpt-5.6-sol"


def test_anthropic_messages_forwards_openai_transport_controls(
    client,
    recording_backend: RecordingBackend,
):
    base = {
        "model": "gpt-5.5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    }
    body_response = client.post(
        "/v1/messages",
        json={**base, "subagent": "agent-1", "memgen_request": False},
    )
    header_response = client.post(
        "/v1/messages",
        json=base,
        headers={
            "x-openai-subagent": "agent-2",
            "x-openai-memgen-request": "true",
        },
    )

    assert body_response.status_code == 200
    assert header_response.status_code == 200
    body_request = recording_backend.requests.get(timeout=1)
    header_request = recording_backend.requests.get(timeout=1)
    assert body_request["headers"]["x-openai-subagent"] == "agent-1"
    assert body_request["headers"]["x-openai-memgen-request"] == "false"
    assert header_request["headers"]["x-openai-subagent"] == "agent-2"
    assert header_request["headers"]["x-openai-memgen-request"] == "true"


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        ({"subagent": "bad value"}, {}),
        ({"subagent": "body-agent"}, {"x-openai-subagent": "header-agent"}),
        ({"memgen_request": "true"}, {}),
        ({"memgen_request": True}, {"x-openai-memgen-request": "false"}),
        ({}, {"x-openai-memgen-request": "TRUE"}),
    ],
)
def test_anthropic_messages_rejects_invalid_transport_controls(client, body, headers):
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            **body,
        },
        headers=headers,
    )
    assert response.status_code == 400


def test_anthropic_messages_forwards_function_strict_and_empty_description(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.5",
            "max_tokens": 1024,
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "name": "lookup",
                    "description": "",
                    "input_schema": {"type": "object"},
                    "strict": True,
                    "eager_input_streaming": None,
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert recording_backend.requests.get(timeout=1)["body"]["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "",
            "parameters": {"type": "object"},
            "strict": True,
        }
    ]


@pytest.mark.parametrize(
    ("disable_parallel_tool_use", "expected_parallel_tool_calls"),
    [(True, False), (False, True)],
)
def test_anthropic_tool_choice_maps_parallel_control_to_provider(
    client,
    recording_backend: RecordingBackend,
    disable_parallel_tool_use,
    expected_parallel_tool_calls,
):
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.5",
            "max_tokens": 1024,
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": {
                "type": "auto",
                "disable_parallel_tool_use": disable_parallel_tool_use,
            },
        },
    )

    assert response.status_code == 200, response.text
    request_body = recording_backend.requests.get(timeout=1)["body"]
    assert request_body["parallel_tool_calls"] is expected_parallel_tool_calls


def test_anthropic_parallel_tool_choice_rejects_lite_before_upstream(
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
            "tool_choice": {
                "type": "auto",
                "disable_parallel_tool_use": False,
            },
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


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


def test_anthropic_messages_without_session_rejects_unforwarded_max_tokens(
    client,
    recording_backend: RecordingBackend,
):
    from fastapi.testclient import TestClient

    with TestClient(client.app, raise_server_exceptions=False) as no_session_client:
        response = no_session_client.post(
            "/v1/messages",
            json={
                "model": "gpt-5.6-sol",
                "max_tokens": 1024,
                "system": "system",
                "messages": [{"role": "user", "content": "hello"}],
                "previous_response_id": None,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.catalog_requests.empty()
    assert recording_backend.requests.empty()


@pytest.mark.parametrize(
    ("route", "payload"),
    [
        (
            "/v1/messages/count_tokens",
            {
                "model": "gpt-5.6-sol",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "hello"}],
            },
        ),
        (
            "/v1/messages/count_tokens",
            {
                "model": "gpt-5.6-sol",
                "cache_control": {"type": "ephemeral"},
                "messages": [{"role": "user", "content": "hello"}],
            },
        ),
        (
            "/v1/messages/compact",
            {
                "model": "gpt-5.6-sol",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "hello"}],
            },
        ),
    ],
)
def test_anthropic_noop_compatibility_fields_require_claude_code_session(
    client,
    recording_backend: RecordingBackend,
    route,
    payload,
):
    from fastapi.testclient import TestClient

    with TestClient(client.app, raise_server_exceptions=False) as no_session_client:
        response = no_session_client.post(route, json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.catalog_requests.empty()
    assert recording_backend.requests.empty()


@pytest.mark.parametrize("route", ["/v1/messages", "/v1/messages/count_tokens", "/v1/messages/compact"])
def test_anthropic_routes_strictly_resolve_claude_code_session_header(
    client,
    recording_backend: RecordingBackend,
    route,
):
    from fastapi.testclient import TestClient

    payload = {
        "model": "gpt-5.6-sol",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    }
    with TestClient(client.app, raise_server_exceptions=False) as no_session_client:
        blank = no_session_client.post(
            route,
            headers={"x-claude-code-session-id": " "},
            json=payload,
        )
        duplicate = no_session_client.post(
            route,
            headers=[
                ("x-claude-code-session-id", "session-a"),
                ("x-claude-code-session-id", "session-b"),
            ],
            json=payload,
        )

    assert blank.status_code == 400
    assert duplicate.status_code == 400
    assert recording_backend.catalog_requests.empty()
    assert recording_backend.requests.empty()


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


def test_anthropic_messages_rejects_message_level_cache_control(
    client,
    recording_backend: RecordingBackend,
):
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": "hello",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    )

    assert response.status_code == 400
    assert recording_backend.requests.empty()


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
            "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
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


def test_anthropic_disabled_thinking_fails_when_live_model_lacks_none_effort(
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
            "messages": [{"role": "user", "content": "look up docs"}],
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "high"},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert recording_backend.requests.empty()


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


def test_anthropic_stream_does_not_block_concurrent_claude_code_requests(
    monkeypatch,
    model_catalog_snapshot,
):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod

    provider_waiting = threading.Event()
    release_provider = threading.Event()

    class GatedProvider:
        model = "gpt-5.6-sol"
        catalog_ttl = 300

        def get_model_catalog(self):
            return model_catalog_snapshot

        def resolve_model(self, *_args, **_kwargs):
            return model_catalog_snapshot, model_catalog_snapshot.model(self.model)

        def preflight_chat(self, messages, **kwargs):
            del messages, kwargs
            return SimpleNamespace(
                snapshot=model_catalog_snapshot,
                capability=model_catalog_snapshot.model(self.model),
                payload={},
                replay_input=(),
            )

        def chat_stream(self, messages, **kwargs):
            del messages, kwargs
            yield {"type": "content", "text": "early"}
            provider_waiting.set()
            if not release_provider.wait(timeout=5):
                raise AssertionError("test did not release the provider stream")
            yield {
                "type": "finish",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            }

    monkeypatch.setattr(server_mod, "_provider", GatedProvider())
    stream_result: dict[str, Any] = {}
    health_result: dict[str, Any] = {}

    with TestClient(
        server_mod.app,
        raise_server_exceptions=False,
        headers={"x-claude-code-session-id": "test-claude-code-session"},
    ) as concurrent_client:

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


def test_anthropic_builtin_claude_model_uses_explicit_configured_backend(
    client,
    recording_backend: RecordingBackend,
):
    import codex_as_api.server as server_mod

    server_mod._provider.model = "gpt-5.5"
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


@pytest.mark.parametrize("reasoning_effort", ["", 0, False])
def test_anthropic_messages_rejects_falsey_invalid_reasoning_effort_before_upstream(
    client,
    recording_backend: RecordingBackend,
    reasoning_effort,
):
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-5.6-sol",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": reasoning_effort,
        },
    )

    assert response.status_code == 400
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
                "Anthropic hosted web_search cannot be represented losslessly by this facade"
            ),
        },
    }
    assert recording_backend.requests.empty()
