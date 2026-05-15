from __future__ import annotations

import json

import pytest

from codex_as_api.auth import ChatGPTOAuthError, ChatGPTOAuthMissingError


@pytest.fixture()
def client():
    from codex_as_api.server import app
    from fastapi.testclient import TestClient
    return TestClient(app, raise_server_exceptions=False)


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
    assert resp.status_code in (200, 401, 500)
    if resp.status_code == 422:
        pytest.fail(f"Schema validation rejected a valid request: {resp.json()}")


def test_chat_completions_auth_error_not_422(client):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code != 422


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
    assert resp.status_code != 422


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
    assert resp.status_code != 422


def test_chat_completions_previous_response_id_field_accepted(client):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        "previous_response_id": "resp-abc123",
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code != 422


def test_chat_completions_all_extended_fields_accepted(client):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        "subagent": "agent-1",
        "memgen_request": False,
        "previous_response_id": "resp-xyz",
        "reasoning_effort": "high",
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code != 422


def test_chat_completions_missing_auth_returns_auth_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_AS_API_AUTH_PATH", str(tmp_path / "nonexistent.json"))
    import importlib
    import codex_as_api.server as server_mod
    server_mod.AUTH_PATH = str(tmp_path / "nonexistent.json")
    server_mod._provider = None
    from codex_as_api.server import app
    from fastapi.testclient import TestClient
    c = TestClient(app, raise_server_exceptions=False)
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    }
    resp = c.post("/v1/chat/completions", json=payload)
    assert resp.status_code in (401, 500)
    body = resp.json()
    assert "error" in body
    assert body["error"]["type"] == "chatgpt_oauth_error"
    server_mod._provider = None


def test_messages_count_tokens_returns_real_provider_value(client, monkeypatch):
    import codex_as_api.server as server_mod

    class DummyProvider:
        def count_tokens(self, messages, *, model=None, tools=None, tool_choice=None, stop=None, reasoning_effort=None):
            assert model == server_mod.MODEL
            assert [m.content for m in messages] == ["You are helpful.", "hello"]
            return 42

    monkeypatch.setattr(server_mod, "_provider", DummyProvider())
    resp = client.post("/v1/messages/count_tokens", json={
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["input_tokens"] == 42
    assert body["context_window"] >= body["auto_compact_token_limit"]
    server_mod._provider = None


def test_messages_count_tokens_falls_back_to_estimate_when_provider_rejects(client, monkeypatch):
    import codex_as_api.server as server_mod

    class DummyProvider:
        def count_tokens(self, messages, **kwargs):
            raise RuntimeError("Unsupported parameter: max_output_tokens")

    monkeypatch.setattr(server_mod, "_provider", DummyProvider())
    resp = client.post("/v1/messages/count_tokens", json={
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["input_tokens"] > 0
    server_mod._provider = None


def test_messages_compact_accepts_anthropic_body(client, monkeypatch):
    import codex_as_api.server as server_mod

    class DummyProvider:
        def compact_messages(self, messages, *, model=None, reasoning_effort=None):
            assert model == server_mod.MODEL
            assert reasoning_effort == "high"
            assert [m.content for m in messages] == ["sys", "hello"]
            return "checkpoint"

    monkeypatch.setattr(server_mod, "_provider", DummyProvider())
    resp = client.post("/v1/messages/compact", json={
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "system": "sys",
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200
    assert resp.json() == {"checkpoint": "checkpoint"}
    server_mod._provider = None


def test_anthropic_messages_uses_codex_model_for_provider_and_client_model_in_response(client, monkeypatch):
    import codex_as_api.server as server_mod
    from codex_as_api.messages import AssistantResponse

    class DummyProvider:
        def chat(self, messages, **kwargs):
            assert kwargs["model"] == server_mod.MODEL
            return AssistantResponse(content="ok")

    monkeypatch.setattr(server_mod, "_provider", DummyProvider())
    resp = client.post("/v1/messages", json={
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200
    assert resp.json()["model"] == "claude-sonnet-4-5"
    server_mod._provider = None
