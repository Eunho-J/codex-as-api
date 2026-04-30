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
