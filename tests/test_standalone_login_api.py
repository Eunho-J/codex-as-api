"""Offline contract tests for the standalone login management API."""

from __future__ import annotations

import json
import socket
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def management_client(monkeypatch, tmp_path):
    import codex_as_api.server as server

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "legacy"))
    monkeypatch.setenv("CODEX_AS_API_AUTH_PATH", str(tmp_path / "project" / "auth.json"))
    monkeypatch.delenv("CODEX_AS_API_HOME", raising=False)
    monkeypatch.delenv("CODEX_AS_API_ADMIN_KEY", raising=False)
    monkeypatch.setattr(server, "_provider", None)

    def no_network(*args, **kwargs):
        pytest.fail("management API attempted a real network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    with TestClient(server.app, raise_server_exceptions=False) as client:
        yield client


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    assert response.headers.get("cache-control") == "no-store"
    assert response.json()["error"]["code"] == code


ROUTES = [
    ("POST", "/auth/login"),
    ("GET", "/auth/login/missing"),
    ("DELETE", "/auth/login/missing"),
    ("GET", "/auth/status"),
]


@pytest.mark.parametrize(("method", "path"), ROUTES)
def test_management_disabled_without_admin_key(management_client, method, path):
    response = management_client.request(method, path, headers={"Authorization": "Bearer anything"})
    assert_error(response, 503, "management_disabled")


@pytest.mark.parametrize(("method", "path"), ROUTES)
@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "Basic secret", "Bearer "])
def test_management_requires_independent_admin_bearer(monkeypatch, management_client, method, path, authorization):
    monkeypatch.setenv("CODEX_AS_API_ADMIN_KEY", "fake-admin-key")
    headers = {} if authorization is None else {"Authorization": authorization}
    response = management_client.request(method, path, headers=headers)
    assert_error(response, 401, "invalid_admin_key")
    assert "fake-admin-key" not in response.text


@pytest.mark.parametrize(("method", "path"), ROUTES)
@pytest.mark.parametrize("origin", ["https://evil.example", "http://testserver:81", "null"])
def test_management_rejects_foreign_origin(monkeypatch, management_client, method, path, origin):
    monkeypatch.setenv("CODEX_AS_API_ADMIN_KEY", "fake-admin-key")
    response = management_client.request(
        method, path, headers={"Authorization": "Bearer fake-admin-key", "Origin": origin}
    )
    assert_error(response, 403, "origin_not_allowed")


@pytest.mark.parametrize("method", ["GET", "DELETE"])
def test_unknown_session_is_safe_and_no_store(monkeypatch, management_client, method):
    monkeypatch.setenv("CODEX_AS_API_ADMIN_KEY", "fake-admin-key")
    response = management_client.request(
        method,
        "/auth/login/not-a-real-session",
        headers={"Authorization": "Bearer fake-admin-key", "Origin": "http://testserver"},
    )
    assert_error(response, 404, "login_not_found")


def test_fake_issuer_actual_store_and_fastapi_success(monkeypatch, management_client, tmp_path, make_jwt):
    import codex_as_api.server as server
    from codex_as_api.device_login import POLL_URL, USERCODE_URL, DeviceLoginManager

    monkeypatch.setenv("CODEX_AS_API_ADMIN_KEY", "fake-admin-key")
    monkeypatch.setattr(server, "AUTH_PATH", None)
    jwt = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "fake-account"}})

    async def transport(url, body, headers, timeout):
        if url == USERCODE_URL:
            result = {"device_auth_id": "fake-device", "user_code": "fake-code"}
        elif url == POLL_URL:
            result = {"authorization_code": "fake-code-auth", "code_verifier": "fake-verifier",
                      "code_challenge": "fake-challenge"}
        else:
            result = {"access_token": jwt, "id_token": jwt, "refresh_token": "fake-refresh"}
        return 200, json.dumps(result).encode()

    monkeypatch.setattr(server, "_login_manager", DeviceLoginManager(server._commit_device_login, transport=transport))
    headers = {"Authorization": "Bearer fake-admin-key"}
    started = management_client.post("/auth/login", headers=headers, json={"method": "device"})
    assert started.status_code == 202
    assert started.json()["user_code"] == "fake-code"
    for _ in range(100):
        result = management_client.get("/auth/login/" + started.json()["id"], headers=headers)
        if result.json()["status"] != "pending":
            break
        time.sleep(0.01)
    assert result.json()["status"] == "succeeded"
    assert result.headers["cache-control"] == "no-store"
    assert "fake-code" not in result.text and "fake-refresh" not in result.text
    path = tmp_path / "project" / "auth.json"
    assert path.is_file() and (path.stat().st_mode & 0o777) == 0o600
    assert management_client.get("/auth/status", headers=headers).json() == {"status": "authenticated"}


def test_status_only_checks_local_project_store(monkeypatch, management_client, tmp_path):
    monkeypatch.setenv("CODEX_AS_API_ADMIN_KEY", "fake-admin-key")
    legacy = tmp_path / "legacy" / "auth.json"
    legacy.parent.mkdir()
    legacy.write_text('{"tokens":{"access_token":"fake-legacy-secret"}}')
    response = management_client.get("/auth/status", headers={"Authorization": "Bearer fake-admin-key"})
    assert response.status_code == 200, response.text
    assert response.headers.get("cache-control") == "no-store"
    assert response.json() == {"status": "unauthenticated"}
    assert "fake-legacy-secret" not in response.text


def test_status_recognizes_valid_local_credential_without_export(
    monkeypatch, management_client, tmp_path, auth_json_factory
):
    monkeypatch.setenv("CODEX_AS_API_ADMIN_KEY", "fake-admin-key")
    project = tmp_path / "project" / "auth.json"
    project.parent.mkdir()
    project.write_bytes(auth_json_factory(refresh_token="fake-refresh-secret").read_bytes())
    response = management_client.get("/auth/status", headers={"Authorization": "Bearer fake-admin-key"})
    assert response.status_code == 200, response.text
    assert response.headers.get("cache-control") == "no-store"
    assert response.json() == {"status": "authenticated"}
    assert "fake-refresh-secret" not in response.text


def test_status_reports_malformed_project_store(monkeypatch, management_client, tmp_path):
    monkeypatch.setenv("CODEX_AS_API_ADMIN_KEY", "fake-admin-key")
    project = tmp_path / "project" / "auth.json"
    project.parent.mkdir()
    project.write_text("not-json fake-private-value")
    response = management_client.get("/auth/status", headers={"Authorization": "Bearer fake-admin-key"})
    assert response.status_code == 200, response.text
    assert response.headers.get("cache-control") == "no-store"
    assert response.json() == {"status": "reauthentication_required"}
    assert "fake-private-value" not in response.text


@pytest.mark.parametrize(
    "body",
    [
        {"issuer": "https://evil.example"},
        {"client_id": "evil"},
        {"redirect_uri": "https://evil.example"},
        {"path": "/tmp/evil"},
        {"access_token": "fake-secret"},
        {"refresh_token": "fake-secret"},
        {"id_token": "fake-secret"},
        [],
        "anything",
        1,
    ],
)
def test_start_rejects_nonempty_or_nonobject_body_without_network(monkeypatch, management_client, body):
    monkeypatch.setenv("CODEX_AS_API_ADMIN_KEY", "fake-admin-key")
    response = management_client.post("/auth/login", headers={"Authorization": "Bearer fake-admin-key"}, json=body)
    assert 400 <= response.status_code < 500, response.text
    assert response.headers.get("cache-control") == "no-store"
    assert "fake-secret" not in response.text
