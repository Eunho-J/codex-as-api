"""Offline tests for management request body limits and error responses."""

import socket

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    import codex_as_api.server as server

    monkeypatch.setenv("CODEX_AS_API_ADMIN_KEY", "test-admin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "legacy"))
    monkeypatch.setenv("CODEX_AS_API_AUTH_PATH", str(tmp_path / "auth.json"))
    monkeypatch.setattr(server, "_login_manager", None)

    def no_network(*args, **kwargs):
        pytest.fail("unexpected network connection")

    class FakeManager:
        _active = None

        async def start(self):
            return {"id": "fake", "status": "pending"}

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(server, "_get_login_manager", lambda: FakeManager())
    monkeypatch.setattr(server, "_get_browser_login_manager", lambda: FakeManager())
    with TestClient(server.app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.mark.parametrize("body", [
    b" " * 4097,
    b"{" + b" " * 4095 + b"}",
    b'{"secret":',
    b'\xff',
    b'[NaN]',
    b'[' * 1500 + b']' * 1500,
])
def test_bad_management_body_returns_json_envelope_no_store(client, body):
    response = client.post("/auth/login", headers={"Authorization": "Bearer test-admin"}, content=body)
    assert response.status_code == 400
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "error": {"code": "invalid_request", "message": "Request body must be empty or an empty object"}
    }


@pytest.mark.parametrize(("body", "method"), [
    (b"{" + b" " * 4094 + b"}", "browser"),
    (b'{"method":"browser"}', "browser"),
    (b'{"method":"device"}', "device"),
])
def test_valid_management_body_selects_method(client, body, method):
    response = client.post("/auth/login", headers={"Authorization": "Bearer test-admin"}, content=body)
    assert response.status_code == 202
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"id": "fake", "method": method, "status": "pending"}


@pytest.mark.parametrize("first,second", [("browser", "device"), ("device", "browser")])
def test_one_active_slot_across_methods(client, monkeypatch, first, second):
    import codex_as_api.server as server

    class Manager:
        _active = None

        async def start(self):
            self._active = True
            return {"id": "fake", "status": "pending"}

    browser, device = Manager(), Manager()
    monkeypatch.setattr(server, "_get_login_manager", lambda: device)
    monkeypatch.setattr(server, "_get_browser_login_manager", lambda: browser)
    headers = {"Authorization": "Bearer test-admin"}
    assert client.post("/auth/login", headers=headers, json={"method": first}).status_code == 202
    response = client.post("/auth/login", headers=headers, json={"method": second})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "login_in_progress"


@pytest.mark.parametrize("body", [b'{"method":"unknown"}', b'{"method":"device","extra":1}', b'{"method":null}'])
def test_invalid_method_rejected(client, body):
    response = client.post("/auth/login", headers={"Authorization": "Bearer test-admin"}, content=body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
