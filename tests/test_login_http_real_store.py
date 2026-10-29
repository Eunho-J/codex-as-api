"""Offline HTTP integration: real FastAPI, login manager, and durable store."""
import asyncio
import json
import os
import socket
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codex_as_api import auth, server
from codex_as_api.device_login import POLL_URL, TOKEN_URL, USERCODE_URL, DeviceLoginManager

HEADERS = {"Authorization": "Bearer fake-admin"}


@pytest.fixture
def setup_login(monkeypatch, tmp_path, make_jwt):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "legacy"))
    monkeypatch.setenv("CODEX_AS_API_AUTH_PATH", str(tmp_path / "project" / "auth.json"))
    monkeypatch.setenv("CODEX_AS_API_ADMIN_KEY", "fake-admin")
    monkeypatch.delenv("CODEX_AS_API_HOME", raising=False)
    monkeypatch.setattr(server, "AUTH_PATH", None)
    monkeypatch.setattr(server, "_provider", None)
    monkeypatch.setattr(server, "_login_manager", None)

    def no_network(*args, **kwargs):
        pytest.fail("unexpected network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    jwt = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "new-account"}})
    tokens = {"access_token": jwt, "id_token": jwt, "refresh_token": "new-secret"}
    path = tmp_path / "project" / "auth.json"
    return path, tokens


def response_for(url, tokens):
    if url == USERCODE_URL:
        body = {"device_auth_id": "device", "user_code": "user-code", "interval": 1}
    elif url == POLL_URL:
        body = {"authorization_code": "code", "code_verifier": "verifier", "code_challenge": "challenge"}
    else:
        assert url == TOKEN_URL
        body = tokens
    return 200, json.dumps(body).encode()


def start(client):
    response = client.post("/auth/login", headers=HEADERS, json={"method": "device"})
    assert response.status_code == 202, response.text
    return "/auth/login/" + response.json()["id"]


def wait_terminal(client, route):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(route, headers=HEADERS)
        assert response.status_code == 200, response.text
        if response.json()["status"] != "pending":
            return response
        time.sleep(0.01)
    pytest.fail("login did not finish")


def test_success_only_after_real_store_is_durable(monkeypatch, setup_login):
    path, tokens = setup_login
    entered, release = threading.Event(), threading.Event()
    original_fsync = os.fsync

    def blocked_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("directory fsync not released")
        return original_fsync(fd)

    async def transport(url, body, headers, timeout):
        return response_for(url, tokens)

    monkeypatch.setattr(auth.os, "fsync", blocked_directory_fsync)
    monkeypatch.setattr(server, "_login_manager", DeviceLoginManager(server._commit_device_login, transport=transport))
    with TestClient(server.app) as client:
        try:
            route = start(client)
            assert entered.wait(5)
            assert client.get(route, headers=HEADERS).json()["status"] == "pending"
            assert auth.load_token_data(path).account_id == "new-account"
        finally:
            release.set()
        response = wait_terminal(client, route)
        assert response.json()["status"] == "succeeded"
        assert response.headers["cache-control"] == "no-store"
        assert "new-secret" not in response.text
        assert path.stat().st_mode & 0o777 == 0o600
        assert client.get("/auth/status", headers=HEADERS).json() == {"status": "authenticated"}


@pytest.mark.parametrize("blocked_url", [POLL_URL, TOKEN_URL], ids=["poll", "exchange"])
def test_delete_blocked_provider_prevents_late_write(monkeypatch, setup_login, blocked_url, make_jwt):
    path, tokens = setup_login
    old_jwt = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "old-account"}})
    auth.commit_login(path, {"access_token": old_jwt, "id_token": old_jwt, "refresh_token": "old-secret"})
    original = path.read_bytes()
    entered, cancelled, release, returned = (threading.Event() for _ in range(4))

    async def transport(url, body, headers, timeout):
        if url == blocked_url:
            entered.set()
            try:
                await asyncio.to_thread(release.wait, 5)
            except asyncio.CancelledError:
                cancelled.set()
                await asyncio.to_thread(release.wait, 5)
            returned.set()
        return response_for(url, tokens)

    monkeypatch.setattr(server, "_login_manager", DeviceLoginManager(server._commit_device_login, transport=transport))
    with TestClient(server.app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        try:
            route = start(client)
            assert entered.wait(5)
            future = executor.submit(client.delete, route, headers=HEADERS)
            assert cancelled.wait(5), "DELETE did not cancel blocked provider call"
        finally:
            release.set()
        response = future.result(timeout=5)
        assert returned.wait(5)
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        assert client.get(route, headers=HEADERS).json()["status"] == "cancelled"
        assert path.read_bytes() == original
        assert list(path.parent.glob(".auth.json.tmp-*")) == []


@pytest.mark.parametrize(
    "stage,expected", [("before_replace", "login_storage_error"), ("after_replace", "login_persistence_uncertain")]
)
def test_real_store_failure_reported_over_http(monkeypatch, setup_login, stage, expected, make_jwt):
    path, tokens = setup_login
    old_jwt = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "old-account"}})
    auth.commit_login(path, {"access_token": old_jwt, "id_token": old_jwt, "refresh_token": "old-secret"})
    old = path.read_bytes()

    async def transport(url, body, headers, timeout):
        return response_for(url, tokens)

    if stage == "before_replace":
        def fail_replace(self, target):
            raise OSError("fake replace failure")
        monkeypatch.setattr(Path, "replace", fail_replace)
    else:
        original_fsync = os.fsync
        def fail_directory_fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("fake directory fsync failure")
            return original_fsync(fd)
        monkeypatch.setattr(auth.os, "fsync", fail_directory_fsync)

    monkeypatch.setattr(server, "_login_manager", DeviceLoginManager(server._commit_device_login, transport=transport))
    with TestClient(server.app) as client:
        response = wait_terminal(client, start(client))
        assert response.json()["status"] == "failed"
        assert response.json()["error_code"] == expected
        assert "secret" not in response.text
        if stage == "before_replace":
            assert path.read_bytes() == old
            assert auth.load_token_data(path).account_id == "old-account"
        else:
            assert auth.load_token_data(path).account_id == "new-account"
        assert list(path.parent.glob(".auth.json.tmp-*")) == []
