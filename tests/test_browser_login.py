"""Offline browser OAuth behavior over a real loopback callback listener."""
import asyncio
import base64
import hashlib
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from codex_as_api.auth import CHATGPT_OAUTH_CLIENT_ID, commit_login
from codex_as_api.browser_login import AUTHORIZE_URL, SCOPES, BrowserLoginManager
from codex_as_api.device_login import TOKEN_URL, LoginError


async def callback(port, target, *, host=None, method="GET"):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        f"{method} {target} HTTP/1.1\r\nHost: {host or f'127.0.0.1:{port}'}\r\n\r\n".encode()
    )
    await writer.drain()
    data = await reader.read()
    writer.close()
    await writer.wait_closed()
    return data


def test_browser_callback_validation_pkce_and_real_commit(tmp_path, make_jwt):
    async def scenario():
        path = tmp_path / "auth.json"
        jwt = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "fake-account"}})
        tokens = {"access_token": jwt, "id_token": jwt, "refresh_token": "fake-refresh"}
        calls = []

        async def transport(url, body, headers, timeout):
            calls.append((url, parse_qs(body.decode()), headers, timeout))
            return 200, json.dumps(tokens).encode()

        async def commit(value, guard):
            return await asyncio.to_thread(commit_login, path, value, guard)

        manager = BrowserLoginManager(commit, transport=transport, ports=(0,))
        try:
            started = await manager.start()
            url = urlsplit(started["auth_url"])
            assert url.scheme + "://" + url.netloc + url.path == AUTHORIZE_URL
            query = parse_qs(url.query)
            assert query["client_id"] == [CHATGPT_OAUTH_CLIENT_ID]
            assert query["scope"] == [SCOPES]
            assert query["code_challenge_method"] == ["S256"]
            redirect = urlsplit(query["redirect_uri"][0])
            assert redirect.hostname == "127.0.0.1" and redirect.path == "/auth/callback"
            port, state = redirect.port, query["state"][0]
            invalid = [
                ("/other?state=" + state + "&code=x", None, "GET"),
                ("/auth/callback?state=wrong&code=x", None, "GET"),
                ("/auth/callback?state=" + state + "&state=" + state + "&code=x", None, "GET"),
                ("/auth/callback?state=" + state + "&code=x&error=denied", None, "GET"),
                ("/auth/callback?state=" + state + "&code=x", "example.com", "GET"),
                ("/auth/callback?state=" + state + "&code=x", None, "POST"),
            ]
            for target, host, method in invalid:
                response = await callback(port, target, host=host, method=method)
                assert b"400 Bad Request" in response
                assert manager.get(started["id"])["status"] == "pending"
            assert not calls and not path.exists()
            response = await callback(port, "/auth/callback?state=" + state + "&code=fake-code")
            assert b"200 OK" in response and b"Cache-Control: no-store" in response
            result = manager.get(started["id"])
            assert result["status"] == "succeeded" and result["method"] == "browser"
            assert "auth_url" not in result and "fake-refresh" not in str(result)
            assert json.loads(path.read_text())["tokens"] == tokens
            assert len(calls) == 1
            endpoint, exchange, headers, timeout = calls[0]
            assert endpoint == TOKEN_URL and headers["Content-Type"] == "application/x-www-form-urlencoded"
            assert 0 < timeout <= 900
            assert exchange["grant_type"] == ["authorization_code"]
            assert exchange["code"] == ["fake-code"]
            assert exchange["redirect_uri"] == query["redirect_uri"]
            assert exchange["client_id"] == query["client_id"]
            challenge = base64.urlsafe_b64encode(hashlib.sha256(exchange["code_verifier"][0].encode()).digest())
            assert challenge.rstrip(b"=").decode() == query["code_challenge"][0]
            assert (await manager.cancel(started["id"]))["status"] == "succeeded"
        finally:
            await manager.close()

    asyncio.run(scenario())


def test_cancel_during_exchange_preserves_existing_store(tmp_path, make_jwt):
    async def scenario():
        path = tmp_path / "auth.json"
        jwt = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "fake-account"}})
        old = {"access_token": jwt, "id_token": jwt, "refresh_token": "old-refresh"}
        commit_login(path, old)
        original = path.read_bytes()
        entered = asyncio.Event()
        blocked = asyncio.Event()

        async def transport(url, body, headers, timeout):
            entered.set()
            await blocked.wait()
            return 200, json.dumps({**old, "refresh_token": "new-refresh"}).encode()

        async def commit(value, guard):
            return await asyncio.to_thread(commit_login, path, value, guard)

        manager = BrowserLoginManager(commit, transport=transport, ports=(0,))
        try:
            started = await manager.start()
            query = parse_qs(urlsplit(started["auth_url"]).query)
            port = urlsplit(query["redirect_uri"][0]).port
            request = asyncio.create_task(callback(port, "/auth/callback?state=" + query["state"][0] + "&code=x"))
            await asyncio.wait_for(entered.wait(), 1)
            assert (await manager.cancel(started["id"]))["status"] == "cancelled"
            blocked.set()
            await request
            assert manager.get(started["id"])["status"] == "cancelled"
            assert path.read_bytes() == original
        finally:
            await manager.close()

    asyncio.run(scenario())


def test_fastapi_default_browser_login_end_to_end(monkeypatch, tmp_path, make_jwt):
    import http.client

    from fastapi.testclient import TestClient

    import codex_as_api.server as server

    path = tmp_path / "auth.json"
    jwt = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "fake-account"}})
    tokens = {"access_token": jwt, "id_token": jwt, "refresh_token": "fake-refresh"}

    async def transport(url, body, headers, timeout):
        assert url == TOKEN_URL
        return 200, json.dumps(tokens).encode()

    monkeypatch.setenv("CODEX_AS_API_ADMIN_KEY", "fake-admin")
    monkeypatch.setattr(server, "AUTH_PATH", str(path))
    monkeypatch.setattr(server, "_login_manager", None)
    monkeypatch.setattr(server, "_browser_login_manager", BrowserLoginManager(
        server._commit_device_login, transport=transport, ports=(0,)
    ))
    with TestClient(server.app) as client:
        headers = {"Authorization": "Bearer fake-admin"}
        response = client.post("/auth/login", headers=headers)
        assert response.status_code == 202
        started = response.json()
        assert started["method"] == "browser" and started["status"] == "pending"
        query = parse_qs(urlsplit(started["auth_url"]).query)
        port = urlsplit(query["redirect_uri"][0]).port
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request("GET", "/auth/callback?state=" + query["state"][0] + "&code=fake-code")
            callback_response = connection.getresponse()
            assert callback_response.status == 200
            callback_response.read()
        finally:
            connection.close()
        status = client.get("/auth/login/" + started["id"], headers=headers).json()
        assert status["method"] == "browser" and status["status"] == "succeeded"
        assert "auth_url" not in status and "fake-refresh" not in str(status)
        assert json.loads(path.read_text())["tokens"] == tokens


def test_commit_value_error_fails_and_releases_login_slot():
    async def scenario():
        async def transport(url, body, headers, timeout):
            return 200, json.dumps({"access_token": "a", "id_token": "i", "refresh_token": "r"}).encode()

        async def commit(tokens, guard):
            raise ValueError("private storage detail")

        manager = BrowserLoginManager(commit, transport=transport, ports=(0,))
        try:
            started = await manager.start()
            query = parse_qs(urlsplit(started["auth_url"]).query)
            port = urlsplit(query["redirect_uri"][0]).port
            response = await callback(port, "/auth/callback?state=" + query["state"][0] + "&code=x")
            assert b"400 Bad Request" in response and b"private storage detail" not in response
            record = manager.get(started["id"])
            assert record["status"] == "failed" and record["error_code"] == "login_storage_error"
            assert (await manager.cancel(started["id"]))["status"] == "failed"
            assert (await manager.start())["status"] == "pending"
        finally:
            await manager.close()

    asyncio.run(scenario())


def test_busy_listener_fallback_and_denial():
    async def scenario():
        async def never_commit(tokens, guard):
            pytest.fail("unexpected commit")

        async def never_transport(*args):
            pytest.fail("unexpected token exchange")

        owner = await asyncio.start_server(lambda reader, writer: writer.close(), "127.0.0.1", 0)
        occupied = owner.sockets[0].getsockname()[1]
        manager = BrowserLoginManager(never_commit, transport=never_transport, ports=(occupied, 0))
        busy = BrowserLoginManager(never_commit, transport=never_transport, ports=(occupied, occupied))
        try:
            with pytest.raises(LoginError) as error:
                await busy.start()
            assert error.value.code == "callback_port_unavailable"
            assert busy._active is None
            started = await manager.start()
            query = parse_qs(urlsplit(started["auth_url"]).query)
            port = urlsplit(query["redirect_uri"][0]).port
            assert port != occupied
            response = await callback(port, "/auth/callback?state=" + query["state"][0] + "&error=access_denied")
            assert b"400 Bad Request" in response
            record = manager.get(started["id"])
            assert record["status"] == "failed" and record["error_code"] == "login_denied"
            assert "access_denied" not in str(record)
            with pytest.raises(OSError):
                await asyncio.open_connection("127.0.0.1", port)
            assert owner.is_serving()
        finally:
            await manager.close()
            await busy.close()
            owner.close()
            await owner.wait_closed()

    asyncio.run(scenario())


def test_expiry_closes_loopback_listener():
    async def scenario():
        wake = asyncio.Event()
        clock = [0.0]

        async def sleep(delay):
            await wake.wait()

        async def never_commit(tokens, guard):
            pytest.fail("unexpected commit")

        manager = BrowserLoginManager(never_commit, ports=(0,), lifetime=1, monotonic=lambda: clock[0], sleep=sleep)
        try:
            started = await manager.start()
            query = parse_qs(urlsplit(started["auth_url"]).query)
            port = urlsplit(query["redirect_uri"][0]).port
            clock[0] = 2
            wake.set()
            for _ in range(10):
                if manager.get(started["id"])["status"] == "expired":
                    break
                await asyncio.sleep(0)
            assert manager.get(started["id"])["status"] == "expired"
            with pytest.raises(OSError):
                await asyncio.open_connection("127.0.0.1", port)
        finally:
            await manager.close()

    asyncio.run(scenario())
