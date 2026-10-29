"""Device login protocol tests: injected transport, clock and commit only."""
from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs

import pytest

from codex_as_api.device_login import (
    POLL_URL,
    REDIRECT_URI,
    TOKEN_URL,
    USERCODE_URL,
    VERIFICATION_URL,
    DeviceLoginManager,
    LoginError,
    default_transport,
)


def run(coro):
    return asyncio.run(coro)


def encoded(value):
    return json.dumps(value).encode()


async def wait_terminal(manager, login_id):
    for _ in range(100):
        result = manager.get(login_id)
        if result["status"] != "pending":
            return result
        await asyncio.sleep(0)
    pytest.fail("login did not reach a terminal state")


def test_default_transport_cancellation_closes_inflight_stream(monkeypatch):
    async def scenario():
        entered = asyncio.Event()
        closed = []

        class Client:
            def __init__(self, **kwargs):
                assert kwargs["follow_redirects"] is False

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                closed.append("client")

            def stream(self, *args, **kwargs):
                return Stream()

        class Stream:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                closed.append("stream")

            async def aiter_bytes(self):
                entered.set()
                await asyncio.Event().wait()
                yield b""

        monkeypatch.setattr("codex_as_api.device_login.httpx.AsyncClient", Client)
        task = asyncio.create_task(default_transport(USERCODE_URL, b"{}", {}, 1))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed == ["stream", "client"]

    run(scenario())


def test_device_success_polling_and_exchange_protocol():
    async def scenario():
        calls = []
        commits = []
        polls = iter([(403, b"ignored"), (404, b"ignored"), (200, encoded({
            "authorization_code": "fake-authorization", "code_verifier": "fake-verifier",
            "code_challenge": "fake-challenge"}))])
        async def transport(url, body, headers, timeout):
            calls.append((url, body, headers, timeout))
            if url == USERCODE_URL:
                return 200, encoded({"device_auth_id": "fake-device-id", "usercode": "fake-code", "interval": "2"})
            if url == POLL_URL:
                return next(polls)
            assert url == TOKEN_URL
            return 200, encoded({"access_token": "fake-access", "refresh_token": "fake-refresh", "id_token": "fake-id"})
        async def commit(tokens, guard):
            assert guard()
            commits.append(tokens)
            return True
        sleeps = []
        async def sleep(delay):
            sleeps.append(delay)
        manager = DeviceLoginManager(commit, transport=transport, sleep=sleep)
        started = await manager.start()
        assert started["verification_url"] == VERIFICATION_URL
        assert started["user_code"] == "fake-code"
        assert started["interval"] == 2
        terminal = await wait_terminal(manager, started["id"])
        assert terminal["status"] == "succeeded"
        assert set(terminal) == {"id", "status", "expires_at"}
        assert (await manager.cancel(started["id"]))["status"] == "succeeded"
        assert sleeps == [2, 2]
        assert [call[0] for call in calls] == [USERCODE_URL, POLL_URL, POLL_URL, POLL_URL, TOKEN_URL]
        assert json.loads(calls[0][1]) == {"client_id": "app_EMoamEEZ73f0CkXaXp7hrann"}
        assert json.loads(calls[1][1]) == {"device_auth_id": "fake-device-id", "user_code": "fake-code"}
        exchange = parse_qs(calls[-1][1].decode())
        assert exchange == {"grant_type": ["authorization_code"], "code": ["fake-authorization"],
                            "redirect_uri": [REDIRECT_URI], "client_id": ["app_EMoamEEZ73f0CkXaXp7hrann"],
                            "code_verifier": ["fake-verifier"]}
        assert all(0 < c[3] <= 900 for c in calls)
        assert commits == [{"access_token": "fake-access", "refresh_token": "fake-refresh", "id_token": "fake-id"}]
        assert "fake-code" not in str(terminal) and "fake-access" not in str(terminal)
        await manager.close()
    run(scenario())


@pytest.mark.parametrize("status,body,expected", [
    (404, b"fake-upstream-secret", "device_login_unsupported"),
    (500, b"fake-upstream-secret", "login_upstream_error"),
    (200, b"not-json fake-upstream-secret", "invalid_login_response"),
    (200, encoded({"device_auth_id": "device", "user_code": "code", "interval": 0}), "invalid_login_response"),
    (200, encoded({"device_auth_id": "device", "user_code": "code", "interval": "-1"}), "invalid_login_response"),
    (200, encoded({"device_auth_id": "device", "user_code": "code", "interval": True}), "invalid_login_response"),
])
def test_initial_failures_are_safe(status, body, expected):
    async def scenario():
        async def transport(*args):
            return status, body
        async def commit(*args):
            pytest.fail("failed initiation must not commit")
        manager = DeviceLoginManager(commit, transport=transport)
        with pytest.raises(LoginError) as error:
            await manager.start()
        assert error.value.code == expected
        assert "fake-upstream-secret" not in str(error.value)
        await manager.close()
    run(scenario())


def test_cancel_interrupts_poll_and_never_commits():
    async def scenario():
        polling = asyncio.Event()
        async def transport(url, *args):
            if url == USERCODE_URL:
                return 200, encoded({"device_auth_id": "device", "user_code": "code"})
            polling.set()
            await asyncio.Event().wait()
            pytest.fail("cancelled poll resumed")
        async def commit(*args):
            pytest.fail("cancelled login must not commit")
        manager = DeviceLoginManager(commit, transport=transport)
        started = await manager.start()
        await polling.wait()
        with pytest.raises(LoginError) as error:
            await manager.start()
        assert error.value.code == "login_in_progress"
        assert (await manager.cancel(started["id"]))["status"] == "cancelled"
        assert (await manager.cancel(started["id"]))["status"] == "cancelled"
        await manager.close()
    run(scenario())


def test_cancel_interrupts_inflight_exchange_without_committing():
    async def scenario():
        entered, interrupted = asyncio.Event(), asyncio.Event()

        async def transport(url, *args):
            if url == USERCODE_URL:
                return 200, encoded({"device_auth_id": "device", "user_code": "code"})
            if url == POLL_URL:
                return 200, encoded({"authorization_code": "code", "code_verifier": "verifier",
                                     "code_challenge": "challenge"})
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                interrupted.set()

        async def commit(*args):
            pytest.fail("cancelled exchange must not commit")

        manager = DeviceLoginManager(commit, transport=transport)
        started = await manager.start()
        await entered.wait()
        assert (await manager.cancel(started["id"]))["status"] == "cancelled"
        assert interrupted.is_set()
        await manager.close()

    run(scenario())


def test_deadline_includes_polling_and_preserves_existing_login():
    async def scenario():
        clock = [0.0]
        existing = ["fake-previous-login"]
        async def transport(url, *args):
            if url == USERCODE_URL:
                return 200, encoded({"device_auth_id": "device", "user_code": "code", "interval": 2})
            return 403, b"pending"
        async def sleep(delay):
            clock[0] += delay
        async def commit(*args):
            existing[:] = ["wrong"]
            return True
        manager = DeviceLoginManager(commit, transport=transport, sleep=sleep, monotonic=lambda: clock[0], lifetime=3)
        started = await manager.start()
        terminal = await wait_terminal(manager, started["id"])
        assert terminal["status"] == "expired"
        assert existing == ["fake-previous-login"]
        await manager.close()
    run(scenario())


def test_huge_interval_is_bounded_by_remaining_deadline():
    async def scenario():
        clock = [0.0]
        sleeps = []

        async def transport(url, *args):
            if url == USERCODE_URL:
                return 200, encoded({"device_auth_id": "device", "user_code": "code", "interval": "9" * 1000})
            return 403, b""

        async def sleep(delay):
            sleeps.append(delay)
            clock[0] += delay

        async def commit(*args):
            pytest.fail("expired login must not commit")

        manager = DeviceLoginManager(commit, transport=transport, sleep=sleep, monotonic=lambda: clock[0], lifetime=3)
        started = await manager.start()
        assert (await wait_terminal(manager, started["id"]))["status"] == "expired"
        assert sleeps == [3]
        await manager.close()

    run(scenario())


def test_cancellation_during_commit_fences_late_write():
    async def scenario():
        entered = asyncio.Event()
        resume = asyncio.Event()
        existing = ["fake-previous-login"]
        async def transport(url, *args):
            if url == USERCODE_URL:
                return 200, encoded({"device_auth_id": "device", "user_code": "code"})
            if url == POLL_URL:
                return 200, encoded({
                    "authorization_code": "code", "code_verifier": "verifier", "code_challenge": "challenge",
                })
            return 200, encoded({"access_token": "access", "refresh_token": "refresh", "id_token": "id"})
        async def commit(tokens, guard):
            entered.set()
            await resume.wait()
            if not guard():
                return False
            existing[:] = [tokens["access_token"]]
            return True
        manager = DeviceLoginManager(commit, transport=transport)
        started = await manager.start()
        await entered.wait()
        cancellation = asyncio.create_task(manager.cancel(started["id"]))
        await asyncio.sleep(0)
        resume.set()
        assert (await cancellation)["status"] == "cancelled"
        assert existing == ["fake-previous-login"]
        await manager.close()
    run(scenario())


def test_commit_failure_is_safe_and_preserves_previous_credentials():
    async def scenario():
        existing = ["fake-previous-login"]
        async def transport(url, *args):
            if url == USERCODE_URL:
                return 200, encoded({"device_auth_id": "device", "user_code": "code"})
            if url == POLL_URL:
                return 200, encoded({
                    "authorization_code": "code", "code_verifier": "verifier", "code_challenge": "challenge",
                })
            return 200, encoded({"access_token": "fake-access", "refresh_token": "fake-refresh", "id_token": "fake-id"})
        async def commit(tokens, guard):
            raise OSError("fake-sensitive-path")
        manager = DeviceLoginManager(commit, transport=transport)
        started = await manager.start()
        terminal = await wait_terminal(manager, started["id"])
        assert terminal["status"] == "failed"
        assert terminal["error_code"] == "login_storage_error"
        assert "fake-sensitive-path" not in str(terminal)
        assert existing == ["fake-previous-login"]
        await manager.close()
    run(scenario())
