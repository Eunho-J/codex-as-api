"""Modified port of OpenAI Codex rust-v0.164.0 device-code authentication.

Adapted from codex-rs/login/src/device_code_auth.rs. Copyright OpenAI Codex
contributors; Apache License, Version 2.0. See the distributed third-party
notices and license. Modified for this project's API and credential store.

The transport, clock and commit operation are injectable for offline testing.
The commit callback is responsible for validation, private durable atomic storage,
and fencing concurrent refreshes. It must evaluate its guard immediately before
replacement, with no intervening await, and return whether replacement occurred.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import secrets
import threading
import time
import urllib.parse
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

from .auth import CHATGPT_OAUTH_CLIENT_ID, AuthDurabilityUnknownError
from .strict_json import strict_json_loads

ISSUER = "https://auth.openai.com"
USERCODE_URL = ISSUER + "/api/accounts/deviceauth/usercode"
POLL_URL = ISSUER + "/api/accounts/deviceauth/token"
TOKEN_URL = ISSUER + "/oauth/token"
VERIFICATION_URL = ISSUER + "/codex/device"
REDIRECT_URI = ISSUER + "/deviceauth/callback"
Transport = Callable[[str, bytes, dict[str, str], float], Awaitable[tuple[int, bytes]]]
Commit = Callable[[dict[str, str], Callable[[], bool]], Awaitable[bool]]


class LoginError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass
class _Attempt:
    id: str
    expires_at: str
    deadline: float
    status: str = "pending"
    error_code: str | None = None
    task: asyncio.Task[None] | None = None
    committing: bool = False
    cancel_requested: bool = False
    commit_claimed: bool = False
    claim_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def public(self) -> dict[str, object]:
        result: dict[str, object] = {"id": self.id, "status": self.status, "expires_at": self.expires_at}
        if self.error_code:
            result["error_code"] = self.error_code
        return result


async def default_transport(url: str, body: bytes, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    # Native asynchronous I/O ensures task cancellation closes the in-flight request.
    async with (
        httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client,
        client.stream("POST", url, content=body, headers=headers) as response,
    ):
        if not 200 <= response.status_code < 300:
            return response.status_code, b""
        data = bytearray()
        async for chunk in response.aiter_bytes():
            data.extend(chunk)
            if len(data) > 1024 * 1024:
                raise LoginError("invalid_login_response")
        return response.status_code, bytes(data)


def _object(body: bytes) -> dict[str, object]:
    if len(body) > 1024 * 1024:
        raise LoginError("invalid_login_response")
    try:
        value = strict_json_loads(body)
    except (ValueError, UnicodeError):
        raise LoginError("invalid_login_response") from None
    if not isinstance(value, dict):
        raise LoginError("invalid_login_response")
    return value


def _string(obj: dict[str, object], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LoginError("invalid_login_response")
    return value


class DeviceLoginManager:
    def __init__(self, commit: Commit, *, transport: Transport = default_transport,
                 monotonic: Callable[[], float] = time.monotonic,
                 now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 lifetime: float = 900.0, max_records: int = 32):
        if lifetime <= 0 or max_records < 1:
            raise ValueError("invalid login manager settings")
        self._commit, self._transport = commit, transport
        self._monotonic, self._now, self._sleep = monotonic, now, sleep
        self._lifetime, self._max_records = lifetime, max_records
        self._records: OrderedDict[str, _Attempt] = OrderedDict()
        self._active: _Attempt | None = None
        self._closed = False

    async def _request(self, attempt: _Attempt, url: str, body: bytes, content_type: str) -> tuple[int, bytes]:
        remaining = attempt.deadline - self._monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        return await asyncio.wait_for(
            self._transport(url, body, {"Content-Type": content_type}, remaining), remaining
        )

    def _finish(self, attempt: _Attempt, status: str, error: str | None = None) -> None:
        attempt.status, attempt.error_code = status, error
        if self._active is attempt:
            self._active = None
        while len(self._records) > self._max_records:
            key = next((key for key, item in self._records.items() if item is not self._active), None)
            if key is None:
                break
            self._records.pop(key)

    async def start(self) -> dict[str, object]:
        if self._closed:
            raise LoginError("login_unavailable")
        if self._active is not None:
            raise LoginError("login_in_progress")
        deadline = self._monotonic() + self._lifetime
        expires = (self._now() + dt.timedelta(seconds=self._lifetime)).isoformat().replace("+00:00", "Z")
        attempt = _Attempt(secrets.token_urlsafe(24), expires, deadline)
        self._active = attempt
        self._records[attempt.id] = attempt
        # Track initiation as well, so shutdown can interrupt an in-flight request.
        attempt.task = asyncio.current_task()
        try:
            status, body = await self._request(attempt, USERCODE_URL,
                json.dumps({"client_id": CHATGPT_OAUTH_CLIENT_ID}).encode(), "application/json")
            if status == 404:
                raise LoginError("device_login_unsupported")
            if not 200 <= status < 300:
                raise LoginError("login_upstream_error")
            payload = _object(body)
            device_id = _string(payload, "device_auth_id")
            user_code = _string(payload, "user_code" if "user_code" in payload else "usercode")
            interval_value = payload.get("interval", 5)
            if isinstance(interval_value, bool) or not (isinstance(interval_value, int) or
                    isinstance(interval_value, str) and interval_value.isascii() and interval_value.isdigit()):
                raise LoginError("invalid_login_response")
            try:
                interval = int(interval_value)
            except ValueError:
                raise LoginError("invalid_login_response") from None
            if interval <= 0:
                raise LoginError("invalid_login_response")
            if self._monotonic() >= deadline:
                raise asyncio.TimeoutError
            attempt.task = asyncio.create_task(self._run(attempt, device_id, user_code, interval))
            return {**attempt.public(), "verification_url": VERIFICATION_URL,
                    "user_code": user_code, "interval": interval}
        except asyncio.CancelledError:
            self._finish(attempt, "cancelled")
            raise
        except asyncio.TimeoutError:
            self._finish(attempt, "expired")
            raise LoginError("login_expired") from None
        except LoginError as exc:
            self._finish(attempt, "failed", exc.code)
            raise
        except Exception:
            self._finish(attempt, "failed", "login_upstream_error")
            raise LoginError("login_upstream_error") from None

    async def _run(self, attempt: _Attempt, device_id: str, user_code: str, interval: int) -> None:
        try:
            while True:
                status, body = await self._request(attempt, POLL_URL,
                    json.dumps({"device_auth_id": device_id, "user_code": user_code}).encode(), "application/json")
                if status in (403, 404):
                    remaining = attempt.deadline - self._monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    await asyncio.wait_for(self._sleep(min(interval, remaining)), remaining)
                    continue
                if not 200 <= status < 300:
                    raise LoginError("login_upstream_error")
                response = _object(body)
                code = _string(response, "authorization_code")
                verifier = _string(response, "code_verifier")
                _string(response, "code_challenge")
                exchange = urllib.parse.urlencode({"grant_type": "authorization_code", "code": code,
                    "redirect_uri": REDIRECT_URI, "client_id": CHATGPT_OAUTH_CLIENT_ID,
                    "code_verifier": verifier}).encode()
                status, body = await self._request(attempt, TOKEN_URL, exchange, "application/x-www-form-urlencoded")
                if not 200 <= status < 300:
                    raise LoginError("login_upstream_error")
                response = _object(body)
                tokens = {name: _string(response, name) for name in ("access_token", "refresh_token", "id_token")}
                def guard() -> bool:
                    # Called from a worker thread. The lock makes the test and claim
                    # indivisible relative to cancellation, including across GIL switches.
                    with attempt.claim_lock:
                        if (self._closed or attempt.cancel_requested or self._active is not attempt
                                or attempt.status != "pending" or self._monotonic() >= attempt.deadline):
                            return False
                        attempt.commit_claimed = True
                        return True
                if self._monotonic() >= attempt.deadline:
                    raise asyncio.TimeoutError
                attempt.committing = True
                commit_task = asyncio.ensure_future(self._commit(tokens, guard))
                while True:
                    try:
                        committed = await asyncio.shield(commit_task)
                        break
                    except asyncio.CancelledError:
                        # Do not abandon a worker that may already have claimed.
                        with attempt.claim_lock:
                            if not attempt.commit_claimed:
                                attempt.cancel_requested = True
                        if commit_task.cancelled():
                            raise
                attempt.committing = False
                if committed:
                    self._finish(attempt, "succeeded")
                elif attempt.cancel_requested:
                    self._finish(attempt, "cancelled")
                elif self._monotonic() >= attempt.deadline:
                    self._finish(attempt, "expired")
                else:
                    self._finish(attempt, "failed", "login_storage_error")
                return
        except asyncio.CancelledError:
            self._finish(attempt, "cancelled")
        except asyncio.TimeoutError:
            self._finish(attempt, "expired")
        except LoginError as exc:
            self._finish(attempt, "failed", exc.code)
        except AuthDurabilityUnknownError:
            self._finish(attempt, "failed", "login_persistence_uncertain")
        except Exception:
            self._finish(attempt, "failed", "login_storage_error" if attempt.committing else "login_upstream_error")

    def get(self, login_id: str) -> dict[str, object]:
        attempt = self._records.get(login_id)
        if attempt is None:
            raise LoginError("login_not_found")
        return attempt.public()

    async def cancel(self, login_id: str) -> dict[str, object]:
        attempt = self._records.get(login_id)
        if attempt is None:
            raise LoginError("login_not_found")
        if attempt.status == "pending":
            with attempt.claim_lock:
                if not attempt.commit_claimed:
                    attempt.cancel_requested = True
            task = attempt.task
            if not attempt.committing:
                attempt.status = "cancelled"
            if task and task is not asyncio.current_task():
                if not attempt.committing:
                    task.cancel()
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    if not task.cancelled():
                        raise
            if attempt.status == "cancelled":
                self._finish(attempt, "cancelled")
        return attempt.public()

    async def close(self) -> None:
        self._closed = True
        if self._active is not None:
            await self.cancel(self._active.id)
