"""Modified browser login based on OpenAI Codex rust-v0.164.0 (Apache-2.0).

Adapted from codex-rs/login/src/server.rs and oauth/{authorization,pkce,client}.rs;
see the distributed third-party notices. The local listener and transport are
injectable, and this API does not launch a browser.
"""
from __future__ import annotations

import asyncio
import base64
import datetime as dt
import hashlib
import secrets
import time
import urllib.parse
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field

from .auth import CHATGPT_OAUTH_CLIENT_ID, AuthDurabilityUnknownError
from .device_login import TOKEN_URL, Commit, LoginError, Transport, _Attempt, _object, _string, default_transport

AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
SCOPES = "openid profile email offline_access api.connectors.read api.connectors.invoke"


@dataclass
class _BrowserAttempt(_Attempt):
    state: str = ""
    verifier: str = ""
    redirect_uri: str = ""
    server: asyncio.Server | None = None
    expiry_task: asyncio.Task[None] | None = None
    handlers: set[asyncio.Task[None]] = field(default_factory=set)
    accepted: bool = False

    def public(self) -> dict[str, object]:
        return {**super().public(), "method": "browser"}


class BrowserLoginManager:
    def __init__(self, commit: Commit, *, transport: Transport = default_transport,
                 monotonic: Callable[[], float] = time.monotonic,
                 now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 lifetime: float = 900.0, max_records: int = 32,
                 ports: tuple[int, ...] = (1455, 1457)):
        if lifetime <= 0 or max_records < 1 or not ports:
            raise ValueError("invalid login manager settings")
        self._commit, self._transport = commit, transport
        self._monotonic, self._now, self._sleep = monotonic, now, sleep
        self._lifetime, self._max_records, self._ports = lifetime, max_records, ports
        self._records: OrderedDict[str, _BrowserAttempt] = OrderedDict()
        self._active: _BrowserAttempt | None = None
        self._closed = False

    def _finish(self, attempt: _BrowserAttempt, status: str, error: str | None = None) -> None:
        attempt.status, attempt.error_code = status, error
        if attempt.server:
            attempt.server.close()
        if attempt.expiry_task and attempt.expiry_task is not asyncio.current_task():
            attempt.expiry_task.cancel()
        if self._active is attempt:
            self._active = None
        while len(self._records) > self._max_records:
            key = next((key for key, value in self._records.items() if value is not self._active), None)
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
        attempt = _BrowserAttempt(secrets.token_urlsafe(24), expires, deadline,
                                  state=secrets.token_urlsafe(32), verifier=secrets.token_urlsafe(64))
        self._active = attempt
        self._records[attempt.id] = attempt
        try:
            for port in self._ports:
                try:
                    attempt.server = await asyncio.start_server(
                        lambda r, w: self._accept(attempt, r, w), "127.0.0.1", port, limit=8192)
                    break
                except OSError:
                    continue
            if attempt.server is None:
                raise LoginError("callback_port_unavailable")
            if self._closed or attempt.cancel_requested:
                raise LoginError("login_unavailable")
            bound = attempt.server.sockets[0].getsockname()[1]
            attempt.redirect_uri = f"http://127.0.0.1:{bound}/auth/callback"
            digest = hashlib.sha256(attempt.verifier.encode()).digest()
            challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
            query = urllib.parse.urlencode({"response_type": "code", "client_id": CHATGPT_OAUTH_CLIENT_ID,
                "redirect_uri": attempt.redirect_uri, "state": attempt.state, "code_challenge": challenge,
                "code_challenge_method": "S256", "scope": SCOPES, "id_token_add_organizations": "true",
                "codex_cli_simplified_flow": "true", "originator": "codex_cli_rs"})
            attempt.expiry_task = asyncio.create_task(self._expire(attempt))
            return {**attempt.public(), "auth_url": AUTHORIZE_URL + "?" + query}
        except BaseException:
            self._finish(attempt, "cancelled" if attempt.cancel_requested else "failed")
            raise

    async def _expire(self, attempt: _BrowserAttempt) -> None:
        try:
            await self._sleep(max(0, attempt.deadline - self._monotonic()))
            if attempt.status != "pending":
                return
            with attempt.claim_lock:
                if not attempt.commit_claimed:
                    attempt.cancel_requested = True
            if attempt.committing:
                return  # commit guard and completion resolve the terminal state
            self._finish(attempt, "expired")
            for task in tuple(attempt.handlers):
                task.cancel()
        except asyncio.CancelledError:
            pass

    def _accept(self, attempt: _BrowserAttempt, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(self._handle(attempt, reader, writer))
        attempt.handlers.add(task)
        task.add_done_callback(attempt.handlers.discard)

    async def _reply(self, writer: asyncio.StreamWriter, success: bool) -> None:
        body = b"Login completed. You may close this window." if success else b"Login could not be completed."
        status = b"200 OK" if success else b"400 Bad Request"
        writer.write(b"HTTP/1.1 " + status + b"\r\nContent-Type: text/plain; charset=utf-8\r\n"
                     b"Cache-Control: no-store\r\nReferrer-Policy: no-referrer\r\nConnection: close\r\n"
                     + b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        await writer.drain()

    async def _handle(
        self, attempt: _BrowserAttempt, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            remaining = attempt.deadline - self._monotonic()
            if remaining <= 0:
                return
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), min(remaining, 10.0))
            if len(raw) > 8192:
                raise ValueError
            lines = raw.decode("ascii").split("\r\n")
            request = lines[0].split(" ")
            if len(request) != 3 or request[0] != "GET" or request[2] not in ("HTTP/1.0", "HTTP/1.1"):
                raise ValueError
            hosts = [line.split(":", 1)[1].strip() for line in lines[1:] if line.lower().startswith("host:")]
            expected_host = urllib.parse.urlsplit(attempt.redirect_uri).netloc
            if hosts != [expected_host]:
                raise ValueError
            target = urllib.parse.urlsplit(request[1])
            if target.path != "/auth/callback" or target.scheme or target.netloc or target.fragment:
                raise ValueError
            query = urllib.parse.parse_qs(target.query, keep_blank_values=True, max_num_fields=64)
            states, codes, errors = (query.get(key, []) for key in ("state", "code", "error"))
            if (len(states) != 1 or not states[0] or not secrets.compare_digest(states[0], attempt.state)
                    or len(codes) > 1 or len(errors) > 1 or bool(codes) == bool(errors)
                    or (codes and not codes[0]) or (errors and not errors[0])
                    or attempt.status != "pending" or attempt.accepted or self._monotonic() >= attempt.deadline):
                raise ValueError
            attempt.accepted = True
            if attempt.server:
                attempt.server.close()
            if errors:
                self._finish(attempt, "failed", "login_denied")
                await self._reply(writer, False)
                return
            attempt.task = asyncio.current_task()
            body = urllib.parse.urlencode({"grant_type": "authorization_code", "client_id": CHATGPT_OAUTH_CLIENT_ID,
                "code": codes[0], "redirect_uri": attempt.redirect_uri, "code_verifier": attempt.verifier}).encode()
            remaining = attempt.deadline - self._monotonic()
            status, data = await asyncio.wait_for(self._transport(TOKEN_URL, body,
                {"Content-Type": "application/x-www-form-urlencoded"}, remaining), remaining)
            if not 200 <= status < 300:
                raise LoginError("login_upstream_error")
            payload = _object(data)
            tokens = {key: _string(payload, key) for key in ("access_token", "refresh_token", "id_token")}
            def guard() -> bool:
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
                    with attempt.claim_lock:
                        if not attempt.commit_claimed:
                            attempt.cancel_requested = True
                    if commit_task.cancelled():
                        raise
            attempt.committing = False
            if committed:
                self._finish(attempt, "succeeded")
            elif self._monotonic() >= attempt.deadline:
                self._finish(attempt, "expired")
            elif attempt.cancel_requested:
                self._finish(attempt, "cancelled")
            else:
                self._finish(attempt, "failed", "login_storage_error")
            await self._reply(writer, committed)
        except asyncio.CancelledError:
            if attempt.status == "pending":
                self._finish(attempt, "cancelled")
        except (ValueError, UnicodeError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            if attempt.accepted and attempt.status == "pending":
                self._finish(attempt, "failed", "login_storage_error" if attempt.committing else "login_upstream_error")
            with suppress(ConnectionError, OSError):
                await self._reply(writer, False)
        except asyncio.TimeoutError:
            if attempt.accepted and attempt.status == "pending":
                self._finish(attempt, "expired")
        except AuthDurabilityUnknownError:
            self._finish(attempt, "failed", "login_persistence_uncertain")
        except LoginError as exc:
            self._finish(attempt, "failed", exc.code)
        except Exception:
            if attempt.accepted and attempt.status == "pending":
                self._finish(attempt, "failed", "login_storage_error" if attempt.committing else "login_upstream_error")
        finally:
            writer.close()
            with suppress(ConnectionError, OSError):
                await writer.wait_closed()

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
            if attempt.server:
                attempt.server.close()
            if not attempt.committing:
                self._finish(attempt, "cancelled")
                for task in tuple(attempt.handlers):
                    task.cancel()
            tasks = [task for task in attempt.handlers if task is not asyncio.current_task()]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        return attempt.public()

    async def close(self) -> None:
        self._closed = True
        if self._active is not None:
            await self.cancel(self._active.id)
