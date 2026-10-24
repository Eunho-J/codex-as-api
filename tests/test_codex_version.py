from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from codex_as_api.auth import ChatGPTOAuthCatalogUnavailableError, ChatGPTOAuthInvalidRequestError
from codex_as_api.codex_version import (
    CODEX_CLI_ORIGINATOR,
    CODEX_NPM_LATEST_URL,
    CODEX_NPM_PACKAGE_NAME,
    CODEX_VERSION_CACHE_TTL_SECONDS,
    CODEX_VERSION_FETCH_TIMEOUT_SECONDS,
    CODEX_VERSION_RETRY_BACKOFF_SECONDS,
    ChatGPTOAuthCodexVersionInvalidError,
    ChatGPTOAuthCodexVersionUnavailableError,
    CodexVersionResolver,
    codex_cli_headers_for_version,
    codex_user_agent_for_version,
    compare_codex_versions,
    get_cached_codex_cli_version,
    is_valid_codex_version,
    parse_codex_npm_latest_payload,
    parse_codex_semver,
    require_explicit_codex_version,
    reset_codex_version_cache,
    resolve_codex_cli_headers,
    resolve_codex_cli_headers_async,
    resolve_codex_cli_version,
    resolve_codex_cli_version_async,
    resolve_latest_codex_version,
    strict_json_loads_no_duplicate_keys,
    validate_codex_version,
)


@pytest.fixture(autouse=True)
def _reset_global_codex_version_resolver() -> Iterator[None]:
    reset_codex_version_cache()
    yield
    reset_codex_version_cache()


def _npm_payload(version: str = "0.118.0", name: str = CODEX_NPM_PACKAGE_NAME) -> bytes:
    return json.dumps({"name": name, "version": version}).encode("utf-8")


def test_parse_and_validate_strict_stable_codex_semver() -> None:
    assert parse_codex_semver("0.0.0") == (0, 0, 0)
    assert parse_codex_semver("0.118.0") == (0, 118, 0)
    assert parse_codex_semver("1.23.456") == (1, 23, 456)
    assert is_valid_codex_version("0.118.0") is True
    assert validate_codex_version("0.118.0") == "0.118.0"
    assert require_explicit_codex_version("0.118.0") == "0.118.0"

    invalid_versions: list[object] = [
        "",
        "0.118",
        "0.118.0.1",
        "v0.118.0",
        "0.118.0-alpha.1",
        "0.118.0-0",
        "0.118.0+build.1",
        "0.118.0-rc.1+build.1",
        "00.118.0",
        "0.0118.0",
        "0.118.00",
        "-1.0.0",
        "0.-1.0",
        "0.1.-1",
        " 0.118.0",
        "0.118.0 ",
        "0. 118.0",
        "0.118.0\n",
        "0.118.0\r\nX-Injected: 1",
        "0.118.0\x00",
        f"1.2.{'9' * 62}",
        "9007199254740992.0.0",
        None,
        118,
        {"version": "0.118.0"},
    ]

    for candidate in invalid_versions:
        assert parse_codex_semver(candidate) is None
        assert is_valid_codex_version(candidate) is False
        with pytest.raises(ChatGPTOAuthCodexVersionUnavailableError):
            validate_codex_version(candidate)
        with pytest.raises((ChatGPTOAuthCodexVersionInvalidError, ChatGPTOAuthInvalidRequestError, ValueError)):
            require_explicit_codex_version(candidate)


def test_compare_codex_versions_numeric_ordering() -> None:
    assert compare_codex_versions("0.99.9", "0.100.0") == -1
    assert compare_codex_versions("0.100.0", "0.99.9") == 1
    assert compare_codex_versions("0.118.0", "0.118.0") == 0
    assert compare_codex_versions("1.0.0", "0.999.999") == 1
    with pytest.raises(ChatGPTOAuthCodexVersionInvalidError):
        compare_codex_versions("0.118.0-alpha.1", "0.118.0")


def test_strict_json_rejects_duplicate_keys_and_invalid_payloads() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        strict_json_loads_no_duplicate_keys(
            '{"name":"@openai/codex","version":"0.118.0","version":"0.119.0"}'
        )
    with pytest.raises(ValueError, match="duplicate key"):
        strict_json_loads_no_duplicate_keys(
            '{"name":"@openai/codex","ver\\u0073ion":"0.118.0","version":"0.119.0"}'
        )
    with pytest.raises(ValueError, match="duplicate key"):
        strict_json_loads_no_duplicate_keys(
            '{"name":"@openai/codex","version":"0.118.0","dist":{"shasum":"a","shasum":"b"}}'
        )

    assert (
        parse_codex_npm_latest_payload(
            b'{"name":"@openai/codex","version":"0.120.2","dist":{"tarball":"https://example.com/a.tgz"}}'
        )
        == "0.120.2"
    )

    invalid_payloads: list[str | bytes] = [
        b"\xff\xfe\xfd",
        "{malformed",
        "null",
        "[]",
        '"0.118.0"',
        '{"name":"codex","version":"0.118.0"}',
        '{"name":"@openai/other","version":"0.118.0"}',
        '{"version":"0.118.0"}',
        '{"name":"@openai/codex"}',
        '{"name":"@openai/codex","version":"0.118.0-beta.1"}',
        '{"name":"@openai/codex","version":"0.118.0","version":"0.119.0"}',
    ]

    for payload in invalid_payloads:
        with pytest.raises(ChatGPTOAuthCodexVersionUnavailableError) as exc_info:
            parse_codex_npm_latest_payload(payload)
        assert isinstance(exc_info.value, ChatGPTOAuthCatalogUnavailableError)


def test_user_agent_and_cli_headers_for_version() -> None:
    ua = codex_user_agent_for_version("0.118.0", codex_as_api_version="0.8.3")
    assert ua.startswith("codex_cli_rs/0.118.0 (")
    assert ua.endswith(") codex-as-api/0.8.3")
    headers = codex_cli_headers_for_version("0.118.0", codex_as_api_version="0.8.3")
    assert headers == {
        "originator": CODEX_CLI_ORIGINATOR,
        "User-Agent": ua,
    }


def test_first_lookup_failure_raises_typed_503_error_without_fallback_or_backoff() -> None:
    attempts = 0
    now = 100.0

    def fake_fetch(url: str, timeout: float) -> bytes:
        nonlocal attempts
        assert url == CODEX_NPM_LATEST_URL
        assert timeout == CODEX_VERSION_FETCH_TIMEOUT_SECONDS
        attempts += 1
        if attempts == 1:
            raise OSError("network unreachable")
        if attempts == 2:
            raise TimeoutError("socket timed out")
        return _npm_payload("0.118.0")

    resolver = CodexVersionResolver(fetch=fake_fetch, clock=lambda: now)

    with pytest.raises(ChatGPTOAuthCodexVersionUnavailableError, match="network unreachable") as first_err:
        resolver.resolve()
    assert isinstance(first_err.value, ChatGPTOAuthCatalogUnavailableError)
    assert resolver.cached_version() is None

    # Immediate retry (without waiting 60s) must hit fetch again because no last_good_version exists.
    now += 1.0
    with pytest.raises(ChatGPTOAuthCodexVersionUnavailableError, match="timed out after 5s"):
        resolver.resolve()
    assert attempts == 2

    now += 1.0
    assert resolver.resolve() == "0.118.0"
    assert resolver.cached_version() == "0.118.0"
    assert attempts == 3


def test_ttl_cache_and_expired_refresh() -> None:
    now = 1_000.0
    calls = 0
    versions = ["0.118.0", "0.119.0"]

    def fake_fetch(_url: str, _timeout: float) -> bytes:
        nonlocal calls
        version = versions[calls]
        calls += 1
        return _npm_payload(version)

    resolver = CodexVersionResolver(fetch=fake_fetch, clock=lambda: now)

    assert resolver.resolve() == "0.118.0"
    assert calls == 1

    now += CODEX_VERSION_CACHE_TTL_SECONDS - 0.001
    assert resolver.resolve() == "0.118.0"
    assert calls == 1

    now += 0.001
    assert resolver.resolve() == "0.119.0"
    assert calls == 2


def test_expired_refresh_failure_retains_last_good_version_and_backs_off_60s() -> None:
    now = 500.0
    calls = 0

    def fake_fetch(_url: str, _timeout: float) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _npm_payload("0.118.0")
        if calls == 2:
            raise ChatGPTOAuthCodexVersionUnavailableError("npm @openai/codex version lookup failed with HTTP 503")
        if calls == 3:
            return b'{"name":"@openai/codex","version":"0.119.0-rc.1"}'
        return _npm_payload("0.119.0")

    resolver = CodexVersionResolver(fetch=fake_fetch, clock=lambda: now)

    assert resolver.resolve() == "0.118.0"
    assert calls == 1

    # TTL expires; refresh #2 fails -> retains 0.118.0 and sets 60s retry backoff.
    now += CODEX_VERSION_CACHE_TTL_SECONDS
    assert resolver.resolve() == "0.118.0"
    assert calls == 2

    # Within 60s backoff window: returns 0.118.0 without calling fetch.
    now += CODEX_VERSION_RETRY_BACKOFF_SECONDS - 0.001
    assert resolver.resolve() == "0.118.0"
    assert calls == 2

    # At 60s boundary: retries (#3), gets invalid prerelease version -> retains 0.118.0 and backs off another 60s.
    now += 0.001
    assert resolver.resolve() == "0.118.0"
    assert calls == 3

    now += CODEX_VERSION_RETRY_BACKOFF_SECONDS - 0.001
    assert resolver.resolve() == "0.118.0"
    assert calls == 3

    # After second 60s backoff: retries (#4) and upgrades to 0.119.0.
    now += 0.001
    assert resolver.resolve() == "0.119.0"
    assert calls == 4


def test_monotonic_no_downgrade_on_stale_upstream_response() -> None:
    now = 100.0
    calls = 0
    sequence = ["0.100.0", "0.99.9", "0.100.0", "0.101.2", "0.101.1"]

    def fake_fetch(_url: str, _timeout: float) -> bytes:
        nonlocal calls
        version = sequence[calls]
        calls += 1
        return _npm_payload(version)

    resolver = CodexVersionResolver(fetch=fake_fetch, clock=lambda: now)

    assert resolver.resolve() == "0.100.0"
    assert calls == 1

    # Stale mirror returns 0.99.9 after TTL expiry -> retains 0.100.0 and extends TTL by 1h.
    now += CODEX_VERSION_CACHE_TTL_SECONDS
    assert resolver.resolve() == "0.100.0"
    assert calls == 2

    # Does not re-fetch after 60s because stale response was a valid payload that refreshed TTL.
    now += CODEX_VERSION_RETRY_BACKOFF_SECONDS + 10.0
    assert resolver.resolve() == "0.100.0"
    assert calls == 2

    now += CODEX_VERSION_CACHE_TTL_SECONDS
    assert resolver.resolve() == "0.100.0"
    assert calls == 3

    now += CODEX_VERSION_CACHE_TTL_SECONDS
    assert resolver.resolve() == "0.101.2"
    assert calls == 4

    now += CODEX_VERSION_CACHE_TTL_SECONDS
    assert resolver.resolve() == "0.101.2"
    assert calls == 5


def test_concurrent_singleflight_across_threads_and_asyncio() -> None:
    calls = 0
    entered_fetch = threading.Event()
    release_fetch = threading.Event()

    def slow_fetch(_url: str, _timeout: float) -> bytes:
        nonlocal calls
        calls += 1
        entered_fetch.set()
        release_fetch.wait(timeout=5.0)
        return _npm_payload("0.118.0")

    resolver = CodexVersionResolver(fetch=slow_fetch)
    results: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            value = resolver.resolve()
            with lock:
                results.append(value)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()

    assert entered_fetch.wait(timeout=2.0)
    release_fetch.set()

    for thread in threads:
        thread.join(timeout=5.0)

    assert errors == []
    assert results == ["0.118.0"] * 8
    assert calls == 1

    # Also verify async singleflight and header resolution via asyncio.gather.
    async def run_async_checks() -> None:
        nonlocal calls
        entered_fetch.clear()
        release_fetch.set()
        resolver.reset()
        versions = await asyncio.gather(*(resolver.resolve_async() for _ in range(6)))
        assert versions == ["0.118.0"] * 6
        assert calls == 2
        headers = await resolver.headers_async(codex_as_api_version="0.8.3")
        assert headers["originator"] == CODEX_CLI_ORIGINATOR
        assert headers["User-Agent"].startswith("codex_cli_rs/0.118.0 ")

    asyncio.run(run_async_checks())


def test_real_loopback_http_io_and_timeout_enforcement() -> None:
    request_count = 0
    observed_headers: dict[str, str] = {}
    observed_path = ""
    state: dict[str, Any] = {"status": 200, "version": "0.121.0"}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            nonlocal request_count, observed_headers, observed_path
            request_count += 1
            observed_path = self.path
            observed_headers = {k.lower(): v for k, v in self.headers.items()}
            status = int(state["status"])
            if status == 302:
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1/redirected")
                self.end_headers()
                return
            payload = json.dumps({"name": CODEX_NPM_PACKAGE_NAME, "version": state["version"]}).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        host, port = server.server_address[:2]
        registry_url = f"http://{host}:{port}/@openai%2fcodex/latest"

        reset_codex_version_cache(registry_url=registry_url)
        assert get_cached_codex_cli_version() is None

        resolved = resolve_codex_cli_version()
        assert resolved == "0.121.0"
        assert resolve_latest_codex_version() == "0.121.0"
        assert asyncio.run(resolve_codex_cli_version_async()) == "0.121.0"
        assert get_cached_codex_cli_version() == "0.121.0"
        assert request_count == 1
        assert observed_path == "/@openai%2fcodex/latest"
        assert observed_headers.get("accept") == "application/json"
        assert "authorization" not in observed_headers

        headers = resolve_codex_cli_headers(codex_as_api_version="0.8.3")
        async_headers = asyncio.run(resolve_codex_cli_headers_async(codex_as_api_version="0.8.3"))
        assert headers == async_headers
        assert headers["originator"] == CODEX_CLI_ORIGINATOR
        assert headers["User-Agent"].startswith("codex_cli_rs/0.121.0 ")

        # Verify redirect is rejected on cold lookup.
        state["status"] = 302
        reset_codex_version_cache(registry_url=registry_url)
        with pytest.raises(ChatGPTOAuthCodexVersionUnavailableError, match="HTTP 302"):
            resolve_codex_cli_version()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)

    # Verify real socket timeout against a stalled TCP listener.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stalled_listener:
        stalled_listener.bind(("127.0.0.1", 0))
        stalled_listener.listen(1)
        stalled_host, stalled_port = stalled_listener.getsockname()[:2]
        accepted_sockets: list[socket.socket] = []

        def accept_and_hold() -> None:
            with contextlib.suppress(OSError):
                conn, _ = stalled_listener.accept()
                accepted_sockets.append(conn)

        accept_thread = threading.Thread(target=accept_and_hold, daemon=True)
        accept_thread.start()
        try:
            stalled_url = f"http://{stalled_host}:{stalled_port}/@openai%2fcodex/latest"
            reset_codex_version_cache(registry_url=stalled_url, timeout_seconds=0.05)
            with pytest.raises(ChatGPTOAuthCodexVersionUnavailableError, match="timed out"):
                resolve_codex_cli_version()
        finally:
            for conn in accepted_sockets:
                conn.close()
            accept_thread.join(timeout=1.0)
