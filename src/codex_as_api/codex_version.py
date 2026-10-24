from __future__ import annotations

import asyncio
import contextlib
import json
import math
import platform
import re
import socket
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from . import __version__
from .auth import ChatGPTOAuthCatalogUnavailableError, ChatGPTOAuthInvalidRequestError
from .strict_json import (
    JS_SAFE_INTEGER,
    _parse_finite_float,
    _parse_safe_integer,
    _reject_nonstandard_constant,
    _validate_strings_and_numbers,
)

CODEX_NPM_LATEST_URL = "https://registry.npmjs.org/@openai%2fcodex/latest"
CODEX_NPM_PACKAGE_NAME = "@openai/codex"
CODEX_CLI_ORIGINATOR = "codex_cli_rs"
CODEX_VERSION_FETCH_TIMEOUT_SECONDS = 5.0
CODEX_VERSION_CACHE_TTL_SECONDS = 3600.0
CODEX_VERSION_RETRY_BACKOFF_SECONDS = 60.0
MAX_CODEX_VERSION_LENGTH = 64
_MAX_METADATA_RESPONSE_BYTES = 1_048_576

_STRICT_STABLE_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")

CodexSemverTuple = tuple[int, int, int]
CodexVersionFetch = Callable[[str, float], bytes | str]
CodexVersionClock = Callable[[], float]
CodexVersionProvider = Callable[[], str]


class ChatGPTOAuthCodexVersionUnavailableError(ChatGPTOAuthCatalogUnavailableError):
    """Raised when the latest @openai/codex version cannot be resolved and no cached version is available."""


class ChatGPTOAuthCodexVersionInvalidError(ChatGPTOAuthInvalidRequestError, ValueError):
    """Raised when an explicit Codex version argument or configuration value is invalid."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def parse_codex_semver(value: object) -> CodexSemverTuple | None:
    if not isinstance(value, str):
        return None
    if not value or len(value) > MAX_CODEX_VERSION_LENGTH or value != value.strip():
        return None
    if any(ch.isspace() or unicodedata.category(ch) == "Cc" for ch in value):
        return None
    match = _STRICT_STABLE_SEMVER_RE.fullmatch(value)
    if match is None:
        return None
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3))
    if (
        major < 0
        or minor < 0
        or patch < 0
        or major > JS_SAFE_INTEGER
        or minor > JS_SAFE_INTEGER
        or patch > JS_SAFE_INTEGER
    ):
        return None
    return (major, minor, patch)


def is_valid_codex_version(value: object) -> bool:
    return parse_codex_semver(value) is not None


def validate_codex_version(value: object, *, source: str = "npm @openai/codex version") -> str:
    if parse_codex_semver(value) is None or not isinstance(value, str):
        raise ChatGPTOAuthCodexVersionUnavailableError(
            f"{source} must be a strict stable three-component semantic version"
        )
    return value


def require_explicit_codex_version(value: object, *, source: str = "Codex compatibility version") -> str:
    if parse_codex_semver(value) is None or not isinstance(value, str):
        raise ChatGPTOAuthCodexVersionInvalidError(
            f"{source} must be a strict stable three-component semantic version"
        )
    return value


def compare_codex_versions(left: str, right: str) -> int:
    left_semver = parse_codex_semver(left)
    right_semver = parse_codex_semver(right)
    if left_semver is None or right_semver is None:
        raise ChatGPTOAuthCodexVersionInvalidError(
            "compared Codex versions must be strict stable three-component semantic versions"
        )
    if left_semver < right_semver:
        return -1
    if left_semver > right_semver:
        return 1
    return 0


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key!r}")
        result[key] = value
    return result


def strict_json_loads_no_duplicate_keys(data: str | bytes | bytearray) -> Any:
    if isinstance(data, bytearray):
        data = bytes(data)
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="strict")
    value = json.loads(
        data,
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=_reject_nonstandard_constant,
        parse_float=_parse_finite_float,
        parse_int=_parse_safe_integer,
    )
    _validate_strings_and_numbers(value)
    return value


def parse_codex_npm_latest_payload(payload: str | bytes | bytearray) -> str:
    try:
        parsed = strict_json_loads_no_duplicate_keys(payload)
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise ChatGPTOAuthCodexVersionUnavailableError(
            "npm @openai/codex metadata response is not valid strict JSON"
        ) from exc

    if not isinstance(parsed, dict):
        raise ChatGPTOAuthCodexVersionUnavailableError(
            "npm @openai/codex metadata response must be a JSON object"
        )

    if parsed.get("name") != CODEX_NPM_PACKAGE_NAME:
        raise ChatGPTOAuthCodexVersionUnavailableError(
            "npm @openai/codex metadata response has unexpected package name"
        )

    return validate_codex_version(parsed.get("version"), source="npm @openai/codex metadata version")


def _codex_os_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "Mac OS"
    return system or "unknown"


def _codex_os_info() -> str:
    return f"{_codex_os_name()} {platform.release() or 'unknown'}; {platform.machine() or 'unknown'}"


def _sanitize_header_value(value: str) -> str:
    return "".join(ch if " " <= ch <= "~" else "_" for ch in value)


def codex_user_agent_for_version(
    version: str,
    *,
    codex_as_api_version: str = __version__,
) -> str:
    normalized_version = require_explicit_codex_version(version, source="Codex compatibility version")
    normalized_package_version = require_explicit_codex_version(
        codex_as_api_version,
        source="codex-as-api package version",
    )
    return _sanitize_header_value(
        f"{CODEX_CLI_ORIGINATOR}/{normalized_version} ({_codex_os_info()}) codex-as-api/{normalized_package_version}"
    )


def codex_cli_headers_for_version(
    version: str,
    *,
    codex_as_api_version: str = __version__,
) -> dict[str, str]:
    return {
        "originator": CODEX_CLI_ORIGINATOR,
        "User-Agent": codex_user_agent_for_version(version, codex_as_api_version=codex_as_api_version),
    }


def _validate_positive_finite_seconds(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChatGPTOAuthCodexVersionInvalidError(f"{name} must be a positive finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ChatGPTOAuthCodexVersionInvalidError(f"{name} must be a positive finite number")
    return numeric


def _validate_registry_url(url: str) -> str:
    if not isinstance(url, str) or not url or url != url.strip():
        raise ChatGPTOAuthCodexVersionInvalidError(
            "registry_url must be a non-empty HTTP(S) URL without surrounding whitespace"
        )
    if any(ch.isspace() or unicodedata.category(ch) == "Cc" for ch in url):
        raise ChatGPTOAuthCodexVersionInvalidError("registry_url must not contain whitespace or control characters")
    try:
        parsed = urllib.parse.urlsplit(url)
        _ = parsed.port
    except (TypeError, ValueError) as exc:
        raise ChatGPTOAuthCodexVersionInvalidError("registry_url must be a valid HTTP(S) URL") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ChatGPTOAuthCodexVersionInvalidError("registry_url must be an HTTP(S) URL without credentials")
    return url


def default_fetch_codex_npm_latest(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with _NO_REDIRECT_OPENER.open(request, timeout=timeout) as response:
            status = getattr(response, "status", None) or response.getcode()
            if status != 200:
                raise ChatGPTOAuthCodexVersionUnavailableError(
                    f"npm @openai/codex version lookup failed with HTTP {status}"
                )
            body = response.read(_MAX_METADATA_RESPONSE_BYTES + 1)
            if not isinstance(body, bytes):
                raise ChatGPTOAuthCodexVersionUnavailableError(
                    "npm @openai/codex version lookup returned a non-bytes body"
                )
            if len(body) > _MAX_METADATA_RESPONSE_BYTES:
                raise ChatGPTOAuthCodexVersionUnavailableError(
                    "npm @openai/codex version response body exceeds maximum allowed size"
                )
            return body
    except ChatGPTOAuthCodexVersionUnavailableError:
        raise
    except urllib.error.HTTPError as exc:
        with contextlib.suppress(Exception):
            exc.close()
        raise ChatGPTOAuthCodexVersionUnavailableError(
            f"npm @openai/codex version lookup failed with HTTP {exc.code}"
        ) from exc
    except TimeoutError as exc:
        raise ChatGPTOAuthCodexVersionUnavailableError(
            f"npm @openai/codex version lookup timed out after {timeout:g}s"
        ) from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise ChatGPTOAuthCodexVersionUnavailableError(
                f"npm @openai/codex version lookup timed out after {timeout:g}s"
            ) from exc
        raise ChatGPTOAuthCodexVersionUnavailableError(
            f"npm @openai/codex version lookup request failed: {reason}"
        ) from exc
    except OSError as exc:
        raise ChatGPTOAuthCodexVersionUnavailableError(
            f"npm @openai/codex version lookup request failed: {exc}"
        ) from exc


class _InFlightLookup:
    __slots__ = ("done", "error", "result", "revision")

    def __init__(self, revision: int) -> None:
        self.revision = revision
        self.done = False
        self.result: str | None = None
        self.error: ChatGPTOAuthCodexVersionUnavailableError | None = None


class CodexVersionResolver:
    """Thread-safe and async-compatible resolver for the latest stable @openai/codex version."""

    def __init__(
        self,
        *,
        fetch: CodexVersionFetch | None = None,
        clock: CodexVersionClock | None = None,
        ttl_seconds: float = CODEX_VERSION_CACHE_TTL_SECONDS,
        retry_backoff_seconds: float = CODEX_VERSION_RETRY_BACKOFF_SECONDS,
        timeout_seconds: float = CODEX_VERSION_FETCH_TIMEOUT_SECONDS,
        registry_url: str = CODEX_NPM_LATEST_URL,
    ) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._fetch: CodexVersionFetch = fetch if fetch is not None else default_fetch_codex_npm_latest
        self._clock: CodexVersionClock = clock if clock is not None else time.monotonic
        self._ttl_seconds = _validate_positive_finite_seconds(ttl_seconds, "ttl_seconds")
        self._retry_backoff_seconds = _validate_positive_finite_seconds(
            retry_backoff_seconds,
            "retry_backoff_seconds",
        )
        self._timeout_seconds = _validate_positive_finite_seconds(timeout_seconds, "timeout_seconds")
        self._registry_url = _validate_registry_url(registry_url)

        self._last_good_version: str | None = None
        self._last_good_semver: CodexSemverTuple | None = None
        self._expires_at: float | None = None
        self._retry_not_before: float | None = None
        self._in_flight: _InFlightLookup | None = None
        self._revision = 0

    def configure(
        self,
        *,
        fetch: CodexVersionFetch | None = None,
        clock: CodexVersionClock | None = None,
        ttl_seconds: float = CODEX_VERSION_CACHE_TTL_SECONDS,
        retry_backoff_seconds: float = CODEX_VERSION_RETRY_BACKOFF_SECONDS,
        timeout_seconds: float = CODEX_VERSION_FETCH_TIMEOUT_SECONDS,
        registry_url: str = CODEX_NPM_LATEST_URL,
    ) -> None:
        validated_ttl = _validate_positive_finite_seconds(ttl_seconds, "ttl_seconds")
        validated_backoff = _validate_positive_finite_seconds(retry_backoff_seconds, "retry_backoff_seconds")
        validated_timeout = _validate_positive_finite_seconds(timeout_seconds, "timeout_seconds")
        validated_url = _validate_registry_url(registry_url)
        with self._condition:
            self._fetch = fetch if fetch is not None else default_fetch_codex_npm_latest
            self._clock = clock if clock is not None else time.monotonic
            self._ttl_seconds = validated_ttl
            self._retry_backoff_seconds = validated_backoff
            self._timeout_seconds = validated_timeout
            self._registry_url = validated_url
            self.reset()

    def reset(self) -> None:
        with self._condition:
            self._revision += 1
            self._last_good_version = None
            self._last_good_semver = None
            self._expires_at = None
            self._retry_not_before = None
            self._in_flight = None
            self._condition.notify_all()

    def cached_version(self) -> str | None:
        with self._condition:
            return self._last_good_version

    def get_version(self) -> str:
        return self.resolve()

    def resolve(self) -> str:
        with self._condition:
            while True:
                now = self._clock()
                if (
                    self._last_good_version is not None
                    and self._expires_at is not None
                    and now < self._expires_at
                ):
                    return self._last_good_version
                if (
                    self._last_good_version is not None
                    and self._retry_not_before is not None
                    and now < self._retry_not_before
                ):
                    return self._last_good_version
                if self._in_flight is None:
                    flight = _InFlightLookup(self._revision)
                    self._in_flight = flight
                    fetch_fn = self._fetch
                    registry_url = self._registry_url
                    timeout_seconds = self._timeout_seconds
                    break
                flight = self._in_flight
                self._condition.wait()
                if flight.done and flight.revision == self._revision:
                    if flight.error is not None:
                        raise flight.error
                    if flight.result is not None:
                        return flight.result

        return self._execute_in_flight(flight, fetch_fn, registry_url, timeout_seconds)

    async def resolve_async(self) -> str:
        return await asyncio.to_thread(self.resolve)

    def headers(self, *, codex_as_api_version: str = __version__) -> dict[str, str]:
        version = self.resolve()
        return codex_cli_headers_for_version(version, codex_as_api_version=codex_as_api_version)

    async def headers_async(self, *, codex_as_api_version: str = __version__) -> dict[str, str]:
        version = await self.resolve_async()
        return codex_cli_headers_for_version(version, codex_as_api_version=codex_as_api_version)

    def _execute_in_flight(
        self,
        flight: _InFlightLookup,
        fetch_fn: CodexVersionFetch,
        registry_url: str,
        timeout_seconds: float,
    ) -> str:
        fetched_version: str | None = None
        fetched_semver: CodexSemverTuple | None = None
        lookup_error: ChatGPTOAuthCodexVersionUnavailableError | None = None

        try:
            raw_payload = fetch_fn(registry_url, timeout_seconds)
            if not isinstance(raw_payload, (str, bytes, bytearray)):
                raise ChatGPTOAuthCodexVersionUnavailableError(
                    "npm @openai/codex metadata fetch must return bytes or str"
                )
            fetched_version = parse_codex_npm_latest_payload(raw_payload)
            fetched_semver = parse_codex_semver(fetched_version)
            if fetched_semver is None:
                raise ChatGPTOAuthCodexVersionUnavailableError("npm @openai/codex metadata version is invalid")
        except ChatGPTOAuthCodexVersionUnavailableError as exc:
            lookup_error = exc
        except TimeoutError as exc:
            lookup_error = ChatGPTOAuthCodexVersionUnavailableError(
                f"npm @openai/codex version lookup timed out after {timeout_seconds:g}s"
            )
            lookup_error.__cause__ = exc
        except Exception as exc:
            lookup_error = ChatGPTOAuthCodexVersionUnavailableError(
                f"failed to resolve latest @openai/codex version from npm: {exc}"
            )
            lookup_error.__cause__ = exc

        with self._condition:
            try:
                if lookup_error is None and fetched_version is not None and fetched_semver is not None:
                    if self._revision != flight.revision:
                        resolved_version = (
                            self._last_good_version
                            if (
                                self._last_good_version is not None
                                and self._last_good_semver is not None
                                and fetched_semver < self._last_good_semver
                            )
                            else fetched_version
                        )
                        flight.result = resolved_version
                        return resolved_version
                    completed_at = self._clock()
                    if (
                        self._last_good_version is None
                        or self._last_good_semver is None
                        or fetched_semver >= self._last_good_semver
                    ):
                        self._last_good_version = fetched_version
                        self._last_good_semver = fetched_semver
                    self._expires_at = completed_at + self._ttl_seconds
                    self._retry_not_before = None
                    flight.result = self._last_good_version
                    return self._last_good_version

                assert lookup_error is not None
                if self._revision == flight.revision and self._last_good_version is not None:
                    self._retry_not_before = self._clock() + self._retry_backoff_seconds
                    flight.result = self._last_good_version
                    return self._last_good_version

                flight.error = lookup_error
                raise lookup_error
            finally:
                flight.done = True
                if self._in_flight is flight:
                    self._in_flight = None
                self._condition.notify_all()


default_codex_version_resolver = CodexVersionResolver()


def resolve_codex_cli_version(
    resolver: CodexVersionResolver = default_codex_version_resolver,
) -> str:
    return resolver.resolve()


async def resolve_codex_cli_version_async(
    resolver: CodexVersionResolver = default_codex_version_resolver,
) -> str:
    return await resolver.resolve_async()


def resolve_latest_codex_version(
    resolver: CodexVersionResolver = default_codex_version_resolver,
) -> str:
    return resolver.resolve()


async def resolve_latest_codex_version_async(
    resolver: CodexVersionResolver = default_codex_version_resolver,
) -> str:
    return await resolver.resolve_async()


def resolve_codex_cli_headers(
    resolver: CodexVersionResolver = default_codex_version_resolver,
    *,
    codex_as_api_version: str = __version__,
) -> dict[str, str]:
    return resolver.headers(codex_as_api_version=codex_as_api_version)


async def resolve_codex_cli_headers_async(
    resolver: CodexVersionResolver = default_codex_version_resolver,
    *,
    codex_as_api_version: str = __version__,
) -> dict[str, str]:
    return await resolver.headers_async(codex_as_api_version=codex_as_api_version)


def get_cached_codex_cli_version(
    resolver: CodexVersionResolver = default_codex_version_resolver,
) -> str | None:
    return resolver.cached_version()


def reset_codex_version_cache(
    *,
    fetch: CodexVersionFetch | None = None,
    clock: CodexVersionClock | None = None,
    ttl_seconds: float = CODEX_VERSION_CACHE_TTL_SECONDS,
    retry_backoff_seconds: float = CODEX_VERSION_RETRY_BACKOFF_SECONDS,
    timeout_seconds: float = CODEX_VERSION_FETCH_TIMEOUT_SECONDS,
    registry_url: str = CODEX_NPM_LATEST_URL,
) -> None:
    default_codex_version_resolver.configure(
        fetch=fetch,
        clock=clock,
        ttl_seconds=ttl_seconds,
        retry_backoff_seconds=retry_backoff_seconds,
        timeout_seconds=timeout_seconds,
        registry_url=registry_url,
    )
