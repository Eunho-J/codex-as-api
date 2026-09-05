from __future__ import annotations

import base64
import binascii
import dataclasses
import datetime as _dt
import http.client
import json
import math
import os
import pathlib
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, cast

from .strict_json import JS_SAFE_INTEGER, strict_json_loads

CHATGPT_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_AUTH_PATH = "~/.codex/auth.json"
DEFAULT_REFRESH_URL = "https://auth.openai.com/oauth/token"
REFRESH_URL_OVERRIDE_ENV = "CODEX_REFRESH_TOKEN_URL_OVERRIDE"
REFRESH_WINDOW = _dt.timedelta(minutes=5)
REFRESH_TIMEOUT_SECONDS = 30.0

_SECRET_KEYS = (
    "access_token",
    "refresh_token",
    "id_token",
    "Authorization",
    "authorization",
    "ChatGPT-Account-Id",
    "chatgpt-account-id",
)

_REFRESH_LOCKS: dict[pathlib.Path, threading.Lock] = {}
_REFRESH_LOCKS_GUARD = threading.Lock()


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _urlopen_no_redirect(request: urllib.request.Request, *, timeout: float) -> Any:
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


def _response_socket(response: Any) -> Any | None:
    stream = response
    for _ in range(2):
        buffered = getattr(stream, "fp", None)
        raw = getattr(buffered, "raw", None)
        sock = getattr(raw, "_sock", None)
        if sock is not None:
            return sock
        if buffered is None:
            break
        stream = buffered
    return None


def _read_response_before_deadline(response: Any, deadline: float) -> bytes:
    sock = _response_socket(response)
    read1 = getattr(response, "read1", None)
    if sock is None or not callable(read1):
        body = response.read()
        if time.monotonic() >= deadline:
            raise TimeoutError("OAuth refresh response body exceeded its total deadline")
        return cast(bytes, body)

    chunks: list[bytes] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("OAuth refresh response body exceeded its total deadline")
        sock.settimeout(remaining)
        chunk = read1(64 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        if callable(getattr(response, "isclosed", None)) and response.isclosed():
            return b"".join(chunks)


def _is_ascii_visible_token(value: str) -> bool:
    return bool(value) and all("!" <= character <= "~" for character in value)


class ChatGPTOAuthError(RuntimeError):
    pass


class ChatGPTOAuthInvalidRequestError(ChatGPTOAuthError):
    pass


class ChatGPTOAuthModelNotFoundError(ChatGPTOAuthError):
    pass


class ChatGPTOAuthCatalogUnavailableError(ChatGPTOAuthError):
    pass


class ChatGPTOAuthProtocolError(ChatGPTOAuthError):
    pass


class ChatGPTOAuthMissingError(ChatGPTOAuthError):
    pass


class ChatGPTOAuthRefreshError(ChatGPTOAuthError):
    pass


class _AuthWriteCleanupError(RuntimeError):
    def __init__(self, primary: BaseException, cleanup: BaseException) -> None:
        self.errors = (primary, cleanup)
        super().__init__(
            f"failed to write ChatGPT OAuth auth file: {primary}; temporary cleanup also failed: {cleanup}"
        )


class ChatGPTOAuthUpstreamError(ChatGPTOAuthError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclasses.dataclass(frozen=True, slots=True)
class ChatGPTTokenData:
    auth_path: pathlib.Path
    access_token: str
    refresh_token: str
    id_token: str
    account_id: str
    plan_type: str | None
    user_id: str | None
    fedramp: bool
    access_expires_at: _dt.datetime | None

    @property
    def expired(self) -> bool:
        return self.access_expires_at is not None and self.access_expires_at <= _dt.datetime.now(_dt.timezone.utc)

    def expires_within(self, window: _dt.timedelta = REFRESH_WINDOW) -> bool:
        return (
            self.access_expires_at is not None and self.access_expires_at <= _dt.datetime.now(_dt.timezone.utc) + window
        )


def resolve_auth_path(raw: str | None = None) -> pathlib.Path:
    if raw is not None:
        if not raw.strip():
            raise ValueError("auth path must be a non-empty string")
        return pathlib.Path(raw).expanduser()
    codex_home = os.getenv("CODEX_HOME")
    if codex_home is not None:
        if not codex_home.strip():
            raise ValueError("CODEX_HOME must be a non-empty string")
        return pathlib.Path(codex_home).expanduser() / "auth.json"
    return pathlib.Path(DEFAULT_AUTH_PATH).expanduser()


def _jwt_claims(jwt: str) -> dict[str, Any]:
    parts = jwt.split(".")
    if len(parts) != 3 or any(not part for part in parts):
        raise ChatGPTOAuthMissingError("invalid ChatGPT OAuth JWT structure")
    payload_segment = parts[1]
    if re.fullmatch(r"[A-Za-z0-9_-]+", payload_segment, flags=re.ASCII) is None or len(payload_segment) % 4 == 1:
        raise ChatGPTOAuthMissingError("invalid ChatGPT OAuth JWT payload")
    payload = payload_segment + "=" * ((4 - len(payload_segment) % 4) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode())
        canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
        if canonical != payload_segment:
            raise ChatGPTOAuthMissingError("invalid ChatGPT OAuth JWT payload")
        value = strict_json_loads(decoded)
    except (
        binascii.Error,
        UnicodeDecodeError,
        UnicodeEncodeError,
        ValueError,
    ) as exc:
        raise ChatGPTOAuthMissingError("invalid ChatGPT OAuth JWT payload") from exc
    if not isinstance(value, dict):
        raise ChatGPTOAuthMissingError("invalid ChatGPT OAuth JWT claims")
    return value


def _expiration(jwt: str) -> _dt.datetime | None:
    claims = _jwt_claims(jwt)
    exp = claims.get("exp")
    if exp is None:
        return None
    if isinstance(exp, bool):
        raise ChatGPTOAuthMissingError("invalid ChatGPT OAuth JWT exp claim")
    if isinstance(exp, int):
        timestamp = exp
    elif isinstance(exp, float) and math.isfinite(exp) and exp.is_integer() and abs(exp) <= JS_SAFE_INTEGER:
        timestamp = int(exp)
    else:
        raise ChatGPTOAuthMissingError("invalid ChatGPT OAuth JWT exp claim")
    if abs(timestamp) > JS_SAFE_INTEGER:
        raise ChatGPTOAuthMissingError("invalid ChatGPT OAuth JWT exp claim")
    try:
        return _dt.datetime.fromtimestamp(timestamp, _dt.timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ChatGPTOAuthMissingError("invalid ChatGPT OAuth JWT exp claim") from exc


def _auth_claims(jwt: str) -> dict[str, Any]:
    claims = _jwt_claims(jwt)
    key = "https://api.openai.com/auth"
    if key not in claims:
        return {}
    value = claims[key]
    if not isinstance(value, dict):
        raise ChatGPTOAuthMissingError("invalid ChatGPT OAuth auth claim")
    return value


def redact_text(text: str, *values: str | None) -> str:
    redacted = str(text)
    secrets = sorted([v for v in values if v], key=len, reverse=True)
    marker = "" if any(value in "***" for value in secrets) else "***"
    for value in secrets:
        redacted = redacted.replace(value, marker)
    while any(value in redacted for value in secrets):
        for value in secrets:
            redacted = redacted.replace(value, "")
    return redacted


def load_token_data(auth_json_path: str | pathlib.Path | None = None) -> ChatGPTTokenData:
    path = resolve_auth_path(str(auth_json_path) if auth_json_path is not None else None)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ChatGPTOAuthMissingError("ChatGPT OAuth auth file not found") from exc
    except (OSError, UnicodeError) as exc:
        raise ChatGPTOAuthMissingError("ChatGPT OAuth auth file is unavailable") from exc
    try:
        data = strict_json_loads(raw)
    except ValueError as exc:
        raise ChatGPTOAuthMissingError("ChatGPT OAuth auth file is invalid JSON") from exc
    return _token_data_from_document(data, path)


def _token_data_from_document(data: object, path: pathlib.Path) -> ChatGPTTokenData:
    if not isinstance(data, dict):
        raise ChatGPTOAuthMissingError("ChatGPT OAuth auth file root must be an object")
    mode = data.get("auth_mode")
    if mode is not None and (not isinstance(mode, str) or mode != "chatgpt"):
        raise ChatGPTOAuthMissingError("ChatGPT OAuth auth_mode is unsupported")
    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        if "tokens" in data:
            raise ChatGPTOAuthMissingError("ChatGPT OAuth auth file tokens must be an object")
        raise ChatGPTOAuthMissingError(_unsupported_auth_schema_message(data))
    if any(key in data for key in ("access_token", "refresh_token", "id_token", "chatgptAuthTokens")):
        raise ChatGPTOAuthMissingError(
            "ChatGPT OAuth auth file mixes canonical tokens with unsupported root credentials"
        )
    access_token = tokens.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise ChatGPTOAuthMissingError("ChatGPT OAuth access_token is missing")
    if not _is_ascii_visible_token(access_token):
        raise ChatGPTOAuthMissingError("ChatGPT OAuth access_token is invalid")
    refresh_token = tokens.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise ChatGPTOAuthMissingError("ChatGPT OAuth refresh_token is missing")
    id_token = tokens.get("id_token")
    if not isinstance(id_token, str) or not id_token.strip():
        raise ChatGPTOAuthMissingError("ChatGPT OAuth id_token is missing")
    id_auth = _auth_claims(id_token)
    access_auth = _auth_claims(access_token)
    account_sources = {
        "account_id": tokens.get("account_id"),
        "id_token": id_auth.get("chatgpt_account_id"),
        "access_token": access_auth.get("chatgpt_account_id"),
    }
    for source, value in account_sources.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ChatGPTOAuthMissingError(f"ChatGPT OAuth {source} account id is invalid")
    account_ids = {value for value in account_sources.values() if isinstance(value, str) and value}
    if not account_ids:
        raise ChatGPTOAuthMissingError("ChatGPT OAuth account id not available; rerun codex login")
    if len(account_ids) != 1:
        raise ChatGPTOAuthMissingError("ChatGPT OAuth token account ids do not match")
    account_id = account_ids.pop()
    if not _is_ascii_visible_token(account_id):
        raise ChatGPTOAuthMissingError("ChatGPT OAuth account id is invalid")
    plan = _consistent_optional_string_claim(
        (
            (id_auth, "chatgpt_plan_type"),
            (access_auth, "chatgpt_plan_type"),
        ),
        "plan type",
    )
    user = _consistent_optional_string_claim(
        (
            (id_auth, "chatgpt_user_id"),
            (id_auth, "user_id"),
            (access_auth, "chatgpt_user_id"),
            (access_auth, "user_id"),
        ),
        "user id",
    )
    fedramp_claims = [
        claims["chatgpt_account_is_fedramp"]
        for claims in (id_auth, access_auth)
        if "chatgpt_account_is_fedramp" in claims
    ]
    if any(not isinstance(value, bool) for value in fedramp_claims):
        raise ChatGPTOAuthMissingError("ChatGPT OAuth fedramp claim must be a boolean")
    if len(set(fedramp_claims)) > 1:
        raise ChatGPTOAuthMissingError("ChatGPT OAuth fedramp claims do not match")
    fedramp = fedramp_claims[0] if fedramp_claims else False
    return ChatGPTTokenData(
        auth_path=path,
        access_token=access_token,
        refresh_token=refresh_token,
        id_token=id_token,
        account_id=account_id,
        plan_type=plan,
        user_id=user,
        fedramp=fedramp,
        access_expires_at=_expiration(access_token),
    )


def _consistent_optional_string_claim(
    sources: tuple[tuple[dict[str, Any], str], ...],
    field: str,
) -> str | None:
    values: list[str] = []
    for claims, key in sources:
        if key not in claims:
            continue
        value = claims[key]
        if not isinstance(value, str) or not value.strip():
            raise ChatGPTOAuthMissingError(f"ChatGPT OAuth {field} claim must be a non-empty string")
        values.append(value)
    if len(set(values)) > 1:
        raise ChatGPTOAuthMissingError(f"ChatGPT OAuth {field} claims do not match")
    return values[0] if values else None


def _unsupported_auth_schema_message(data: dict[str, Any]) -> str:
    if "personal_access_token" in data and "tokens" not in data:
        return (
            "ChatGPT OAuth personal_access_token-only auth is not supported; "
            "rerun codex login to create file-backed tokens"
        )
    if "agent_identity" in data and "tokens" not in data:
        return "ChatGPT OAuth agent_identity-only auth is not supported; rerun codex login to create file-backed tokens"
    if "bedrock_api_key" in data and "tokens" not in data:
        return "ChatGPT OAuth bedrock_api_key-only auth is not supported by the ChatGPT OAuth backend"
    return "ChatGPT OAuth file-backed ChatGPT OAuth tokens are required; rerun codex login"


def is_auth_locally_available(auth_json_path: str | pathlib.Path | None = None) -> bool:
    try:
        data = load_token_data(auth_json_path)
    except ChatGPTOAuthError:
        return False
    return bool(data.access_token and data.account_id)


def _refresh_lock(path: pathlib.Path) -> threading.Lock:
    resolved = path.expanduser()
    with _REFRESH_LOCKS_GUARD:
        lock = _REFRESH_LOCKS.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _REFRESH_LOCKS[resolved] = lock
        return lock


def _write_auth_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(tmp, flags, 0o600)
    failure: BaseException | None = None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
        if os.name != "nt":
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except BaseException as exc:
        failure = exc
    try:
        tmp.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    except OSError as cleanup:
        failure = cleanup if failure is None else _AuthWriteCleanupError(failure, cleanup)
    if failure is not None:
        raise failure


def token_for_request(auth_json_path: str | pathlib.Path | None = None) -> ChatGPTTokenData:
    current = load_token_data(auth_json_path)
    if not current.expires_within():
        return current
    return _refresh_token(current, refresh_if_expiring=True)


def refresh_after_unauthorized(current: ChatGPTTokenData) -> ChatGPTTokenData:
    return _refresh_token(current, refresh_if_expiring=False)


def refresh_token(auth_json_path: str | pathlib.Path | None = None) -> ChatGPTTokenData:
    return _refresh_token(load_token_data(auth_json_path), refresh_if_expiring=False)


def _fail_if_account_changed(current: ChatGPTTokenData, latest: ChatGPTTokenData) -> None:
    if current.account_id != latest.account_id:
        raise ChatGPTOAuthRefreshError("ChatGPT OAuth account changed while refreshing credentials")


def _access_token_changed(current: ChatGPTTokenData, latest: ChatGPTTokenData) -> bool:
    return current.access_token != latest.access_token


def _other_credentials_changed(current: ChatGPTTokenData, latest: ChatGPTTokenData) -> bool:
    return current.refresh_token != latest.refresh_token or current.id_token != latest.id_token


def _validate_refreshed_token_accounts(payload: dict[str, Any], account_id: str) -> None:
    for name in ("access_token", "id_token"):
        value = payload.get(name)
        if not isinstance(value, str) or not value:
            continue
        claim_account = _auth_claims(value).get("chatgpt_account_id")
        if claim_account is not None and (not isinstance(claim_account, str) or claim_account != account_id):
            raise ChatGPTOAuthRefreshError(f"ChatGPT OAuth refreshed {name} account id does not match current account")


def _refresh_url_from_environment() -> str:
    value = os.getenv(REFRESH_URL_OVERRIDE_ENV)
    if value is None:
        return DEFAULT_REFRESH_URL
    if not value.strip() or value != value.strip():
        raise ChatGPTOAuthRefreshError(f"{REFRESH_URL_OVERRIDE_ENV} must be non-empty without surrounding whitespace")
    if any(character.isspace() or unicodedata.category(character) == "Cc" for character in value):
        raise ChatGPTOAuthRefreshError(f"{REFRESH_URL_OVERRIDE_ENV} must not contain whitespace or control characters")
    if re.search(r"%(?![0-9A-Fa-f]{2})", value):
        raise ChatGPTOAuthRefreshError(f"{REFRESH_URL_OVERRIDE_ENV} contains an invalid percent escape")
    try:
        parsed = urllib.parse.urlsplit(value)
        _ = parsed.port
    except (TypeError, ValueError) as exc:
        raise ChatGPTOAuthRefreshError(f"{REFRESH_URL_OVERRIDE_ENV} must be a valid HTTP(S) URL") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ChatGPTOAuthRefreshError(
            f"{REFRESH_URL_OVERRIDE_ENV} must be an absolute HTTP(S) URL without credentials, query, or fragment"
        )
    return value


def validate_auth_environment() -> None:
    _refresh_url_from_environment()


def _refresh_token(current: ChatGPTTokenData, *, refresh_if_expiring: bool) -> ChatGPTTokenData:
    lock = _refresh_lock(current.auth_path)
    with lock:
        latest = load_token_data(current.auth_path)
        _fail_if_account_changed(current, latest)
        if _access_token_changed(current, latest):
            return latest
        if refresh_if_expiring and not latest.expires_within():
            return latest
        current = latest
        endpoint = _refresh_url_from_environment()
        body = json.dumps(
            {
                "client_id": CHATGPT_OAUTH_CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": current.refresh_token,
            }
        ).encode()
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        deadline = time.monotonic() + REFRESH_TIMEOUT_SECONDS
        try:
            with _urlopen_no_redirect(request, timeout=REFRESH_TIMEOUT_SECONDS) as response:
                payload = strict_json_loads(_read_response_before_deadline(response, deadline))
        except urllib.error.HTTPError as exc:
            try:
                text = _read_response_before_deadline(exc, deadline).decode("utf-8", "replace")
            except (OSError, ValueError, TimeoutError, http.client.HTTPException):
                text = "could not read upstream refresh error body"
            redacted = redact_text(
                text,
                current.access_token,
                current.refresh_token,
                current.id_token,
                current.account_id,
            )
            if exc.code in {400, 401}:
                raise ChatGPTOAuthRefreshError(
                    f"ChatGPT OAuth refresh token is invalid; rerun codex login: {redacted}"
                ) from exc
            raise ChatGPTOAuthUpstreamError(
                exc.code,
                f"ChatGPT OAuth token refresh failed: HTTP {exc.code}: {redacted}",
            ) from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ) as exc:
            redacted = redact_text(
                str(exc),
                current.access_token,
                current.refresh_token,
                current.id_token,
                current.account_id,
            )
            raise ChatGPTOAuthUpstreamError(502, f"ChatGPT OAuth token refresh failed: {redacted}") from exc
        except (UnicodeError, ValueError) as exc:
            raise ChatGPTOAuthProtocolError("ChatGPT OAuth token refresh returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ChatGPTOAuthProtocolError("ChatGPT OAuth token refresh returned invalid JSON")
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise ChatGPTOAuthProtocolError("ChatGPT OAuth token refresh response is missing access_token")
        if not _is_ascii_visible_token(access_token):
            raise ChatGPTOAuthProtocolError("ChatGPT OAuth token refresh response has invalid access_token")
        for name in ("id_token", "refresh_token"):
            if name not in payload:
                continue
            value = payload[name]
            if not isinstance(value, str) or not value.strip():
                raise ChatGPTOAuthProtocolError(f"ChatGPT OAuth token refresh response has invalid {name}")
        _validate_refreshed_token_accounts(payload, current.account_id)

        try:
            data = strict_json_loads(current.auth_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("failed to re-read ChatGPT OAuth auth file") from exc
        try:
            exact = _token_data_from_document(data, current.auth_path)
        except ChatGPTOAuthError as exc:
            raise RuntimeError("ChatGPT OAuth auth file changed to an invalid schema") from exc
        _fail_if_account_changed(current, exact)
        if _access_token_changed(current, exact):
            return exact
        if _other_credentials_changed(current, exact):
            raise ChatGPTOAuthRefreshError("ChatGPT OAuth credentials changed while token refresh was in flight")
        tokens = data.get("tokens")
        if not isinstance(tokens, dict):
            raise RuntimeError("ChatGPT OAuth auth file changed to an invalid schema")
        if payload.get("id_token"):
            tokens["id_token"] = payload["id_token"]
        tokens["access_token"] = access_token
        if payload.get("refresh_token"):
            tokens["refresh_token"] = payload["refresh_token"]
        data["last_refresh"] = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
        _write_auth_json(current.auth_path, data)
        return load_token_data(current.auth_path)
