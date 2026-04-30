from __future__ import annotations

import base64
import dataclasses
import datetime as _dt
import json
import os
import pathlib
import threading
import urllib.error
import urllib.request
from typing import Any

CHATGPT_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_AUTH_PATH = "~/.codex/auth.json"
DEFAULT_REFRESH_URL = "https://auth.openai.com/oauth/token"
REFRESH_URL_OVERRIDE_ENV = "CODEX_REFRESH_TOKEN_URL_OVERRIDE"

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


class ChatGPTOAuthError(RuntimeError):
    pass


class ChatGPTOAuthMissingError(ChatGPTOAuthError):
    pass


class ChatGPTOAuthRefreshError(ChatGPTOAuthError):
    pass


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
        return self.access_expires_at is not None and self.access_expires_at <= _dt.datetime.now(_dt.UTC)


def resolve_auth_path(raw: str | None = None) -> pathlib.Path:
    value = raw or os.getenv("CODEX_HOME")
    if value and raw is None:
        return pathlib.Path(value).expanduser() / "auth.json"
    return pathlib.Path(raw or DEFAULT_AUTH_PATH).expanduser()


def _jwt_claims(jwt: str) -> dict[str, Any]:
    parts = jwt.split(".")
    if len(parts) < 2 or not parts[1]:
        return {}
    payload = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode())
        value = json.loads(decoded)
    except Exception as exc:  # noqa: BLE001 - invalid JWT is auth-data invalid
        raise ChatGPTOAuthError("invalid ChatGPT OAuth JWT payload") from exc
    if not isinstance(value, dict):
        raise ChatGPTOAuthError("invalid ChatGPT OAuth JWT claims")
    return value


def _expiration(jwt: str) -> _dt.datetime | None:
    claims = _jwt_claims(jwt)
    exp = claims.get("exp")
    if not isinstance(exp, int):
        return None
    return _dt.datetime.fromtimestamp(exp, _dt.UTC)


def _auth_claims(jwt: str) -> dict[str, Any]:
    claims = _jwt_claims(jwt)
    value = claims.get("https://api.openai.com/auth")
    return value if isinstance(value, dict) else {}


def register_token_secrets(*values: str | None) -> None:
    """No-op in standalone mode; secrets are redacted inline via redact_text()."""
    pass


def redact_text(text: str, *values: str | None) -> str:
    redacted = str(text)
    for value in sorted([v for v in values if v], key=len, reverse=True):
        redacted = redacted.replace(value, "***")
    return redacted


def load_token_data(auth_json_path: str | pathlib.Path | None = None) -> ChatGPTTokenData:
    path = resolve_auth_path(str(auth_json_path) if auth_json_path is not None else None)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ChatGPTOAuthMissingError(f"ChatGPT OAuth auth file not found: {path}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ChatGPTOAuthError(f"ChatGPT OAuth auth file is invalid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ChatGPTOAuthError("ChatGPT OAuth auth file root must be an object")
    mode = data.get("auth_mode")
    if mode not in {"chatgpt", "Chatgpt", "chatgpt_auth_tokens", "ChatgptAuthTokens", None}:
        raise ChatGPTOAuthError(f"ChatGPT OAuth auth_mode required, got {mode!r}")
    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        raise ChatGPTOAuthError("ChatGPT OAuth token data is not available")
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    id_token = tokens.get("id_token")
    for name, value in (("access_token", access_token), ("refresh_token", refresh_token), ("id_token", id_token)):
        if not isinstance(value, str) or value == "":
            raise ChatGPTOAuthError(f"ChatGPT OAuth {name} is missing")
    register_token_secrets(access_token, refresh_token, id_token)
    id_auth = _auth_claims(id_token)
    access_auth = _auth_claims(access_token)
    account_id = tokens.get("account_id") or id_auth.get("chatgpt_account_id") or access_auth.get("chatgpt_account_id")
    if not isinstance(account_id, str) or account_id == "":
        raise ChatGPTOAuthError("ChatGPT OAuth account id not available; rerun codex login")
    register_token_secrets(account_id)
    plan = id_auth.get("chatgpt_plan_type") or access_auth.get("chatgpt_plan_type")
    user = id_auth.get("chatgpt_user_id") or id_auth.get("user_id") or access_auth.get("chatgpt_user_id") or access_auth.get("user_id")
    fedramp = bool(id_auth.get("chatgpt_account_is_fedramp") or access_auth.get("chatgpt_account_is_fedramp"))
    return ChatGPTTokenData(
        auth_path=path,
        access_token=access_token,
        refresh_token=refresh_token,
        id_token=id_token,
        account_id=account_id,
        plan_type=plan if isinstance(plan, str) else None,
        user_id=user if isinstance(user, str) else None,
        fedramp=fedramp,
        access_expires_at=_expiration(access_token),
    )


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
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def refresh_token(auth_json_path: str | pathlib.Path | None = None) -> ChatGPTTokenData:
    current = load_token_data(auth_json_path)
    lock = _refresh_lock(current.auth_path)
    with lock:
        current = load_token_data(auth_json_path)
        endpoint = os.getenv(REFRESH_URL_OVERRIDE_ENV, DEFAULT_REFRESH_URL)
        body = json.dumps({
            "client_id": CHATGPT_OAUTH_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": current.refresh_token,
        }).encode()
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")
            redacted = redact_text(text, current.access_token, current.refresh_token, current.id_token)
            if exc.code == 401:
                raise ChatGPTOAuthRefreshError(f"ChatGPT OAuth refresh token is invalid; rerun codex login: {redacted}") from exc
            raise ChatGPTOAuthRefreshError(f"ChatGPT OAuth token refresh failed: HTTP {exc.code}: {redacted}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ChatGPTOAuthRefreshError(f"ChatGPT OAuth token refresh failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise ChatGPTOAuthRefreshError("ChatGPT OAuth token refresh returned invalid JSON")
        data = json.loads(current.auth_path.read_text(encoding="utf-8"))
        tokens = data.setdefault("tokens", {})
        if payload.get("id_token"):
            tokens["id_token"] = payload["id_token"]
        if payload.get("access_token"):
            tokens["access_token"] = payload["access_token"]
        if payload.get("refresh_token"):
            tokens["refresh_token"] = payload["refresh_token"]
        data["last_refresh"] = _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z")
        _write_auth_json(current.auth_path, data)
        return load_token_data(auth_json_path)
