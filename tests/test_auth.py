from __future__ import annotations

import base64
import io
import json
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from codex_as_api.auth import (
    ChatGPTOAuthError,
    ChatGPTOAuthMissingError,
    ChatGPTOAuthProtocolError,
    ChatGPTOAuthRefreshError,
    ChatGPTOAuthUpstreamError,
    _auth_claims,
    _AuthWriteCleanupError,
    _expiration,
    _jwt_claims,
    _refresh_url_from_environment,
    _write_auth_json,
    is_auth_locally_available,
    load_token_data,
    redact_text,
    refresh_after_unauthorized,
    refresh_token,
    resolve_auth_path,
    token_for_request,
    validate_auth_environment,
)

# ---------------------------------------------------------------------------
# resolve_auth_path
# ---------------------------------------------------------------------------


def test_resolve_auth_path_default():
    path = resolve_auth_path(None)
    assert path == pathlib.Path("~/.codex/auth.json").expanduser()


def test_resolve_auth_path_codex_home_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    path = resolve_auth_path(None)
    assert path == tmp_path / "auth.json"


def test_resolve_auth_path_explicit(tmp_path):
    explicit = str(tmp_path / "custom.json")
    path = resolve_auth_path(explicit)
    assert path == pathlib.Path(explicit)


def test_resolve_auth_path_explicit_ignores_codex_home(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    explicit = str(tmp_path / "explicit.json")
    path = resolve_auth_path(explicit)
    assert path == pathlib.Path(explicit)


@pytest.mark.parametrize("value", ["", "   "])
def test_resolve_auth_path_rejects_empty_explicit_path(value):
    with pytest.raises(ValueError, match="non-empty"):
        resolve_auth_path(value)


@pytest.mark.parametrize("value", ["", "   "])
def test_resolve_auth_path_rejects_empty_codex_home(monkeypatch, value):
    monkeypatch.setenv("CODEX_HOME", value)

    with pytest.raises(ValueError, match="CODEX_HOME"):
        resolve_auth_path(None)


# ---------------------------------------------------------------------------
# _jwt_claims
# ---------------------------------------------------------------------------


def _jwt_with_raw_payload(payload: str) -> str:
    encoded = base64.urlsafe_b64encode(payload.encode()).rstrip(b"=").decode()
    return f"header.{encoded}.sig"


def test_jwt_claims_valid(make_jwt):
    payload = {"sub": "user1", "exp": 12345}
    token = make_jwt(payload)
    claims = _jwt_claims(token)
    assert claims["sub"] == "user1"
    assert claims["exp"] == 12345


def test_jwt_claims_missing_payload_raises():
    with pytest.raises(ChatGPTOAuthError, match="JWT structure"):
        _jwt_claims("onlyonepart")


@pytest.mark.parametrize("token", ["header..sig", ".payload.sig", "header.payload.", "a.b.c.d"])
def test_jwt_claims_empty_or_extra_parts_raise(token):
    with pytest.raises(ChatGPTOAuthError, match="JWT structure"):
        _jwt_claims(token)


def test_jwt_claims_invalid_base64_raises(make_jwt):
    with pytest.raises(ChatGPTOAuthError):
        _jwt_claims("header.!!!invalid!!!.sig")


@pytest.mark.parametrize(
    "payload",
    [
        "e30=",  # padded base64url
        "e+0",  # standard-base64 alphabet
        "e/0",  # standard-base64 alphabet
        "a",  # impossible base64url length
        "e31",  # non-canonical trailing bits for b"{}"
    ],
)
def test_jwt_claims_rejects_noncanonical_base64url_payloads(payload):
    with pytest.raises(ChatGPTOAuthMissingError, match="JWT payload"):
        _jwt_claims(f"header.{payload}.sig")


def test_jwt_claims_non_dict_raises(make_jwt):
    payload = base64.urlsafe_b64encode(b'"just a string"').rstrip(b"=").decode()
    with pytest.raises(ChatGPTOAuthError):
        _jwt_claims(f"header.{payload}.sig")


# ---------------------------------------------------------------------------
# _expiration
# ---------------------------------------------------------------------------


def test_expiration_returns_datetime(make_jwt):
    token = make_jwt({"exp": 2000000000})
    dt = _expiration(token)
    assert dt is not None
    import datetime as _dt

    assert dt.tzinfo is _dt.UTC


def test_expiration_missing_exp_returns_none(make_jwt):
    token = make_jwt({"sub": "user"})
    assert _expiration(token) is None


def test_expiration_non_int_exp_raises(make_jwt):
    token = make_jwt({"exp": "not-an-int"})
    with pytest.raises(ChatGPTOAuthMissingError, match="exp claim"):
        _expiration(token)


@pytest.mark.parametrize("raw_exp", ["1.0", "1e0"])
def test_expiration_accepts_integral_json_numbers(raw_exp):
    expiration = _expiration(_jwt_with_raw_payload(f'{{"exp":{raw_exp}}}'))
    assert expiration is not None
    assert expiration.timestamp() == 1


@pytest.mark.parametrize("raw_exp", ["1.5", "true", "9007199254740992"])
def test_expiration_rejects_nonintegral_unsafe_and_boolean_numbers(raw_exp):
    with pytest.raises(ChatGPTOAuthMissingError):
        _expiration(_jwt_with_raw_payload(f'{{"exp":{raw_exp}}}'))


# ---------------------------------------------------------------------------
# _auth_claims
# ---------------------------------------------------------------------------


def test_auth_claims_extracts_openai_auth(make_jwt):
    payload = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": "acc-xyz",
            "chatgpt_plan_type": "plus",
        }
    }
    token = make_jwt(payload)
    claims = _auth_claims(token)
    assert claims["chatgpt_account_id"] == "acc-xyz"


def test_auth_claims_missing_key_returns_empty(make_jwt):
    token = make_jwt({"sub": "user"})
    assert _auth_claims(token) == {}


def test_auth_claims_non_dict_value_raises(make_jwt):
    token = make_jwt({"https://api.openai.com/auth": "bad"})
    with pytest.raises(ChatGPTOAuthMissingError, match="auth claim"):
        _auth_claims(token)


# ---------------------------------------------------------------------------
# redact_text
# ---------------------------------------------------------------------------


def test_redact_text_replaces_secrets():
    result = redact_text("Bearer mytoken123 and refresh-abc", "mytoken123", "refresh-abc")
    assert result == "Bearer *** and ***"


def test_redact_text_longer_secret_replaced_first():
    result = redact_text("prefix123456789suffix", "12345", "123456789")
    assert "123456789" not in result


def test_redact_text_none_values_skipped():
    result = redact_text("hello world", None, "world")
    assert result == "hello ***"


def test_redact_text_handles_marker_secrets_and_replacement_boundaries():
    assert redact_text("*** token *", "*") == " token "
    assert redact_text("*** token ***", "***") == " token "
    boundary_safe = redact_text("ab", "a*", "b")
    assert "a*" not in boundary_safe
    assert "b" not in boundary_safe


def test_redact_text_no_match_unchanged():
    result = redact_text("nothing to hide", "secret")
    assert result == "nothing to hide"


# ---------------------------------------------------------------------------
# load_token_data
# ---------------------------------------------------------------------------


def test_load_token_data_valid(auth_json_factory):
    path = auth_json_factory()
    data = load_token_data(str(path))
    assert data.access_token
    assert data.refresh_token == "refresh-tok"
    assert data.id_token
    assert data.account_id == "acc-123"
    assert data.plan_type == "plus"
    assert data.user_id == "user-abc"
    assert data.fedramp is False
    assert data.auth_path == path


def test_load_token_data_missing_file_raises(tmp_path):
    with pytest.raises(ChatGPTOAuthMissingError):
        load_token_data(str(tmp_path / "nonexistent.json"))


def test_load_token_data_invalid_json_raises(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text("not json {{{")
    with pytest.raises(ChatGPTOAuthError):
        load_token_data(str(p))


def test_load_token_data_rejects_nonstandard_json_numbers(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text('{"tokens":{"access_token":NaN}}')

    with pytest.raises(ChatGPTOAuthMissingError, match="invalid JSON"):
        load_token_data(str(p))


def test_load_token_data_invalid_utf8_is_authentication_error(tmp_path):
    p = tmp_path / "auth.json"
    p.write_bytes(b"\xff")

    with pytest.raises(ChatGPTOAuthMissingError) as caught:
        load_token_data(str(p))
    assert str(caught.value) == "ChatGPT OAuth auth file is unavailable"
    assert str(p) not in str(caught.value)


def test_load_token_data_unreadable_file_does_not_expose_path(monkeypatch):
    secret_path = pathlib.Path("/private/secret/auth.json")

    def fail_read(_self, *args, **kwargs):
        raise PermissionError("permission denied for private auth path")

    monkeypatch.setattr(pathlib.Path, "read_text", fail_read)
    with pytest.raises(ChatGPTOAuthMissingError) as caught:
        load_token_data(secret_path)
    assert str(caught.value) == "ChatGPT OAuth auth file is unavailable"
    assert str(secret_path) not in str(caught.value)


def test_write_auth_json_preserves_primary_and_cleanup_failures(monkeypatch, tmp_path):
    primary = OSError("simulated rename failure")
    cleanup = OSError("simulated cleanup failure")

    monkeypatch.setattr(pathlib.Path, "replace", lambda *_args: (_ for _ in ()).throw(primary))
    monkeypatch.setattr(pathlib.Path, "unlink", lambda *_args, **_kwargs: (_ for _ in ()).throw(cleanup))

    with pytest.raises(_AuthWriteCleanupError) as caught:
        _write_auth_json(tmp_path / "auth.json", {"tokens": {}})
    assert caught.value.errors == (primary, cleanup)


def test_write_auth_json_surfaces_cleanup_only_failure(monkeypatch, tmp_path):
    cleanup = OSError("simulated cleanup failure")
    monkeypatch.setattr(pathlib.Path, "unlink", lambda *_args, **_kwargs: (_ for _ in ()).throw(cleanup))

    with pytest.raises(OSError) as caught:
        _write_auth_json(tmp_path / "auth.json", {"tokens": {}})
    assert caught.value is cleanup


def test_load_token_data_root_not_dict_raises(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text('["list", "not", "dict"]')
    with pytest.raises(ChatGPTOAuthError):
        load_token_data(str(p))


def test_load_token_data_missing_tokens_raises(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({}))
    with pytest.raises(ChatGPTOAuthError, match="file-backed ChatGPT OAuth tokens are required"):
        load_token_data(str(p))


def test_load_token_data_rejects_root_token_fields(tmp_path, make_jwt):
    id_token = make_jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acc-root",
                "chatgpt_plan_type": "plus",
                "chatgpt_user_id": "user-root",
            }
        }
    )
    access_token = make_jwt({"exp": 9999999999})
    p = tmp_path / "auth.json"
    p.write_text(
        json.dumps(
            {
                "access_token": access_token,
                "refresh_token": "refresh-root",
                "id_token": id_token,
            }
        )
    )

    with pytest.raises(ChatGPTOAuthMissingError, match="file-backed ChatGPT OAuth tokens"):
        load_token_data(str(p))


@pytest.mark.parametrize(
    "root_key",
    ["access_token", "refresh_token", "id_token", "chatgptAuthTokens"],
)
def test_load_token_data_rejects_mixed_nested_and_root_credentials(
    auth_json_factory,
    root_key,
):
    path = auth_json_factory(extra={root_key: "unsupported-root-value"})

    with pytest.raises(ChatGPTOAuthMissingError, match="mixes canonical tokens"):
        load_token_data(path)


def test_load_token_data_rejects_present_null_tokens_instead_of_using_root_fields(
    tmp_path,
    make_jwt,
):
    auth_claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "acc-root"}}
    p = tmp_path / "auth.json"
    p.write_text(
        json.dumps(
            {
                "tokens": None,
                "access_token": make_jwt({"exp": 9999999999, **auth_claims}),
                "refresh_token": "refresh-root",
                "id_token": make_jwt(auth_claims),
            }
        )
    )

    with pytest.raises(ChatGPTOAuthMissingError, match="tokens must be an object"):
        load_token_data(p)


def test_load_token_data_pat_only_has_specific_error(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({"personal_access_token": "pat-only"}))

    with pytest.raises(ChatGPTOAuthError, match="personal_access_token-only auth is not supported"):
        load_token_data(str(p))


def test_load_token_data_agent_identity_only_has_specific_error(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({"agent_identity": {"id": "agent-only"}}))

    with pytest.raises(ChatGPTOAuthError, match="agent_identity-only auth is not supported"):
        load_token_data(str(p))


def test_load_token_data_bedrock_only_has_specific_error(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({"bedrock_api_key": "bedrock-only"}))

    with pytest.raises(ChatGPTOAuthError, match="bedrock_api_key-only auth is not supported"):
        load_token_data(str(p))


def test_load_token_data_missing_access_token_raises(tmp_path, make_jwt):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({"tokens": {"refresh_token": "r", "id_token": make_jwt({})}}))
    with pytest.raises(ChatGPTOAuthError):
        load_token_data(str(p))


@pytest.mark.parametrize("field", ["access_token", "refresh_token", "id_token"])
def test_load_token_data_rejects_whitespace_credentials(
    auth_json_factory,
    field,
):
    path = auth_json_factory()
    document = json.loads(path.read_text(encoding="utf-8"))
    document["tokens"][field] = "   "
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ChatGPTOAuthMissingError):
        load_token_data(path)


@pytest.mark.parametrize("suffix", [" ", "\n", "é"])
def test_load_token_data_rejects_header_unsafe_access_token(
    auth_json_factory,
    suffix,
):
    path = auth_json_factory()
    document = json.loads(path.read_text(encoding="utf-8"))
    document["tokens"]["access_token"] += suffix
    unsafe_token = document["tokens"]["access_token"]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ChatGPTOAuthMissingError, match="access_token is invalid") as caught:
        load_token_data(path)
    assert unsafe_token not in str(caught.value)


def test_load_token_data_rejects_header_unsafe_final_account_id(tmp_path, make_jwt):
    account_id = "bad account"
    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": make_jwt(claims),
                    "refresh_token": "refresh",
                    "id_token": make_jwt(claims),
                    "account_id": account_id,
                }
            }
        )
    )

    with pytest.raises(ChatGPTOAuthMissingError, match="account id is invalid") as caught:
        load_token_data(path)
    assert account_id not in str(caught.value)


def test_load_token_data_invalid_auth_mode_raises(tmp_path, auth_json_factory):
    secret = "access-token-sentinel"
    p = auth_json_factory(extra={"auth_mode": secret})
    with pytest.raises(ChatGPTOAuthError) as caught:
        load_token_data(str(p))
    assert secret not in str(caught.value)


@pytest.mark.parametrize(
    "auth_mode",
    [
        "Chatgpt",
        "ChatgptAuthTokens",
        "chatgpt_auth_tokens",
        "chatgptAuthTokens",
        ["chatgpt"],
        {"mode": "chatgpt"},
    ],
)
def test_load_token_data_rejects_non_managed_auth_modes(auth_json_factory, auth_mode):
    path = auth_json_factory(extra={"auth_mode": auth_mode})
    with pytest.raises(ChatGPTOAuthMissingError, match="auth_mode is unsupported"):
        load_token_data(path)


def test_external_auth_mode_is_rejected_before_refresh_transport(monkeypatch, auth_json_factory):
    path = auth_json_factory(extra={"auth_mode": "chatgptAuthTokens"})
    monkeypatch.setattr(
        "codex_as_api.auth._urlopen_no_redirect",
        lambda *_args, **_kwargs: pytest.fail("external auth must not reach managed refresh transport"),
    )
    with pytest.raises(ChatGPTOAuthMissingError, match="auth_mode is unsupported"):
        refresh_token(path)


@pytest.mark.parametrize("auth_mode", [None, "chatgpt"])
def test_load_token_data_accepts_official_managed_auth_modes(auth_json_factory, auth_mode):
    path = auth_json_factory(extra={"auth_mode": auth_mode})
    assert load_token_data(path).account_id == "acc-123"


def test_load_token_data_expiration_extracted(auth_json_factory):
    import datetime as _dt

    future_exp = int(_dt.datetime(2099, 1, 1, tzinfo=_dt.UTC).timestamp())
    path = auth_json_factory(access_payload={"exp": future_exp})
    data = load_token_data(str(path))
    assert data.access_expires_at is not None
    assert data.access_expires_at.year == 2099


def test_load_token_data_fedramp_flag(make_jwt, tmp_path):
    id_payload = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": "acc-fed",
            "chatgpt_account_is_fedramp": True,
        }
    }
    access_payload = {"exp": 9999999999}
    id_token = make_jwt(id_payload)
    access_token = make_jwt(access_payload)
    p = tmp_path / "auth.json"
    p.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "r",
                    "id_token": id_token,
                }
            }
        )
    )
    data = load_token_data(str(p))
    assert data.fedramp is True


@pytest.mark.parametrize(
    ("id_value", "access_value"),
    [("false", None), (True, False)],
)
def test_load_token_data_rejects_invalid_or_conflicting_fedramp_claims(
    auth_json_factory,
    id_value,
    access_value,
):
    id_auth = {"chatgpt_account_id": "acc-123"}
    access_auth = {"chatgpt_account_id": "acc-123"}
    if id_value is not None:
        id_auth["chatgpt_account_is_fedramp"] = id_value
    if access_value is not None:
        access_auth["chatgpt_account_is_fedramp"] = access_value
    path = auth_json_factory(
        id_payload={"https://api.openai.com/auth": id_auth},
        access_payload={
            "exp": 9999999999,
            "https://api.openai.com/auth": access_auth,
        },
    )

    with pytest.raises(ChatGPTOAuthMissingError, match="fedramp"):
        load_token_data(path)


@pytest.mark.parametrize(
    ("claim", "id_value", "access_value", "message"),
    [
        ("chatgpt_plan_type", 42, "plus", "plan type"),
        ("chatgpt_plan_type", "plus", "pro", "plan type"),
        ("chatgpt_user_id", None, "user-abc", "user id"),
        ("chatgpt_user_id", "user-a", "user-b", "user id"),
    ],
)
def test_load_token_data_rejects_invalid_or_conflicting_metadata_claims(
    auth_json_factory,
    claim,
    id_value,
    access_value,
    message,
):
    id_auth = {"chatgpt_account_id": "acc-123", claim: id_value}
    access_auth = {"chatgpt_account_id": "acc-123", claim: access_value}
    path = auth_json_factory(
        id_payload={"https://api.openai.com/auth": id_auth},
        access_payload={
            "exp": 9999999999,
            "https://api.openai.com/auth": access_auth,
        },
    )

    with pytest.raises(ChatGPTOAuthMissingError, match=message):
        load_token_data(path)


# ---------------------------------------------------------------------------
# is_auth_locally_available
# ---------------------------------------------------------------------------


def test_is_auth_locally_available_true(auth_json_factory):
    path = auth_json_factory()
    assert is_auth_locally_available(str(path)) is True


def test_is_auth_locally_available_false_missing(tmp_path):
    assert is_auth_locally_available(str(tmp_path / "gone.json")) is False


def test_is_auth_locally_available_false_invalid_json(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text("!!!")
    assert is_auth_locally_available(str(p)) is False


def test_proactive_refresh_coalesces_and_preserves_partial_response(monkeypatch, tmp_path, make_jwt):
    import datetime as _dt

    account_claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "acc-refresh"}}
    old_access = make_jwt({"exp": int(time.time()) + 240})
    id_token = make_jwt(account_claims)
    refreshed_access = make_jwt(
        {
            "exp": int(time.time()) + 3600,
            **account_claims,
        }
    )
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": old_access,
                    "refresh_token": "refresh-preserved",
                    "id_token": id_token,
                }
            }
        )
    )
    calls = 0
    calls_lock = threading.Lock()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"access_token": refreshed_access}).encode()

    def fake_urlopen(_request, timeout):
        nonlocal calls
        assert timeout == 30
        with calls_lock:
            calls += 1
        threading.Event().wait(0.05)
        return Response()

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", fake_urlopen)
    barrier = threading.Barrier(3)
    results = []

    def worker():
        barrier.wait()
        results.append(token_for_request(auth_path))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert calls == 1
    assert len(results) == 2
    assert all(result.access_token == refreshed_access for result in results)
    assert all(result.refresh_token == "refresh-preserved" for result in results)
    assert all(result.id_token == id_token for result in results)
    assert results[0].access_expires_at > _dt.datetime.now(_dt.UTC) + _dt.timedelta(minutes=5)


@pytest.mark.parametrize("name", ["id_token", "refresh_token"])
def test_refresh_rejects_explicit_null_optional_token(monkeypatch, tmp_path, make_jwt, name):

    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "acc-refresh"}}
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": make_jwt({"exp": 1, **claims}),
                    "refresh_token": "refresh-old",
                    "id_token": make_jwt(claims),
                }
            }
        )
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"access_token": make_jwt({"exp": 9999999999, **claims}), name: None}).encode()

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", lambda *_args, **_kwargs: Response())

    with pytest.raises(ChatGPTOAuthProtocolError, match=f"invalid {name}"):
        refresh_token(auth_path)


def test_refresh_rejects_header_unsafe_access_token_before_persist(
    monkeypatch,
    auth_json_factory,
):

    auth_path = auth_json_factory()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"access_token": "bad token"}).encode()

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", lambda *_args, **_kwargs: Response())
    before = auth_path.read_text(encoding="utf-8")

    with pytest.raises(ChatGPTOAuthProtocolError, match="invalid access_token") as caught:
        refresh_token(auth_path)
    assert "bad token" not in str(caught.value)
    assert auth_path.read_text(encoding="utf-8") == before


def test_unauthorized_reload_reuses_matching_account_file_update(monkeypatch, auth_json_factory, make_jwt):

    auth_path = auth_json_factory(account_id="acc-shared")
    observed = load_token_data(auth_path)
    refreshed_access = make_jwt(
        {
            "exp": 9999999999,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acc-shared"},
        }
    )
    refreshed_id = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acc-shared"}})
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": refreshed_access,
                    "refresh_token": "refresh-from-file",
                    "id_token": refreshed_id,
                }
            }
        )
    )

    def unexpected_refresh(*_args, **_kwargs):
        raise AssertionError("matching file update must avoid another token rotation")

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", unexpected_refresh)
    result = refresh_after_unauthorized(observed)

    assert result.access_token == refreshed_access
    assert result.refresh_token == "refresh-from-file"


def test_refresh_rejects_root_credential_layout_without_request(monkeypatch, tmp_path, make_jwt):

    account_claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "acc-root"}}
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "access_token": make_jwt({"exp": 1, **account_claims}),
                "refresh_token": "refresh-root",
                "id_token": make_jwt(account_claims),
                "unrelated": "preserved",
            }
        )
    )
    monkeypatch.setattr(
        "codex_as_api.auth._urlopen_no_redirect",
        lambda *_args, **_kwargs: pytest.fail("invalid root credentials must not reach refresh transport"),
    )

    with pytest.raises(ChatGPTOAuthMissingError, match="file-backed ChatGPT OAuth tokens"):
        refresh_token(auth_path)


def test_refresh_rejects_account_switch_before_request(monkeypatch, tmp_path, make_jwt):

    def credentials(account: str, access_suffix: str) -> dict:
        claims = {"https://api.openai.com/auth": {"chatgpt_account_id": account}}
        return {
            "access_token": make_jwt({"exp": 9999999999, "suffix": access_suffix, **claims}),
            "refresh_token": f"refresh-{account}",
            "id_token": make_jwt(claims),
        }

    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"tokens": credentials("acc-old", "old")}))
    observed = load_token_data(auth_path)
    auth_path.write_text(json.dumps({"tokens": credentials("acc-new", "new")}))
    monkeypatch.setattr(
        "codex_as_api.auth._urlopen_no_redirect",
        lambda *_args, **_kwargs: pytest.fail("account switch must fail before refresh transport"),
    )

    with pytest.raises(ChatGPTOAuthRefreshError, match="account changed"):
        refresh_after_unauthorized(observed)


def test_refresh_uses_latest_same_account_refresh_token(monkeypatch, tmp_path, make_jwt):

    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "acc-shared"}}
    access = make_jwt({"exp": 9999999999, **claims})
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access,
                    "refresh_token": "refresh-old",
                    "id_token": make_jwt({**claims, "version": "old"}),
                }
            }
        )
    )
    observed = load_token_data(auth_path)
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access,
                    "refresh_token": "refresh-latest",
                    "id_token": make_jwt({**claims, "version": "latest"}),
                }
            }
        )
    )
    refreshed_access = make_jwt({"exp": 9999999999, "version": "refreshed", **claims})

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"access_token": refreshed_access}).encode()

    def fake_urlopen(request, **_kwargs):
        assert json.loads(request.data)["refresh_token"] == "refresh-latest"
        return Response()

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", fake_urlopen)

    assert refresh_after_unauthorized(observed).access_token == refreshed_access


def test_refresh_compare_and_set_reuses_concurrent_access_update(monkeypatch, tmp_path, make_jwt):

    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "acc-shared"}}
    auth_path = tmp_path / "auth.json"
    current = {
        "access_token": make_jwt({"exp": 1, "version": "old", **claims}),
        "refresh_token": "refresh-old",
        "id_token": make_jwt(claims),
    }
    auth_path.write_text(json.dumps({"tokens": current}))
    external_access = make_jwt({"exp": 9999999999, "version": "external", **claims})
    response_access = make_jwt({"exp": 9999999999, "version": "response", **claims})

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            auth_path.write_text(json.dumps({"tokens": {**current, "access_token": external_access}}))
            return json.dumps({"access_token": response_access}).encode()

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", lambda *_args, **_kwargs: Response())

    result = refresh_token(auth_path)

    assert result.access_token == external_access
    assert load_token_data(auth_path).access_token == external_access


def test_refresh_compare_and_set_rejects_other_credential_change(monkeypatch, tmp_path, make_jwt):

    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "acc-shared"}}
    auth_path = tmp_path / "auth.json"
    current = {
        "access_token": make_jwt({"exp": 1, **claims}),
        "refresh_token": "refresh-old",
        "id_token": make_jwt(claims),
    }
    auth_path.write_text(json.dumps({"tokens": current}))

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            auth_path.write_text(json.dumps({"tokens": {**current, "refresh_token": "refresh-raced"}}))
            return json.dumps({"access_token": make_jwt({"exp": 9999999999, **claims})}).encode()

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", lambda *_args, **_kwargs: Response())

    with pytest.raises(ChatGPTOAuthRefreshError, match="changed while token refresh was in flight"):
        refresh_token(auth_path)


def test_refresh_revalidates_exact_document_before_mutation(monkeypatch, tmp_path, make_jwt):
    def credentials(account: str, version: str, *, exp: int) -> dict[str, str]:
        claims = {"https://api.openai.com/auth": {"chatgpt_account_id": account}}
        return {
            "access_token": make_jwt({"exp": exp, "version": version, **claims}),
            "refresh_token": f"refresh-{account}-{version}",
            "id_token": make_jwt({"version": version, **claims}),
        }

    auth_path = tmp_path / "auth.json"
    account_a = {"tokens": credentials("account-a", "old", exp=1)}
    account_b = {"tokens": credentials("account-b", "replacement", exp=9_999_999_999)}
    auth_path.write_text(json.dumps(account_a), encoding="utf-8")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            auth_path.write_text(json.dumps(account_b), encoding="utf-8")
            claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "account-a"}}
            return json.dumps(
                {"access_token": make_jwt({"exp": 9_999_999_999, "version": "response", **claims})}
            ).encode()

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", lambda *_args, **_kwargs: Response())

    with pytest.raises(ChatGPTOAuthRefreshError, match="account changed"):
        refresh_token(auth_path)
    assert json.loads(auth_path.read_text(encoding="utf-8")) == account_b


def test_load_and_refresh_reject_account_claim_mismatches(monkeypatch, tmp_path, make_jwt):

    old_claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "acc-old"}}
    new_claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "acc-new"}}
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": make_jwt({"exp": 1, **new_claims}),
                    "refresh_token": "refresh-old",
                    "id_token": make_jwt(old_claims),
                }
            }
        )
    )
    with pytest.raises(ChatGPTOAuthError, match="account ids do not match"):
        load_token_data(auth_path)

    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": make_jwt({"exp": 1, **old_claims}),
                    "refresh_token": "refresh-old",
                    "id_token": make_jwt(old_claims),
                }
            }
        )
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"access_token": make_jwt({"exp": 9999999999, **new_claims})}).encode()

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", lambda *_args, **_kwargs: Response())
    with pytest.raises(ChatGPTOAuthRefreshError, match="does not match current account"):
        refresh_token(auth_path)


def test_refresh_error_redacts_account_id(monkeypatch, auth_json_factory):
    import urllib.error
    import urllib.request

    auth_path = auth_json_factory(account_id="acc-secret")

    def fail(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://example.invalid/token",
            400,
            "bad request",
            {},
            io.BytesIO(b"account=acc-secret"),
        )

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", fail)

    with pytest.raises(ChatGPTOAuthRefreshError) as caught:
        refresh_token(auth_path)
    assert "acc-secret" not in str(caught.value)
    assert "account=***" in str(caught.value)


@pytest.mark.parametrize("status", [500, 429])
def test_refresh_preserves_non_auth_upstream_status(monkeypatch, auth_json_factory, status):
    import urllib.error

    auth_path = auth_json_factory()

    def fail(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://example.invalid/token",
            status,
            "upstream failure",
            {},
            io.BytesIO(b'{"error":"temporary"}'),
        )

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", fail)

    with pytest.raises(ChatGPTOAuthUpstreamError) as caught:
        refresh_token(auth_path)
    assert caught.value.status == status


def test_refresh_transport_failure_is_upstream_error(monkeypatch, auth_json_factory):
    import urllib.error

    auth_path = auth_json_factory()
    monkeypatch.setattr(
        "codex_as_api.auth._urlopen_no_redirect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("connection reset")),
    )

    with pytest.raises(ChatGPTOAuthUpstreamError) as caught:
        refresh_token(auth_path)
    assert caught.value.status == 502


def test_refresh_malformed_success_is_protocol_error(monkeypatch, auth_json_factory):
    auth_path = auth_json_factory()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"not-json"

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", lambda *_args, **_kwargs: Response())

    with pytest.raises(ChatGPTOAuthProtocolError, match="invalid JSON"):
        refresh_token(auth_path)


def test_refresh_local_persistence_failure_is_internal(monkeypatch, auth_json_factory, make_jwt):
    auth_path = auth_json_factory()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"access_token": make_jwt({"exp": 9999999999})}).encode()

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", lambda *_args, **_kwargs: Response())
    persistence_failure = OSError("disk full")
    monkeypatch.setattr(
        "codex_as_api.auth._write_auth_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(persistence_failure),
    )

    with pytest.raises(OSError) as caught:
        refresh_token(auth_path)
    assert caught.value is persistence_failure


def test_refresh_http_error_body_read_failure_preserves_upstream_status(
    monkeypatch,
    auth_json_factory,
):
    import urllib.error
    import urllib.request

    auth_path = auth_json_factory()

    class BrokenBody:
        def read(self):
            raise OSError("secret local detail")

        def close(self):
            pass

    monkeypatch.setattr(
        "codex_as_api.auth._urlopen_no_redirect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.HTTPError(
                "https://example.invalid/token",
                429,
                "rate limited",
                {},
                BrokenBody(),
            )
        ),
    )

    with pytest.raises(ChatGPTOAuthUpstreamError) as caught:
        refresh_token(auth_path)
    assert caught.value.status == 429
    assert "could not read upstream refresh error body" in str(caught.value)
    assert "secret local detail" not in str(caught.value)


def test_refresh_response_body_has_a_total_deadline(monkeypatch, auth_json_factory):
    auth_path = auth_json_factory()
    clock = iter([0.0, 0.1, 10.0, 30.1])
    monkeypatch.setattr("codex_as_api.auth.time.monotonic", lambda: next(clock))

    class Socket:
        def settimeout(self, _timeout):
            pass

    class Raw:
        _sock = Socket()

    class Buffered:
        raw = Raw()

    class Response:
        fp = Buffered()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read1(self, _size):
            return b"{" if next(clock) < 30 else b""

        def isclosed(self):
            return False

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", lambda *_args, **_kwargs: Response())

    with pytest.raises(ChatGPTOAuthUpstreamError) as caught:
        refresh_token(auth_path)
    assert caught.value.status == 502
    assert "total deadline" in str(caught.value)


def test_refresh_does_not_hide_programming_errors(monkeypatch, auth_json_factory):

    auth_path = auth_json_factory(account_id="acc-programming-error")
    programming_error = RuntimeError("simulated programming error")

    def fail(*_args, **_kwargs):
        raise programming_error

    monkeypatch.setattr("codex_as_api.auth._urlopen_no_redirect", fail)
    with pytest.raises(RuntimeError) as caught:
        refresh_token(auth_path)
    assert caught.value is programming_error


def test_refresh_endpoint_environment_is_strict_and_path_and_port_are_allowed(monkeypatch):
    monkeypatch.delenv("CODEX_REFRESH_TOKEN_URL_OVERRIDE", raising=False)
    assert _refresh_url_from_environment() == "https://auth.openai.com/oauth/token"

    endpoint = "http://127.0.0.1:18081/oauth/token"
    monkeypatch.setenv("CODEX_REFRESH_TOKEN_URL_OVERRIDE", endpoint)
    validate_auth_environment()
    assert _refresh_url_from_environment() == endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        " ",
        " https://auth.openai.com/oauth/token",
        "ftp://auth.openai.com/oauth/token",
        "https://user:secret@auth.openai.com/oauth/token",
        "https://auth.openai.com/oauth/token?tenant=one",
        "https://auth.openai.com/oauth/token#fragment",
        "https:///oauth/token",
        "https://auth.openai.com:invalid/oauth/token",
        "http://auth.openai.com\n.evil/oauth/token",
        "https://auth.openai.com/\u0080oauth/token",
        "https://auth.openai.com/\u009foauth/token",
        "https://auth.openai.com/a path",
        "https://auth.openai.com/bad%escape",
    ],
)
def test_refresh_endpoint_environment_rejects_unsafe_values(monkeypatch, endpoint):
    monkeypatch.setenv("CODEX_REFRESH_TOKEN_URL_OVERRIDE", endpoint)
    with pytest.raises(ChatGPTOAuthRefreshError, match="CODEX_REFRESH_TOKEN_URL_OVERRIDE"):
        validate_auth_environment()


def test_refresh_endpoint_allows_percent_encoded_path(monkeypatch):
    endpoint = "https://auth.openai.com/oauth%20token"
    monkeypatch.setenv("CODEX_REFRESH_TOKEN_URL_OVERRIDE", endpoint)
    assert _refresh_url_from_environment() == endpoint


def test_refresh_request_does_not_follow_redirect(monkeypatch, auth_json_factory):
    redirected_requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/oauth/token":
                self.send_response(307)
                self.send_header("Location", "/redirect-target")
                self.end_headers()
                return
            redirected_requests.append(self.path)
            self.send_response(200)
            self.end_headers()

        def log_message(self, _format, *args):  # noqa: ANN001
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv(
            "CODEX_REFRESH_TOKEN_URL_OVERRIDE",
            f"http://127.0.0.1:{server.server_port}/oauth/token",
        )
        observed = load_token_data(auth_json_factory(account_id="acc-redirect"))
        with pytest.raises(ChatGPTOAuthUpstreamError, match="HTTP 307") as caught:
            refresh_after_unauthorized(observed)
        assert caught.value.status == 307
        assert redirected_requests == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
