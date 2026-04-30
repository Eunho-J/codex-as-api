from __future__ import annotations

import json
import os
import pathlib

import pytest

from codex_as_api.auth import (
    ChatGPTOAuthError,
    ChatGPTOAuthMissingError,
    _auth_claims,
    _expiration,
    _jwt_claims,
    is_auth_locally_available,
    load_token_data,
    redact_text,
    register_token_secrets,
    resolve_auth_path,
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


# ---------------------------------------------------------------------------
# _jwt_claims
# ---------------------------------------------------------------------------


def test_jwt_claims_valid(make_jwt):
    payload = {"sub": "user1", "exp": 12345}
    token = make_jwt(payload)
    claims = _jwt_claims(token)
    assert claims["sub"] == "user1"
    assert claims["exp"] == 12345


def test_jwt_claims_missing_payload_returns_empty():
    claims = _jwt_claims("onlyonepart")
    assert claims == {}


def test_jwt_claims_empty_second_part_returns_empty():
    claims = _jwt_claims("header..sig")
    assert claims == {}


def test_jwt_claims_invalid_base64_raises(make_jwt):
    with pytest.raises(ChatGPTOAuthError):
        _jwt_claims("header.!!!invalid!!!.sig")


def test_jwt_claims_non_dict_raises(make_jwt):
    import base64
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


def test_expiration_non_int_exp_returns_none(make_jwt):
    token = make_jwt({"exp": "not-an-int"})
    assert _expiration(token) is None


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


def test_auth_claims_non_dict_value_returns_empty(make_jwt):
    token = make_jwt({"https://api.openai.com/auth": "bad"})
    assert _auth_claims(token) == {}


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


def test_load_token_data_root_not_dict_raises(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text('["list", "not", "dict"]')
    with pytest.raises(ChatGPTOAuthError):
        load_token_data(str(p))


def test_load_token_data_missing_tokens_raises(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({}))
    with pytest.raises(ChatGPTOAuthError):
        load_token_data(str(p))


def test_load_token_data_missing_access_token_raises(tmp_path, make_jwt):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({"tokens": {"refresh_token": "r", "id_token": make_jwt({})}}))
    with pytest.raises(ChatGPTOAuthError):
        load_token_data(str(p))


def test_load_token_data_invalid_auth_mode_raises(tmp_path, auth_json_factory):
    p = auth_json_factory(extra={"auth_mode": "unknown_mode"})
    with pytest.raises(ChatGPTOAuthError):
        load_token_data(str(p))


def test_load_token_data_expiration_extracted(auth_json_factory):
    import datetime as _dt
    future_exp = int(_dt.datetime(2099, 1, 1, tzinfo=_dt.UTC).timestamp())
    path = auth_json_factory(access_payload={"exp": future_exp})
    data = load_token_data(str(path))
    assert data.access_expires_at is not None
    assert data.access_expires_at.year == 2099


def test_load_token_data_fedramp_flag(make_jwt, tmp_path):
    import base64
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
    p.write_text(json.dumps({"tokens": {
        "access_token": access_token,
        "refresh_token": "r",
        "id_token": id_token,
    }}))
    data = load_token_data(str(p))
    assert data.fedramp is True


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


# ---------------------------------------------------------------------------
# register_token_secrets
# ---------------------------------------------------------------------------


def test_register_token_secrets_is_noop():
    register_token_secrets("tok1", "tok2", None)
