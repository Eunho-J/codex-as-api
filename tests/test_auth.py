from __future__ import annotations

import io
import json
import pathlib
import threading
import time

import pytest

from codex_as_api.auth import (
    ChatGPTOAuthError,
    ChatGPTOAuthMissingError,
    ChatGPTOAuthRefreshError,
    _auth_claims,
    _expiration,
    _jwt_claims,
    is_auth_locally_available,
    load_token_data,
    redact_text,
    refresh_after_unauthorized,
    refresh_token,
    register_token_secrets,
    resolve_auth_path,
    token_for_request,
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
    with pytest.raises(ChatGPTOAuthError, match="file-backed ChatGPT OAuth tokens are required"):
        load_token_data(str(p))


def test_load_token_data_supports_latest_root_token_fields(tmp_path, make_jwt):
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
                "personal_access_token": "pat-present-but-not-primary",
                "agent_identity": {"id": "agent"},
            }
        )
    )

    data = load_token_data(str(p))

    assert data.access_token == access_token
    assert data.refresh_token == "refresh-root"
    assert data.account_id == "acc-root"


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


def test_proactive_refresh_coalesces_and_preserves_partial_response(monkeypatch, tmp_path, make_jwt):
    import datetime as _dt
    import urllib.request

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

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
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
    import urllib.request

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
            return json.dumps(
                {"access_token": make_jwt({"exp": 9999999999, **claims}), name: None}
            ).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: Response())

    with pytest.raises(ChatGPTOAuthRefreshError, match=f"invalid {name}"):
        refresh_token(auth_path)


def test_unauthorized_reload_reuses_matching_account_file_update(monkeypatch, auth_json_factory, make_jwt):
    import urllib.request

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

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_refresh)
    result = refresh_after_unauthorized(observed)

    assert result.access_token == refreshed_access
    assert result.refresh_token == "refresh-from-file"


def test_refresh_preserves_root_credential_layout(monkeypatch, tmp_path, make_jwt):
    import urllib.request

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
    refreshed_access = make_jwt({"exp": 9999999999, **account_claims})

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"access_token": refreshed_access}).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: Response())

    result = refresh_token(auth_path)
    stored = json.loads(auth_path.read_text())

    assert result.access_token == refreshed_access
    assert stored["access_token"] == refreshed_access
    assert stored["refresh_token"] == "refresh-root"
    assert stored["unrelated"] == "preserved"
    assert "tokens" not in stored


def test_refresh_rejects_account_switch_before_request(monkeypatch, tmp_path, make_jwt):
    import urllib.request

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
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("account switch must fail before refresh transport"),
    )

    with pytest.raises(ChatGPTOAuthRefreshError, match="account changed"):
        refresh_after_unauthorized(observed)


def test_refresh_uses_latest_same_account_refresh_token(monkeypatch, tmp_path, make_jwt):
    import urllib.request

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

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert refresh_after_unauthorized(observed).access_token == refreshed_access


def test_refresh_compare_and_set_reuses_concurrent_access_update(monkeypatch, tmp_path, make_jwt):
    import urllib.request

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

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: Response())

    result = refresh_token(auth_path)

    assert result.access_token == external_access
    assert load_token_data(auth_path).access_token == external_access


def test_refresh_compare_and_set_rejects_other_credential_change(monkeypatch, tmp_path, make_jwt):
    import urllib.request

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

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: Response())

    with pytest.raises(ChatGPTOAuthRefreshError, match="changed while token refresh was in flight"):
        refresh_token(auth_path)


def test_load_and_refresh_reject_account_claim_mismatches(monkeypatch, tmp_path, make_jwt):
    import urllib.request

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

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
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

    monkeypatch.setattr(urllib.request, "urlopen", fail)

    with pytest.raises(ChatGPTOAuthRefreshError) as caught:
        refresh_token(auth_path)
    assert "acc-secret" not in str(caught.value)
    assert "account=***" in str(caught.value)
