import json
import threading

import pytest

from codex_as_api import auth


def test_same_account_relogin_fences_inflight_refresh(tmp_path, make_jwt, monkeypatch):
    path = tmp_path / "auth.json"
    jwt = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "account"}})
    original = {"access_token": jwt, "id_token": jwt, "refresh_token": "old"}
    auth.commit_login(path, original)
    current = auth.load_token_data(path)
    entered = threading.Event()
    release = threading.Event()

    class Response:
        def __enter__(self):
            entered.set()
            assert release.wait(5)
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"access_token": jwt, "refresh_token": "stale"}).encode()

    monkeypatch.setattr(auth, "_urlopen_no_redirect", lambda *args, **kwargs: Response())
    errors = []

    def refresh():
        try:
            auth.refresh_after_unauthorized(current)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=refresh)
    worker.start()
    assert entered.wait(5)
    try:
        # Identical credential values still represent a distinct login generation.
        auth.commit_login(path, original)
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], auth.ChatGPTOAuthRefreshError)
    assert auth.load_token_data(path).refresh_token == "old"
    assert auth.load_token_data(path).login_generation != current.login_generation


def test_bad_login_does_not_replace_existing_credentials(tmp_path, make_jwt):
    path = tmp_path / "auth.json"
    jwt = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "account"}})
    auth.commit_login(path, {"access_token": jwt, "id_token": jwt, "refresh_token": "old"})
    before = path.read_bytes()
    with pytest.raises(auth.ChatGPTOAuthMissingError):
        auth.commit_login(path, {"access_token": "invalid", "id_token": jwt, "refresh_token": "new"})
    assert path.read_bytes() == before


def test_cancel_at_replace_preserves_previous_store(tmp_path, make_jwt):
    path = tmp_path / "auth.json"
    jwt = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "account"}})
    tokens = {"access_token": jwt, "id_token": jwt, "refresh_token": "old"}
    auth.commit_login(path, tokens)
    before = path.read_bytes()
    assert auth.commit_login(path, {**tokens, "refresh_token": "new"}, lambda: False) is False
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".auth.json.tmp-*")) == []


def test_replace_failure_preserves_previous_store(tmp_path, make_jwt, monkeypatch):
    path = tmp_path / "auth.json"
    jwt = make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "account"}})
    tokens = {"access_token": jwt, "id_token": jwt, "refresh_token": "old"}
    auth.commit_login(path, tokens)
    before = path.read_bytes()

    def fail_replace(*args):
        raise OSError("fake failure")

    monkeypatch.setattr(type(path), "replace", fail_replace)
    with pytest.raises(OSError):
        auth.commit_login(path, {**tokens, "refresh_token": "new"})
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".auth.json.tmp-*")) == []
