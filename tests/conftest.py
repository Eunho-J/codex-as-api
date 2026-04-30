from __future__ import annotations

import base64
import json
import pathlib
import pytest


def _make_jwt(payload: dict) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{header}.{body}.fakesig"


@pytest.fixture()
def make_jwt():
    return _make_jwt


@pytest.fixture()
def auth_json_factory(tmp_path):
    def _factory(
        access_payload: dict | None = None,
        id_payload: dict | None = None,
        refresh_token: str = "refresh-tok",
        account_id: str | None = None,
        extra: dict | None = None,
    ) -> pathlib.Path:
        ap = access_payload or {"exp": 9999999999}
        ip = id_payload or {
            "exp": 9999999999,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": account_id or "acc-123",
                "chatgpt_plan_type": "plus",
                "chatgpt_user_id": "user-abc",
            },
        }
        access_token = _make_jwt(ap)
        id_token = _make_jwt(ip)
        data: dict = {
            "tokens": {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "id_token": id_token,
            }
        }
        if extra:
            data.update(extra)
        p = tmp_path / "auth.json"
        p.write_text(json.dumps(data))
        return p

    return _factory
