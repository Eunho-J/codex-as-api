from __future__ import annotations

import base64
import json
import pathlib
import time
from copy import deepcopy

import pytest

from codex_as_api.model_capabilities import parse_model_catalog


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


def _catalog_model(
    slug: str,
    priority: int,
    *,
    default_effort: str = "medium",
    lite: bool = False,
    original_detail: bool = True,
    context_window: int = 272_000,
    max_context_window: int = 272_000,
    service_tier: bool = False,
) -> dict:
    efforts = ("low", "medium", "high", "xhigh", "max")
    return {
        "slug": slug,
        "display_name": slug,
        "description": f"Test catalog entry for {slug}",
        "priority": priority,
        "visibility": "list",
        "supported_in_api": slug != "gpt-5.3-codex-spark",
        "default_reasoning_level": default_effort,
        "supported_reasoning_levels": [{"effort": effort, "description": effort} for effort in efforts],
        "context_window": context_window,
        "max_context_window": max_context_window,
        "auto_compact_token_limit": None,
        "input_modalities": ["text", "image"],
        "service_tiers": ([{"id": "priority", "name": "Priority", "description": "Fast"}] if service_tier else []),
        "default_service_tier": None,
        "use_responses_lite": lite,
        "supports_image_detail_original": original_detail,
        "support_verbosity": True,
        "default_verbosity": "medium",
        "multi_agent_reasoning_effort": "max",
    }


@pytest.fixture()
def model_catalog_document() -> dict:
    models = [
        _catalog_model(
            "gpt-5.6-sol",
            0,
            default_effort="low",
            lite=True,
            service_tier=True,
        ),
        _catalog_model("gpt-5.6-terra", 1, lite=True, service_tier=True),
        _catalog_model("gpt-5.6-luna", 2, lite=True, service_tier=True),
        _catalog_model("gpt-5.5", 3, service_tier=True),
        _catalog_model(
            "gpt-5.4",
            4,
            max_context_window=1_000_000,
            service_tier=True,
        ),
        _catalog_model("gpt-5.4-mini", 5),
        _catalog_model("gpt-5.2", 6, original_detail=False),
        _catalog_model("gpt-5.3-codex", 7, original_detail=False),
        _catalog_model(
            "gpt-5.3-codex-spark",
            8,
            original_detail=False,
        ),
    ]
    return {"models": deepcopy(models)}


@pytest.fixture()
def model_catalog_snapshot(model_catalog_document):
    now = time.time()
    return parse_model_catalog(
        model_catalog_document,
        key=("test-account", "https://example.test/codex", "0.153.3"),
        etag='"test-etag"',
        fetched_at=time.time(),
        expires_at=now + 3600,
    )
