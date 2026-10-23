"""The proxy speed extension must not alias Ultra Fast to ordinary priority."""

import pytest

from codex_as_api.auth import ChatGPTOAuthInvalidRequestError
from codex_as_api.server import _anthropic_service_tier

# Reuse the real provider + loopback HTTP backend, rather than mocking the
# provider methods (which would bypass live-catalog tier validation).
from tests.test_server import client, recording_backend  # noqa: F401

# Imported pytest fixtures are intentionally injected by argument name.
# ruff: noqa: F811


@pytest.fixture(autouse=True)
def _isolate_ultrafast_http_environment(monkeypatch):
    monkeypatch.setenv("CODEX_AS_API_RESPONSES_LITE", "auto")


@pytest.fixture()
def ultrafast_catalog(model_catalog_document):
    model = next(
        model for model in model_catalog_document["models"]
        if model["slug"] == "gpt-5.6-sol"
    )
    model["service_tiers"].append({
        "id": "ultrafast",
        "name": "Ultra Fast",
        "description": "Test-only advertised Ultra Fast tier",
    })
    return model_catalog_document


@pytest.mark.parametrize("fields", [
    {"speed": "ultrafast"},
    {"speed": "ultrafast", "service_tier": "ultrafast"},
    {"service_tier": "ultrafast"},
])
def test_ultrafast_speed_resolves_without_downgrade(fields):
    assert _anthropic_service_tier(fields) == "ultrafast"


@pytest.mark.parametrize("fields", [
    {"speed": "ultrafast", "service_tier": "priority"},
    {"speed": "fast", "service_tier": "ultrafast"},
    {"speed": "standard", "service_tier": "ultrafast"},
    {"speed": {}},
    {"speed": []},
])
def test_ultrafast_conflicts_and_malformed_speed_are_invalid_requests(fields):
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        _anthropic_service_tier(fields)


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize("fields", [
    {"speed": "ultrafast"},
    {"service_tier": "ultrafast"},
    {"speed": "ultrafast", "service_tier": "ultrafast"},
], ids=["speed-alias", "direct-tier", "matching-controls"])
def test_messages_ultrafast_reaches_recording_upstream(
    ultrafast_catalog, client, recording_backend, stream, fields,
):
    response = client.post("/v1/messages", json={
        "model": "gpt-5.6-sol",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
        **fields,
    })

    assert response.status_code == 200, response.text
    if stream:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "event: message_stop" in response.text
        assert "backend-ok" in response.text
        assert "event: error" not in response.text
    else:
        assert response.json()["content"][0]["type"] == "text"
        assert response.json()["content"][0]["text"] == "backend-ok"
    recorded = recording_backend.requests.get(timeout=1)
    assert recorded["path"] == "/responses"
    assert recorded["body"]["service_tier"] == "ultrafast"
    assert "speed" not in recorded["body"]
    assert recording_backend.requests.empty()


@pytest.mark.parametrize("route, fields", [
    ("/v1/messages/compact", {"speed": "ultrafast"}),
    ("/v1/messages/compact", {"service_tier": "ultrafast"}),
    ("/v1/messages/compact", {"speed": "ultrafast", "service_tier": "ultrafast"}),
    ("/v1/compact", {"service_tier": "ultrafast"}),
], ids=["speed-alias", "direct-tier", "matching-controls", "openai-compact"])
def test_compact_ultrafast_reaches_recording_upstream(
    ultrafast_catalog, client, recording_backend, route, fields,
):
    body = {
        "model": "gpt-5.6-sol",
        "messages": [{"role": "user", "content": "hello"}],
        **fields,
    }
    if route == "/v1/messages/compact":
        body["max_tokens"] = 64
    response = client.post(route, json=body)

    assert response.status_code == 200, response.text
    recorded = recording_backend.requests.get(timeout=1)
    assert recorded["path"] == "/responses/compact"
    assert recorded["body"]["service_tier"] == "ultrafast"
    assert "speed" not in recorded["body"]
    assert recording_backend.requests.empty()


@pytest.mark.parametrize("route, stream", [
    ("/v1/messages", False),
    ("/v1/messages", True),
    ("/v1/messages/compact", False),
])
@pytest.mark.parametrize("fields", [
    {"speed": "ultrafast"},
    {"service_tier": "ultrafast"},
], ids=["speed-alias", "direct-tier"])
def test_unadvertised_ultrafast_rejected_before_recording_upstream(
    client, recording_backend, route, stream, fields,
):
    # Default catalog advertises priority only: do not globally grant Ultra Fast
    # in the common fixture, or this rejection guard would become untestable.
    body = {
        "model": "gpt-5.6-sol",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
        **fields,
    }
    if route == "/v1/messages":
        body["stream"] = stream
    response = client.post(route, json=body)

    assert response.status_code == 400, response.text
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "service_tier" in response.json()["error"]["message"]
    assert recording_backend.requests.empty()
    # Catalog discovery is permitted; no generation/compact POST may occur.
    assert not recording_backend.catalog_requests.empty()


def test_openai_compact_unadvertised_ultrafast_rejected_before_upstream(
    client, recording_backend,
):
    response = client.post("/v1/compact", json={
        "model": "gpt-5.6-sol",
        "messages": [{"role": "user", "content": "hello"}],
        "service_tier": "ultrafast",
    })

    assert response.status_code == 400, response.text
    assert "service_tier" in response.json()["error"]["message"]
    assert recording_backend.requests.empty()
    assert not recording_backend.catalog_requests.empty()
