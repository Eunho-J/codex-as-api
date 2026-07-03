from __future__ import annotations

import json
import os
import pathlib
import uuid
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Literal

RESPONSES_LITE_ENV = "CODEX_AS_API_RESPONSES_LITE"
CODEX_METADATA_ENV = "CODEX_AS_API_CODEX_METADATA"
LITE_HEADER_NAME = "x-openai-internal-codex-responses-lite"
LITE_HEADER_VALUE = "true"
TURN_METADATA_KEY = "x-codex-turn-metadata"
INSTALLATION_ID_KEY = "x-codex-installation-id"
WINDOW_ID_KEY = "x-codex-window-id"
SESSION_ID_KEY = "session_id"
THREAD_ID_KEY = "thread_id"
TURN_ID_KEY = "turn_id"

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CAPABILITIES_PATH = _PROJECT_ROOT / "config" / "model-capabilities.json"
_PACKAGE_CAPABILITIES_PATH = pathlib.Path(__file__).resolve().with_name("model-capabilities.json")
_INSTALLATION_NAMESPACE = uuid.UUID("d2c81270-8f15-5e8d-a5c4-4cdbf2c21fd0")
_SESSION_ID = str(uuid.uuid4())
_THREAD_ID = str(uuid.uuid4())
_WINDOW_ID = str(uuid.uuid4())

ResponsesLiteMode = Literal["off", "on", "auto"]


@dataclass(frozen=True, slots=True)
class ModelCapability:
    use_responses_lite: bool
    supports_parallel_tool_calls: bool
    support_verbosity: bool
    default_verbosity: str | None
    service_tiers: tuple[str, ...]
    default_service_tier: str | None
    source: str


UNKNOWN_CAPABILITY = ModelCapability(
    use_responses_lite=False,
    supports_parallel_tool_calls=False,
    support_verbosity=False,
    default_verbosity=None,
    service_tiers=(),
    default_service_tier=None,
    source="unknown",
)


def load_model_capabilities() -> dict[str, ModelCapability]:
    path = _CAPABILITIES_PATH if _CAPABILITIES_PATH.exists() else _PACKAGE_CAPABILITIES_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models")
    if not isinstance(models, dict):
        raise RuntimeError("model capabilities JSON must contain a models object")
    return {str(name): _capability_from_mapping(value) for name, value in models.items()}


def capability_for_model(model: str | None) -> ModelCapability:
    if not model:
        return UNKNOWN_CAPABILITY
    return load_model_capabilities().get(model, UNKNOWN_CAPABILITY)


def resolve_responses_lite_mode(value: bool | str | None = None) -> ResponsesLiteMode:
    raw: object = value if value is not None else os.getenv(RESPONSES_LITE_ENV, "auto")
    if isinstance(raw, bool):
        return "on" if raw else "off"
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return "on"
        if normalized in {"false", "0", "no"}:
            return "off"
        if normalized in {"on", "off", "auto"}:
            return normalized  # type: ignore[return-value]
    raise ValueError("responses_lite must be one of: off, on, auto")


def use_responses_lite(model: str, value: bool | str | None = None) -> bool:
    mode = resolve_responses_lite_mode(value)
    if mode == "on":
        return True
    if mode == "off":
        return False
    return capability_for_model(model).use_responses_lite


def resolve_codex_metadata_enabled(value: bool | None = None) -> bool:
    if value is not None:
        return value
    raw = os.getenv(CODEX_METADATA_ENV, "off").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError("codex_metadata must be on or off")


def apply_model_capability_fields(
    payload: MutableMapping[str, Any],
    *,
    model: str,
    text: Mapping[str, Any] | None,
    service_tier: str | None,
) -> None:
    capability = capability_for_model(model)
    if capability.support_verbosity:
        merged_text: dict[str, Any] = dict(text or {})
        if "verbosity" not in merged_text and capability.default_verbosity is not None:
            merged_text["verbosity"] = capability.default_verbosity
        if merged_text:
            payload["text"] = merged_text
    elif text is not None:
        payload["text"] = dict(text)

    if service_tier is not None and service_tier != "default" and service_tier in capability.service_tiers:
        payload["service_tier"] = service_tier


def should_enable_parallel_tool_calls(
    *,
    model: str,
    requested: bool | None,
    responses_lite: bool,
) -> bool:
    if responses_lite or requested is not True:
        return False
    return capability_for_model(model).supports_parallel_tool_calls


def build_codex_client_metadata(
    *,
    auth_json_path: str | os.PathLike[str] | None,
    existing: Mapping[str, str] | None,
) -> dict[str, str]:
    metadata = dict(existing or {})
    auth_path = pathlib.Path(auth_json_path or "~/.codex/auth.json").expanduser()
    absolute_path = os.path.abspath(os.fspath(auth_path))
    installation_id = str(uuid.uuid5(_INSTALLATION_NAMESPACE, f"codex-as-api:{absolute_path}"))
    turn_id = str(uuid.uuid4())
    turn_metadata = {
        "installation_id": installation_id,
        "session_id": _SESSION_ID,
        "thread_id": _THREAD_ID,
        "turn_id": turn_id,
        "window_id": _WINDOW_ID,
        "source": "codex-as-api",
    }
    metadata.update(
        {
            INSTALLATION_ID_KEY: installation_id,
            SESSION_ID_KEY: _SESSION_ID,
            THREAD_ID_KEY: _THREAD_ID,
            TURN_ID_KEY: turn_id,
            WINDOW_ID_KEY: _WINDOW_ID,
            TURN_METADATA_KEY: json.dumps(turn_metadata, separators=(",", ":"), sort_keys=True),
        }
    )
    return metadata


def strip_image_detail_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_image_detail_fields(item) for item in value]
    if isinstance(value, dict):
        return {
            key: strip_image_detail_fields(child)
            for key, child in value.items()
            if not (key == "detail" and value.get("type") == "input_image")
        }
    return value


def _capability_from_mapping(value: object) -> ModelCapability:
    if not isinstance(value, dict):
        raise RuntimeError("model capability entry must be an object")
    use_responses_lite = value.get("use_responses_lite")
    supports_parallel_tool_calls = value.get("supports_parallel_tool_calls")
    support_verbosity = value.get("support_verbosity")
    if not isinstance(use_responses_lite, bool):
        raise RuntimeError("model capability use_responses_lite must be a boolean")
    if not isinstance(supports_parallel_tool_calls, bool):
        raise RuntimeError("model capability supports_parallel_tool_calls must be a boolean")
    if not isinstance(support_verbosity, bool):
        raise RuntimeError("model capability support_verbosity must be a boolean")
    service_tiers = value.get("service_tiers")
    tiers = tuple(str(item) for item in service_tiers) if isinstance(service_tiers, list) else ()
    default_verbosity = value.get("default_verbosity")
    default_service_tier = value.get("default_service_tier")
    source = value.get("source")
    return ModelCapability(
        use_responses_lite=use_responses_lite,
        supports_parallel_tool_calls=supports_parallel_tool_calls,
        support_verbosity=support_verbosity,
        default_verbosity=default_verbosity if isinstance(default_verbosity, str) else None,
        service_tiers=tiers,
        default_service_tier=default_service_tier if isinstance(default_service_tier, str) else None,
        source=source if isinstance(source, str) else "unknown",
    )
