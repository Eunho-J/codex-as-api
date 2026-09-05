from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from .auth import (
    ChatGPTOAuthCatalogUnavailableError,
    ChatGPTOAuthInvalidRequestError,
    ChatGPTOAuthModelNotFoundError,
    resolve_auth_path,
)

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

DEFAULT_MODEL_CATALOG_TTL_SECONDS = 300.0
DEFAULT_MODEL_CATALOG_TIMEOUT_SECONDS = 5.0
DEFAULT_EFFECTIVE_CONTEXT_WINDOW_PERCENT = 95
_INSTALLATION_NAMESPACE = uuid.UUID("d2c81270-8f15-5e8d-a5c4-4cdbf2c21fd0")
_WINDOW_ID = str(uuid.uuid4())

ResponsesLiteMode = Literal["off", "on", "auto"]
CatalogKey = tuple[str, str, str]
_JS_SAFE_INTEGER = (1 << 53) - 1


@dataclass(frozen=True, slots=True)
class ModelServiceTier:
    id: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class ModelReasoningLevel:
    effort: str
    description: str


@dataclass(frozen=True, slots=True)
class ModelCapability:
    slug: str
    display_name: str
    description: str | None
    priority: int
    visibility: str
    supported_in_api: bool
    supported_reasoning_levels: tuple[ModelReasoningLevel, ...]
    default_reasoning_level: str | None
    multi_agent_reasoning_effort: str | None
    context_window: int | None
    max_context_window: int | None
    effective_context_window_percent: int
    auto_compact_token_limit: int | None
    input_modalities: tuple[str, ...]
    service_tiers: tuple[ModelServiceTier, ...]
    default_service_tier: str | None
    use_responses_lite: bool
    supports_image_detail_original: bool
    support_verbosity: bool
    default_verbosity: str | None
    comp_hash: str | None
    supports_reasoning_summary_parameter: bool = True
    default_reasoning_summary: str = "auto"

    @property
    def service_tier_ids(self) -> tuple[str, ...]:
        return tuple(tier.id for tier in self.service_tiers)

    @property
    def reasoning_effort_ids(self) -> tuple[str, ...]:
        return tuple(level.effort for level in self.supported_reasoning_levels)


@dataclass(frozen=True, slots=True)
class ModelCatalogSnapshot:
    key: CatalogKey
    models: tuple[ModelCapability, ...]
    models_by_slug: Mapping[str, ModelCapability]
    etag: str | None
    fetched_at: float
    expires_at: float

    @property
    def account_id(self) -> str:
        return self.key[0]

    @property
    def base_url(self) -> str:
        return self.key[1]

    @property
    def client_version(self) -> str:
        return self.key[2]

    def model(self, slug: str) -> ModelCapability:
        try:
            return self.models_by_slug[slug]
        except KeyError as exc:
            raise ChatGPTOAuthModelNotFoundError("requested model was not found in the authenticated catalog") from exc

    def default_model(self) -> ModelCapability:
        visible = [model for model in self.models if model.visibility == "list"]
        if not visible:
            raise ChatGPTOAuthCatalogUnavailableError(
                "upstream model catalog has no model with list visibility for default selection"
            )
        return min(enumerate(visible), key=lambda item: (item[1].priority, item[0]))[1]


@dataclass(frozen=True, slots=True)
class CatalogLoadResult:
    document: object
    etag: str | None


class ModelCatalogCache:
    """Fresh-only, process-local model catalog cache with per-key single flight."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._snapshots: dict[CatalogKey, ModelCatalogSnapshot] = {}
        self._deadlines: dict[CatalogKey, float] = {}
        self._refreshing: set[CatalogKey] = set()
        self._revisions: dict[CatalogKey, int] = {}
        self._failures: dict[CatalogKey, tuple[int, Exception]] = {}

    def get(
        self,
        key: CatalogKey,
        loader: Callable[[], CatalogLoadResult],
        *,
        ttl_seconds: float,
    ) -> ModelCatalogSnapshot:
        if (
            not isinstance(ttl_seconds, (int, float))
            or isinstance(ttl_seconds, bool)
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError("model catalog TTL must be a positive finite number")
        with self._condition:
            monotonic_now = time.monotonic()
            cached = self._snapshots.get(key)
            if cached is not None and monotonic_now < self._deadlines[key]:
                return cached
            observed_revision = self._revisions.get(key, 0)
            if key in self._refreshing:
                while key in self._refreshing:
                    self._condition.wait()
                monotonic_now = time.monotonic()
                cached = self._snapshots.get(key)
                if cached is not None and monotonic_now < self._deadlines[key]:
                    return cached
                failure = self._failures.get(key)
                if failure is not None and failure[0] > observed_revision:
                    raise failure[1]
            load_revision = self._revisions.get(key, 0)
            self._refreshing.add(key)

        try:
            loaded = loader()
            now = time.time()
            snapshot = parse_model_catalog(
                loaded.document,
                key=key,
                etag=loaded.etag,
                fetched_at=now,
                expires_at=now + ttl_seconds,
            )
        except Exception as exc:
            with self._condition:
                revision = self._revisions.get(key, 0) + 1
                self._revisions[key] = revision
                self._failures[key] = (revision, exc)
                self._refreshing.discard(key)
                self._condition.notify_all()
            raise

        with self._condition:
            if self._revisions.get(key, 0) != load_revision:
                error = ChatGPTOAuthCatalogUnavailableError("model catalog refresh was invalidated while in flight")
                revision = self._revisions.get(key, 0) + 1
                self._revisions[key] = revision
                self._failures[key] = (revision, error)
                self._refreshing.discard(key)
                self._condition.notify_all()
                raise error
            self._snapshots[key] = snapshot
            self._deadlines[key] = time.monotonic() + ttl_seconds
            self._failures.pop(key, None)
            self._revisions[key] = self._revisions.get(key, 0) + 1
            self._refreshing.discard(key)
            self._condition.notify_all()
            return snapshot

    def invalidate(self, key: CatalogKey) -> None:
        with self._condition:
            self._snapshots.pop(key, None)
            self._deadlines.pop(key, None)
            self._revisions[key] = self._revisions.get(key, 0) + 1

    def invalidate_on_etag_mismatch(self, key: CatalogKey, response_etag: str | None) -> None:
        if response_etag is None or not (normalized_etag := response_etag.strip()):
            return
        with self._condition:
            cached = self._snapshots.get(key)
            mismatch = cached is not None and cached.etag != normalized_etag
            if mismatch:
                self._snapshots.pop(key, None)
                self._deadlines.pop(key, None)
                self._revisions[key] = self._revisions.get(key, 0) + 1

    def clear(self) -> None:
        with self._condition:
            for key in self._refreshing:
                self._revisions[key] = self._revisions.get(key, 0) + 1
            self._snapshots.clear()
            self._deadlines.clear()
            self._failures.clear()


def parse_model_catalog(
    document: object,
    *,
    key: CatalogKey,
    etag: str | None,
    fetched_at: float,
    expires_at: float,
) -> ModelCatalogSnapshot:
    if not isinstance(document, dict):
        raise ChatGPTOAuthCatalogUnavailableError("model catalog response must be a JSON object")
    raw_models = document.get("models")
    if not isinstance(raw_models, list):
        raise ChatGPTOAuthCatalogUnavailableError("model catalog response must contain a models array")

    models: list[ModelCapability] = []
    by_slug: dict[str, ModelCapability] = {}
    for index, value in enumerate(raw_models):
        model = _model_from_mapping(value, index=index)
        if model.slug in by_slug:
            raise ChatGPTOAuthCatalogUnavailableError("model catalog contains a duplicate model slug")
        by_slug[model.slug] = model
        models.append(model)
    return ModelCatalogSnapshot(
        key=key,
        models=tuple(models),
        models_by_slug=MappingProxyType(by_slug),
        etag=etag.strip() if isinstance(etag, str) and etag.strip() else None,
        fetched_at=fetched_at,
        expires_at=expires_at,
    )


def resolve_model(
    snapshot: ModelCatalogSnapshot,
    requested: str | None,
    configured: str | None,
    *,
    anthropic_facade: bool = False,
) -> ModelCapability:
    def configured_capability(slug: str) -> ModelCapability:
        try:
            return snapshot.model(slug)
        except ChatGPTOAuthModelNotFoundError as exc:
            raise ChatGPTOAuthCatalogUnavailableError(
                "configured model is unavailable in the authenticated catalog"
            ) from exc

    candidate = _non_empty_optional_model(requested, "model")
    try:
        configured_model = _non_empty_optional_model(configured, "configured model")
    except ChatGPTOAuthInvalidRequestError as exc:
        raise ChatGPTOAuthCatalogUnavailableError("configured model is unusable") from exc
    if candidate is not None and anthropic_facade and candidate.startswith("claude-"):
        if configured_model is None:
            raise ChatGPTOAuthInvalidRequestError(
                "claude-* facade models require CODEX_AS_API_MODEL or config.toml model"
            )
        return configured_capability(configured_model)
    if candidate is not None:
        return snapshot.model(candidate)
    if configured_model is not None:
        return configured_capability(configured_model)
    capability = snapshot.default_model()
    if not capability.slug.strip() or capability.slug != capability.slug.strip():
        raise ChatGPTOAuthCatalogUnavailableError("default model publishes an unusable slug")
    return capability


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


def use_responses_lite(capability: ModelCapability, value: bool | str | None = None) -> bool:
    mode = resolve_responses_lite_mode(value)
    if mode == "on":
        return True
    if mode == "off":
        return False
    return capability.use_responses_lite


def apply_model_capability_fields(
    payload: MutableMapping[str, Any],
    *,
    capability: ModelCapability,
    text: Mapping[str, Any] | None,
    service_tier: str | None,
) -> None:
    if text is not None and text.get("verbosity") is not None:
        verbosity = text["verbosity"]
        if not isinstance(verbosity, str) or verbosity not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError("text.verbosity must be one of: low, medium, high")
    if capability.support_verbosity:
        merged_text: dict[str, Any] = dict(text or {})
        if merged_text.get("verbosity") is None:
            merged_text.pop("verbosity", None)
        if "verbosity" not in merged_text and capability.default_verbosity is not None:
            merged_text["verbosity"] = capability.default_verbosity
        if merged_text:
            payload["text"] = merged_text
    elif text is not None:
        if text.get("verbosity") is not None:
            raise ValueError("verbosity is not supported for the requested model")
        forwarded_text = dict(text)
        forwarded_text.pop("verbosity", None)
        if forwarded_text:
            payload["text"] = forwarded_text

    if service_tier is None or service_tier == "default":
        return
    wire_service_tier = "priority" if service_tier == "fast" else service_tier
    if wire_service_tier not in capability.service_tier_ids:
        raise ValueError("service_tier is not supported for the requested model")
    payload["service_tier"] = wire_service_tier


def resolve_codex_metadata_enabled(value: bool | None = None) -> bool:
    if value is not None:
        return value
    configured = os.getenv(CODEX_METADATA_ENV)
    if configured is None:
        return False
    raw = configured.strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError("codex_metadata must be on or off")


def validate_model_capability_environment() -> None:
    if RESPONSES_LITE_ENV in os.environ:
        resolve_responses_lite_mode()
    if CODEX_METADATA_ENV in os.environ:
        resolve_codex_metadata_enabled()


def build_codex_client_metadata(
    *,
    auth_json_path: str | os.PathLike[str] | None,
    existing: Mapping[str, str] | None,
) -> dict[str, str]:
    metadata = dict(existing or {})
    session_id = metadata.get(SESSION_ID_KEY)
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("codex_metadata requires a non-empty client_metadata.session_id")
    thread_id = metadata.get(THREAD_ID_KEY)
    if thread_id is None:
        thread_id = session_id
    elif not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("client_metadata.thread_id must be a non-empty string when provided")

    auth_path = resolve_auth_path(os.fspath(auth_json_path) if auth_json_path is not None else None)
    absolute_path = os.path.abspath(os.fspath(auth_path))
    installation_id = str(uuid.uuid5(_INSTALLATION_NAMESPACE, f"codex-as-api:{absolute_path}"))
    turn_id = str(uuid.uuid4())
    turn_metadata = {
        "installation_id": installation_id,
        "session_id": session_id,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "window_id": _WINDOW_ID,
        "source": "codex-as-api",
    }
    metadata.update(
        {
            INSTALLATION_ID_KEY: installation_id,
            SESSION_ID_KEY: session_id,
            THREAD_ID_KEY: thread_id,
            TURN_ID_KEY: turn_id,
            WINDOW_ID_KEY: _WINDOW_ID,
            TURN_METADATA_KEY: json.dumps(turn_metadata, separators=(",", ":"), sort_keys=True),
        }
    )
    return metadata


def _model_from_mapping(value: object, *, index: int) -> ModelCapability:
    where = f"models[{index}]"
    if not isinstance(value, dict):
        raise ChatGPTOAuthCatalogUnavailableError(f"{where} must be an object")
    slug = _required_string(value, "slug", where, allow_empty=True)
    display_name = _required_string(value, "display_name", where, allow_empty=True)
    description = _optional_string(value, "description", where, allow_empty=True)
    priority = _required_i32(value, "priority", where)
    visibility = _required_string(value, "visibility", where)
    if visibility not in {"list", "hide", "none"}:
        raise ChatGPTOAuthCatalogUnavailableError(f"{where}.visibility is invalid")
    supported_in_api = _required_bool(value, "supported_in_api", where)
    reasoning = _reasoning_levels(value.get("supported_reasoning_levels"), where)
    default_reasoning = _optional_reasoning_effort(value, "default_reasoning_level", where)
    multi_agent_reasoning_effort = _optional_reasoning_effort(
        value,
        "multi_agent_reasoning_effort",
        where,
    )
    context_window = _optional_safe_int(value, "context_window", where)
    max_context_window = _optional_safe_int(value, "max_context_window", where)
    effective_percent = value.get(
        "effective_context_window_percent",
        DEFAULT_EFFECTIVE_CONTEXT_WINDOW_PERCENT,
    )
    parsed_effective_percent = _as_safe_integer(effective_percent)
    if parsed_effective_percent is None:
        raise ChatGPTOAuthCatalogUnavailableError(
            f"{where}.effective_context_window_percent must be a JavaScript-safe integer"
        )
    effective_percent = parsed_effective_percent
    auto_compact = _optional_safe_int(value, "auto_compact_token_limit", where)
    modalities = _string_array(
        value.get("input_modalities", ["text", "image"]),
        f"{where}.input_modalities",
        allow_empty=True,
    )
    invalid_modalities = [modality for modality in modalities if modality not in {"text", "image", "audio"}]
    if invalid_modalities:
        raise ChatGPTOAuthCatalogUnavailableError(f"{where}.input_modalities contains an unsupported value")
    service_tiers = _service_tiers(value.get("service_tiers", []), where)
    default_service_tier = _optional_string(value, "default_service_tier", where, allow_empty=True)
    default_verbosity = _optional_string(value, "default_verbosity", where)
    if default_verbosity is not None and default_verbosity not in {
        "low",
        "medium",
        "high",
    }:
        raise ChatGPTOAuthCatalogUnavailableError(f"{where}.default_verbosity must be one of: low, medium, high")
    comp_hash = _optional_string(value, "comp_hash", where, allow_empty=True)
    supports_reasoning_summary_parameter = _optional_bool(
        value,
        "supports_reasoning_summary_parameter",
        where,
        default=True,
    )
    default_reasoning_summary = value.get("default_reasoning_summary", "auto")
    if default_reasoning_summary not in {"auto", "concise", "detailed", "none"}:
        raise ChatGPTOAuthCatalogUnavailableError(
            f"{where}.default_reasoning_summary must be one of: auto, concise, detailed, none"
        )
    return ModelCapability(
        slug=slug,
        display_name=display_name,
        description=description,
        priority=priority,
        visibility=visibility,
        supported_in_api=supported_in_api,
        supported_reasoning_levels=reasoning,
        default_reasoning_level=default_reasoning,
        multi_agent_reasoning_effort=multi_agent_reasoning_effort,
        context_window=context_window,
        max_context_window=max_context_window,
        effective_context_window_percent=effective_percent,
        auto_compact_token_limit=auto_compact,
        input_modalities=modalities,
        service_tiers=service_tiers,
        default_service_tier=default_service_tier,
        use_responses_lite=_optional_bool(value, "use_responses_lite", where, default=False),
        supports_image_detail_original=_optional_bool(
            value,
            "supports_image_detail_original",
            where,
            default=False,
        ),
        support_verbosity=_required_bool(value, "support_verbosity", where),
        default_verbosity=default_verbosity,
        comp_hash=comp_hash,
        supports_reasoning_summary_parameter=supports_reasoning_summary_parameter,
        default_reasoning_summary=default_reasoning_summary,
    )


def _reasoning_levels(
    value: object,
    where: str,
) -> tuple[ModelReasoningLevel, ...]:
    if not isinstance(value, list):
        raise ChatGPTOAuthCatalogUnavailableError(f"{where}.supported_reasoning_levels must be an array")
    levels: list[ModelReasoningLevel] = []
    for index, item in enumerate(value):
        item_where = f"{where}.supported_reasoning_levels[{index}]"
        if not isinstance(item, dict):
            raise ChatGPTOAuthCatalogUnavailableError(f"{item_where} must be an object")
        effort = _required_reasoning_effort(item, "effort", item_where)
        description = _required_string(
            item,
            "description",
            item_where,
            allow_empty=True,
        )
        levels.append(ModelReasoningLevel(effort=effort, description=description))
    return tuple(levels)


def _service_tiers(value: object, where: str) -> tuple[ModelServiceTier, ...]:
    if not isinstance(value, list):
        raise ChatGPTOAuthCatalogUnavailableError(f"{where}.service_tiers must be an array")
    tiers: list[ModelServiceTier] = []
    for index, item in enumerate(value):
        item_where = f"{where}.service_tiers[{index}]"
        if not isinstance(item, dict):
            raise ChatGPTOAuthCatalogUnavailableError(f"{item_where} must be an object")
        tier_id = _required_string(item, "id", item_where, allow_empty=True)
        tiers.append(
            ModelServiceTier(
                id=tier_id,
                name=_required_string(item, "name", item_where, allow_empty=True),
                description=_required_string(item, "description", item_where, allow_empty=True),
            )
        )
    return tuple(tiers)


def _string_array(value: object, where: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        requirement = "an array" if allow_empty else "a non-empty array"
        raise ChatGPTOAuthCatalogUnavailableError(f"{where} must be {requirement}")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ChatGPTOAuthCatalogUnavailableError(f"{where}[{index}] must be a non-empty string")
        result.append(item)
    return tuple(result)


def _required_string(
    value: Mapping[str, object],
    key: str,
    where: str,
    *,
    allow_empty: bool = False,
) -> str:
    item = value.get(key)
    if not isinstance(item, str) or (not allow_empty and not item.strip()):
        suffix = "a string" if allow_empty else "a non-empty string"
        raise ChatGPTOAuthCatalogUnavailableError(f"{where}.{key} must be {suffix}")
    return item


def _required_reasoning_effort(
    value: Mapping[str, object],
    key: str,
    where: str,
) -> str:
    item = value.get(key)
    if not isinstance(item, str) or item == "":
        raise ChatGPTOAuthCatalogUnavailableError(f"{where}.{key} must be a non-empty string")
    return item


def _optional_string(
    value: Mapping[str, object],
    key: str,
    where: str,
    *,
    allow_empty: bool = False,
) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str) or (not allow_empty and not item.strip()):
        raise ChatGPTOAuthCatalogUnavailableError(
            f"{where}.{key} must be " + ("a string or null" if allow_empty else "a non-empty string or null")
        )
    return item


def _optional_reasoning_effort(
    value: Mapping[str, object],
    key: str,
    where: str,
) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str) or item == "":
        raise ChatGPTOAuthCatalogUnavailableError(f"{where}.{key} must be a non-empty string or null")
    return item


def _required_bool(value: Mapping[str, object], key: str, where: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise ChatGPTOAuthCatalogUnavailableError(f"{where}.{key} must be a boolean")
    return item


def _optional_bool(
    value: Mapping[str, object],
    key: str,
    where: str,
    *,
    default: bool,
) -> bool:
    if key not in value:
        return default
    return _required_bool(value, key, where)


def _required_i32(value: Mapping[str, object], key: str, where: str) -> int:
    item = value.get(key)
    parsed = _as_safe_integer(item)
    if parsed is None or not -(1 << 31) <= parsed < (1 << 31):
        raise ChatGPTOAuthCatalogUnavailableError(f"{where}.{key} must be a signed 32-bit integer")
    return parsed


def _optional_safe_int(
    value: Mapping[str, object],
    key: str,
    where: str,
) -> int | None:
    item = value.get(key)
    if item is None:
        return None
    parsed = _as_safe_integer(item)
    if parsed is None:
        raise ChatGPTOAuthCatalogUnavailableError(f"{where}.{key} must be a JavaScript-safe integer or null")
    return parsed


def _as_safe_integer(item: object) -> int | None:
    if isinstance(item, bool):
        return None
    if isinstance(item, int):
        parsed = item
    elif isinstance(item, float) and math.isfinite(item) and item.is_integer():
        parsed = int(item)
    else:
        return None
    if not -_JS_SAFE_INTEGER <= parsed <= _JS_SAFE_INTEGER:
        return None
    return parsed


def _non_empty_optional_model(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ChatGPTOAuthInvalidRequestError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise ChatGPTOAuthInvalidRequestError(f"{field} must not contain surrounding whitespace")
    return value
