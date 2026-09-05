from __future__ import annotations

import http.client
import json
import math
import os
import pathlib
import platform
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, NoReturn, cast

from . import __version__
from .auth import (
    ChatGPTOAuthCatalogUnavailableError,
    ChatGPTOAuthError,
    ChatGPTOAuthInvalidRequestError,
    ChatGPTOAuthProtocolError,
    ChatGPTOAuthRefreshError,
    ChatGPTOAuthUpstreamError,
    redact_text,
    refresh_after_unauthorized,
    token_for_request,
    validate_auth_environment,
)
from .messages import AssistantResponse, Message, MessageRole, ToolCall, ToolSchema, Usage
from .model_capabilities import (
    DEFAULT_MODEL_CATALOG_TIMEOUT_SECONDS,
    DEFAULT_MODEL_CATALOG_TTL_SECONDS,
    LITE_HEADER_NAME,
    LITE_HEADER_VALUE,
    RESPONSES_LITE_ENV,
    SESSION_ID_KEY,
    THREAD_ID_KEY,
    CatalogKey,
    CatalogLoadResult,
    ModelCapability,
    ModelCatalogCache,
    ModelCatalogSnapshot,
    apply_model_capability_fields,
    build_codex_client_metadata,
    resolve_codex_metadata_enabled,
    resolve_model,
    use_responses_lite,
    validate_model_capability_environment,
)
from .protocol import (
    reasoning_from_response_items,
    reasoning_parts_from_response_items,
    response_failure_message,
)
from .strict_json import strict_json_loads

CHATGPT_OAUTH_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
CHATGPT_OAUTH_DEFAULT_TIMEOUT = 300.0
REMOTE_COMPACTION_MARKER = "[Remote Responses compacted history]"
CODEX_CLI_ORIGINATOR = "codex_cli_rs"
CODEX_CLI_VERSION_ENV = "CODEX_AS_API_CODEX_CLI_VERSION"
KNOWN_REASONING_MODES = frozenset({"standard", "pro"})
KNOWN_REASONING_CONTEXTS = frozenset({"auto", "current_turn", "all_turns"})
KNOWN_IMAGE_DETAILS = frozenset({"auto", "low", "high", "original"})
KNOWN_VERBOSITY_VALUES = frozenset({"low", "medium", "high"})
_RESPONSE_EVENT_TYPES = frozenset(
    {
        "response.created",
        "response.metadata",
        "codex.response.metadata",
        "responsesapi.websocket_timing",
        "response.in_progress",
        "response.queued",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
        "response.image_generation_call.in_progress",
        "response.image_generation_call.generating",
        "response.image_generation_call.partial_image",
        "response.image_generation_call.completed",
        "response.failed",
        "response.incomplete",
        "response.completed",
    }
)
_UNSUPPORTED_RESPONSE_EVENT_TYPES = frozenset(
    {
        "response.file_search_call.in_progress",
        "response.file_search_call.searching",
        "response.file_search_call.completed",
        "response.code_interpreter_call.in_progress",
        "response.code_interpreter_call.interpreting",
        "response.code_interpreter_call_code.delta",
        "response.code_interpreter_call_code.done",
        "response.code_interpreter_call.completed",
        "response.mcp_call.in_progress",
        "response.mcp_call_arguments.delta",
        "response.mcp_call_arguments.done",
        "response.mcp_call.completed",
        "response.mcp_call.failed",
        "response.mcp_list_tools.in_progress",
        "response.mcp_list_tools.completed",
        "response.mcp_list_tools.failed",
        "response.shell_call_command.added",
        "response.shell_call_command.delta",
        "response.shell_call_command.done",
        "response.shell_call_output_content.delta",
        "response.shell_call_output_content.done",
        "response.audio.delta",
        "response.audio.done",
        "response.audio.transcript.delta",
        "response.audio.transcript.done",
        "response.refusal.delta",
        "response.refusal.done",
        "response.output_text.annotation.added",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
    }
)
RESPONSE_CHAIN_CAPACITY = 256
_CODEX_CLI_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}(?:[-+][0-9A-Za-z.-]+)?$")
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_UPSTREAM_CONTRACT_PATH = _PROJECT_ROOT / "config" / "codex-upstream-contract.json"
_PACKAGE_UPSTREAM_CONTRACT_PATH = pathlib.Path(__file__).resolve().with_name("codex-upstream-contract.json")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _urlopen_no_redirect(request: urllib.request.Request, *, timeout: float) -> Any:
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


def _load_codex_compatibility_version() -> str:
    path = _PACKAGE_UPSTREAM_CONTRACT_PATH if _PACKAGE_UPSTREAM_CONTRACT_PATH.exists() else _UPSTREAM_CONTRACT_PATH
    try:
        document = strict_json_loads(path.read_text(encoding="utf-8"))
        version = document["upstream"]["version"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"invalid bundled Codex upstream contract: {exc}") from exc
    if not isinstance(version, str) or _CODEX_CLI_VERSION_RE.fullmatch(version) is None:
        raise RuntimeError("bundled Codex upstream contract has an invalid version")
    return version


CODEX_COMPATIBILITY_VERSION = _load_codex_compatibility_version()


class _ResponseChainStore:
    """Thread-safe, process-local replay history for public response IDs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._chains: OrderedDict[
            tuple[str, str],
            tuple[list[dict[str, Any]], list[dict[str, Any]], str | None],
        ] = OrderedDict()

    def resolve(
        self,
        response_id: str,
        *,
        account_id: str,
        current_comp_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            key = (account_id, response_id)
            chain = self._chains.get(key)
            if chain is None:
                raise ChatGPTOAuthInvalidRequestError("previous_response_id is unknown or has been evicted")
            request_input, response_output, source_comp_hash = chain
            if (
                source_comp_hash is not None
                and current_comp_hash is not None
                and source_comp_hash != current_comp_hash
            ):
                raise ChatGPTOAuthInvalidRequestError(
                    "previous_response_id requires compaction because the model compatibility hash changed"
                )
            self._chains.move_to_end(key)
            return deepcopy([*request_input, *response_output])

    def commit(
        self,
        response_id: str,
        request_input: Sequence[dict[str, Any]],
        response_output: Sequence[dict[str, Any]],
        *,
        account_id: str,
        comp_hash: str | None = None,
    ) -> None:
        with self._lock:
            key = (account_id, response_id)
            self._chains[key] = (
                deepcopy(list(request_input)),
                deepcopy(list(response_output)),
                comp_hash,
            )
            self._chains.move_to_end(key)
            while len(self._chains) > RESPONSE_CHAIN_CAPACITY:
                self._chains.popitem(last=False)


@dataclass(frozen=True, slots=True)
class PreparedChatRequest:
    payload: dict[str, Any]
    replay_input: tuple[dict[str, Any], ...]
    snapshot: ModelCatalogSnapshot
    capability: ModelCapability


class _TransportHeaders(dict[str, str]):
    def __init__(self, *args: Any, catalog_key: CatalogKey | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.catalog_key = catalog_key


def resolve_codex_cli_version() -> str:
    raw = os.getenv(CODEX_CLI_VERSION_ENV)
    if raw is not None:
        raise ValueError(
            f"{CODEX_CLI_VERSION_ENV} is not supported; the private wire version is pinned to "
            f"{CODEX_COMPATIBILITY_VERSION}"
        )
    return CODEX_COMPATIBILITY_VERSION


def codex_cli_headers_for_version(version: str | None) -> dict[str, str]:
    if version is not None and version != CODEX_COMPATIBILITY_VERSION:
        raise ValueError(f"Codex compatibility version is pinned to {CODEX_COMPATIBILITY_VERSION}")
    normalized = CODEX_COMPATIBILITY_VERSION
    return {
        "originator": CODEX_CLI_ORIGINATOR,
        "User-Agent": _sanitize_header_value(
            f"{CODEX_CLI_ORIGINATOR}/{normalized} ({_codex_os_info()}) codex-as-api/{__version__}"
        ),
    }


def _codex_cli_headers() -> dict[str, str]:
    return codex_cli_headers_for_version(PINNED_CODEX_CLI_VERSION)


def _codex_os_info() -> str:
    return f"{_codex_os_name()} {platform.release() or 'unknown'}; {platform.machine() or 'unknown'}"


def _codex_os_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "Mac OS"
    return system or "unknown"


def _sanitize_header_value(value: str) -> str:
    return "".join(ch if " " <= ch <= "~" else "_" for ch in value)


PINNED_CODEX_CLI_VERSION = resolve_codex_cli_version()


def _normalize_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base_url must be a non-empty HTTP(S) URL")
    if value != value.strip():
        raise ValueError("base_url must not contain surrounding whitespace")
    if any(character.isspace() or unicodedata.category(character) == "Cc" for character in value):
        raise ValueError("base_url must not contain whitespace or control characters")
    if re.search(r"%(?![0-9A-Fa-f]{2})", value):
        raise ValueError("base_url contains an invalid percent escape")
    try:
        parsed = urllib.parse.urlsplit(value)
        _ = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError("base_url must be a valid HTTP(S) URL") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        raise ValueError("base_url must be an HTTP(S) URL without credentials, query, or fragment")
    return value.rstrip("/")


def _validate_header_token(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or not all("!" <= char <= "~" for char in value):
        raise ChatGPTOAuthInvalidRequestError(f"{field} must contain only visible ASCII characters without spaces")
    return value


def _response_socket(response: Any) -> Any | None:
    stream = response
    for _ in range(2):
        buffered = getattr(stream, "fp", None)
        raw = getattr(buffered, "raw", None)
        sock = getattr(raw, "_sock", None)
        if sock is not None:
            return sock
        if buffered is None:
            break
        stream = buffered
    return None


def _read_response_before_deadline(response: Any, deadline: float) -> bytes:
    sock = _response_socket(response)
    read1 = getattr(response, "read1", None)
    if sock is None or not callable(read1):
        body = response.read()
        if time.monotonic() >= deadline:
            raise TimeoutError("upstream response body exceeded its total deadline")
        return cast(bytes, body)

    chunks: list[bytes] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("upstream response body exceeded its total deadline")
        sock.settimeout(remaining)
        chunk = read1(64 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        if callable(getattr(response, "isclosed", None)) and response.isclosed():
            return b"".join(chunks)


def _http_error_body(
    exc: urllib.error.HTTPError,
    token_values: tuple[str | None, ...],
    *,
    deadline: float | None = None,
) -> str:
    try:
        raw = exc.read() if deadline is None else _read_response_before_deadline(exc, deadline)
        body = raw.decode("utf-8", "replace")
    except (OSError, ValueError, TimeoutError, http.client.HTTPException):
        return "could not read upstream error body"
    return redact_text(body, *token_values)


def _redact_failure_event(
    value: Any,
    token_values: tuple[str | None, ...],
) -> Any:
    if isinstance(value, str):
        return redact_text(value, *token_values)
    if isinstance(value, list):
        return [_redact_failure_event(item, token_values) for item in value]
    if isinstance(value, dict):
        return {key: _redact_failure_event(item, token_values) for key, item in value.items()}
    return value


class ChatGPTOAuthProvider:
    name: str = "chatgpt_oauth"
    provider_namespace: str = "agent.provider.chatgpt_oauth"
    supports_prompt_cache_key: bool = True

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str = CHATGPT_OAUTH_DEFAULT_BASE_URL,
        auth_json_path: str | None = None,
        timeout: float | None = None,
        catalog_timeout: float | None = None,
        catalog_ttl: float | None = None,
    ) -> None:
        resolve_codex_cli_version()
        validate_auth_environment()
        validate_model_capability_environment()
        if model is not None and (not isinstance(model, str) or not model.strip() or model != model.strip()):
            raise ValueError("model must be a non-empty string when provided")
        if auth_json_path is not None and not auth_json_path.strip():
            raise ValueError("auth_json_path must be a non-empty string")
        self.model = model
        self.base_url = _normalize_base_url(base_url)
        self.auth_json_path = auth_json_path
        self.timeout = (
            CHATGPT_OAUTH_DEFAULT_TIMEOUT if timeout is None else _validate_positive_finite(timeout, "timeout")
        )
        self.catalog_timeout = (
            DEFAULT_MODEL_CATALOG_TIMEOUT_SECONDS
            if catalog_timeout is None
            else _validate_positive_finite(catalog_timeout, "catalog_timeout")
        )
        self.catalog_ttl = (
            DEFAULT_MODEL_CATALOG_TTL_SECONDS
            if catalog_ttl is None
            else _validate_positive_finite(catalog_ttl, "catalog_ttl")
        )
        self.api_key = None
        self._active_response_lock = threading.Lock()
        self._active_responses: set[Any] = set()
        self._model_catalog_cache = ModelCatalogCache()
        self._response_chains = _ResponseChainStore()

    def cancel_current_requests(self) -> None:
        with self._active_response_lock:
            responses = list(self._active_responses)
        failures: list[Exception] = []
        for response in responses:
            try:
                response.close()
            except Exception as exc:  # noqa: BLE001 - close every active response before reporting
                failures.append(exc)
        if failures:
            raise RuntimeError(
                f"failed to close {len(failures)} active response(s); first failure: {failures[0]}"
            ) from failures[0]

    def get_model_catalog(self) -> ModelCatalogSnapshot:
        token = token_for_request(self.auth_json_path)
        version = PINNED_CODEX_CLI_VERSION
        key = (token.account_id, self.base_url, version)
        return self._model_catalog_cache.get(
            key,
            lambda: self._fetch_model_catalog(token, version),
            ttl_seconds=self.catalog_ttl,
        )

    def resolve_model(
        self,
        requested: str | None = None,
        *,
        anthropic_facade: bool = False,
        snapshot: ModelCatalogSnapshot | None = None,
    ) -> tuple[ModelCatalogSnapshot, ModelCapability]:
        if requested is not None and (
            not isinstance(requested, str) or not requested or requested != requested.strip()
        ):
            raise ChatGPTOAuthInvalidRequestError("model must be a non-empty string without surrounding whitespace")
        if requested is not None and anthropic_facade and requested.startswith("claude-") and self.model is None:
            raise ChatGPTOAuthInvalidRequestError(
                "claude-* facade models require CODEX_AS_API_MODEL or config.toml model"
            )
        catalog = snapshot or self.get_model_catalog()
        capability = resolve_model(
            catalog,
            requested,
            self.model,
            anthropic_facade=anthropic_facade,
        )
        return catalog, capability

    def _fetch_model_catalog(self, initial_token: Any, version: str) -> CatalogLoadResult:
        token = initial_token
        initial_account_id = initial_token.account_id
        token_values: tuple[str | None, ...] = (None,)
        query = urllib.parse.urlencode({"client_version": version})
        url = f"{self.base_url}/models?{query}"
        for attempt in range(2):
            headers = self._headers(token)
            headers["Accept"] = "application/json"
            token_values = (token.access_token, token.refresh_token, token.id_token, token.account_id)
            request = urllib.request.Request(url, headers=headers, method="GET")
            deadline = time.monotonic() + self.catalog_timeout
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("model catalog request exceeded its total deadline")
                with _urlopen_no_redirect(request, timeout=remaining) as response:
                    try:
                        document = strict_json_loads(_read_response_before_deadline(response, deadline))
                    except (UnicodeDecodeError, ValueError) as exc:
                        raise ChatGPTOAuthCatalogUnavailableError(
                            f"model catalog returned invalid JSON: {exc}"
                        ) from exc
                    return CatalogLoadResult(
                        document=document,
                        etag=response.headers.get("ETag"),
                    )
            except urllib.error.HTTPError as exc:
                if exc.code == 401 and attempt == 0:
                    exc.close()
                    token = refresh_after_unauthorized(token)
                    if token.account_id != initial_account_id:
                        raise ChatGPTOAuthRefreshError(
                            "authenticated account changed during model catalog refresh"
                        ) from exc
                    continue
                redacted = _http_error_body(exc, token_values, deadline=deadline)
                if exc.code == 401:
                    raise ChatGPTOAuthUpstreamError(
                        exc.code,
                        f"ChatGPT OAuth model catalog authentication failed after credential refresh: {redacted}",
                    ) from exc
                raise ChatGPTOAuthUpstreamError(
                    exc.code, f"ChatGPT OAuth model catalog request failed: HTTP {exc.code}: {redacted}"
                ) from exc
            except ChatGPTOAuthError:
                raise
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
                raise ChatGPTOAuthCatalogUnavailableError(
                    f"ChatGPT OAuth model catalog request failed: {redact_text(str(exc), *token_values)}"
                ) from exc
        raise AssertionError("unreachable model catalog retry state")

    def preflight_chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_context: str | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        prompt_cache_key: str | None = None,
        previous_response_id: str | None = None,
        service_tier: str | None = None,
        text: dict | None = None,
        client_metadata: dict[str, str] | None = None,
        codex_metadata: bool | None = None,
        responses_lite: bool | str | None = None,
        parallel_tool_calls: bool | None = None,
        safety_identifier: str | None = None,
        prompt_cache_options: dict[str, Any] | None = None,
        verbosity: str | None = None,
        _resolved_model: tuple[ModelCatalogSnapshot, ModelCapability] | None = None,
    ) -> PreparedChatRequest:
        """Prepare and validate one request without opening an upstream stream."""
        return self._prepare_responses_request(
            messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            reasoning_mode=reasoning_mode,
            reasoning_context=reasoning_context,
            max_tokens=max_tokens,
            stop=stop,
            prompt_cache_key=prompt_cache_key,
            previous_response_id=previous_response_id,
            service_tier=service_tier,
            text=text,
            client_metadata=client_metadata,
            codex_metadata=codex_metadata,
            responses_lite=responses_lite,
            parallel_tool_calls=parallel_tool_calls,
            safety_identifier=safety_identifier,
            prompt_cache_options=prompt_cache_options,
            verbosity=verbosity,
            _resolved_model=_resolved_model,
        )

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_context: str | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        prompt_cache_key: str | None = None,
        subagent: str | None = None,
        memgen_request: bool | None = None,
        previous_response_id: str | None = None,
        service_tier: str | None = None,
        text: dict | None = None,
        client_metadata: dict[str, str] | None = None,
        codex_metadata: bool | None = None,
        responses_lite: bool | str | None = None,
        parallel_tool_calls: bool | None = None,
        safety_identifier: str | None = None,
        prompt_cache_options: dict[str, Any] | None = None,
        verbosity: str | None = None,
        _prepared_request: PreparedChatRequest | None = None,
    ) -> AssistantResponse:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason: str | None = None
        saw_finish = False
        raw_events: list[dict[str, Any]] = []
        usage: Usage | None = None
        response_id: str | None = None
        tool_call_ids: set[str] = set()
        for event in self.chat_stream(
            messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            reasoning_mode=reasoning_mode,
            reasoning_context=reasoning_context,
            max_tokens=max_tokens,
            stop=stop,
            prompt_cache_key=prompt_cache_key,
            subagent=subagent,
            memgen_request=memgen_request,
            previous_response_id=previous_response_id,
            service_tier=service_tier,
            text=text,
            client_metadata=client_metadata,
            codex_metadata=codex_metadata,
            responses_lite=responses_lite,
            parallel_tool_calls=parallel_tool_calls,
            safety_identifier=safety_identifier,
            prompt_cache_options=prompt_cache_options,
            verbosity=verbosity,
            _prepared_request=_prepared_request,
        ):
            raw_events.append(dict(event))
            if event.get("type") == "content":
                content_parts.append(_required_event_text(event, "content"))
            elif event.get("type") in {"reasoning_delta", "reasoning_raw_delta"}:
                reasoning_parts.append(_required_event_text(event, str(event["type"])))
            elif event.get("type") == "tool_call":
                event_id = event.get("id")
                event_name = event.get("name")
                event_arguments = event.get("arguments")
                if not isinstance(event_id, str):
                    raise ChatGPTOAuthProtocolError("tool_call event requires a string id")
                if not isinstance(event_name, str):
                    raise ChatGPTOAuthProtocolError("tool_call event requires a string name")
                if not isinstance(event_arguments, str):
                    raise ChatGPTOAuthProtocolError("tool_call event arguments must be a string")
                if event_id in tool_call_ids:
                    raise ChatGPTOAuthProtocolError(f"provider response contains duplicate call_id {event_id!r}")
                tool_call_ids.add(event_id)
                tool_calls.append(ToolCall(id=event_id, name=event_name, arguments=event_arguments))
            elif event.get("type") == "finish":
                saw_finish = True
                event_finish_reason = event.get("finish_reason")
                if event_finish_reason is not None and (
                    not isinstance(event_finish_reason, str) or not event_finish_reason
                ):
                    raise ChatGPTOAuthProtocolError("finish event finish_reason must be non-empty or null")
                finish_reason = event_finish_reason
                if isinstance(event.get("reasoning_content"), str):
                    reasoning_parts = [str(event["reasoning_content"])]
                usage = _usage_from_response(event.get("usage"))
                event_response_id = event.get("response_id")
                if not isinstance(event_response_id, str) or not event_response_id:
                    raise ChatGPTOAuthProtocolError("finish event requires a non-empty response_id")
                response_id = event_response_id
        if not saw_finish or response_id is None:
            raise ChatGPTOAuthProtocolError("provider stream ended without a complete finish event")
        return AssistantResponse(
            content="".join(content_parts),
            tool_calls=tuple(tool_calls),
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content="".join(reasoning_parts) or None,
            response_id=response_id,
            raw={"events": _compact_raw_events(raw_events)},
        )

    def chat_stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_context: str | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        prompt_cache_key: str | None = None,
        subagent: str | None = None,
        memgen_request: bool | None = None,
        previous_response_id: str | None = None,
        service_tier: str | None = None,
        text: dict | None = None,
        client_metadata: dict[str, str] | None = None,
        codex_metadata: bool | None = None,
        responses_lite: bool | str | None = None,
        parallel_tool_calls: bool | None = None,
        safety_identifier: str | None = None,
        prompt_cache_options: dict[str, Any] | None = None,
        verbosity: str | None = None,
        _prepared_request: PreparedChatRequest | None = None,
    ) -> Iterator[dict[str, Any]]:
        if _prepared_request is None:
            prepared = self._prepare_responses_request(
                messages,
                model=model,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                reasoning_mode=reasoning_mode,
                reasoning_context=reasoning_context,
                stop=stop,
                prompt_cache_key=prompt_cache_key,
                max_tokens=max_tokens,
                previous_response_id=previous_response_id,
                service_tier=service_tier,
                text=text,
                client_metadata=client_metadata,
                codex_metadata=codex_metadata,
                responses_lite=responses_lite,
                parallel_tool_calls=parallel_tool_calls,
                safety_identifier=safety_identifier,
                prompt_cache_options=prompt_cache_options,
                verbosity=verbosity,
            )
        else:
            prepared = _prepared_request
        payload = deepcopy(prepared.payload)
        replay_input = list(prepared.replay_input)
        extra_headers = _responses_transport_headers(
            payload,
            catalog_key=prepared.snapshot.key,
        )
        if subagent is not None:
            extra_headers["x-openai-subagent"] = _validate_header_token(subagent, "subagent")
        if memgen_request is not None:
            extra_headers["x-openai-memgen-request"] = "true" if memgen_request else "false"
        stream = self._post_sse(
            "/responses",
            payload,
            extra_headers=extra_headers,
        )
        final_output: list[dict[str, Any]] = []
        reasoning_summary_parts: list[str] = []
        reasoning_raw_parts: list[str] = []
        streamed_text_parts: list[str] = []
        yielded_web_search_ids: set[str] = set()
        yielded_tool_call_ids: set[str] = set()
        saw_text_delta = False
        saw_reasoning_summary_delta = False
        saw_reasoning_raw_delta = False
        saw_function_tool_call = False
        for event in stream:
            _validate_response_event(event)
            typ = event.get("type")
            if typ == "response.output_text.delta":
                delta = event.get("delta")
                if not isinstance(delta, str):
                    raise ChatGPTOAuthProtocolError("response.output_text.delta requires a string delta")
                if delta:
                    saw_text_delta = True
                    streamed_text_parts.append(delta)
                    yield {"type": "content", "text": delta}
            elif typ == "response.output_item.done":
                item = event.get("item")
                if not isinstance(item, dict):
                    raise ChatGPTOAuthProtocolError("response.output_item.done must contain an object item")
                if item.get("type") == "image_generation_call":
                    raise ChatGPTOAuthProtocolError("image_generation_call cannot be represented by normal chat")
                final_output.append(item)
                tool = _tool_call_from_response_item(item)
                if tool is not None:
                    if tool.id in yielded_tool_call_ids:
                        raise ChatGPTOAuthProtocolError(
                            f"provider response contains duplicate call_id {tool.id!r}"
                        )
                    saw_function_tool_call = True
                    yielded_tool_call_ids.add(tool.id)
                    yield {"type": "tool_call", "id": tool.id, "name": tool.name, "arguments": tool.arguments}
                web_search = _web_search_event_from_response_item(item)
                if web_search is not None:
                    yielded_web_search_ids.add(str(web_search["id"]))
                    yield web_search
            elif typ == "response.reasoning_summary_part.added":
                yield {
                    "type": "reasoning_section_break",
                    "summary_index": event.get("summary_index"),
                }
            elif typ == "response.reasoning_summary_text.delta":
                delta = event.get("delta")
                if not isinstance(delta, str):
                    raise ChatGPTOAuthProtocolError("response.reasoning_summary_text.delta requires a string delta")
                saw_reasoning_summary_delta = True
                reasoning_summary_parts.append(delta)
                if delta:
                    yield {
                        "type": "reasoning_delta",
                        "text": delta,
                        "summary_index": event.get("summary_index"),
                    }
            elif typ == "response.reasoning_text.delta":
                delta = event.get("delta")
                if not isinstance(delta, str):
                    raise ChatGPTOAuthProtocolError("response.reasoning_text.delta requires a string delta")
                saw_reasoning_raw_delta = True
                reasoning_raw_parts.append(delta)
                if delta:
                    yield {
                        "type": "reasoning_raw_delta",
                        "text": delta,
                        "content_index": event.get("content_index"),
                    }
            elif typ == "response.failed":
                raise ChatGPTOAuthUpstreamError(
                    502,
                    response_failure_message(event, "failed"),
                )
            elif typ == "response.incomplete":
                raise ChatGPTOAuthUpstreamError(
                    502,
                    response_failure_message(event, "incomplete"),
                )
            elif typ == "response.completed":
                response = _validated_completed_response(event)
                usage = response.get("usage")
                _usage_from_response(usage)
                display_output = deepcopy(final_output)
                for item in display_output:
                    tool = _tool_call_from_response_item(item)
                    if tool is not None and tool.id not in yielded_tool_call_ids:
                        yielded_tool_call_ids.add(tool.id)
                        saw_function_tool_call = True
                        yield {
                            "type": "tool_call",
                            "id": tool.id,
                            "name": tool.name,
                            "arguments": tool.arguments,
                        }
                    web_search = _web_search_event_from_response_item(item)
                    if web_search is not None and str(web_search["id"]) not in yielded_web_search_ids:
                        yielded_web_search_ids.add(str(web_search["id"]))
                        yield web_search
                final_text = _text_from_response_items(display_output)
                if saw_text_delta and "".join(streamed_text_parts) != final_text:
                    raise ChatGPTOAuthProtocolError(
                        "response.completed output text does not match streamed output text"
                    )
                if not saw_text_delta and final_text:
                    saw_text_delta = True
                    yield {"type": "content", "text": final_text}
                completed_summary, completed_raw = reasoning_parts_from_response_items(display_output)
                if saw_reasoning_summary_delta:
                    if "".join(reasoning_summary_parts) != completed_summary:
                        raise ChatGPTOAuthProtocolError(
                            "response.completed reasoning summary does not match streamed reasoning summary"
                        )
                elif completed_summary:
                    reasoning_summary_parts.append(completed_summary)
                    yield {"type": "reasoning_delta", "text": completed_summary}
                if saw_reasoning_raw_delta:
                    if "".join(reasoning_raw_parts) != completed_raw:
                        raise ChatGPTOAuthProtocolError(
                            "response.completed reasoning content does not match streamed reasoning content"
                        )
                elif completed_raw:
                    reasoning_raw_parts.append(completed_raw)
                    yield {"type": "reasoning_raw_delta", "text": completed_raw}
                self._response_chains.commit(
                    response["id"],
                    replay_input,
                    final_output,
                    account_id=prepared.snapshot.account_id,
                    comp_hash=prepared.capability.comp_hash,
                )
                end_turn = response.get("end_turn")
                finish_event: dict[str, Any] = {
                    "type": "finish",
                    "finish_reason": (
                        "tool_calls"
                        if saw_function_tool_call
                        else None
                        if end_turn is False
                        else "stop"
                    ),
                    "reasoning_content": (
                        "".join(reasoning_summary_parts) + "".join(reasoning_raw_parts)
                    ) or None,
                    "response_id": response["id"],
                }
                if usage is not None:
                    finish_event["usage"] = usage
                yield finish_event
                return
        raise ChatGPTOAuthProtocolError("ChatGPT OAuth response stream ended before response.completed")

    def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        reference_images: Sequence[dict[str, Any]] = (),
        size: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_context: str | None = None,
        responses_lite: bool | str | None = None,
        safety_identifier: str | None = None,
        prompt_cache_options: dict[str, Any] | None = None,
        verbosity: str | None = None,
        _resolved_model: tuple[ModelCatalogSnapshot, ModelCapability] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(prompt, str) or prompt.strip() == "":
            raise ChatGPTOAuthInvalidRequestError("image generation prompt is required")
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend(_validate_image_content_items(reference_images))
        if size is not None and size != "auto":
            raise ChatGPTOAuthInvalidRequestError("image size is not supported by the ChatGPT Codex OAuth transport")
        snapshot, capability = _resolved_model or self.resolve_model(model)
        if "image" not in capability.input_modalities:
            raise ChatGPTOAuthInvalidRequestError(
                "image generation is not supported by the requested model"
            )
        request_model = capability.slug
        payload = {
            "model": request_model,
            "instructions": (
                "Use the image_generation tool to create the requested image. "
                "Return the generated image through an image_generation_call result."
            ),
            "input": [{"type": "message", "role": "user", "content": content}],
            "tools": [{"type": "image_generation", "output_format": "png"}],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "stream": True,
            "store": False,
            "include": [],
        }
        _finalize_responses_payload(
            payload,
            capability=capability,
            reasoning_effort=reasoning_effort,
            reasoning_mode=reasoning_mode,
            reasoning_context=reasoning_context,
            responses_lite=responses_lite,
            safety_identifier=safety_identifier,
            prompt_cache_options=prompt_cache_options,
            verbosity=verbosity,
        )
        output_items = self._collect_response_output_items(payload, catalog_key=snapshot.key)
        generated = _image_generations_from_response_items(output_items)
        if not generated:
            raise ChatGPTOAuthProtocolError("image generation response returned no image_generation_call")
        return generated

    def inspect_images(
        self,
        prompt: str,
        *,
        model: str | None = None,
        images: Sequence[dict[str, Any]],
        reasoning_effort: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_context: str | None = None,
        responses_lite: bool | str | None = None,
        safety_identifier: str | None = None,
        prompt_cache_options: dict[str, Any] | None = None,
        verbosity: str | None = None,
        _resolved_model: tuple[ModelCatalogSnapshot, ModelCapability] | None = None,
    ) -> str:
        if not isinstance(prompt, str) or prompt.strip() == "":
            raise ChatGPTOAuthInvalidRequestError("image inspection prompt is required")
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend(_validate_image_content_items(images))
        snapshot, capability = _resolved_model or self.resolve_model(model)
        request_model = capability.slug
        payload = {
            "model": request_model,
            "instructions": "Inspect the attached image(s) and answer the user's review prompt directly.",
            "input": [{"type": "message", "role": "user", "content": content}],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "stream": True,
            "store": False,
            "include": [],
        }
        _finalize_responses_payload(
            payload,
            capability=capability,
            reasoning_effort=reasoning_effort,
            reasoning_mode=reasoning_mode,
            reasoning_context=reasoning_context,
            responses_lite=responses_lite,
            safety_identifier=safety_identifier,
            prompt_cache_options=prompt_cache_options,
            verbosity=verbosity,
        )
        output_items = self._collect_response_output_items(payload, catalog_key=snapshot.key)
        text = _inspection_text_from_response_items(output_items)
        if text == "":
            raise ChatGPTOAuthProtocolError("image inspection response returned empty content")
        return text

    def _collect_response_output_items(
        self,
        payload: dict[str, Any],
        *,
        catalog_key: CatalogKey | None = None,
    ) -> list[dict[str, Any]]:
        output_items: list[dict[str, Any]] = []

        for event in self._post_sse(
            "/responses",
            payload,
            extra_headers=_responses_transport_headers(
                payload,
                catalog_key=catalog_key,
            ),
        ):
            _validate_response_event(event)
            typ = event.get("type")
            if typ == "response.output_item.done":
                item = event.get("item")
                if not isinstance(item, dict):
                    raise ChatGPTOAuthProtocolError("response.output_item.done must contain an object item")
                output_items.append(item)
            elif typ == "response.failed":
                raise ChatGPTOAuthUpstreamError(
                    502,
                    response_failure_message(event, "failed"),
                )
            elif typ == "response.incomplete":
                raise ChatGPTOAuthUpstreamError(
                    502,
                    response_failure_message(event, "incomplete"),
                )
            elif typ == "response.completed":
                _validated_completed_response(event)
                return output_items
        raise ChatGPTOAuthProtocolError("ChatGPT OAuth response stream ended before response.completed")

    def compact_messages(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        reasoning_effort: str | None = None,
        responses_lite: bool | str | None = None,
        previous_response_id: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_options: dict[str, Any] | None = None,
        service_tier: str | None = None,
        text: dict[str, Any] | None = None,
        verbosity: str | None = None,
        _resolved_model: tuple[ModelCatalogSnapshot, ModelCapability] | None = None,
    ) -> str:
        snapshot, capability = _resolved_model or self.resolve_model(model)
        request_model = capability.slug
        base_instructions, input_items = _split_instructions_and_input(messages)
        if previous_response_id is not None:
            validated_previous_response_id = _validate_previous_response_id(previous_response_id)
            input_items = (
                self._response_chains.resolve(
                    validated_previous_response_id,
                    account_id=snapshot.account_id,
                    current_comp_hash=capability.comp_hash,
                )
                + input_items
            )
        tools_payload = [] if tools is None else [_tool_schema_to_response_dict(tool) for tool in tools]
        payload = {
            "model": request_model,
            "input": input_items,
            "tools": tools_payload,
            "parallel_tool_calls": False,
        }
        if base_instructions:
            payload["instructions"] = base_instructions
        if prompt_cache_key is not None:
            payload["prompt_cache_key"] = _validate_non_empty_string(
                prompt_cache_key,
                "prompt_cache_key",
            )
        _finalize_responses_payload(
            payload,
            capability=capability,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            text=text,
            verbosity=verbosity,
            responses_lite=responses_lite,
            include_encrypted_content=False,
            prompt_cache_options=prompt_cache_options,
        )
        data = self._post_json(
            "/responses/compact",
            payload,
            extra_headers=_responses_transport_headers(
                payload,
                catalog_key=snapshot.key,
            ),
        )
        output = data.get("output")
        if not isinstance(output, list):
            raise ChatGPTOAuthProtocolError("remote compact response missing output array")
        compacted_history = _filter_compacted_history_items(output)
        # Preserve the installed replacement-history items for the ChatGPT OAuth provider. The marker
        # is deliberately not a fallback summary; it is expanded back into Response items later.
        return (
            REMOTE_COMPACTION_MARKER
            + "\n"
            + json.dumps(
                compacted_history,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    def _responses_payload(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_context: str | None = None,
        stop: Sequence[str] | None = None,
        prompt_cache_key: str | None = None,
        max_tokens: int | None = None,
        previous_response_id: str | None = None,
        service_tier: str | None = None,
        text: dict | None = None,
        client_metadata: dict[str, str] | None = None,
        codex_metadata: bool | None = None,
        responses_lite: bool | str | None = None,
        parallel_tool_calls: bool | None = None,
        safety_identifier: str | None = None,
        prompt_cache_options: dict[str, Any] | None = None,
        verbosity: str | None = None,
    ) -> dict[str, Any]:
        return self._prepare_responses_request(
            messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            reasoning_mode=reasoning_mode,
            reasoning_context=reasoning_context,
            stop=stop,
            prompt_cache_key=prompt_cache_key,
            max_tokens=max_tokens,
            previous_response_id=previous_response_id,
            service_tier=service_tier,
            text=text,
            client_metadata=client_metadata,
            codex_metadata=codex_metadata,
            responses_lite=responses_lite,
            parallel_tool_calls=parallel_tool_calls,
            safety_identifier=safety_identifier,
            prompt_cache_options=prompt_cache_options,
            verbosity=verbosity,
        ).payload

    def _prepare_responses_request(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_context: str | None = None,
        stop: Sequence[str] | None = None,
        prompt_cache_key: str | None = None,
        max_tokens: int | None = None,
        previous_response_id: str | None = None,
        service_tier: str | None = None,
        text: dict | None = None,
        client_metadata: dict[str, str] | None = None,
        codex_metadata: bool | None = None,
        responses_lite: bool | str | None = None,
        parallel_tool_calls: bool | None = None,
        safety_identifier: str | None = None,
        prompt_cache_options: dict[str, Any] | None = None,
        verbosity: str | None = None,
        _resolved_model: tuple[ModelCatalogSnapshot, ModelCapability] | None = None,
    ) -> PreparedChatRequest:
        if temperature is not None:
            raise ChatGPTOAuthInvalidRequestError("temperature is not supported by the ChatGPT Codex OAuth transport")
        if max_tokens is not None:
            raise ChatGPTOAuthInvalidRequestError("max_tokens is not supported by the ChatGPT Codex OAuth transport")
        _reject_unsupported_stop(stop)
        instructions, input_items = _split_instructions_and_input(messages)
        snapshot, capability = _resolved_model or self.resolve_model(model)
        request_model = capability.slug
        if previous_response_id is not None:
            validated_previous_response_id = _validate_previous_response_id(previous_response_id)
            input_items = (
                self._response_chains.resolve(
                    validated_previous_response_id,
                    account_id=snapshot.account_id,
                    current_comp_hash=capability.comp_hash,
                )
                + input_items
            )
        tools_payload = [] if tools is None else [_tool_schema_to_response_dict(tool) for tool in tools]
        payload: dict[str, Any] = {
            "model": request_model,
            "input": input_items,
            "tools": tools_payload,
            "tool_choice": "auto" if tool_choice is None else tool_choice,
            "parallel_tool_calls": parallel_tool_calls is True,
            "stream": True,
            "store": False,
            "include": [],
        }
        if instructions:
            payload["instructions"] = instructions
        if any(tool.get("type") == "web_search" for tool in payload["tools"]):
            payload["include"] = ["web_search_call.action.sources"]
        metadata = dict(client_metadata) if client_metadata is not None else None
        if metadata is not None:
            for key in (SESSION_ID_KEY, THREAD_ID_KEY):
                if key in metadata and not metadata[key].strip():
                    raise ChatGPTOAuthInvalidRequestError(
                        f"client_metadata.{key} must be a non-empty string when provided"
                    )
        if resolve_codex_metadata_enabled(codex_metadata):
            try:
                metadata = build_codex_client_metadata(
                    auth_json_path=self.auth_json_path,
                    existing=metadata,
                )
            except ValueError as exc:
                raise ChatGPTOAuthInvalidRequestError(str(exc)) from exc
        effective_prompt_cache_key = prompt_cache_key
        if effective_prompt_cache_key is None and metadata is not None:
            session_id = metadata.get(SESSION_ID_KEY)
            if isinstance(session_id, str) and session_id.strip():
                effective_prompt_cache_key = session_id
        if effective_prompt_cache_key is not None:
            payload["prompt_cache_key"] = _validate_non_empty_string(
                effective_prompt_cache_key,
                "prompt_cache_key",
            )
        if metadata is not None:
            payload["client_metadata"] = metadata
        _finalize_responses_payload(
            payload,
            capability=capability,
            reasoning_effort=reasoning_effort,
            reasoning_mode=reasoning_mode,
            reasoning_context=reasoning_context,
            text=text,
            verbosity=verbosity,
            service_tier=service_tier,
            responses_lite=responses_lite,
            safety_identifier=safety_identifier,
            prompt_cache_options=prompt_cache_options,
        )
        return PreparedChatRequest(
            payload=payload,
            replay_input=tuple(deepcopy(input_items)),
            snapshot=snapshot,
            capability=capability,
        )

    def _headers(self, token: Any | None = None) -> dict[str, str]:
        token = token or token_for_request(self.auth_json_path)
        headers = {
            **_codex_cli_headers(),
            "Authorization": f"Bearer {token.access_token}",
            "ChatGPT-Account-Id": token.account_id,
            "Content-Type": "application/json",
        }
        if token.fedramp:
            headers["X-OpenAI-Fedramp"] = "true"
        return headers

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
        catalog_key: CatalogKey | None = None,
    ) -> dict[str, Any]:
        if catalog_key is None:
            catalog_key = getattr(extra_headers, "catalog_key", None)
        raw = self._request_json(
            path,
            payload,
            extra_headers=extra_headers,
            catalog_key=catalog_key,
        )
        try:
            data = strict_json_loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ChatGPTOAuthProtocolError(f"ChatGPT OAuth response returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ChatGPTOAuthProtocolError("ChatGPT OAuth response must be a JSON object")
        return data

    def _post_sse(
        self,
        path: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
        catalog_key: CatalogKey | None = None,
    ) -> Iterator[dict[str, Any]]:
        if catalog_key is None:
            catalog_key = getattr(extra_headers, "catalog_key", None)
        yield from self._request_sse(
            path,
            payload,
            extra_headers=extra_headers,
            catalog_key=catalog_key,
        )

    def _request_sse(
        self,
        path: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
        catalog_key: CatalogKey | None = None,
    ) -> Iterator[dict[str, Any]]:
        token_values: tuple[str | None, ...] = (None,)
        for attempt in range(2):
            token = token_for_request(self.auth_json_path)
            _require_catalog_account(token.account_id, catalog_key)
            headers = self._headers(token)
            headers["Accept"] = "text/event-stream"
            if extra_headers:
                headers.update(extra_headers)
            token_values = (token.access_token, token.refresh_token, token.id_token, token.account_id)
            req = urllib.request.Request(
                self.base_url + path,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            error_deadline = time.monotonic() + self.timeout
            try:
                with _urlopen_no_redirect(req, timeout=self.timeout) as response:
                    if catalog_key is not None:
                        self._model_catalog_cache.invalidate_on_etag_mismatch(
                            catalog_key,
                            response.headers.get("X-Models-Etag"),
                        )
                    with self._active_response_lock:
                        self._active_responses.add(response)
                    block: list[str] = []
                    try:
                        while True:
                            raw_line = response.readline()
                            if raw_line == b"":
                                if block:
                                    event = _decode_sse_block(block)
                                    if event is not None:
                                        if event.get("type") in {
                                            "error",
                                            "response.failed",
                                            "response.incomplete",
                                        }:
                                            event = _redact_failure_event(event, token_values)
                                        yield event
                                return
                            try:
                                line = raw_line.decode("utf-8").rstrip("\r\n")
                            except UnicodeDecodeError as exc:
                                raise ChatGPTOAuthProtocolError(
                                    "ChatGPT OAuth SSE response was not valid UTF-8"
                                ) from exc
                            if line == "":
                                event = _decode_sse_block(block)
                                block = []
                                if event is not None:
                                    if event.get("type") in {
                                        "error",
                                        "response.failed",
                                        "response.incomplete",
                                    }:
                                        event = _redact_failure_event(event, token_values)
                                    yield event
                                continue
                            block.append(line)
                    finally:
                        with self._active_response_lock:
                            self._active_responses.discard(response)
            except urllib.error.HTTPError as exc:
                if exc.code == 401 and attempt == 0:
                    exc.close()
                    refresh_after_unauthorized(token)
                    continue
                redacted = _http_error_body(exc, token_values, deadline=error_deadline)
                raise ChatGPTOAuthUpstreamError(
                    exc.code,
                    f"ChatGPT OAuth request failed: HTTP {exc.code}: {redacted}",
                ) from exc
            except ChatGPTOAuthError:
                raise
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
                raise ChatGPTOAuthUpstreamError(
                    502,
                    f"ChatGPT OAuth transport failed: {redact_text(str(exc), *token_values)}",
                ) from exc
            return

    def _request_json(
        self,
        path: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
        catalog_key: CatalogKey | None = None,
    ) -> bytes:
        token_values: tuple[str | None, ...] = (None,)
        for attempt in range(2):
            token = token_for_request(self.auth_json_path)
            _require_catalog_account(token.account_id, catalog_key)
            headers = self._headers(token)
            if extra_headers:
                headers.update(extra_headers)
            token_values = (token.access_token, token.refresh_token, token.id_token, token.account_id)
            req = urllib.request.Request(
                self.base_url + path,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            deadline = time.monotonic() + self.timeout
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("ChatGPT OAuth request exceeded its total deadline")
                with _urlopen_no_redirect(req, timeout=remaining) as response:
                    if catalog_key is not None:
                        self._model_catalog_cache.invalidate_on_etag_mismatch(
                            catalog_key,
                            response.headers.get("X-Models-Etag"),
                        )
                    return _read_response_before_deadline(response, deadline)
            except urllib.error.HTTPError as exc:
                if exc.code == 401 and attempt == 0:
                    exc.close()
                    refresh_after_unauthorized(token)
                    continue
                redacted = _http_error_body(exc, token_values, deadline=deadline)
                raise ChatGPTOAuthUpstreamError(
                    exc.code,
                    f"ChatGPT OAuth request failed: HTTP {exc.code}: {redacted}",
                ) from exc
            except ChatGPTOAuthError:
                raise
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
                raise ChatGPTOAuthUpstreamError(
                    502,
                    f"ChatGPT OAuth transport failed: {redact_text(str(exc), *token_values)}",
                ) from exc
        raise AssertionError("unreachable ChatGPT OAuth request retry state")


def _validate_non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ChatGPTOAuthInvalidRequestError(f"{field} must be a non-empty string")
    return value


def _required_event_text(event: dict[str, Any], event_type: str) -> str:
    value = event.get("text")
    if not isinstance(value, str):
        raise ChatGPTOAuthProtocolError(f"{event_type} event requires string text")
    return value


def _require_catalog_account(
    account_id: str,
    catalog_key: CatalogKey | None,
) -> None:
    if catalog_key is not None and account_id != catalog_key[0]:
        raise ChatGPTOAuthRefreshError("ChatGPT OAuth account changed after model catalog preflight")


def _validate_previous_response_id(value: object) -> str:
    response_id = _validate_non_empty_string(value, "previous_response_id")
    if response_id.strip() == "":
        raise ChatGPTOAuthInvalidRequestError("previous_response_id must be a non-empty string")
    return response_id


def _reject_safety_identifier(_value: object) -> None:
    raise ChatGPTOAuthInvalidRequestError("safety_identifier is not supported by the ChatGPT Codex OAuth transport")


def _reject_prompt_cache_options(_value: object) -> None:
    raise ChatGPTOAuthInvalidRequestError("prompt_cache_options is not supported by the ChatGPT Codex OAuth transport")


def _merge_text_and_verbosity(
    text: dict[str, Any] | None,
    verbosity: str | None,
) -> dict[str, Any] | None:
    if text is not None and not isinstance(text, dict):
        raise ChatGPTOAuthInvalidRequestError("text must be an object")
    merged = dict(text) if text is not None else None
    if merged is not None:
        unknown = sorted(set(merged) - {"format", "verbosity"})
        if unknown:
            raise ChatGPTOAuthInvalidRequestError("text contains unsupported fields: " + ", ".join(unknown))
        if "format" in merged:
            _validate_text_format(merged["format"])
    nested_verbosity = None
    if merged is not None and "verbosity" in merged:
        nested_verbosity = merged["verbosity"]
        if nested_verbosity is None:
            merged.pop("verbosity")
        elif not isinstance(nested_verbosity, str) or nested_verbosity not in KNOWN_VERBOSITY_VALUES:
            raise ChatGPTOAuthInvalidRequestError("text.verbosity must be one of: low, medium, high")
    if verbosity is None:
        return merged
    if not isinstance(verbosity, str) or verbosity not in KNOWN_VERBOSITY_VALUES:
        raise ChatGPTOAuthInvalidRequestError("verbosity must be one of: low, medium, high")
    if nested_verbosity is not None and nested_verbosity != verbosity:
        raise ChatGPTOAuthInvalidRequestError("verbosity conflicts with text.verbosity")
    if merged is None:
        merged = {}
    merged["verbosity"] = verbosity
    return merged


def _validate_text_format(value: object) -> None:
    if not isinstance(value, dict):
        raise ChatGPTOAuthInvalidRequestError("text.format must be an object")
    typ = value.get("type")
    if typ in {"text", "json_object"}:
        unknown = sorted(set(value) - {"type"})
    elif typ == "json_schema":
        unknown = sorted(set(value) - {"type", "name", "description", "schema", "strict"})
        if not isinstance(value.get("schema"), dict):
            raise ChatGPTOAuthInvalidRequestError("text.format.schema must be an object")
        name = value.get("name")
        if name is not None and (not isinstance(name, str) or not name):
            raise ChatGPTOAuthInvalidRequestError("text.format.name must be a non-empty string when provided")
        description = value.get("description")
        if description is not None and not isinstance(description, str):
            raise ChatGPTOAuthInvalidRequestError("text.format.description must be a string when provided")
        strict = value.get("strict")
        if strict is not None and not isinstance(strict, bool):
            raise ChatGPTOAuthInvalidRequestError("text.format.strict must be a boolean when provided")
    else:
        raise ChatGPTOAuthInvalidRequestError("text.format.type must be one of: text, json_object, json_schema")
    if unknown:
        raise ChatGPTOAuthInvalidRequestError("text.format contains unsupported fields: " + ", ".join(unknown))


def _reject_prompt_cache_breakpoint(_value: object) -> None:
    raise ChatGPTOAuthInvalidRequestError(
        "prompt_cache_breakpoint is not supported by the ChatGPT Codex OAuth transport"
    )


def _validate_image_content_items(images: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ChatGPTOAuthInvalidRequestError(f"image reference {index} must be an object")
        image_url = image.get("image_url")
        if not isinstance(image_url, str) or image_url.strip() == "":
            raise ChatGPTOAuthInvalidRequestError(f"image reference {index} requires image_url")
        if not image_url.startswith("data:image/"):
            raise ChatGPTOAuthInvalidRequestError(f"image reference {index} must be a data:image URL")
        unknown = sorted(set(image) - {"image_url", "detail", "prompt_cache_breakpoint"})
        if unknown:
            raise ChatGPTOAuthInvalidRequestError(
                f"image reference {index} contains unsupported fields: " + ", ".join(unknown)
            )
        item: dict[str, Any] = {"type": "input_image", "image_url": image_url}
        if image.get("detail") is not None:
            detail = image["detail"]
            if not isinstance(detail, str) or detail not in KNOWN_IMAGE_DETAILS:
                raise ChatGPTOAuthInvalidRequestError("image detail must be one of: auto, low, high, original")
            item["detail"] = detail
        if image.get("prompt_cache_breakpoint") is not None:
            _reject_prompt_cache_breakpoint(image["prompt_cache_breakpoint"])
        items.append(item)
    return items


def _image_generation_from_item(item: dict[str, Any]) -> dict[str, Any] | None:
    if item.get("type") != "image_generation_call":
        return None
    result = item.get("result")
    if not isinstance(result, str):
        raise ChatGPTOAuthProtocolError("image_generation_call requires a string result")
    item_id = item.get("id")
    if item_id is not None and not isinstance(item_id, str):
        raise ChatGPTOAuthProtocolError("image_generation_call id must be a string or null")
    status = item.get("status")
    if not isinstance(status, str):
        raise ChatGPTOAuthProtocolError("image_generation_call requires a string status")
    revised_prompt = item.get("revised_prompt")
    if revised_prompt is not None and not isinstance(revised_prompt, str):
        raise ChatGPTOAuthProtocolError("image_generation_call revised_prompt must be a string or null")
    image = {
        "status": status,
        "result": result,
    }
    if item_id is not None:
        image["id"] = item_id
    if revised_prompt is not None:
        image["revised_prompt"] = revised_prompt
    return image


def _image_generations_from_response_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    for item in items:
        item_type = item.get("type")
        if item_type == "reasoning":
            _validate_response_output_item(item)
            continue
        if item_type != "image_generation_call":
            raise ChatGPTOAuthProtocolError("image generation response contains an unsupported output item")
        image = _image_generation_from_item(item)
        if image is not None:  # pragma: no branch - item type checked above
            generated.append(image)
    return generated


def _inspection_text_from_response_items(items: Sequence[dict[str, Any]]) -> str:
    for item in items:
        if item.get("type") not in {"reasoning", "message"}:
            raise ChatGPTOAuthProtocolError("image inspection response contains an unsupported output item")
        _validate_response_output_item(item)
    return _text_from_response_items(items).strip()


def _decode_sse_block(lines: list[str]) -> dict[str, Any] | None:
    data_lines = [line[5:].strip() for line in lines if line.startswith("data:")]
    if not data_lines:
        return None
    joined = "\n".join(data_lines)
    if joined == "[DONE]":
        return None
    try:
        event = strict_json_loads(joined)
    except ValueError as exc:
        raise ChatGPTOAuthProtocolError("ChatGPT OAuth SSE event must contain valid JSON") from exc
    if not isinstance(event, dict):
        raise ChatGPTOAuthProtocolError("SSE event JSON must be an object")
    return event


def _validate_response_event(event: object) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ChatGPTOAuthProtocolError("ChatGPT OAuth SSE event must be a JSON object")
    event_type = event.get("type")
    if not isinstance(event_type, str):
        raise ChatGPTOAuthProtocolError("ChatGPT OAuth SSE event requires a string type")
    if event_type == "error":
        raise ChatGPTOAuthUpstreamError(
            502,
            response_failure_message(event, "error"),
        )
    if event_type in _UNSUPPORTED_RESPONSE_EVENT_TYPES:
        raise ChatGPTOAuthProtocolError("ChatGPT OAuth SSE event has an unsupported semantic type")
    if event_type not in _RESPONSE_EVENT_TYPES:
        return event
    if event_type == "response.created":
        if not isinstance(event.get("response"), dict):
            raise ChatGPTOAuthProtocolError("response.created must contain an object response")
    elif event_type == "response.output_item.added":
        item = event.get("item")
        if not isinstance(item, dict):
            raise ChatGPTOAuthProtocolError("response.output_item.added must contain an object item")
        _validate_added_response_output_item(item)
    elif event_type == "response.output_item.done":
        item = event.get("item")
        if not isinstance(item, dict):
            raise ChatGPTOAuthProtocolError(f"{event_type} must contain an object item")
        _validate_response_output_item(item)
    elif event_type in {"response.content_part.added", "response.content_part.done"}:
        part = event.get("part")
        if not isinstance(part, dict):
            raise ChatGPTOAuthProtocolError(f"{event_type} must contain an object part")
        part_type = part.get("type")
        if part_type != "output_text":
            raise ChatGPTOAuthProtocolError(f"{event_type} has an unsupported semantic part type")
        if not isinstance(part.get("text"), str):
            raise ChatGPTOAuthProtocolError(f"{event_type} output_text part requires a text string")
        if "annotations" in part and not isinstance(part["annotations"], list):
            raise ChatGPTOAuthProtocolError(f"{event_type} output_text annotations must be an array")
        if "logprobs" in part and not isinstance(part["logprobs"], list):
            raise ChatGPTOAuthProtocolError(f"{event_type} output_text logprobs must be an array")
    elif event_type == "response.output_text.delta":
        delta = event.get("delta")
        if not isinstance(delta, str):
            raise ChatGPTOAuthProtocolError(f"{event_type} requires a delta string")
    elif event_type == "response.reasoning_summary_text.delta":
        delta = event.get("delta")
        summary_index = event.get("summary_index")
        if not isinstance(delta, str) or not isinstance(summary_index, int) or isinstance(summary_index, bool):
            raise ChatGPTOAuthProtocolError(
                "response.reasoning_summary_text.delta requires a string delta and integer summary_index"
            )
    elif event_type == "response.reasoning_summary_text.done":
        item_id = event.get("item_id")
        text = event.get("text")
        summary_index = event.get("summary_index")
        if (
            not isinstance(item_id, str)
            or not isinstance(text, str)
            or not isinstance(summary_index, int)
            or isinstance(summary_index, bool)
        ):
            raise ChatGPTOAuthProtocolError(
                "response.reasoning_summary_text.done requires string item_id/text and integer summary_index"
            )
    elif event_type == "response.reasoning_text.delta":
        delta = event.get("delta")
        content_index = event.get("content_index")
        if not isinstance(delta, str) or not isinstance(content_index, int) or isinstance(content_index, bool):
            raise ChatGPTOAuthProtocolError(
                "response.reasoning_text.delta requires a string delta and integer content_index"
            )
    elif event_type == "response.reasoning_summary_part.added":
        value = event.get("summary_index")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ChatGPTOAuthProtocolError("response.reasoning_summary_part.added requires integer summary_index")
    elif event_type == "response.completed":
        _validated_completed_response(event)
    return event


def _validate_added_response_output_item(item: dict[str, Any]) -> None:
    item_type = item.get("type")
    if not isinstance(item_type, str) or not item_type:
        raise ChatGPTOAuthProtocolError("response.output_item.added item requires a non-empty string type")
    if item_type == "custom_tool_call":
        raise ChatGPTOAuthProtocolError("custom_tool_call is not supported by the public tool contract")
    _validate_response_item_optional_fields(item, item_type)
    if item_type == "function_call":
        for field in ("name", "arguments", "call_id"):
            if not isinstance(item.get(field), str):
                raise ChatGPTOAuthProtocolError(f"response.output_item.added {item_type} requires string {field}")
        return
    if item_type == "message":
        if not isinstance(item.get("role"), str) or not isinstance(item.get("content"), list):
            raise ChatGPTOAuthProtocolError("response.output_item.added message requires string role and content array")
        for index, part in enumerate(item["content"]):
            if not isinstance(part, dict):
                raise ChatGPTOAuthProtocolError(
                    f"response.output_item.added message content[{index}] must be an object"
                )
            part_type = part.get("type")
            if not isinstance(part_type, str):
                raise ChatGPTOAuthProtocolError(f"response.output_item.added message content[{index}] is invalid")
            value_field = {
                "input_text": "text",
                "output_text": "text",
                "input_image": "image_url",
                "input_audio": "audio_url",
            }.get(part_type)
            if value_field is None or not isinstance(part.get(value_field), str):
                raise ChatGPTOAuthProtocolError(f"response.output_item.added message content[{index}] is invalid")
        return
    if item_type == "reasoning":
        reasoning_from_response_items([item])
        encrypted = item.get("encrypted_content")
        if encrypted is not None and not isinstance(encrypted, str):
            raise ChatGPTOAuthProtocolError(
                "response.output_item.added reasoning encrypted_content must be a string or null"
            )
        return
    if item_type == "web_search_call":
        status = item.get("status")
        action = item.get("action")
        if status is not None and not isinstance(status, str):
            raise ChatGPTOAuthProtocolError(
                "response.output_item.added web_search_call status must be a string or null"
            )
        if action is not None and not isinstance(action, dict):
            raise ChatGPTOAuthProtocolError(
                "response.output_item.added web_search_call action must be an object or null"
            )
        return
    if item_type == "image_generation_call":
        _image_generation_from_item(item)
        return
    raise ChatGPTOAuthProtocolError("response.output_item.added item has an unsupported type")


def _validate_response_output_item(item: dict[str, Any]) -> None:
    item_type = item.get("type")
    if not isinstance(item_type, str) or not item_type:
        raise ChatGPTOAuthProtocolError("response output item requires a non-empty string type")
    if item_type == "custom_tool_call":
        raise ChatGPTOAuthProtocolError("custom_tool_call is not supported by the public tool contract")
    _validate_response_item_optional_fields(item, item_type)
    if item_type == "function_call":
        _tool_call_from_response_item(item)
        return
    if item_type == "image_generation_call":
        _image_generation_from_item(item)
        return
    if item_type == "web_search_call":
        _web_search_event_from_response_item(item)
        return
    if item_type == "reasoning":
        reasoning_from_response_items([item])
        encrypted = item.get("encrypted_content")
        if encrypted is not None and not isinstance(encrypted, str):
            raise ChatGPTOAuthProtocolError(
                "response.output_item.done reasoning encrypted_content must be a string or null"
            )
        return
    if item_type != "message":
        raise ChatGPTOAuthProtocolError("response output item has an unsupported type")
    if item.get("role") != "assistant":
        raise ChatGPTOAuthProtocolError("response message item role must be 'assistant'")
    content = item.get("content")
    if not isinstance(content, list):
        raise ChatGPTOAuthProtocolError("response message item requires a content array")
    for index, part in enumerate(content):
        if not isinstance(part, dict):
            raise ChatGPTOAuthProtocolError(f"response message content[{index}] must be an object")
        part_type = part.get("type")
        if part_type != "output_text":
            raise ChatGPTOAuthProtocolError("response message content has an unsupported type")
        if not isinstance(part.get("text"), str):
            raise ChatGPTOAuthProtocolError(f"response message content[{index}] requires string text")


def _validate_response_item_optional_fields(item: dict[str, Any], item_type: str) -> None:
    item_id = item.get("id")
    if item_id is not None and not isinstance(item_id, str):
        raise ChatGPTOAuthProtocolError(f"{item_type} id must be a string or null")

    metadata = item.get("internal_chat_message_metadata_passthrough")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ChatGPTOAuthProtocolError(
                f"{item_type} internal_chat_message_metadata_passthrough must be an object or null"
            )
        turn_id = metadata.get("turn_id")
        if turn_id is not None and not isinstance(turn_id, str):
            raise ChatGPTOAuthProtocolError(
                f"{item_type} internal_chat_message_metadata_passthrough.turn_id must be a string or null"
            )
        create_time = metadata.get("create_time")
        if create_time is not None and (
            not isinstance(create_time, (int, float))
            or isinstance(create_time, bool)
            or not math.isfinite(create_time)
        ):
            raise ChatGPTOAuthProtocolError(
                f"{item_type} internal_chat_message_metadata_passthrough.create_time must be a JSON number or null"
            )

    def nullable_string(field: str) -> None:
        value = item.get(field)
        if value is not None and not isinstance(value, str):
            raise ChatGPTOAuthProtocolError(f"{item_type} {field} must be a string or null")

    if item_type == "message":
        phase = item.get("phase")
        if phase is not None and phase not in {"commentary", "final_answer"}:
            raise ChatGPTOAuthProtocolError(
                "message phase must be commentary, final_answer, or null"
            )
    elif item_type == "reasoning":
        nullable_string("encrypted_content")
    elif item_type == "function_call":
        nullable_string("namespace")
        encrypted_args = item.get("encrypted_function_args")
        if encrypted_args is not None and (
            not isinstance(encrypted_args, list)
            or any(not isinstance(value, str) for value in encrypted_args)
        ):
            raise ChatGPTOAuthProtocolError(
                "function_call encrypted_function_args must be a string array or null"
            )
    elif item_type == "web_search_call":
        nullable_string("status")
    elif item_type == "image_generation_call":
        nullable_string("revised_prompt")


def _validated_completed_response(event: dict[str, Any]) -> dict[str, Any]:
    response = event.get("response")
    if not isinstance(response, dict):
        raise ChatGPTOAuthProtocolError("response.completed must contain a response with a non-empty id")
    response_id = response.get("id")
    if not isinstance(response_id, str) or response_id == "":
        raise ChatGPTOAuthProtocolError("response.completed must contain a response with a non-empty id")
    if response.get("end_turn") is not None and not isinstance(response["end_turn"], bool):
        raise ChatGPTOAuthProtocolError("response.completed response.end_turn must be a boolean or null")
    _usage_from_response(response.get("usage"))
    return response


def _split_instructions_and_input(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    input_messages: list[Message] = []
    for msg in messages:
        if msg.role is MessageRole.SYSTEM and not msg.content.startswith(REMOTE_COMPACTION_MARKER):
            if msg.content_parts is not None and any(
                part.get("prompt_cache_breakpoint") is not None for part in msg.content_parts
            ):
                _reject_prompt_cache_breakpoint(None)
            instructions.append(msg.content)
        else:
            input_messages.append(msg)
    return "\n\n".join(instructions), _messages_to_response_items(input_messages)


def _messages_to_response_items(messages: Sequence[Message]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role is MessageRole.SYSTEM and message.content.startswith(REMOTE_COMPACTION_MARKER):
            raw = message.content[len(REMOTE_COMPACTION_MARKER) :].strip()
            try:
                parsed = strict_json_loads(raw)
            except ValueError as exc:
                raise ChatGPTOAuthInvalidRequestError("remote compaction marker contains invalid JSON") from exc
            if not isinstance(parsed, list):
                raise ChatGPTOAuthInvalidRequestError("remote compaction marker must contain a response item array")
            items.extend(_filter_compacted_history_items(parsed, source="marker"))
            continue
        if message.role is MessageRole.TOOL:
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
            continue
        if message.role is MessageRole.ASSISTANT and message.tool_calls:
            if message.content or message.content_parts is not None:
                items.append(
                    _message_item(
                        "assistant",
                        message.content,
                        content_parts=message.content_parts,
                    )
                )
            for tool_call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                )
            continue
        if message.role is MessageRole.ASSISTANT:
            role = "assistant"
        elif message.role is MessageRole.DEVELOPER:
            role = "developer"
        elif message.role is MessageRole.USER:
            role = "user"
        else:
            raise ChatGPTOAuthInvalidRequestError(f"unsupported internal message role {message.role.value!r}")
        items.append(
            _message_item(
                role,
                message.content,
                message.images,
                message.content_parts,
            )
        )
    return items


def _message_item(
    role: str,
    content: str,
    images: tuple[str, ...] = (),
    content_parts: tuple[dict[str, object], ...] | None = None,
) -> dict[str, Any]:
    if content_parts is not None:
        if any(part.get("prompt_cache_breakpoint") is not None for part in content_parts):
            _reject_prompt_cache_breakpoint(None)
        normalized_parts: list[dict[str, object]] = []
        for part in content_parts:
            normalized = dict(part)
            if normalized.get("prompt_cache_breakpoint") is None:
                normalized.pop("prompt_cache_breakpoint", None)
            if normalized.get("type") == "input_image" and normalized.get("detail") is None:
                normalized.pop("detail", None)
            normalized_parts.append(normalized)
        return {
            "type": "message",
            "role": role,
            "content": normalized_parts,
        }
    typ = "output_text" if role == "assistant" else "input_text"
    content_items: list[dict[str, Any]] = [{"type": typ, "text": content}]
    for image_url in images:
        content_items.append({"type": "input_image", "image_url": image_url})
    return {"type": "message", "role": role, "content": content_items}


def _tool_schema_to_response_dict(tool: ToolSchema) -> dict[str, Any]:
    if tool.parameters.get("__codex_as_api_tool_type") == "web_search":
        openai_tool = tool.parameters.get("openai_tool")
        if not isinstance(openai_tool, dict):
            raise ChatGPTOAuthInvalidRequestError("web_search tool requires an openai_tool object")
        return dict(openai_tool)
    result: dict[str, Any] = {
        "type": "function",
        "name": tool.name,
        "parameters": tool.parameters,
        "strict": tool.strict,
    }
    if tool.description is not None:
        result["description"] = tool.description
    return result


def _finalize_responses_payload(
    payload: dict[str, Any],
    *,
    capability: ModelCapability,
    reasoning_effort: str | None,
    reasoning_mode: str | None = None,
    reasoning_context: str | None = None,
    text: dict[str, Any] | None = None,
    verbosity: str | None = None,
    service_tier: str | None = None,
    responses_lite: bool | str | None = None,
    include_encrypted_content: bool = True,
    safety_identifier: str | None = None,
    prompt_cache_options: dict[str, Any] | None = None,
) -> None:
    _require_input_modalities(capability, payload)
    if not capability.supports_image_detail_original and _has_original_image_detail(payload):
        raise ChatGPTOAuthInvalidRequestError("image detail 'original' is not supported for the requested model")
    if safety_identifier is not None:
        _reject_safety_identifier(safety_identifier)
    if prompt_cache_options is not None:
        _reject_prompt_cache_options(prompt_cache_options)
    merged_text = _merge_text_and_verbosity(text, verbosity)
    try:
        apply_model_capability_fields(
            payload,
            capability=capability,
            text=merged_text,
            service_tier=service_tier,
        )
    except ValueError as exc:
        raise ChatGPTOAuthInvalidRequestError(str(exc)) from exc
    effective_effort = resolve_model_reasoning_effort(
        capability,
        reasoning_effort if reasoning_effort is not None else capability.default_reasoning_level,
    )
    _set_reasoning_payload(
        payload,
        effective_effort,
        reasoning_mode=reasoning_mode,
        reasoning_context=reasoning_context,
        include_encrypted_content=include_encrypted_content,
    )
    reasoning = payload.get("reasoning")
    if capability.supports_reasoning_summary_parameter and capability.default_reasoning_summary != "none":
        if reasoning is None:
            reasoning = {}
            payload["reasoning"] = reasoning
        if not isinstance(reasoning, dict):
            raise ChatGPTOAuthInvalidRequestError("reasoning must be an object")
        reasoning["summary"] = capability.default_reasoning_summary
        if include_encrypted_content:
            _ensure_reasoning_encrypted_content(payload)
    elif isinstance(reasoning, dict):
        reasoning.pop("summary", None)
    if include_encrypted_content:
        _ensure_reasoning_encrypted_content(payload)
    try:
        lite = use_responses_lite(capability, responses_lite)
    except ValueError as exc:
        raise ChatGPTOAuthInvalidRequestError(str(exc)) from exc
    if not lite:
        return
    if payload.get("parallel_tool_calls") is True:
        raise ChatGPTOAuthInvalidRequestError("Responses Lite cannot preserve parallel_tool_calls=true")
    if _has_any_image_detail(payload):
        raise ChatGPTOAuthInvalidRequestError("Responses Lite cannot preserve image detail")

    if reasoning_context is not None and reasoning_context != "all_turns":
        raise ChatGPTOAuthInvalidRequestError(
            "Responses Lite requires reasoning.context to be all_turns when explicitly provided"
        )

    raw_tools = payload.get("tools")
    if not isinstance(raw_tools, list) or any(not isinstance(tool, dict) for tool in raw_tools):
        raise ChatGPTOAuthInvalidRequestError("Responses Lite tools must be an object array")
    tools_payload = [dict(tool) for tool in raw_tools]
    _apply_responses_lite_payload(payload, tools_payload)


def _has_original_image_detail(value: object) -> bool:
    return any(part.get("detail") == "original" for part in _input_image_parts(value))


def _has_any_image_detail(value: object) -> bool:
    return any(part.get("detail") is not None for part in _input_image_parts(value))


def _has_input_images(value: object) -> bool:
    return next(iter(_input_image_parts(value)), None) is not None


def _input_image_parts(value: object) -> Iterator[dict[str, Any]]:
    input_items = value.get("input") if isinstance(value, dict) else value
    if not isinstance(input_items, list):
        return
    for item in input_items:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "input_image":
                yield part


def _require_input_modalities(capability: ModelCapability, payload: dict[str, Any]) -> None:
    required: set[str] = set()

    def collect_parts(parts: object) -> None:
        if not isinstance(parts, list):
            return
        for part in parts:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"input_text", "output_text"}:
                required.add("text")
            elif part_type == "input_image":
                required.add("image")
            elif part_type == "input_audio":
                required.add("audio")

    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions:
        required.add("text")
    input_items = payload.get("input")
    if isinstance(input_items, list):
        for item in input_items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                collect_parts(item.get("content"))
            elif item_type in {"function_call_output", "custom_tool_call_output"}:
                output = item.get("output")
                if isinstance(output, str):
                    required.add("text")
                else:
                    collect_parts(output)
    unsupported = required.difference(capability.input_modalities)
    if unsupported:
        modality = sorted(unsupported)[0]
        raise ChatGPTOAuthInvalidRequestError(
            f"the requested model does not accept {modality} input"
        )


def resolve_model_reasoning_effort(
    capability: ModelCapability,
    requested: str | None,
) -> str | None:
    if requested is None:
        return None
    if not isinstance(requested, str) or not requested or requested != requested.strip():
        raise ChatGPTOAuthInvalidRequestError("reasoning_effort must be a non-empty string when provided")
    supported = capability.reasoning_effort_ids
    if requested == "ultra":
        preferred = capability.multi_agent_reasoning_effort
        if preferred is not None and preferred != "ultra" and preferred in supported:
            return preferred
        if "max" in supported:
            return "max"
        non_ultra = [effort for effort in supported if effort != "ultra"]
        if non_ultra:
            return non_ultra[-1]
        raise ChatGPTOAuthInvalidRequestError("the requested model has no wire reasoning effort for ultra")
    if requested not in supported:
        raise ChatGPTOAuthInvalidRequestError("reasoning_effort is not supported for the requested model")
    return "disabled" if requested == "persistent" else requested


def _reject_unsupported_stop(stop: Sequence[str] | None) -> None:
    if stop is None:
        return
    raise ChatGPTOAuthInvalidRequestError("stop is not supported by the private Codex OAuth HTTP transport")


def _apply_responses_lite_payload(payload: dict[str, Any], tools_payload: Sequence[dict[str, Any]]) -> None:
    raw_tools = payload.get("tools")
    if not isinstance(raw_tools, list) or any(not isinstance(tool, dict) for tool in raw_tools):
        raise ChatGPTOAuthInvalidRequestError("Responses Lite tools must be an object array")
    hosted_tool_types = sorted(
        {str(tool["type"]) for tool in tools_payload if tool.get("type") in {"web_search", "image_generation"}}
    )
    if hosted_tool_types:
        raise ChatGPTOAuthInvalidRequestError(
            "Responses Lite cannot execute hosted tools without a standalone runtime: "
            + ", ".join(hosted_tool_types)
            + f"; set {RESPONSES_LITE_ENV}=off to use classic Responses"
        )

    if "instructions" in payload:
        instructions = payload.pop("instructions")
        if not isinstance(instructions, str):
            raise ChatGPTOAuthInvalidRequestError("Responses Lite instructions must be a string")
    else:
        instructions = ""
    payload.pop("tools")
    if "tool_choice" in payload and payload["tool_choice"] != "auto":
        raise ChatGPTOAuthInvalidRequestError("Responses Lite tool_choice must be the exact string 'auto'")
    payload["parallel_tool_calls"] = False
    input_items = payload.get("input")
    if not isinstance(input_items, list):
        raise ChatGPTOAuthInvalidRequestError("Responses Lite input must be an array")
    developer_items: list[dict[str, Any]] = [
        {
            "type": "additional_tools",
            "role": "developer",
            "tools": list(tools_payload),
        }
    ]
    if instructions:
        developer_items.append(
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": instructions}],
            }
        )
    payload["input"] = [*developer_items, *input_items]
    if "reasoning" not in payload:
        reasoning: dict[str, Any] = {}
        payload["reasoning"] = reasoning
    else:
        reasoning = payload["reasoning"]
    if not isinstance(reasoning, dict):
        raise ChatGPTOAuthInvalidRequestError("Responses Lite reasoning must be an object")
    reasoning["context"] = "all_turns"
    payload["_codex_as_api_responses_lite"] = True


def _ensure_reasoning_encrypted_content(payload: dict[str, Any]) -> None:
    if "include" not in payload:
        include: list[Any] = []
        payload["include"] = include
    else:
        include = payload["include"]
    if not isinstance(include, list):
        raise ChatGPTOAuthInvalidRequestError("include must be an array")
    if "reasoning.encrypted_content" not in include:
        include.append("reasoning.encrypted_content")


def _responses_transport_headers(
    payload: dict[str, Any],
    *,
    catalog_key: CatalogKey | None = None,
) -> _TransportHeaders:
    if payload.pop("_codex_as_api_responses_lite", False) is True:
        return _TransportHeaders(
            {LITE_HEADER_NAME: LITE_HEADER_VALUE},
            catalog_key=catalog_key,
        )
    return _TransportHeaders(catalog_key=catalog_key)


def _compact_raw_events(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [event for event in events if event.get("type") == "web_search_call"]
    for event in events[-20:]:
        if event not in keep:
            keep.append(event)
    return keep


_CONTEXTUAL_USER_MARKER_PAIRS = (
    ("# agents.md instructions", "</instructions>"),
    ("<environment_context>", "</environment_context>"),
    ("<skill>", "</skill>"),
    ("<user_shell_command>", "</user_shell_command>"),
    ("<turn_aborted>", "</turn_aborted>"),
    ("<subagent_notification>", "</subagent_notification>"),
    ("<recommended_plugins>", "</recommended_plugins>"),
)
_HOOK_PROMPT_RE = re.compile(
    r'^<hook_prompt\s+[^>]*hook_run_id="([^"]+)"[^>]*>[\s\S]*</hook_prompt>$',
    re.IGNORECASE,
)
_EXTERNAL_CONTEXT_RE = re.compile(r"^<external_([^>]+)>[\s\S]*</external_([^>]+)>$")
_INTERNAL_CONTEXT_RE = re.compile(
    r'^<codex_internal_context source="[a-z][a-z0-9_]*">[\s\S]*</codex_internal_context>$'
)


def _is_hook_prompt_text(text: str) -> bool:
    match = _HOOK_PROMPT_RE.fullmatch(text.strip())
    return match is not None and match.group(1).strip() != ""


def _is_contextual_user_text(text: str) -> bool:
    trimmed = text.strip()
    lowered = trimmed.lower()
    if any(lowered.startswith(start) and lowered.endswith(end) for start, end in _CONTEXTUAL_USER_MARKER_PAIRS):
        return True
    external = _EXTERNAL_CONTEXT_RE.fullmatch(trimmed)
    if external is not None and external.group(1) == external.group(2):
        return True
    if _INTERNAL_CONTEXT_RE.fullmatch(trimmed) is not None:
        return True
    if lowered.startswith("<goal_context>") and lowered.endswith("</goal_context>"):
        return True
    return (
        trimmed.startswith("Warning: The maximum number of unified exec processes you can keep open is")
        or (
            trimmed.startswith("Warning: apply_patch was requested via ")
            and trimmed.endswith("Use the apply_patch tool instead of exec_command.")
        )
        or trimmed.startswith("Warning: Your account was flagged for potentially high-risk cyber activity")
    )


def _is_real_user_or_hook_message(content: list[dict[str, Any]]) -> bool:
    texts = [part["text"] for part in content if part.get("type") == "input_text" and isinstance(part.get("text"), str)]
    has_visible_hook = any(_is_hook_prompt_text(text) for text in texts)
    if (
        has_visible_hook
        and len(texts) == len(content)
        and all(_is_hook_prompt_text(text) or _is_contextual_user_text(text) for text in texts)
    ):
        return True
    return not any(_is_hook_prompt_text(text) or _is_contextual_user_text(text) for text in texts)


def _validate_compacted_history_item(
    item: dict[str, Any],
    *,
    index: int,
    source: str,
) -> None:
    item_type = item.get("type")
    if not isinstance(item_type, str):
        _raise_compaction_error(
            source,
            f"remote compact {source} item {index} requires a string type",
        )
    try:
        _validate_response_item_optional_fields(item, item_type)
    except ChatGPTOAuthProtocolError as exc:
        _raise_compaction_error(
            source,
            f"remote compact {source} {item_type} item {index} is invalid: {exc}",
        )
    if item_type == "agent_message":
        author = item.get("author")
        recipient = item.get("recipient")
        content = item.get("content")
        if not isinstance(author, str) or not isinstance(recipient, str) or not isinstance(content, list):
            _raise_compaction_error(
                source,
                f"remote compact {source} agent_message item {index} requires "
                "string author/recipient and content array",
            )
        for part_index, part in enumerate(content):
            if not isinstance(part, dict):
                _raise_compaction_error(
                    source,
                    f"remote compact {source} agent_message item {index} content part {part_index} must be an object",
                )
            part_type = part.get("type")
            valid_text = part_type == "input_text" and isinstance(part.get("text"), str)
            valid_encrypted = part_type == "encrypted_content" and isinstance(part.get("encrypted_content"), str)
            if not valid_text and not valid_encrypted:
                _raise_compaction_error(
                    source, f"remote compact {source} agent_message item {index} content part {part_index} is invalid"
                )
        return
    if item_type in {"compaction", "compaction_summary"}:
        if not isinstance(item.get("encrypted_content"), str):
            _raise_compaction_error(
                source, f"remote compact {source} {item_type} item {index} requires string encrypted_content"
            )
        return
    if item_type == "context_compaction":
        encrypted_content = item.get("encrypted_content")
        if encrypted_content is not None and not isinstance(encrypted_content, str):
            _raise_compaction_error(
                source, f"remote compact {source} context_compaction item {index} encrypted_content must be a string"
            )
        return
    if item_type == "additional_tools":
        tools = item.get("tools")
        if (
            item.get("role") != "developer"
            or not isinstance(tools, list)
            or any(not isinstance(tool, dict) for tool in tools)
        ):
            _raise_compaction_error(
                source,
                f"remote compact {source} additional_tools item {index} requires developer role and object tools",
            )
        return
    if item_type in {"reasoning", "function_call"}:
        try:
            _validate_response_output_item(item)
        except ChatGPTOAuthProtocolError as exc:
            _raise_compaction_error(
                source,
                f"remote compact {source} {item_type} item {index} is invalid: {exc}",
            )
        return
    if item_type != "message":
        _raise_compaction_error(
            source,
            f"remote compact {source} item {index} has an unsupported type",
        )
    role = item.get("role")
    if role not in {"user", "assistant", "developer"}:
        _raise_compaction_error(
            source,
            f"remote compact {source} message item {index} has an unsupported role",
        )
    content = item.get("content")
    if not isinstance(content, list):
        _raise_compaction_error(source, f"remote compact {source} message item {index} must have a content array")
    for part_index, part in enumerate(content):
        if not isinstance(part, dict):
            _raise_compaction_error(
                source, f"remote compact {source} message item {index} content part {part_index} must be an object"
            )
        part_type = part.get("type")
        valid_text = part_type in {"input_text", "output_text"} and isinstance(part.get("text"), str)
        valid_image = part_type == "input_image" and isinstance(part.get("image_url"), str)
        valid_audio = part_type == "input_audio" and isinstance(part.get("audio_url"), str)
        if valid_image:
            detail = part.get("detail")
            valid_image = detail is None or (isinstance(detail, str) and detail in {"auto", "low", "high", "original"})
        if not valid_text and not valid_image and not valid_audio:
            _raise_compaction_error(
                source, f"remote compact {source} message item {index} content part {part_index} is invalid"
            )


def _raise_compaction_error(source: str, message: str) -> NoReturn:
    if source == "output":
        raise ChatGPTOAuthProtocolError(message)
    raise ChatGPTOAuthInvalidRequestError(message)


def _filter_compacted_history_items(
    items: Sequence[Any],
    *,
    source: str = "output",
) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            _raise_compaction_error(
                source,
                f"remote compact {source} item {index} must be an object",
            )
        _validate_compacted_history_item(item, index=index, source=source)
        item_type = item.get("type")
        role = item.get("role")
        keep = (
            (item_type == "message" and role == "assistant")
            or (item_type == "message" and role == "user" and _is_real_user_or_hook_message(item["content"]))
            or item_type in {"agent_message", "compaction", "compaction_summary", "context_compaction"}
        )
        if keep:
            compacted.append(item)
    return compacted


def _web_search_event_from_response_item(
    item: dict[str, Any],
) -> dict[str, Any] | None:
    if item.get("type") != "web_search_call":
        return None
    tool_id = item.get("id")
    if not isinstance(tool_id, str):
        raise ChatGPTOAuthProtocolError("web_search_call requires a string id")
    raw_action = item.get("action")
    if not isinstance(raw_action, dict):
        raise ChatGPTOAuthProtocolError("web_search_call requires an action object")
    action = cast(dict[str, Any], raw_action)
    sources = _web_search_sources_from_action(action)
    return {
        "type": "web_search_call",
        "id": tool_id,
        "input": {"query": _web_search_query_from_action(action)},
        "content": sources,
    }


def _web_search_query_from_action(action: dict[str, Any]) -> str:
    action_type = action.get("type")
    if not isinstance(action_type, str):
        raise ChatGPTOAuthProtocolError("web_search_call action type must be a string")
    if action_type != "search":
        raise ChatGPTOAuthProtocolError(
            f"web_search_call action type {action_type!r} cannot be represented by this facade"
        )

    queries = action.get("queries")
    if queries is not None:
        if not isinstance(queries, list):
            raise ChatGPTOAuthProtocolError("web_search_call action queries must be an array")
        if any(not isinstance(value, str) for value in queries):
            raise ChatGPTOAuthProtocolError("web_search_call action queries must contain only strings")
    query = action.get("query")
    if query is not None and not isinstance(query, str):
        raise ChatGPTOAuthProtocolError("web_search_call action query must be a string")
    if isinstance(queries, list) and len(queries) > 1:
        raise ChatGPTOAuthProtocolError(
            "web_search_call action contains multiple queries that cannot be represented by this facade"
        )
    if isinstance(query, str):
        if isinstance(queries, list) and queries and queries[0] != query:
            raise ChatGPTOAuthProtocolError("web_search_call action query conflicts with queries")
        return query
    if isinstance(queries, list) and queries:
        return cast(str, queries[0])
    raise ChatGPTOAuthProtocolError("web_search_call action requires a query")


def _web_search_sources_from_action(action: dict[str, Any]) -> list[dict[str, Any]]:
    sources = action.get("sources")
    if not isinstance(sources, list):
        raise ChatGPTOAuthProtocolError("web_search_call action requires a sources array")
    return _normalize_web_search_sources(sources)


def _normalize_web_search_sources(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ChatGPTOAuthProtocolError("web search sources must be an array")
    out: list[dict[str, Any]] = []
    for index, source in enumerate(value):
        if not isinstance(source, dict):
            raise ChatGPTOAuthProtocolError(f"web search source {index} must be an object")
        url = source.get("url")
        if not isinstance(url, str) or not url:
            raise ChatGPTOAuthProtocolError(f"web search source {index} requires a non-empty url")
        title = source.get("title")
        if title is not None and not isinstance(title, str):
            raise ChatGPTOAuthProtocolError(f"web search source {index} title must be a string or null")
        result: dict[str, Any] = {
            "type": "web_search_result",
            "url": url,
        }
        if title is not None:
            result["title"] = title
        page_age = source.get("page_age")
        if page_age is not None and not isinstance(page_age, str):
            raise ChatGPTOAuthProtocolError(f"web search source {index} page_age must be a string or null")
        if page_age is not None:
            result["page_age"] = page_age
        out.append(result)
    return out


def _set_reasoning_payload(
    payload: dict[str, Any],
    reasoning_effort: str | None,
    *,
    reasoning_mode: str | None = None,
    reasoning_context: str | None = None,
    include_encrypted_content: bool = True,
) -> None:
    if reasoning_effort is None and reasoning_mode is None and reasoning_context is None:
        return
    existing = payload.get("reasoning")
    if "reasoning" in payload and not isinstance(existing, dict):
        raise ChatGPTOAuthInvalidRequestError("reasoning must be an object")
    reasoning = dict(existing) if isinstance(existing, dict) else {}
    existing_mode = reasoning.get("mode")
    if existing_mode is not None:
        if not isinstance(existing_mode, str) or existing_mode not in KNOWN_REASONING_MODES:
            raise ChatGPTOAuthInvalidRequestError("reasoning.mode must be one of: standard, pro")
        raise ChatGPTOAuthInvalidRequestError("reasoning.mode is not supported by the ChatGPT Codex OAuth transport")
    reasoning.pop("mode", None)
    if reasoning_effort is not None:
        if not isinstance(reasoning_effort, str) or reasoning_effort == "":
            raise ChatGPTOAuthInvalidRequestError("reasoning_effort must be a non-empty string when provided")
        reasoning["effort"] = reasoning_effort
    if reasoning_mode is not None:
        if not isinstance(reasoning_mode, str) or reasoning_mode not in KNOWN_REASONING_MODES:
            raise ChatGPTOAuthInvalidRequestError("reasoning.mode must be one of: standard, pro")
        raise ChatGPTOAuthInvalidRequestError("reasoning.mode is not supported by the ChatGPT Codex OAuth transport")
    if reasoning_context is not None:
        if not isinstance(reasoning_context, str) or reasoning_context not in KNOWN_REASONING_CONTEXTS:
            raise ChatGPTOAuthInvalidRequestError("reasoning.context must be one of: auto, current_turn, all_turns")
        reasoning["context"] = reasoning_context
    if reasoning:
        payload["reasoning"] = reasoning
    if reasoning and include_encrypted_content:
        _ensure_reasoning_encrypted_content(payload)


def _validate_positive_finite(value: float, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return float(value)


def _tool_call_from_response_item(item: dict[str, Any]) -> ToolCall | None:
    item_type = item.get("type")
    if item_type == "custom_tool_call":
        raise ChatGPTOAuthProtocolError("custom_tool_call is not supported by the public tool contract")
    if item_type != "function_call":
        return None
    name = item.get("name")
    if not isinstance(name, str):
        raise ChatGPTOAuthProtocolError(f"{item.get('type')} requires a string name")
    field = "arguments"
    raw_args = item.get(field)
    if not isinstance(raw_args, str):
        raise ChatGPTOAuthProtocolError(f"{item_type} requires a string {field}")
    call_id = item.get("call_id")
    if not isinstance(call_id, str):
        raise ChatGPTOAuthProtocolError(f"{item_type} requires a string call_id")
    return ToolCall(id=call_id, name=name, arguments=raw_args)


def _text_from_response_items(items: Sequence[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in items:
        item_type = item.get("type")
        if item_type != "message":
            if item_type not in {
                "function_call",
                "image_generation_call",
                "reasoning",
                "web_search_call",
            }:
                raise ChatGPTOAuthProtocolError("response output item has an unsupported type")
            continue
        if item.get("role") != "assistant":
            raise ChatGPTOAuthProtocolError("response message item role must be 'assistant'")
        content = item.get("content")
        if not isinstance(content, list):
            raise ChatGPTOAuthProtocolError("response message item requires a content array")
        for index, part in enumerate(content):
            if not isinstance(part, dict):
                raise ChatGPTOAuthProtocolError(f"response message content[{index}] must be an object")
            part_type = part.get("type")
            if part_type != "output_text":
                raise ChatGPTOAuthProtocolError("response message content has an unsupported type")
            text = part.get("text")
            if not isinstance(text, str):
                raise ChatGPTOAuthProtocolError(f"response message content[{index}] requires string text")
            if text:
                parts.append(text)
    return "".join(parts)


def _usage_from_response(value: Any) -> Usage | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ChatGPTOAuthProtocolError("response usage must be an object")
    unsupported_aliases = sorted(
        set(value)
        & {
            "prompt_tokens",
            "completion_tokens",
            "prompt_tokens_details",
            "cached_input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        }
    )
    if unsupported_aliases:
        raise ChatGPTOAuthProtocolError(
            "response usage contains unsupported token aliases: " + ", ".join(unsupported_aliases)
        )
    prompt = value.get("input_tokens")
    completion = value.get("output_tokens")
    total = value.get("total_tokens")
    if (
        not isinstance(prompt, int)
        or isinstance(prompt, bool)
        or prompt < 0
        or not isinstance(completion, int)
        or isinstance(completion, bool)
        or completion < 0
    ):
        raise ChatGPTOAuthProtocolError("response usage requires non-negative integer input_tokens and output_tokens")
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        raise ChatGPTOAuthProtocolError("response usage requires non-negative integer total_tokens")
    if total != prompt + completion:
        raise ChatGPTOAuthProtocolError("response usage total_tokens must equal input_tokens plus output_tokens")
    token_details = value.get("input_tokens_details")
    if token_details is not None and not isinstance(token_details, dict):
        raise ChatGPTOAuthProtocolError("response usage input_tokens_details must be an object or null")
    cached_tokens = None
    cache_write_tokens = None
    if isinstance(token_details, dict):
        cached_tokens = _usage_integer(
            token_details.get("cached_tokens"),
            "cached_tokens",
        )
        if token_details.get("cache_write_tokens") is not None:
            cache_write_tokens = _usage_integer(
                token_details["cache_write_tokens"],
                "cache_write_tokens",
            )
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def _usage_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ChatGPTOAuthProtocolError(f"response usage {field} must be a non-negative integer")
    return value
