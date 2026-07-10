from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import threading
import urllib.error
import urllib.request
import uuid
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from copy import deepcopy
from typing import Any, cast

from .auth import (
    ChatGPTOAuthError,
    ChatGPTOAuthInvalidRequestError,
    load_token_data,
    redact_text,
    refresh_token,
    register_token_secrets,
)
from .messages import AssistantResponse, Message, MessageRole, ToolCall, ToolSchema, Usage
from .model_capabilities import (
    LITE_HEADER_NAME,
    LITE_HEADER_VALUE,
    RESPONSES_LITE_ENV,
    apply_model_capability_fields,
    build_codex_client_metadata,
    capability_for_model,
    resolve_codex_metadata_enabled,
    resolve_model_for_backend,
    should_enable_parallel_tool_calls,
    strip_image_detail_fields,
    use_responses_lite,
)
from .protocol import (
    reasoning_from_response_items,
    response_failure_message,
)

CHATGPT_OAUTH_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
CHATGPT_OAUTH_DEFAULT_MODEL = "gpt-5.5"
REMOTE_COMPACTION_MARKER = "[Remote Responses compacted history]"
CODEX_CLI_ORIGINATOR = "codex_cli_rs"
CODEX_CLI_VERSION_ENV = "CODEX_AS_API_CODEX_CLI_VERSION"
CODEX_CLI_NPM_PACKAGE = "@openai/codex"
KNOWN_REASONING_EFFORT_VALUES = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"})
KNOWN_REASONING_MODES = frozenset({"standard", "pro"})
KNOWN_REASONING_CONTEXTS = frozenset({"auto", "current_turn", "all_turns"})
KNOWN_IMAGE_DETAILS = frozenset({"auto", "low", "high", "original"})
KNOWN_VERBOSITY_VALUES = frozenset({"low", "medium", "high"})
RESPONSE_CHAIN_CAPACITY = 256
_CODEX_CLI_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}(?:[-+][0-9A-Za-z.-]+)?$")
_CODEX_CLI_VERSION_UNSET = object()
_codex_cli_version_cache: str | None | object = _CODEX_CLI_VERSION_UNSET
_codex_cli_version_lock = threading.Lock()


class _ResponseChainStore:
    """Thread-safe, process-local replay history for public response IDs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._chains: OrderedDict[
            str,
            tuple[list[dict[str, Any]], list[dict[str, Any]]],
        ] = OrderedDict()

    def resolve(self, response_id: str) -> list[dict[str, Any]]:
        with self._lock:
            chain = self._chains.get(response_id)
            if chain is None:
                raise ChatGPTOAuthInvalidRequestError(
                    f"previous_response_id {response_id!r} is unknown or has been evicted"
                )
            self._chains.move_to_end(response_id)
            request_input, response_output = chain
            return deepcopy([*request_input, *response_output])

    def commit(
        self,
        response_id: str,
        request_input: Sequence[dict[str, Any]],
        response_output: Sequence[dict[str, Any]],
    ) -> None:
        with self._lock:
            self._chains[response_id] = (
                deepcopy(list(request_input)),
                deepcopy(list(response_output)),
            )
            self._chains.move_to_end(response_id)
            while len(self._chains) > RESPONSE_CHAIN_CAPACITY:
                self._chains.popitem(last=False)


def prime_codex_cli_version_cache() -> None:
    resolve_codex_cli_version()


def resolve_codex_cli_version() -> str | None:
    override = _normalize_codex_cli_version(os.getenv(CODEX_CLI_VERSION_ENV))
    if override:
        return override
    global _codex_cli_version_cache
    with _codex_cli_version_lock:
        if _codex_cli_version_cache is not _CODEX_CLI_VERSION_UNSET:
            return _codex_cli_version_cache if isinstance(_codex_cli_version_cache, str) else None
        _codex_cli_version_cache = _fetch_latest_codex_cli_version_from_npm()
        return _codex_cli_version_cache if isinstance(_codex_cli_version_cache, str) else None


def _fetch_latest_codex_cli_version_from_npm() -> str | None:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed executable and arguments.
            ["npm", "view", CODEX_CLI_NPM_PACKAGE, "version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return _normalize_codex_cli_version(completed.stdout)


def _normalize_codex_cli_version(value: str | None) -> str | None:
    version = value.strip() if value else ""
    if not version:
        return None
    return version if _CODEX_CLI_VERSION_RE.match(version) else None


def codex_cli_headers_for_version(version: str | None) -> dict[str, str]:
    headers = {"originator": CODEX_CLI_ORIGINATOR}
    normalized = _normalize_codex_cli_version(version)
    if normalized:
        headers["User-Agent"] = _sanitize_header_value(
            f"{CODEX_CLI_ORIGINATOR}/{normalized} ({_codex_os_info()}) codex-as-api"
        )
    return headers


def _codex_cli_headers() -> dict[str, str]:
    return codex_cli_headers_for_version(resolve_codex_cli_version())


def _codex_os_info() -> str:
    return f"{_codex_os_name()} {platform.release() or 'unknown'}; {platform.machine() or 'unknown'}"


def _codex_os_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "Mac OS"
    return system or "unknown"


def _sanitize_header_value(value: str) -> str:
    return "".join(ch if " " <= ch <= "~" else "_" for ch in value)


class ChatGPTOAuthProvider:
    name: str = "chatgpt_oauth"
    provider_namespace: str = "agent.provider.chatgpt_oauth"
    supports_prompt_cache_key: bool = True

    def __init__(
        self,
        *,
        model: str = CHATGPT_OAUTH_DEFAULT_MODEL,
        base_url: str = CHATGPT_OAUTH_DEFAULT_BASE_URL,
        auth_json_path: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.auth_json_path = auth_json_path
        # ``None`` is intentional: Codex Responses can run for minutes while it
        # streams thinking/tool progress.  A client-side read timeout aborts a
        # still-healthy turn and leaves workflow state half-transitioned.
        self.timeout = timeout
        self.api_key = None
        self._active_response_lock = threading.Lock()
        self._active_responses: set[Any] = set()
        self._response_chains = _ResponseChainStore()

    def cancel_current_requests(self) -> None:
        with self._active_response_lock:
            responses = list(self._active_responses)
        for response in responses:
            try:
                response.close()
            except Exception:
                pass

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
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Prepare and validate one request without opening an upstream stream."""
        replay_input: list[dict[str, Any]] = []
        payload = self._responses_payload(
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
            _history_input_target=replay_input,
        )
        return payload, replay_input

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
    ) -> AssistantResponse:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason = "stop"
        raw_events: list[dict[str, Any]] = []
        usage: Usage | None = None
        response_id: str | None = None
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
        ):
            raw_events.append(dict(event))
            if event.get("type") == "content":
                content_parts.append(str(event.get("text", "")))
            elif event.get("type") in {"reasoning_delta", "reasoning_raw_delta"}:
                reasoning_parts.append(str(event.get("text", "")))
            elif event.get("type") == "tool_call":
                tool_calls.append(
                    ToolCall(id=str(event["id"]), name=str(event["name"]), arguments=dict(event.get("arguments") or {}))
                )
            elif event.get("type") == "finish":
                finish_reason = str(event.get("finish_reason") or finish_reason)
                if isinstance(event.get("reasoning_content"), str):
                    reasoning_parts = [str(event["reasoning_content"])]
                usage = _usage_from_response(event.get("usage")) or usage
                if isinstance(event.get("response_id"), str):
                    response_id = str(event["response_id"])
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
        _prepared_request: tuple[dict[str, Any], list[dict[str, Any]]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        if _prepared_request is None:
            replay_input: list[dict[str, Any]] = []
            payload = self._responses_payload(
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
                _history_input_target=replay_input,
            )
        else:
            payload, replay_input = _prepared_request
        extra_headers = _responses_transport_headers(payload)
        if subagent is not None:
            extra_headers["x-openai-subagent"] = subagent
        if memgen_request is not None:
            extra_headers["x-openai-memgen-request"] = "true" if memgen_request else "false"
        stream = self._post_sse("/responses", payload, extra_headers=extra_headers)
        final_output: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []
        yielded_web_search_ids: set[str] = set()
        saw_text_delta = False
        saw_reasoning_delta = False
        saw_function_tool_call = False
        for event in stream:
            typ = event.get("type")
            if typ == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    saw_text_delta = True
                    yield {"type": "content", "text": delta}
            elif typ == "response.output_item.done":
                item = event.get("item")
                if not isinstance(item, dict):
                    raise ChatGPTOAuthError("response.output_item.done must contain an object item")
                final_output.append(item)
                tool = _tool_call_from_response_item(item)
                if tool is not None:
                    saw_function_tool_call = True
                    yield {"type": "tool_call", "id": tool.id, "name": tool.name, "arguments": tool.arguments}
                web_search = _web_search_event_from_response_item(item)
                if web_search is not None:
                    yielded_web_search_ids.add(str(web_search["id"]))
                    yield web_search
            elif typ == "response.reasoning_summary_part.added":
                yield {
                    "type": "reasoning_section_break",
                    "summary_index": event.get("summary_index"),
                    "part_index": event.get("part_index"),
                }
            elif typ == "response.reasoning_summary_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    saw_reasoning_delta = True
                    reasoning_parts.append(delta)
                    yield {
                        "type": "reasoning_delta",
                        "text": delta,
                        "summary_index": event.get("summary_index"),
                    }
            elif typ == "response.reasoning_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    saw_reasoning_delta = True
                    reasoning_parts.append(delta)
                    yield {
                        "type": "reasoning_raw_delta",
                        "text": delta,
                        "summary_index": event.get("summary_index"),
                    }
            elif typ == "response.failed":
                raise ChatGPTOAuthError(response_failure_message(event, "failed"))
            elif typ == "response.incomplete":
                raise ChatGPTOAuthError(response_failure_message(event, "incomplete"))
            elif typ == "response.completed":
                response = _validated_completed_response(event)
                completed_output = deepcopy(final_output)
                self._response_chains.commit(
                    response["id"],
                    replay_input,
                    completed_output,
                )
                usage = response.get("usage")
                display_output = completed_output
                for item in display_output:
                    web_search = _web_search_event_from_response_item(item, display_output)
                    if web_search is not None and str(web_search["id"]) not in yielded_web_search_ids:
                        yielded_web_search_ids.add(str(web_search["id"]))
                        yield web_search
                if not saw_text_delta:
                    final_text = _text_from_response_items(display_output)
                    if final_text:
                        saw_text_delta = True
                        yield {"type": "content", "text": final_text}
                if not saw_reasoning_delta:
                    completed_reasoning = reasoning_from_response_items(display_output)
                    if completed_reasoning:
                        saw_reasoning_delta = True
                        reasoning_parts.append(completed_reasoning)
                        yield {"type": "reasoning_delta", "text": completed_reasoning}
                yield {
                    "type": "finish",
                    "finish_reason": "tool_calls" if saw_function_tool_call else "stop",
                    "usage": usage,
                    "reasoning_content": "".join(reasoning_parts) or None,
                    "response_id": response["id"],
                }
                return
        raise ChatGPTOAuthError("ChatGPT OAuth response stream ended before response.completed")

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
    ) -> list[dict[str, Any]]:
        if not isinstance(prompt, str) or prompt.strip() == "":
            raise ChatGPTOAuthError("image generation prompt is required")
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend(_validate_image_content_items(reference_images))
        if size and size != "auto":
            content[0]["text"] = f"{prompt}\n\nRequested output size/aspect: {size}"
        request_model = resolve_model_for_backend(model or self.model)
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
            "prompt_cache_key": str(uuid.uuid4()),
        }
        _finalize_responses_payload(
            payload,
            model=request_model,
            reasoning_effort=reasoning_effort,
            reasoning_mode=reasoning_mode,
            reasoning_context=reasoning_context,
            responses_lite=responses_lite,
            safety_identifier=safety_identifier,
            prompt_cache_options=prompt_cache_options,
            verbosity=verbosity,
        )
        output_items = self._collect_response_output_items(payload)
        images = [_image_generation_from_item(item) for item in output_items]
        generated = [image for image in images if image is not None]
        if not generated:
            raise ChatGPTOAuthError("image generation response returned no image_generation_call")
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
    ) -> str:
        if not isinstance(prompt, str) or prompt.strip() == "":
            raise ChatGPTOAuthError("image inspection prompt is required")
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend(_validate_image_content_items(images))
        request_model = resolve_model_for_backend(model or self.model)
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
            "prompt_cache_key": str(uuid.uuid4()),
        }
        _finalize_responses_payload(
            payload,
            model=request_model,
            reasoning_effort=reasoning_effort,
            reasoning_mode=reasoning_mode,
            reasoning_context=reasoning_context,
            responses_lite=responses_lite,
            safety_identifier=safety_identifier,
            prompt_cache_options=prompt_cache_options,
            verbosity=verbosity,
        )
        output_items = self._collect_response_output_items(payload)
        text = _text_from_response_items(output_items).strip()
        if text == "":
            raise ChatGPTOAuthError("image inspection response returned empty content")
        return text

    def _collect_response_output_items(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        output_items: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        def append_item(item: dict[str, Any]) -> None:
            key_parts = [str(item.get("type") or "")]
            for field in ("id", "call_id"):
                if isinstance(item.get(field), str) and item[field]:
                    key_parts.append(str(item[field]))
                    break
            else:
                key_parts.append(json.dumps(item, sort_keys=True, ensure_ascii=False, default=str))
            key = "\x1f".join(key_parts)
            if key in seen_keys:
                return
            seen_keys.add(key)
            output_items.append(item)

        for event in self._post_sse(
            "/responses",
            payload,
            extra_headers=_responses_transport_headers(payload),
        ):
            typ = event.get("type")
            if typ == "response.output_item.done":
                item = event.get("item")
                if not isinstance(item, dict):
                    raise ChatGPTOAuthError("response.output_item.done must contain an object item")
                append_item(item)
            elif typ == "response.failed":
                raise ChatGPTOAuthError(response_failure_message(event, "failed"))
            elif typ == "response.incomplete":
                raise ChatGPTOAuthError(response_failure_message(event, "incomplete"))
            elif typ == "response.completed":
                _validated_completed_response(event)
                return output_items
        raise ChatGPTOAuthError("ChatGPT OAuth response stream ended before response.completed")

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
    ) -> str:
        request_model = resolve_model_for_backend(model or self.model)
        base_instructions, input_items = _split_instructions_and_input(messages)
        if previous_response_id is not None:
            validated_previous_response_id = _validate_previous_response_id(previous_response_id)
            input_items = self._response_chains.resolve(validated_previous_response_id) + input_items
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
            model=request_model,
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
            extra_headers=_responses_transport_headers(payload),
        )
        output = data.get("output")
        if not isinstance(output, list):
            raise ChatGPTOAuthError("remote compact response missing output array")
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
        _history_input_target: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del temperature  # ChatGPT Codex backend rejects explicit temperature for this endpoint.
        del max_tokens  # ChatGPT Codex backend rejects max_output_tokens for this endpoint.
        _reject_unsupported_stop(stop)
        instructions, input_items = _split_instructions_and_input(messages)
        if instructions == "":
            raise ChatGPTOAuthError("ChatGPT OAuth Responses request requires system instructions")
        request_model = resolve_model_for_backend(model or self.model)
        if previous_response_id is not None:
            validated_previous_response_id = _validate_previous_response_id(previous_response_id)
            input_items = self._response_chains.resolve(validated_previous_response_id) + input_items
        if _history_input_target is not None:
            _history_input_target.extend(deepcopy(input_items))
        tools_payload = [] if tools is None else [_tool_schema_to_response_dict(tool) for tool in tools]
        payload: dict[str, Any] = {
            "model": request_model,
            "instructions": instructions,
            "input": input_items,
            "tools": tools_payload,
            "tool_choice": "auto" if tool_choice is None else tool_choice,
            "parallel_tool_calls": should_enable_parallel_tool_calls(
                model=request_model,
                requested=parallel_tool_calls,
                responses_lite=False,
            ),
            "stream": True,
            "store": False,
            "include": [],
        }
        if any(tool.get("type") == "web_search" for tool in payload["tools"]):
            payload["include"] = ["web_search_call.action.sources"]
        if prompt_cache_key is not None:
            payload["prompt_cache_key"] = _validate_non_empty_string(
                prompt_cache_key,
                "prompt_cache_key",
            )
        if client_metadata is not None:
            payload["client_metadata"] = client_metadata
        if resolve_codex_metadata_enabled(codex_metadata):
            payload["client_metadata"] = build_codex_client_metadata(
                auth_json_path=self.auth_json_path,
                existing=payload.get("client_metadata") if isinstance(payload.get("client_metadata"), dict) else None,
            )
        _finalize_responses_payload(
            payload,
            model=request_model,
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
        return payload

    def _headers(self) -> dict[str, str]:
        token = load_token_data(self.auth_json_path)
        register_token_secrets(token.access_token, token.refresh_token, token.id_token, token.account_id)
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
    ) -> dict[str, Any]:
        raw = self._request_json(path, payload, extra_headers=extra_headers)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ChatGPTOAuthError("ChatGPT OAuth response must be a JSON object")
        return data

    def _post_sse(
        self, path: str, payload: dict[str, Any], extra_headers: dict[str, str] | None = None
    ) -> Iterator[dict[str, Any]]:
        yield from self._request_sse(path, payload, extra_headers=extra_headers)

    def _request_sse(
        self, path: str, payload: dict[str, Any], extra_headers: dict[str, str] | None = None
    ) -> Iterator[dict[str, Any]]:
        token_values: tuple[str | None, ...] = (None,)
        for attempt in range(2):
            headers = self._headers()
            headers["Accept"] = "text/event-stream"
            if extra_headers:
                headers.update(extra_headers)
            token = load_token_data(self.auth_json_path)
            token_values = (token.access_token, token.refresh_token, token.id_token, token.account_id)
            req = urllib.request.Request(
                self.base_url + path,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
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
                                        yield event
                                return
                            line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
                            if line == "":
                                event = _decode_sse_block(block)
                                block = []
                                if event is not None:
                                    yield event
                                continue
                            block.append(line)
                    finally:
                        with self._active_response_lock:
                            self._active_responses.discard(response)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                redacted = redact_text(body, *token_values)
                if exc.code == 401 and attempt == 0:
                    refresh_token(self.auth_json_path)
                    continue
                raise ChatGPTOAuthError(f"ChatGPT OAuth request failed: HTTP {exc.code}: {redacted}") from exc
            except Exception as exc:  # noqa: BLE001
                raise ChatGPTOAuthError(
                    f"ChatGPT OAuth request failed: {redact_text(str(exc), *token_values)}"
                ) from exc
            return

    def _request_json(
        self,
        path: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> bytes:
        token_values: tuple[str | None, ...] = (None,)
        for attempt in range(2):
            headers = self._headers()
            if extra_headers:
                headers.update(extra_headers)
            token = load_token_data(self.auth_json_path)
            token_values = (token.access_token, token.refresh_token, token.id_token, token.account_id)
            req = urllib.request.Request(
                self.base_url + path,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    return bytes(response.read())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                redacted = redact_text(body, *token_values)
                if exc.code == 401 and attempt == 0:
                    refresh_token(self.auth_json_path)
                    continue
                raise ChatGPTOAuthError(f"ChatGPT OAuth request failed: HTTP {exc.code}: {redacted}") from exc
            except Exception as exc:  # noqa: BLE001
                raise ChatGPTOAuthError(
                    f"ChatGPT OAuth request failed: {redact_text(str(exc), *token_values)}"
                ) from exc
        raise AssertionError("unreachable ChatGPT OAuth request retry state")


def _validate_non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ChatGPTOAuthInvalidRequestError(f"{field} must be a non-empty string")
    return value


def _validate_previous_response_id(value: object) -> str:
    response_id = _validate_non_empty_string(value, "previous_response_id")
    if response_id.strip() == "":
        raise ChatGPTOAuthInvalidRequestError("previous_response_id must be a non-empty string")
    return response_id


def _is_gpt_5_6_model(model: str) -> bool:
    return model == "gpt-5.6" or model.startswith("gpt-5.6-")


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
    nested_verbosity = merged.get("verbosity") if merged is not None else None
    if nested_verbosity is not None and (
        not isinstance(nested_verbosity, str) or nested_verbosity not in KNOWN_VERBOSITY_VALUES
    ):
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


def _reject_prompt_cache_breakpoint(_value: object) -> None:
    raise ChatGPTOAuthInvalidRequestError(
        "prompt_cache_breakpoint is not supported by the ChatGPT Codex OAuth transport"
    )


def _validate_image_content_items(images: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ChatGPTOAuthError(f"image reference {index} must be an object")
        image_url = image.get("image_url")
        if not isinstance(image_url, str) or image_url.strip() == "":
            raise ChatGPTOAuthError(f"image reference {index} requires image_url")
        if not image_url.startswith("data:image/"):
            raise ChatGPTOAuthError(f"image reference {index} must be a data:image URL")
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
    if not isinstance(result, str) or result.strip() == "":
        raise ChatGPTOAuthError("image_generation_call returned empty result")
    return {
        "id": str(item.get("id") or uuid.uuid4().hex),
        "status": str(item.get("status") or "completed"),
        "revised_prompt": item.get("revised_prompt") if isinstance(item.get("revised_prompt"), str) else None,
        "result": result,
    }


def _decode_sse_block(lines: list[str]) -> dict[str, Any] | None:
    data_lines = [line[5:].strip() for line in lines if line.startswith("data:")]
    if not data_lines:
        return None
    joined = "\n".join(data_lines)
    if joined == "[DONE]":
        return None
    try:
        event = json.loads(joined)
    except json.JSONDecodeError as exc:
        raise ChatGPTOAuthError(f"invalid SSE event JSON: {joined[:80]}") from exc
    if not isinstance(event, dict):
        raise ChatGPTOAuthError("SSE event JSON must be an object")
    return event


def _validated_completed_response(event: dict[str, Any]) -> dict[str, Any]:
    response = event.get("response")
    if not isinstance(response, dict):
        raise ChatGPTOAuthError("response.completed must contain a response with a non-empty id")
    response_id = response.get("id")
    if not isinstance(response_id, str) or response_id == "":
        raise ChatGPTOAuthError("response.completed must contain a response with a non-empty id")
    return response


def _split_instructions_and_input(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    input_messages: list[Message] = []
    for msg in messages:
        if msg.role is MessageRole.SYSTEM and not msg.content.startswith(REMOTE_COMPACTION_MARKER):
            if any(part.get("prompt_cache_breakpoint") is not None for part in msg.content_parts):
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
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ChatGPTOAuthError("remote compaction marker contains invalid JSON") from exc
            if not isinstance(parsed, list):
                raise ChatGPTOAuthError("remote compaction marker must contain a response item array")
            items.extend(_filter_compacted_history_items(parsed, source="marker"))
            continue
        if message.role is MessageRole.TOOL:
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id or message.name or "tool-call",
                    "output": message.content,
                }
            )
            continue
        if message.role is MessageRole.ASSISTANT and message.tool_calls:
            if message.content or message.content_parts:
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
                        "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                    }
                )
            continue
        role = "assistant" if message.role is MessageRole.ASSISTANT else "user"
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
    content_parts: tuple[dict[str, object], ...] = (),
) -> dict[str, Any]:
    if content_parts:
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
    content_items: list[dict[str, Any]] = [{"type": typ, "text": content or ""}]
    for image_url in images:
        content_items.append({"type": "input_image", "image_url": image_url})
    return {"type": "message", "role": role, "content": content_items}


def _tool_schema_to_response_dict(tool: ToolSchema) -> dict[str, Any]:
    if tool.parameters.get("__codex_as_api_tool_type") == "web_search":
        openai_tool = tool.parameters.get("openai_tool")
        if isinstance(openai_tool, dict):
            return dict(openai_tool)
        return {"type": "web_search", "external_web_access": True}
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": False,
    }


def _finalize_responses_payload(
    payload: dict[str, Any],
    *,
    model: str,
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
    capability = capability_for_model(model)
    if not capability.supports_image_detail_original and _has_original_image_detail(payload):
        raise ChatGPTOAuthInvalidRequestError(f"image detail 'original' is not supported for model {model!r}")
    if _has_prompt_cache_breakpoint(payload.get("input")):
        _reject_prompt_cache_breakpoint(None)
    if safety_identifier is not None:
        _reject_safety_identifier(safety_identifier)
    if prompt_cache_options is not None:
        _reject_prompt_cache_options(prompt_cache_options)
    merged_text = _merge_text_and_verbosity(text, verbosity)
    try:
        apply_model_capability_fields(payload, model=model, text=merged_text, service_tier=service_tier)
    except ValueError as exc:
        raise ChatGPTOAuthInvalidRequestError(str(exc)) from exc
    effective_effort = (
        reasoning_effort
        if reasoning_effort is not None
        else "medium"
        if reasoning_mode is not None
        else capability.default_reasoning_effort
    )
    _set_reasoning_payload(
        payload,
        effective_effort,
        reasoning_mode=reasoning_mode,
        reasoning_context=reasoning_context,
        model=model,
        include_encrypted_content=include_encrypted_content,
    )
    try:
        lite = use_responses_lite(model, responses_lite)
    except ValueError as exc:
        raise ChatGPTOAuthInvalidRequestError(str(exc)) from exc
    if not lite:
        return

    if reasoning_context is not None and reasoning_context != "all_turns":
        raise ChatGPTOAuthInvalidRequestError(
            "Responses Lite requires reasoning.context to be all_turns when explicitly provided"
        )

    raw_tools = payload.get("tools")
    tools_payload = [dict(tool) for tool in raw_tools] if isinstance(raw_tools, list) else []
    _apply_responses_lite_payload(payload, tools_payload)


def _has_prompt_cache_breakpoint(value: object) -> bool:
    if isinstance(value, dict):
        return value.get("prompt_cache_breakpoint") is not None or any(
            _has_prompt_cache_breakpoint(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_has_prompt_cache_breakpoint(item) for item in value)
    return False


def _has_original_image_detail(value: object) -> bool:
    if isinstance(value, dict):
        if value.get("type") == "input_image" and value.get("detail") == "original":
            return True
        return any(_has_original_image_detail(child) for child in value.values())
    if isinstance(value, list):
        return any(_has_original_image_detail(item) for item in value)
    return False


def _reject_unsupported_stop(stop: Sequence[str] | None) -> None:
    if stop is None or not any(value != "" for value in stop):
        return
    raise ChatGPTOAuthInvalidRequestError(
        "stop is not supported by the private Codex OAuth HTTP transport"
    )


def _apply_responses_lite_payload(payload: dict[str, Any], tools_payload: Sequence[dict[str, Any]]) -> None:
    hosted_tool_types = sorted(
        {str(tool["type"]) for tool in tools_payload if tool.get("type") in {"web_search", "image_generation"}}
    )
    if hosted_tool_types:
        raise ChatGPTOAuthInvalidRequestError(
            "Responses Lite cannot execute hosted tools without a standalone runtime: "
            + ", ".join(hosted_tool_types)
            + f"; set {RESPONSES_LITE_ENV}=off to use classic Responses"
        )

    instructions = str(payload.pop("instructions", ""))
    payload.pop("tools", None)
    if "tool_choice" in payload and payload["tool_choice"] != "auto":
        raise ChatGPTOAuthInvalidRequestError("Responses Lite tool_choice must be the exact string 'auto'")
    payload["parallel_tool_calls"] = False
    input_items = list(payload.get("input") or [])
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
    payload["input"] = strip_image_detail_fields([*developer_items, *input_items])
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict):
        reasoning["context"] = "all_turns"
    payload["_codex_as_api_responses_lite"] = True


def _ensure_reasoning_encrypted_content(payload: dict[str, Any]) -> None:
    include = payload.setdefault("include", [])
    if isinstance(include, list) and "reasoning.encrypted_content" not in include:
        include.append("reasoning.encrypted_content")


def _responses_transport_headers(payload: dict[str, Any]) -> dict[str, str]:
    if payload.pop("_codex_as_api_responses_lite", False) is True:
        return {LITE_HEADER_NAME: LITE_HEADER_VALUE}
    return {}


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
        raise ChatGPTOAuthError(f"remote compact {source} item {index} requires a string type")
    if item_type == "agent_message":
        author = item.get("author")
        recipient = item.get("recipient")
        content = item.get("content")
        if not isinstance(author, str) or not isinstance(recipient, str) or not isinstance(content, list):
            raise ChatGPTOAuthError(
                f"remote compact {source} agent_message item {index} requires string author/recipient and content array"
            )
        for part_index, part in enumerate(content):
            if not isinstance(part, dict):
                raise ChatGPTOAuthError(
                    f"remote compact {source} agent_message item {index} content part {part_index} must be an object"
                )
            part_type = part.get("type")
            valid_text = part_type == "input_text" and isinstance(part.get("text"), str)
            valid_encrypted = part_type == "encrypted_content" and isinstance(part.get("encrypted_content"), str)
            if not valid_text and not valid_encrypted:
                raise ChatGPTOAuthError(
                    f"remote compact {source} agent_message item {index} content part {part_index} is invalid"
                )
        return
    if item_type in {"compaction", "compaction_summary"}:
        if not isinstance(item.get("encrypted_content"), str):
            raise ChatGPTOAuthError(
                f"remote compact {source} {item_type} item {index} requires string encrypted_content"
            )
        return
    if item_type == "context_compaction":
        encrypted_content = item.get("encrypted_content")
        if encrypted_content is not None and not isinstance(encrypted_content, str):
            raise ChatGPTOAuthError(
                f"remote compact {source} context_compaction item {index} encrypted_content must be a string"
            )
        return
    if item_type != "message":
        return
    role = item.get("role")
    content = item.get("content")
    if not isinstance(role, str) or not isinstance(content, list):
        raise ChatGPTOAuthError(
            f"remote compact {source} message item {index} must have a string role and content array"
        )
    for part_index, part in enumerate(content):
        if not isinstance(part, dict):
            raise ChatGPTOAuthError(
                f"remote compact {source} message item {index} content part {part_index} must be an object"
            )
        part_type = part.get("type")
        valid_text = part_type in {"input_text", "output_text"} and isinstance(part.get("text"), str)
        valid_image = part_type == "input_image" and isinstance(part.get("image_url"), str)
        if valid_image:
            detail = part.get("detail")
            valid_image = detail is None or (isinstance(detail, str) and detail in {"auto", "low", "high", "original"})
        if not valid_text and not valid_image:
            raise ChatGPTOAuthError(
                f"remote compact {source} message item {index} content part {part_index} is invalid"
            )


def _filter_compacted_history_items(
    items: Sequence[Any],
    *,
    source: str = "output",
) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ChatGPTOAuthError(f"remote compact {source} item {index} must be an object")
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
    all_items: Sequence[dict[str, Any]] = (),
) -> dict[str, Any] | None:
    if item.get("type") != "web_search_call":
        return None
    raw_id = str(item.get("id") or item.get("call_id") or uuid.uuid4().hex)
    tool_id = (
        raw_id
        if raw_id.startswith("srvtoolu_")
        else "srvtoolu_" + "".join(ch for ch in raw_id if ch.isalnum() or ch == "_")
    )
    raw_action = item.get("action")
    action = cast(dict[str, Any], raw_action) if isinstance(raw_action, dict) else {}
    sources = _web_search_sources_from_action(action)
    if not sources and all_items:
        sources.extend(_web_search_sources_from_annotations(all_items))
    return {
        "type": "web_search_call",
        "id": tool_id,
        "input": {"query": _web_search_query_from_action(action)},
        "content": sources,
    }


def _web_search_query_from_action(action: dict[str, Any]) -> str:
    query = action.get("query")
    if isinstance(query, str):
        return query
    queries = action.get("queries")
    if isinstance(queries, list):
        for q in queries:
            if isinstance(q, str) and q:
                return q
    url = action.get("url")
    return url if isinstance(url, str) else ""


def _web_search_sources_from_action(action: dict[str, Any]) -> list[dict[str, Any]]:
    return _normalize_web_search_sources(action.get("sources"))


def _web_search_sources_from_annotations(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_sources: list[dict[str, Any]] = []
    for item in items:
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            annotations = part.get("annotations")
            if not isinstance(annotations, list):
                continue
            for ann in annotations:
                if isinstance(ann, dict) and ann.get("type") == "url_citation":
                    raw_sources.append(ann)
    return _normalize_web_search_sources(raw_sources)


def _normalize_web_search_sources(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in value:
        if not isinstance(source, dict):
            continue
        url = source.get("url")
        if not isinstance(url, str) or not url or url in seen:
            continue
        seen.add(url)
        result: dict[str, Any] = {
            "type": "web_search_result",
            "url": url,
            "title": source.get("title") if isinstance(source.get("title"), str) else url,
        }
        if isinstance(source.get("page_age"), str):
            result["page_age"] = source["page_age"]
        out.append(result)
    return out


def _set_reasoning_payload(
    payload: dict[str, Any],
    reasoning_effort: str | None,
    *,
    reasoning_mode: str | None = None,
    reasoning_context: str | None = None,
    model: str | None = None,
    include_encrypted_content: bool = True,
) -> None:
    if reasoning_effort is None and reasoning_mode is None and reasoning_context is None:
        return
    reasoning: dict[str, Any] = {}
    if reasoning_effort is not None:
        if not isinstance(reasoning_effort, str) or reasoning_effort == "":
            raise ChatGPTOAuthError("reasoning_effort must be a non-empty string when provided")
        reasoning["effort"] = "max" if reasoning_effort == "ultra" else reasoning_effort
    if reasoning_mode is not None:
        if not isinstance(reasoning_mode, str) or reasoning_mode not in KNOWN_REASONING_MODES:
            raise ChatGPTOAuthInvalidRequestError("reasoning.mode must be one of: standard, pro")
        if model is not None and not _is_gpt_5_6_model(model):
            raise ChatGPTOAuthInvalidRequestError("reasoning.mode is only supported for GPT-5.6 models")
        if reasoning_mode == "pro":
            raise ChatGPTOAuthInvalidRequestError(
                "reasoning.mode 'pro' is not supported by the ChatGPT Codex OAuth transport"
            )
    if reasoning_context is not None:
        if not isinstance(reasoning_context, str) or reasoning_context not in KNOWN_REASONING_CONTEXTS:
            raise ChatGPTOAuthInvalidRequestError("reasoning.context must be one of: auto, current_turn, all_turns")
        reasoning["context"] = reasoning_context
    if reasoning:
        payload["reasoning"] = reasoning
    if reasoning and include_encrypted_content:
        _ensure_reasoning_encrypted_content(payload)


def _tool_call_from_response_item(item: dict[str, Any]) -> ToolCall | None:
    if item.get("type") not in {"function_call", "custom_tool_call"}:
        return None
    name = item.get("name")
    if not isinstance(name, str) or name == "":
        return None
    raw_args = item.get("arguments") or item.get("input") or "{}"
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {"input": raw_args}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {}
    call_id = item.get("call_id") or item.get("id") or uuid.uuid4().hex
    return ToolCall(id=str(call_id), name=name, arguments=args)


def _text_from_response_items(items: Sequence[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in items:
        item_type = item.get("type")
        if item_type in {"output_text", "text"}:
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
            continue
        if item_type != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, str):
                if part:
                    parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type not in {"output_text", "text"}:
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts)


def _usage_from_response(value: Any) -> Usage | None:
    if not isinstance(value, dict):
        return None
    prompt = value.get("input_tokens", value.get("prompt_tokens"))
    completion = value.get("output_tokens", value.get("completion_tokens"))
    total = value.get("total_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    token_details = value.get("input_tokens_details", value.get("prompt_tokens_details"))
    cached_tokens = 0
    cache_write_tokens = 0
    if isinstance(token_details, dict) and isinstance(token_details.get("cached_tokens"), int):
        cached_tokens = int(token_details["cached_tokens"])
    if isinstance(token_details, dict) and isinstance(token_details.get("cache_write_tokens"), int):
        cache_write_tokens = int(token_details["cache_write_tokens"])
    elif isinstance(value.get("cached_input_tokens"), int):
        cached_tokens = int(value["cached_input_tokens"])
    elif isinstance(value.get("cache_read_input_tokens"), int):
        cached_tokens = int(value["cache_read_input_tokens"])
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total if isinstance(total, int) else None,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
    )
