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
from collections.abc import Iterator, Sequence
from typing import Any

from .auth import (
    ChatGPTOAuthError,
    load_token_data,
    redact_text,
    refresh_token,
    register_token_secrets,
)
from .messages import AssistantResponse, Message, MessageRole, ToolCall, ToolSchema, Usage
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
REASONING_EFFORT_VALUES = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
_CODEX_CLI_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}(?:[-+][0-9A-Za-z.-]+)?$")
_CODEX_CLI_VERSION_UNSET = object()
_codex_cli_version_cache: str | None | object = _CODEX_CLI_VERSION_UNSET
_codex_cli_version_lock = threading.Lock()


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

    def cancel_current_requests(self) -> None:
        with self._active_response_lock:
            responses = list(self._active_responses)
        for response in responses:
            try:
                response.close()
            except Exception:
                pass

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        prompt_cache_key: str | None = None,
        subagent: str | None = None,
        memgen_request: bool | None = None,
        previous_response_id: str | None = None,
        service_tier: str | None = None,
        text: dict | None = None,
        client_metadata: dict[str, str] | None = None,
    ) -> AssistantResponse:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason = "stop"
        raw_events: list[dict[str, Any]] = []
        usage: Usage | None = None
        for event in self.chat_stream(
            messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            stop=stop,
            prompt_cache_key=prompt_cache_key,
            subagent=subagent,
            memgen_request=memgen_request,
            previous_response_id=previous_response_id,
            service_tier=service_tier,
            text=text,
            client_metadata=client_metadata,
        ):
            raw_events.append(dict(event))
            if event.get("type") == "content":
                content_parts.append(str(event.get("text", "")))
            elif event.get("type") in {"reasoning_delta", "reasoning_raw_delta"}:
                reasoning_parts.append(str(event.get("text", "")))
            elif event.get("type") == "tool_call":
                tool_calls.append(ToolCall(id=str(event["id"]), name=str(event["name"]), arguments=dict(event.get("arguments") or {})))
            elif event.get("type") == "finish":
                finish_reason = str(event.get("finish_reason") or finish_reason)
                if isinstance(event.get("reasoning_content"), str):
                    reasoning_parts = [str(event["reasoning_content"])]
                usage = _usage_from_response(event.get("usage")) or usage
        return AssistantResponse(
            content="".join(content_parts),
            tool_calls=tuple(tool_calls),
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content="".join(reasoning_parts) or None,
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
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        prompt_cache_key: str | None = None,
        subagent: str | None = None,
        memgen_request: bool | None = None,
        previous_response_id: str | None = None,
        service_tier: str | None = None,
        text: dict | None = None,
        client_metadata: dict[str, str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        payload = self._responses_payload(
            messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            stop=stop,
            prompt_cache_key=prompt_cache_key,
            max_tokens=max_tokens,
            previous_response_id=previous_response_id,
            service_tier=service_tier,
            text=text,
            client_metadata=client_metadata,
        )
        extra_headers: dict[str, str] = {}
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
        for event in stream:
            typ = event.get("type")
            if typ == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    saw_text_delta = True
                    yield {"type": "content", "text": delta}
            elif typ == "response.output_item.done":
                item = event.get("item")
                if isinstance(item, dict):
                    final_output.append(item)
                    tool = _tool_call_from_response_item(item)
                    if tool is not None:
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
                response = event.get("response")
                if isinstance(response, dict):
                    usage = response.get("usage")
                    if not final_output and isinstance(response.get("output"), list):
                        final_output.extend(item for item in response["output"] if isinstance(item, dict))
                    for item in final_output:
                        web_search = _web_search_event_from_response_item(item, final_output)
                        if web_search is not None and str(web_search["id"]) not in yielded_web_search_ids:
                            yielded_web_search_ids.add(str(web_search["id"]))
                            yield web_search
                    if not saw_text_delta:
                        final_text = _text_from_response_items(final_output)
                        if final_text:
                            saw_text_delta = True
                            yield {"type": "content", "text": final_text}
                    if not saw_reasoning_delta:
                        completed_reasoning = reasoning_from_response_items(final_output)
                        if completed_reasoning:
                            saw_reasoning_delta = True
                            reasoning_parts.append(completed_reasoning)
                            yield {"type": "reasoning_delta", "text": completed_reasoning}
                yield {
                    "type": "finish",
                    "finish_reason": "stop",
                    "usage": usage if isinstance(response, dict) else None,
                    "reasoning_content": "".join(reasoning_parts) or None,
                }

    def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        reference_images: Sequence[dict[str, str]] = (),
        size: str | None = None,
        reasoning_effort: str | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(prompt, str) or prompt.strip() == "":
            raise ChatGPTOAuthError("image generation prompt is required")
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend(_validate_image_content_items(reference_images))
        if size and size != "auto":
            content[0]["text"] = f"{prompt}\n\nRequested output size/aspect: {size}"
        payload = {
            "model": model or self.model,
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
        _set_reasoning_payload(payload, reasoning_effort)
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
        images: Sequence[dict[str, str]],
        reasoning_effort: str | None = None,
    ) -> str:
        if not isinstance(prompt, str) or prompt.strip() == "":
            raise ChatGPTOAuthError("image inspection prompt is required")
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend(_validate_image_content_items(images))
        payload = {
            "model": model or self.model,
            "instructions": "Inspect the attached image(s) and answer the user's review prompt directly.",
            "input": [{"type": "message", "role": "user", "content": content}],
            "tools": [],
            "parallel_tool_calls": False,
            "stream": True,
            "store": False,
            "include": [],
            "prompt_cache_key": str(uuid.uuid4()),
        }
        _set_reasoning_payload(payload, reasoning_effort)
        output_items = self._collect_response_output_items(payload)
        text = _text_from_response_items(output_items).strip()
        if text == "":
            raise ChatGPTOAuthError("image inspection response returned empty content")
        return text

    def _collect_response_output_items(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        output_items: list[dict[str, Any]] = []
        completed_output_seen = False
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

        for event in self._post_sse("/responses", payload):
            typ = event.get("type")
            if typ == "response.output_item.done":
                item = event.get("item")
                if isinstance(item, dict):
                    append_item(item)
            elif typ == "response.failed":
                raise ChatGPTOAuthError(response_failure_message(event, "failed"))
            elif typ == "response.incomplete":
                raise ChatGPTOAuthError(response_failure_message(event, "incomplete"))
            elif typ == "response.completed":
                response = event.get("response")
                if isinstance(response, dict) and isinstance(response.get("output"), list):
                    completed_output_seen = True
                    for item in response["output"]:
                        if isinstance(item, dict):
                            append_item(item)
        if not output_items and not completed_output_seen:
            raise ChatGPTOAuthError("ChatGPT OAuth response returned no output items")
        return output_items


    def compact_messages(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        payload = {
            "model": model or self.model,
            "input": _messages_to_response_items(messages),
            "instructions": "Create a compact checkpoint of this conversation for continuation.",
            "tools": [],
            "parallel_tool_calls": False,
        }
        _set_reasoning_payload(payload, reasoning_effort)
        data = self._post_json("/responses/compact", payload)
        output = data.get("output")
        if not isinstance(output, list):
            raise ChatGPTOAuthError("remote compact response missing output array")
        # Preserve the raw response items for the ChatGPT OAuth provider. The marker is deliberately
        # not a fallback summary; it is expanded back into Response items on subsequent requests.
        return REMOTE_COMPACTION_MARKER + "\n" + json.dumps(output, ensure_ascii=False, separators=(",", ":"))

    def _responses_payload(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        tools: Sequence[ToolSchema] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        stop: Sequence[str] | None = None,
        prompt_cache_key: str | None = None,
        max_tokens: int | None = None,
        previous_response_id: str | None = None,
        service_tier: str | None = None,
        text: dict | None = None,
        client_metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del temperature  # ChatGPT Codex backend rejects explicit temperature for this endpoint.
        del max_tokens  # ChatGPT Codex backend rejects max_output_tokens for this endpoint.
        instructions, input_items = _split_instructions_and_input(messages)
        if instructions == "":
            raise ChatGPTOAuthError("ChatGPT OAuth Responses request requires system instructions")
        payload: dict[str, Any] = {
            "model": model or self.model,
            "instructions": instructions,
            "input": input_items,
            "tools": [] if tools is None else [_tool_schema_to_response_dict(tool) for tool in tools],
            "tool_choice": tool_choice or "auto",
            "parallel_tool_calls": False,
            "stream": True,
            "store": False,
            "include": [],
        }
        if any(tool.get("type") == "web_search" for tool in payload["tools"]):
            payload["include"] = ["web_search_call.action.sources"]
        if prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if stop is not None:
            payload["stop"] = list(stop)
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if service_tier is not None:
            payload["service_tier"] = service_tier
        if text is not None:
            payload["text"] = text
        if client_metadata is not None:
            payload["client_metadata"] = client_metadata
        _set_reasoning_payload(payload, reasoning_effort)
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

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        raw = self._request_json(path, payload)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ChatGPTOAuthError("ChatGPT OAuth response must be a JSON object")
        return data

    def _post_sse(self, path: str, payload: dict[str, Any], extra_headers: dict[str, str] | None = None) -> Iterator[dict[str, Any]]:
        yield from self._request_sse(path, payload, extra_headers=extra_headers)

    def _request_sse(self, path: str, payload: dict[str, Any], extra_headers: dict[str, str] | None = None) -> Iterator[dict[str, Any]]:
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
                raise ChatGPTOAuthError(f"ChatGPT OAuth request failed: {redact_text(str(exc), *token_values)}") from exc
            return

    def _request_json(self, path: str, payload: dict[str, Any]) -> bytes:
        token_values: tuple[str | None, ...] = (None,)
        for attempt in range(2):
            headers = self._headers()
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
                    return response.read()
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                redacted = redact_text(body, *token_values)
                if exc.code == 401 and attempt == 0:
                    refresh_token(self.auth_json_path)
                    continue
                raise ChatGPTOAuthError(f"ChatGPT OAuth request failed: HTTP {exc.code}: {redacted}") from exc
            except Exception as exc:  # noqa: BLE001
                raise ChatGPTOAuthError(f"ChatGPT OAuth request failed: {redact_text(str(exc), *token_values)}") from exc
        raise AssertionError("unreachable ChatGPT OAuth request retry state")


def _validate_image_content_items(images: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ChatGPTOAuthError(f"image reference {index} must be an object")
        image_url = image.get("image_url")
        if not isinstance(image_url, str) or image_url.strip() == "":
            raise ChatGPTOAuthError(f"image reference {index} requires image_url")
        if not image_url.startswith("data:image/"):
            raise ChatGPTOAuthError(f"image reference {index} must be a data:image URL")
        items.append({"type": "input_image", "image_url": image_url})
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
    return event if isinstance(event, dict) else None


def _split_instructions_and_input(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    input_messages: list[Message] = []
    for msg in messages:
        if msg.role is MessageRole.SYSTEM and not msg.content.startswith(REMOTE_COMPACTION_MARKER):
            instructions.append(msg.content)
        else:
            input_messages.append(msg)
    return "\n\n".join(instructions), _messages_to_response_items(input_messages)


def _messages_to_response_items(messages: Sequence[Message]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role is MessageRole.SYSTEM and message.content.startswith(REMOTE_COMPACTION_MARKER):
            raw = message.content[len(REMOTE_COMPACTION_MARKER):].strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ChatGPTOAuthError("remote compaction marker must contain a response item array")
            for index, item in enumerate(parsed):
                if not isinstance(item, dict):
                    raise ChatGPTOAuthError(
                        f"remote compaction marker item {index} must be an object"
                    )
                items.append(item)
            continue
        if message.role is MessageRole.TOOL:
            items.append({
                "type": "function_call_output",
                "call_id": message.tool_call_id or message.name or "tool-call",
                "output": message.content,
            })
            continue
        if message.role is MessageRole.ASSISTANT and message.tool_calls:
            if message.content:
                items.append(_message_item("assistant", message.content))
            for tool_call in message.tool_calls:
                items.append({
                    "type": "function_call",
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                })
            continue
        role = "assistant" if message.role is MessageRole.ASSISTANT else "user"
        items.append(_message_item(role, message.content, message.images))
    return items


def _message_item(role: str, content: str, images: tuple[str, ...] = ()) -> dict[str, Any]:
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
    return {"type": "function", "name": tool.name, "description": tool.description, "parameters": tool.parameters, "strict": False}


def _compact_raw_events(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [event for event in events if event.get("type") == "web_search_call"]
    for event in events[-20:]:
        if event not in keep:
            keep.append(event)
    return keep


def _web_search_event_from_response_item(
    item: dict[str, Any],
    all_items: Sequence[dict[str, Any]] = (),
) -> dict[str, Any] | None:
    if item.get("type") != "web_search_call":
        return None
    raw_id = str(item.get("id") or item.get("call_id") or uuid.uuid4().hex)
    tool_id = raw_id if raw_id.startswith("srvtoolu_") else "srvtoolu_" + "".join(
        ch for ch in raw_id if ch.isalnum() or ch == "_"
    )
    action = item.get("action") if isinstance(item.get("action"), dict) else {}
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


def _set_reasoning_payload(payload: dict[str, Any], reasoning_effort: str | None) -> None:
    if reasoning_effort is None:
        return
    if not isinstance(reasoning_effort, str) or reasoning_effort == "":
        raise ChatGPTOAuthError("reasoning_effort must be a non-empty string when provided")
    effort = reasoning_effort.lower()
    if effort not in REASONING_EFFORT_VALUES:
        raise ChatGPTOAuthError(
            "reasoning_effort must be one of: " + ", ".join(sorted(REASONING_EFFORT_VALUES))
        )
    payload["reasoning"] = {"effort": effort}


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


def _to_openai_chat_completion_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    parsed = _usage_from_response(value)
    prompt_tokens = (
        parsed.prompt_tokens
        if parsed is not None
        else int(value.get("prompt_tokens") or value.get("input_tokens") or 0)
    )
    completion_tokens = (
        parsed.completion_tokens
        if parsed is not None
        else int(value.get("completion_tokens") or value.get("output_tokens") or 0)
    )
    total_tokens = (
        parsed.total_tokens
        if parsed is not None and parsed.total_tokens is not None
        else int(value.get("total_tokens") or prompt_tokens + completion_tokens)
    )
    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens > 0:
        prompt_tokens = total_tokens
    if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
        total_tokens = prompt_tokens + completion_tokens
    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


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
    if isinstance(token_details, dict) and isinstance(token_details.get("cached_tokens"), int):
        cached_tokens = int(token_details["cached_tokens"])
    elif isinstance(value.get("cached_input_tokens"), int):
        cached_tokens = int(value["cached_input_tokens"])
    elif isinstance(value.get("cache_read_input_tokens"), int):
        cached_tokens = int(value["cache_read_input_tokens"])
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total if isinstance(total, int) else None,
        cached_tokens=cached_tokens,
    )
