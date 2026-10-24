from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Mapping
from typing import Any

from .strict_json import strict_json_loads

REASONING_SIGNATURE_PREFIX = "codex-reasoning-v1:"
REASONING_SIGNATURE_RECOGNIZED_PREFIX = "codex-reasoning"
REASONING_SIGNATURE_VERSION = 1
REASONING_COMPATIBLE_MODELS: frozenset[str] = frozenset(
    {
        "gpt-6.1-sol",
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-daybreak-blue-latest",
    }
)

_BASE64URL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_RECOGNIZED_SIGNATURE_PREFIX_RE = re.compile(r"^\s*codex[-_]reasoning", re.IGNORECASE)


def is_recognized_reasoning_signature(signature: str | None) -> bool:
    return isinstance(signature, str) and _RECOGNIZED_SIGNATURE_PREFIX_RE.match(signature) is not None


is_codex_reasoning_signature = is_recognized_reasoning_signature


def is_reasoning_model_compatible(source_model: str, target_model: str) -> bool:
    if not isinstance(source_model, str) or not isinstance(target_model, str):
        return False
    if not source_model or not target_model:
        return False
    if source_model == target_model:
        return True
    return source_model in REASONING_COMPATIBLE_MODELS and target_model in REASONING_COMPATIBLE_MODELS


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON object key {key!r}")
        obj[key] = value
    return obj


def _strict_json_loads_no_duplicates(raw: str) -> Any:
    try:
        parsed = strict_json_loads(raw)
        json.loads(raw, object_pairs_hook=_reject_duplicate_object_keys)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError("invalid JSON in reasoning signature") from exc
    return parsed


def encode_reasoning_signature(model: str, item: Mapping[str, Any]) -> str | None:
    if not isinstance(model, str) or not model or model != model.strip():
        raise ValueError("reasoning signature model must be a non-empty string")
    if not isinstance(item, Mapping) or item.get("type") != "reasoning":
        raise ValueError("reasoning signature item must have type 'reasoning'")
    encrypted_content = item.get("encrypted_content")
    if encrypted_content is None or encrypted_content == "":
        return None
    if not isinstance(encrypted_content, str):
        raise ValueError("reasoning item encrypted_content must be a string")

    item_id = item.get("id")
    if item_id is not None and not isinstance(item_id, str):
        raise ValueError("reasoning item id must be a string when provided")

    raw_summary = item.get("summary")
    summary_parts: list[dict[str, str]] = []
    if raw_summary is not None:
        if not isinstance(raw_summary, list):
            raise ValueError("reasoning item summary must be an array")
        for index, part in enumerate(raw_summary):
            if (
                not isinstance(part, Mapping)
                or part.get("type") != "summary_text"
                or not isinstance(part.get("text"), str)
            ):
                raise ValueError(f"reasoning item summary[{index}] must be a summary_text object")
            summary_parts.append({"type": "summary_text", "text": part["text"]})

    canonical_item: dict[str, Any] = {"type": "reasoning"}
    if isinstance(item_id, str) and item_id:
        canonical_item["id"] = item_id
    canonical_item["encrypted_content"] = encrypted_content
    canonical_item["summary"] = summary_parts

    envelope: dict[str, Any] = {
        "v": REASONING_SIGNATURE_VERSION,
        "model": model,
        "item": canonical_item,
    }
    raw_bytes = json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw_bytes).rstrip(b"=").decode("ascii")
    return f"{REASONING_SIGNATURE_PREFIX}{encoded}"


def decode_reasoning_signature(signature: str) -> dict[str, Any]:
    if not isinstance(signature, str) or not signature.startswith(REASONING_SIGNATURE_PREFIX):
        raise ValueError("Anthropic thinking signature has an invalid codex-reasoning prefix")
    encoded = signature[len(REASONING_SIGNATURE_PREFIX) :]
    if not encoded or len(encoded) % 4 == 1 or _BASE64URL_RE.fullmatch(encoded) is None:
        raise ValueError("Anthropic thinking signature must use unpadded base64url encoding")
    padded = encoded + ("=" * (-len(encoded) % 4))
    try:
        raw_bytes = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Anthropic thinking signature is not valid base64url") from exc
    if base64.urlsafe_b64encode(raw_bytes).rstrip(b"=").decode("ascii") != encoded:
        raise ValueError("Anthropic thinking signature must use canonical unpadded base64url encoding")
    try:
        raw_json = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Anthropic thinking signature payload must be valid UTF-8") from exc

    parsed = _strict_json_loads_no_duplicates(raw_json)
    if not isinstance(parsed, dict):
        raise ValueError("Anthropic thinking signature payload must be a JSON object")
    if set(parsed.keys()) != {"v", "model", "item"}:
        raise ValueError("Anthropic thinking signature payload contains invalid fields")
    version = parsed.get("v")
    if type(version) is not int or version != REASONING_SIGNATURE_VERSION:
        raise ValueError("Anthropic thinking signature version is unsupported")
    model = parsed.get("model")
    if not isinstance(model, str) or not model or model != model.strip():
        raise ValueError("Anthropic thinking signature model must be a non-empty string")

    item = parsed.get("item")
    if not isinstance(item, dict):
        raise ValueError("Anthropic thinking signature item must be an object")
    unknown_item_keys = set(item.keys()) - {"type", "id", "encrypted_content", "summary"}
    missing_item_keys = {"type", "encrypted_content", "summary"} - set(item.keys())
    if unknown_item_keys or missing_item_keys:
        raise ValueError("Anthropic thinking signature item contains invalid fields")
    if item.get("type") != "reasoning":
        raise ValueError("Anthropic thinking signature item type must be 'reasoning'")

    canonical_item: dict[str, Any] = {"type": "reasoning"}
    if "id" in item:
        item_id = item["id"]
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("Anthropic thinking signature item id must be a non-empty string when present")
        canonical_item["id"] = item_id

    encrypted_content = item.get("encrypted_content")
    if not isinstance(encrypted_content, str) or not encrypted_content:
        raise ValueError("Anthropic thinking signature item encrypted_content must be a non-empty string")
    canonical_item["encrypted_content"] = encrypted_content

    summary = item.get("summary")
    if not isinstance(summary, list):
        raise ValueError("Anthropic thinking signature item summary must be an array")
    normalized_summary: list[dict[str, str]] = []
    for index, part in enumerate(summary):
        if not isinstance(part, dict) or set(part.keys()) != {"type", "text"}:
            raise ValueError(f"Anthropic thinking signature item summary[{index}] has invalid fields")
        if part.get("type") != "summary_text" or not isinstance(part.get("text"), str):
            raise ValueError(f"Anthropic thinking signature item summary[{index}] must be a summary_text object")
        normalized_summary.append({"type": "summary_text", "text": part["text"]})
    canonical_item["summary"] = normalized_summary

    return {
        "v": REASONING_SIGNATURE_VERSION,
        "model": model,
        "item": canonical_item,
    }


def extract_reasoning_item_for_model(
    entry: Mapping[str, Any],
    *,
    backend_model: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        raise ValueError("reasoning_items entry must be a mapping")
    raw_item_obj = entry.get("item") if "item" in entry else entry
    if not isinstance(raw_item_obj, Mapping) or raw_item_obj.get("type") != "reasoning":
        raise ValueError("reasoning_items entry must contain a reasoning item")
    entry_model = entry.get("model")
    if entry_model is not None and (
        not isinstance(entry_model, str) or not entry_model or entry_model != entry_model.strip()
    ):
        raise ValueError("reasoning_items entry model must be a non-empty string when provided")
    if (
        backend_model is not None
        and entry_model is not None
        and not is_reasoning_model_compatible(entry_model, backend_model)
    ):
        return None

    encrypted_content = raw_item_obj.get("encrypted_content")
    if not isinstance(encrypted_content, str) or not encrypted_content:
        return None

    item_id = raw_item_obj.get("id")
    if item_id is not None and (not isinstance(item_id, str) or not item_id):
        raise ValueError("reasoning item id must be a non-empty string when provided")

    raw_summary = raw_item_obj.get("summary")
    if not isinstance(raw_summary, list):
        raise ValueError("reasoning item summary must be an array")
    normalized_summary: list[dict[str, str]] = []
    for index, part in enumerate(raw_summary):
        if (
            not isinstance(part, Mapping)
            or set(part.keys()) != {"type", "text"}
            or part.get("type") != "summary_text"
            or not isinstance(part.get("text"), str)
        ):
            raise ValueError(f"reasoning item summary[{index}] must be a summary_text object")
        normalized_summary.append({"type": "summary_text", "text": part["text"]})

    canonical_item: dict[str, Any] = {"type": "reasoning"}
    if isinstance(item_id, str) and item_id:
        canonical_item["id"] = item_id
    canonical_item["encrypted_content"] = encrypted_content
    canonical_item["summary"] = normalized_summary
    return canonical_item
