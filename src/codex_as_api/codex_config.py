from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[import-not-found]


_MAX_SIGNED_64_BIT_INTEGER = (1 << 63) - 1


@dataclass(frozen=True)
class CodexConfig:
    codex_home: str
    config_path: str
    model: str | None = None
    model_reasoning_effort: str | None = None
    model_context_window: int | None = None
    model_auto_compact_token_limit: int | None = None


def _expand_home(path: str) -> str:
    if path == "~":
        return str(Path.home())
    if path.startswith("~/"):
        return str(Path.home() / path[2:])
    return path


def resolve_codex_home(raw: str | None = None) -> str:
    return _expand_home(raw or os.getenv("CODEX_HOME") or str(Path.home() / ".codex"))


def load_codex_config(raw_codex_home: str | None = None) -> CodexConfig:
    codex_home = resolve_codex_home(raw_codex_home)
    config_path = str(Path(codex_home) / "config.toml")
    try:
        text = Path(config_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return CodexConfig(codex_home=codex_home, config_path=config_path)
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"failed to read Codex config {config_path}: {exc}") from exc

    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid Codex config {config_path}: {exc}") from exc
    return CodexConfig(
        codex_home=codex_home,
        config_path=config_path,
        model=_optional_root_string(parsed, "model", empty_as_none=True),
        model_reasoning_effort=_optional_root_string(parsed, "model_reasoning_effort"),
        model_context_window=_optional_root_integer(parsed, "model_context_window"),
        model_auto_compact_token_limit=_optional_root_integer(
            parsed,
            "model_auto_compact_token_limit",
        ),
    )


def _optional_root_string(
    parsed: dict[str, Any],
    key: str,
    *,
    empty_as_none: bool = False,
) -> str | None:
    if key not in parsed:
        return None
    value = parsed[key]
    if not isinstance(value, str):
        raise ValueError(f"Codex config {key} must be a string")
    if value == "":
        if empty_as_none:
            return None
        raise ValueError(f"Codex config {key} must be a non-empty string")
    return value


def _optional_root_integer(parsed: dict[str, Any], key: str) -> int | None:
    if key not in parsed:
        return None
    value = parsed[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Codex config {key} must be an integer")
    if value <= 0:
        raise ValueError(f"Codex config {key} must be greater than zero")
    if value > _MAX_SIGNED_64_BIT_INTEGER:
        raise ValueError(f"Codex config {key} must fit in a signed 64-bit integer")
    return value
