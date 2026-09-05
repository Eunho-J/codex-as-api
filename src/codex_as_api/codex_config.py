from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[import-not-found]


_JS_SAFE_INTEGER = 9_007_199_254_740_991


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
    if raw is not None:
        if not raw.strip():
            raise ValueError("Codex home must be a non-empty string")
        return _expand_home(raw)
    configured = os.getenv("CODEX_HOME")
    if configured is not None:
        if not configured.strip():
            raise ValueError("CODEX_HOME must be a non-empty string")
        return _expand_home(configured)
    return str(Path.home() / ".codex")


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
        model=_optional_root_string(parsed, "model"),
        model_reasoning_effort=_optional_root_string(parsed, "model_reasoning_effort"),
        model_context_window=_optional_root_integer(
            parsed,
            "model_context_window",
            allow_zero=False,
            allow_negative=False,
        ),
        model_auto_compact_token_limit=_optional_root_integer(
            parsed,
            "model_auto_compact_token_limit",
            allow_zero=True,
            allow_negative=True,
        ),
    )


def _optional_root_string(
    parsed: dict[str, Any],
    key: str,
) -> str | None:
    if key not in parsed:
        return None
    value = parsed[key]
    if not isinstance(value, str):
        raise ValueError(f"Codex config {key} must be a string")
    if not value.strip():
        raise ValueError(f"Codex config {key} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"Codex config {key} must not contain surrounding whitespace")
    return value


def _optional_root_integer(
    parsed: dict[str, Any],
    key: str,
    *,
    allow_zero: bool,
    allow_negative: bool,
) -> int | None:
    if key not in parsed:
        return None
    value = parsed[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Codex config {key} must be an integer")
    if (value < 0 and not allow_negative) or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"Codex config {key} must be {qualifier}")
    if abs(value) > _JS_SAFE_INTEGER:
        raise ValueError(f"Codex config {key} must be a JavaScript-safe integer")
    return value
