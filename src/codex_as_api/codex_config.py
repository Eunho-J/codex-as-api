from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodexConfig:
    codex_home: str
    config_path: str
    model: str | None = None
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
    except OSError:
        return CodexConfig(codex_home=codex_home, config_path=config_path)

    return CodexConfig(
        codex_home=codex_home,
        config_path=config_path,
        model=_parse_toml_string(text, "model"),
        model_context_window=_parse_toml_integer(text, "model_context_window"),
        model_auto_compact_token_limit=_parse_toml_integer(text, "model_auto_compact_token_limit"),
    )


def _parse_toml_string(text: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*[\"']([^\"']+)[\"']\s*(?:#.*)?$", text, re.MULTILINE)
    return match.group(1) if match else None


def _parse_toml_integer(text: str, key: str) -> int | None:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*([0-9][0-9_]*)\s*(?:#.*)?$", text, re.MULTILINE)
    if not match:
        return None
    value = int(match.group(1).replace("_", ""))
    return value if value > 0 else None
