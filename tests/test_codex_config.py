from __future__ import annotations

from pathlib import Path

import pytest

from codex_as_api.codex_config import load_codex_config


def test_load_codex_config_parses_reasoning_and_context_settings(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '\n'.join(
            [
                'model = "gpt-5.6-sol"',
                'model_reasoning_effort = "ultra"',
                "model_context_window = 372_000",
                "model_auto_compact_token_limit = 330_000",
            ]
        ),
        encoding="utf-8",
    )

    config = load_codex_config(str(tmp_path))

    assert config.model == "gpt-5.6-sol"
    assert config.model_reasoning_effort == "ultra"
    assert config.model_context_window == 372_000
    assert config.model_auto_compact_token_limit == 330_000


def test_load_codex_config_rejects_empty_reasoning_effort(tmp_path) -> None:
    (tmp_path / "config.toml").write_text('model_reasoning_effort = ""\n', encoding="utf-8")

    with pytest.raises(ValueError, match="model_reasoning_effort must be a non-empty string"):
        load_codex_config(str(tmp_path))


def test_load_codex_config_ignores_inactive_profile_reasoning_effort(tmp_path) -> None:
    (tmp_path / "config.toml").write_text(
        'model = "gpt-5.6-sol"\n\n[profiles.expensive]\nmodel_reasoning_effort = "ultra"\n',
        encoding="utf-8",
    )

    config = load_codex_config(str(tmp_path))

    assert config.model == "gpt-5.6-sol"
    assert config.model_reasoning_effort is None


def test_load_codex_config_decodes_toml_string_escapes(tmp_path) -> None:
    (tmp_path / "config.toml").write_text(
        'model_reasoning_effort = "\\u0075ltra"\n',
        encoding="utf-8",
    )

    config = load_codex_config(str(tmp_path))

    assert config.model_reasoning_effort == "ultra"


def test_load_codex_config_rejects_non_string_root_reasoning_effort(tmp_path) -> None:
    (tmp_path / "config.toml").write_text(
        "model_reasoning_effort = 42\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_reasoning_effort must be a string"):
        load_codex_config(str(tmp_path))


def test_load_codex_config_ignores_wrong_type_inside_inactive_profile(tmp_path) -> None:
    (tmp_path / "config.toml").write_text(
        '[profiles.expensive]\nmodel_reasoning_effort = 42\n',
        encoding="utf-8",
    )

    config = load_codex_config(str(tmp_path))

    assert config.model_reasoning_effort is None


def test_load_codex_config_surfaces_read_failures_with_path(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "gpt-5.6-sol"\n', encoding="utf-8")
    original_read_text = Path.read_text

    def denied_read_text(path, *args, **kwargs):
        if path == config_path:
            raise PermissionError("denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied_read_text)

    with pytest.raises(RuntimeError, match=rf"failed to read Codex config {config_path}.*denied"):
        load_codex_config(str(tmp_path))


def test_load_codex_config_rejects_integer_larger_than_signed_64_bit(tmp_path) -> None:
    (tmp_path / "config.toml").write_text(
        f"model_context_window = {1 << 63}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_context_window must fit in a signed 64-bit integer"):
        load_codex_config(str(tmp_path))
