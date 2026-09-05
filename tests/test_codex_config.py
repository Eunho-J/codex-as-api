from __future__ import annotations

from pathlib import Path

import pytest

from codex_as_api.codex_config import load_codex_config, resolve_codex_home


@pytest.mark.parametrize("value", ["", "   "])
def test_resolve_codex_home_rejects_empty_explicit_path(value) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        resolve_codex_home(value)


@pytest.mark.parametrize("value", ["", "   "])
def test_resolve_codex_home_rejects_empty_environment(monkeypatch, value) -> None:
    monkeypatch.setenv("CODEX_HOME", value)

    with pytest.raises(ValueError, match="CODEX_HOME"):
        resolve_codex_home()


def test_load_codex_config_parses_reasoning_and_context_settings(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "gpt-5.6-sol"',
                'model_reasoning_effort = "ultra"',
                "model_context_window = 272_000",
                "model_auto_compact_token_limit = 244_800",
            ]
        ),
        encoding="utf-8",
    )

    config = load_codex_config(str(tmp_path))

    assert config.model == "gpt-5.6-sol"
    assert config.model_reasoning_effort == "ultra"
    assert config.model_context_window == 272_000
    assert config.model_auto_compact_token_limit == 244_800


@pytest.mark.parametrize("value", ["", "   "])
def test_load_codex_config_rejects_empty_reasoning_effort(tmp_path, value) -> None:
    (tmp_path / "config.toml").write_text(
        f'model_reasoning_effort = "{value}"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_reasoning_effort must be a non-empty string"):
        load_codex_config(str(tmp_path))


@pytest.mark.parametrize("key", ["model", "model_reasoning_effort"])
def test_load_codex_config_rejects_identifier_surrounding_whitespace(tmp_path, key) -> None:
    (tmp_path / "config.toml").write_text(
        f'{key} = " gpt-value "\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="surrounding whitespace"):
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
        "[profiles.expensive]\nmodel_reasoning_effort = 42\n",
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


@pytest.mark.parametrize("field", ["model_context_window", "model_auto_compact_token_limit"])
def test_load_codex_config_requires_javascript_safe_integers(tmp_path, field) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(f"{field} = 9007199254740991\n", encoding="utf-8")
    config = load_codex_config(str(tmp_path))
    assert getattr(config, field) == 9_007_199_254_740_991

    config_path.write_text(f"{field} = 9007199254740992\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{field} must be a JavaScript-safe integer"):
        load_codex_config(str(tmp_path))


def test_load_codex_config_preserves_zero_auto_compact_limit(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("model_auto_compact_token_limit = 0\n", encoding="utf-8")

    assert load_codex_config(str(tmp_path)).model_auto_compact_token_limit == 0

    config_path.write_text("model_auto_compact_token_limit = -1\n", encoding="utf-8")
    assert load_codex_config(str(tmp_path)).model_auto_compact_token_limit == -1
