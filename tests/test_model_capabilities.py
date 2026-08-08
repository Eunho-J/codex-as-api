from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_as_api.model_capabilities import (
    UNKNOWN_CAPABILITY,
    capability_for_model,
    resolve_model_for_backend,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CATALOG_PATH = _PROJECT_ROOT / "config" / "model-capabilities.json"
_UPSTREAM_CONTRACT_PATH = _PROJECT_ROOT / "config" / "codex-upstream-contract.json"
_CONTRACT_FIELDS = (
    "context_window",
    "max_context_window",
    "default_reasoning_effort",
    "use_responses_lite",
    "support_verbosity",
    "default_verbosity",
    "supports_parallel_tool_calls",
    "supports_image_detail_original",
    "service_tiers",
)


def test_catalog_matches_the_pinned_codex_0_147_model_contract() -> None:
    catalog = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))["models"]
    contract = json.loads(_UPSTREAM_CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["upstream"]["repository"] == "openai/codex"
    assert contract["upstream"]["version"] == "0.147.0"
    for upstream_model in contract["models"]:
        slug = upstream_model["slug"]
        assert {field: catalog[slug][field] for field in _CONTRACT_FIELDS} == {
            field: upstream_model[field] for field in _CONTRACT_FIELDS
        }


@pytest.mark.parametrize(
    ("model", "default_effort"),
    [
        ("gpt-5.6-sol", "low"),
        ("gpt-5.6-terra", "medium"),
        ("gpt-5.6-luna", "medium"),
    ],
)
def test_gpt_5_6_capabilities_are_loaded(model: str, default_effort: str) -> None:
    capability = capability_for_model(model)

    assert capability.use_responses_lite is True
    assert capability.supports_parallel_tool_calls is True
    assert capability.default_reasoning_effort == default_effort
    assert capability.context_window == 272_000
    assert capability.max_context_window == 272_000


def test_public_gpt_5_6_alias_uses_sol_capabilities_and_wire_model() -> None:
    capability = capability_for_model("gpt-5.6")

    assert capability.use_responses_lite is True
    assert capability.default_reasoning_effort == "low"
    assert capability.context_window == 272_000
    assert resolve_model_for_backend("gpt-5.6") == "gpt-5.6-sol"


def test_unknown_model_has_no_reasoning_or_context_defaults() -> None:
    assert capability_for_model("not-in-the-catalog") == UNKNOWN_CAPABILITY


@pytest.mark.parametrize(
    "model",
    ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "gpt-5.4", "gpt-5.4-mini"],
)
def test_catalog_models_supporting_original_image_detail(model: str) -> None:
    assert capability_for_model(model).supports_image_detail_original is True


@pytest.mark.parametrize(
    "model",
    ["gpt-5.2", "gpt-5.3-codex", "gpt-5.3-codex-spark", "not-in-the-catalog"],
)
def test_catalog_models_without_verified_original_image_detail_support(model: str) -> None:
    assert capability_for_model(model).supports_image_detail_original is False


@pytest.mark.parametrize(
    ("model", "maximum"),
    [
        ("gpt-5.5", 272_000),
        ("gpt-5.4", 1_000_000),
        ("gpt-5.4-mini", 272_000),
        ("gpt-5.2", 272_000),
    ],
)
def test_current_existing_models_have_official_context_and_effort_defaults(model: str, maximum: int) -> None:
    capability = capability_for_model(model)

    assert capability.default_reasoning_effort == "medium"
    assert capability.context_window == 272_000
    assert capability.max_context_window == maximum
