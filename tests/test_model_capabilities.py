from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from codex_as_api.auth import (
    ChatGPTOAuthCatalogUnavailableError,
    ChatGPTOAuthInvalidRequestError,
    ChatGPTOAuthModelNotFoundError,
)
from codex_as_api.model_capabilities import (
    CODEX_METADATA_ENV,
    RESPONSES_LITE_ENV,
    CatalogLoadResult,
    ModelCatalogCache,
    apply_model_capability_fields,
    parse_model_catalog,
    resolve_model,
    validate_model_capability_environment,
)


def _parse(document: object, *, account: str = "account-a"):
    now = time.time()
    return parse_model_catalog(
        document,
        key=(account, "https://example.test/codex", "0.153.3"),
        etag='"etag-a"',
        fetched_at=1_000.0,
        expires_at=now + 60,
    )


def test_live_catalog_parser_preserves_wire_capabilities(model_catalog_document) -> None:
    snapshot = _parse(model_catalog_document)
    model = snapshot.model("gpt-5.6-sol")

    assert model.slug == "gpt-5.6-sol"
    assert model.default_reasoning_level == "low"
    assert model.reasoning_effort_ids == (
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )
    assert model.supported_reasoning_levels[0].description == "low"
    assert model.supports_reasoning_summary_parameter is True
    assert model.default_reasoning_summary == "auto"
    assert model.context_window == 272_000
    assert model.effective_context_window_percent == 95
    assert model.service_tier_ids == ("priority",)
    assert model.input_modalities == ("text", "image")


def test_catalog_parser_diagnostics_do_not_reflect_slug_or_modality_values(
    model_catalog_document,
) -> None:
    secret = "access-token-sentinel"
    duplicate = deepcopy(model_catalog_document)
    duplicate["models"] = [deepcopy(duplicate["models"][0])] * 2
    duplicate["models"][0]["slug"] = secret
    duplicate["models"][1]["slug"] = secret

    unsupported_modality = deepcopy(model_catalog_document)
    unsupported_modality["models"][0]["input_modalities"] = [secret]

    for document in [duplicate, unsupported_modality]:
        with pytest.raises(ChatGPTOAuthCatalogUnavailableError) as caught:
            _parse(document)
        assert secret not in str(caught.value)

    valid = deepcopy(model_catalog_document)
    valid["models"][0]["slug"] = "preserved-model"
    valid["models"][0]["input_modalities"] = ["audio", "text"]
    parsed = _parse(valid)
    assert parsed.models[0].slug == "preserved-model"
    assert parsed.models[0].input_modalities == ("audio", "text")


def test_invalid_model_control_diagnostics_do_not_reflect_values(
    model_catalog_document,
) -> None:
    secret = "access-token-sentinel"
    document = deepcopy(model_catalog_document)
    model = document["models"][0]
    model["slug"] = secret
    model["support_verbosity"] = False
    model["service_tiers"] = []
    capability = _parse(document).models[0]

    invocations = [
        lambda: apply_model_capability_fields(
            {},
            capability=capability,
            text={"verbosity": "high"},
            service_tier=None,
        ),
        lambda: apply_model_capability_fields(
            {},
            capability=capability,
            text=None,
            service_tier=secret,
        ),
    ]
    for invoke in invocations:
        with pytest.raises(ValueError) as caught:
            invoke()
        assert secret not in str(caught.value)

    with pytest.raises(ChatGPTOAuthModelNotFoundError) as caught:
        _parse(model_catalog_document).model(secret)
    assert secret not in str(caught.value)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document["models"].append(deepcopy(document["models"][0])),
        lambda document: document["models"][0].pop("slug"),
        lambda document: document["models"][0].update(priority="zero"),
        lambda document: document["models"][0].update(service_tiers=["priority"]),
        lambda document: document["models"][0].update(supported_reasoning_levels=[{"effort": "low"}]),
        lambda document: document["models"][0].update(input_modalities=["video"]),
    ],
)
def test_catalog_refresh_is_atomic_on_schema_failure(
    model_catalog_document,
    mutation,
) -> None:
    mutation(model_catalog_document)

    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        _parse(model_catalog_document)


def test_catalog_parser_applies_only_official_optional_field_defaults(
    model_catalog_document,
) -> None:
    model = model_catalog_document["models"][0]
    model["supported_reasoning_levels"] = []
    for field in (
        "description",
        "default_reasoning_level",
        "default_verbosity",
        "context_window",
        "max_context_window",
        "auto_compact_token_limit",
        "default_service_tier",
        "multi_agent_reasoning_effort",
        "service_tiers",
        "use_responses_lite",
        "supports_image_detail_original",
        "effective_context_window_percent",
        "input_modalities",
        "supports_reasoning_summary_parameter",
        "default_reasoning_summary",
        "comp_hash",
    ):
        model.pop(field, None)

    capability = _parse(model_catalog_document).model("gpt-5.6-sol")

    assert capability.description is None
    assert capability.supported_reasoning_levels == ()
    assert capability.default_reasoning_level is None
    assert capability.default_verbosity is None
    assert capability.context_window is None
    assert capability.max_context_window is None
    assert capability.auto_compact_token_limit is None
    assert capability.default_service_tier is None
    assert capability.multi_agent_reasoning_effort is None
    assert capability.service_tiers == ()
    assert capability.use_responses_lite is False
    assert capability.supports_image_detail_original is False
    assert capability.effective_context_window_percent == 95
    assert capability.input_modalities == ("text", "image")
    assert capability.supports_reasoning_summary_parameter is True
    assert capability.default_reasoning_summary == "auto"
    assert capability.comp_hash is None


def test_catalog_parser_preserves_reasoning_summary_controls(model_catalog_document) -> None:
    model = model_catalog_document["models"][0]
    model["supports_reasoning_summary_parameter"] = False
    model["default_reasoning_summary"] = "detailed"

    snapshot = _parse(model_catalog_document)
    capability = snapshot.models[0]
    assert capability.supports_reasoning_summary_parameter is False
    assert capability.default_reasoning_summary == "detailed"

    for field, value in (
        ("supports_reasoning_summary_parameter", None),
        ("supports_reasoning_summary_parameter", "true"),
        ("default_reasoning_summary", None),
        ("default_reasoning_summary", "future"),
    ):
        invalid = deepcopy(model_catalog_document)
        invalid["models"][0][field] = value
        with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
            _parse(invalid)


def test_catalog_parser_accepts_empty_cosmetic_strings(model_catalog_document) -> None:
    model = model_catalog_document["models"][0]
    model["display_name"] = ""
    model["description"] = ""
    model["supported_reasoning_levels"][0]["description"] = ""
    model["service_tiers"][0]["name"] = ""
    model["service_tiers"][0]["description"] = ""

    capability = _parse(model_catalog_document).model("gpt-5.6-sol")
    assert capability.display_name == ""
    assert capability.description == ""
    assert capability.supported_reasoning_levels[0].description == ""
    assert capability.service_tiers[0].name == ""
    assert capability.service_tiers[0].description == ""


def test_catalog_parser_preserves_comp_hash_empty_slugs_and_custom_reasoning_values(
    model_catalog_document,
) -> None:
    model = model_catalog_document["models"][0]
    model["slug"] = " "
    model["comp_hash"] = " compatibility family "
    model["default_reasoning_level"] = " "
    model["multi_agent_reasoning_effort"] = " custom "
    model["supported_reasoning_levels"] = [{"effort": " ", "description": "custom"}]

    snapshot = _parse(model_catalog_document)
    capability = snapshot.models[0]

    assert capability.slug == " "
    assert capability.comp_hash == " compatibility family "
    assert capability.default_reasoning_level == " "
    assert capability.multi_agent_reasoning_effort == " custom "
    assert capability.reasoning_effort_ids == (" ",)
    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        resolve_model(snapshot, None, None)
    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        resolve_model(snapshot, None, " ")
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        resolve_model(snapshot, " ", None)

    for field in ("default_reasoning_level", "multi_agent_reasoning_effort"):
        invalid = deepcopy(model_catalog_document)
        invalid["models"][0][field] = ""
        with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
            _parse(invalid)
    invalid = deepcopy(model_catalog_document)
    invalid["models"][0]["supported_reasoning_levels"][0]["effort"] = ""
    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        _parse(invalid)


def test_catalog_parser_accepts_empty_models_but_default_selection_fails() -> None:
    snapshot = _parse({"models": []})

    assert snapshot.models == ()
    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        resolve_model(snapshot, None, None)


def test_catalog_parser_preserves_remaining_officially_unconstrained_values(model_catalog_document) -> None:
    model = model_catalog_document["models"][0]
    model["supported_reasoning_levels"] = [
        {"effort": "low", "description": "first"},
        {"effort": "low", "description": "second"},
    ]
    model["default_reasoning_level"] = "low"
    model["multi_agent_reasoning_effort"] = "also-not-listed"
    model["service_tiers"] = [
        {"id": "same", "name": "first", "description": ""},
        {"id": "same", "name": "second", "description": ""},
    ]
    model["default_service_tier"] = "not-listed"
    model["context_window"] = -1
    model["max_context_window"] = -2
    model["auto_compact_token_limit"] = 0
    model["effective_context_window_percent"] = -100

    capability = _parse(model_catalog_document).model("gpt-5.6-sol")
    assert [level.description for level in capability.supported_reasoning_levels] == [
        "first",
        "second",
    ]
    assert [tier.name for tier in capability.service_tiers] == ["first", "second"]
    assert capability.default_reasoning_level == "low"
    assert capability.multi_agent_reasoning_effort == "also-not-listed"
    assert capability.default_service_tier == "not-listed"
    assert capability.context_window == -1
    assert capability.max_context_window == -2
    assert capability.auto_compact_token_limit == 0
    assert capability.effective_context_window_percent == -100


def test_catalog_parser_preserves_default_reasoning_outside_supported_levels(model_catalog_document) -> None:
    model_catalog_document["models"][0]["default_reasoning_level"] = "not-listed"

    capability = _parse(model_catalog_document).model("gpt-5.6-sol")

    assert capability.default_reasoning_level == "not-listed"


def test_catalog_parser_preserves_default_ultra_without_a_wire_mapping(model_catalog_document) -> None:
    model = model_catalog_document["models"][0]
    model["supported_reasoning_levels"] = [{"effort": "ultra", "description": "ultra"}]
    model["default_reasoning_level"] = "ultra"
    model["multi_agent_reasoning_effort"] = "ultra"

    capability = _parse(model_catalog_document).model("gpt-5.6-sol")

    assert capability.default_reasoning_level == "ultra"
    assert capability.reasoning_effort_ids == ("ultra",)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("priority", 1 << 31),
        ("priority", -(1 << 31) - 1),
        ("context_window", 1 << 53),
        ("max_context_window", -(1 << 53)),
        ("auto_compact_token_limit", 1 << 53),
        ("effective_context_window_percent", -(1 << 53)),
    ],
)
def test_catalog_parser_rejects_cross_runtime_unsafe_integers(
    model_catalog_document,
    field,
    value,
) -> None:
    model_catalog_document["models"][0][field] = value
    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        _parse(model_catalog_document)


def test_catalog_parser_accepts_integral_json_numbers(model_catalog_document) -> None:
    model = model_catalog_document["models"][0]
    model["priority"] = 2.0
    model["context_window"] = json.loads("100000e0")
    model["max_context_window"] = 120_000.0
    model["auto_compact_token_limit"] = json.loads("80000e0")
    model["effective_context_window_percent"] = 95.0

    capability = _parse(model_catalog_document).models[0]
    assert capability.priority == 2
    assert capability.context_window == 100_000
    assert capability.max_context_window == 120_000
    assert capability.auto_compact_token_limit == 80_000
    assert capability.effective_context_window_percent == 95


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("priority", 1.5),
        ("priority", True),
        ("priority", (1 << 31)),
        ("context_window", 1.5),
        ("context_window", True),
        ("effective_context_window_percent", 1.5),
        ("effective_context_window_percent", True),
    ],
)
def test_catalog_parser_rejects_nonintegral_boolean_and_out_of_range_integers(
    model_catalog_document,
    field,
    value,
) -> None:
    model_catalog_document["models"][0][field] = value
    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        _parse(model_catalog_document)


def test_catalog_parser_preserves_duplicate_input_modalities(
    model_catalog_document,
) -> None:
    model_catalog_document["models"][0]["input_modalities"] = [
        "text",
        "image",
        "text",
    ]

    capability = _parse(model_catalog_document).model("gpt-5.6-sol")

    assert capability.input_modalities == ("text", "image", "text")


def test_catalog_parser_preserves_unconstrained_service_tier_strings(
    model_catalog_document,
) -> None:
    model = model_catalog_document["models"][0]
    model["service_tiers"] = [{"id": "", "name": "", "description": ""}]
    model["default_service_tier"] = ""

    capability = _parse(model_catalog_document).model("gpt-5.6-sol")

    assert capability.service_tiers[0].id == ""
    assert capability.default_service_tier == ""


def test_catalog_parser_ignores_unused_additive_model_fields(
    model_catalog_document,
) -> None:
    model_catalog_document["models"][0].update(
        supports_reasoning_summaries="not-a-bool",
        available_in_plans=[1],
        prefer_websockets={"future": True},
        requires_sandboxed_review="unknown",
        minimal_client_version=153,
        future_backend_field={"shape": "unknown"},
    )

    capability = _parse(model_catalog_document).model("gpt-5.6-sol")

    assert capability.slug == "gpt-5.6-sol"


@pytest.mark.parametrize("verbosity", ["verbose", "", 1, True, [], {}])
def test_capability_fields_reject_invalid_verbosity(
    model_catalog_document,
    verbosity,
) -> None:
    capability = _parse(model_catalog_document).model("gpt-5.6-sol")

    with pytest.raises(ValueError, match="text.verbosity must be one of"):
        apply_model_capability_fields(
            {},
            capability=capability,
            text={"verbosity": verbosity},
            service_tier=None,
        )


def test_model_resolution_uses_live_default_and_exact_membership(
    model_catalog_document,
) -> None:
    snapshot = _parse(model_catalog_document)

    assert resolve_model(snapshot, None, None).slug == "gpt-5.6-sol"
    assert resolve_model(snapshot, "gpt-5.6-sol", None).slug == "gpt-5.6-sol"
    with pytest.raises(ChatGPTOAuthModelNotFoundError):
        resolve_model(snapshot, "gpt-5.6", None)
    assert (
        resolve_model(
            snapshot,
            "claude-sonnet-4-6",
            "gpt-5.5",
            anthropic_facade=True,
        ).slug
        == "gpt-5.5"
    )

    with_literal_row = deepcopy(model_catalog_document)
    literal_row = deepcopy(with_literal_row["models"][0])
    literal_row["slug"] = "gpt-5.6"
    with_literal_row["models"].append(literal_row)
    with_literal_snapshot = _parse(with_literal_row)
    assert resolve_model(with_literal_snapshot, "gpt-5.6", None).slug == "gpt-5.6"
    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        resolve_model(
            snapshot,
            "claude-sonnet-4-6",
            "gpt-5.6",
            anthropic_facade=True,
        )


def test_unknown_model_never_falls_back(model_catalog_document) -> None:
    snapshot = _parse(model_catalog_document)

    with pytest.raises(ChatGPTOAuthModelNotFoundError):
        resolve_model(snapshot, "gpt-5.6-so1", None)
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        resolve_model(snapshot, " gpt-5.6-sol ", None)


def test_hidden_only_catalog_has_no_implicit_default_but_allows_explicit_selection(
    model_catalog_document,
) -> None:
    hidden_only = deepcopy(model_catalog_document)
    for model in hidden_only["models"]:
        model["visibility"] = "hide"
    snapshot = _parse(hidden_only)

    assert resolve_model(snapshot, "gpt-5.5", None).slug == "gpt-5.5"
    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        resolve_model(snapshot, None, None)


def test_missing_operator_configured_model_is_catalog_unavailable(
    model_catalog_document,
) -> None:
    snapshot = _parse(model_catalog_document)

    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        resolve_model(snapshot, None, "removed-model")
    with pytest.raises(ChatGPTOAuthModelNotFoundError):
        resolve_model(snapshot, "removed-model", None)


def test_claude_facade_requires_explicit_configured_backend(
    model_catalog_document,
) -> None:
    snapshot = _parse(model_catalog_document)

    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="require"):
        resolve_model(
            snapshot,
            "claude-sonnet-4-6",
            None,
            anthropic_facade=True,
        )


def test_claude_prefix_remains_a_facade_even_if_catalog_contains_that_slug(
    model_catalog_document,
) -> None:
    claude_row = deepcopy(model_catalog_document["models"][0])
    claude_row["slug"] = "claude-sonnet-4-6"
    model_catalog_document["models"].append(claude_row)
    snapshot = _parse(model_catalog_document)

    assert (
        resolve_model(
            snapshot,
            "claude-sonnet-4-6",
            "gpt-5.5",
            anthropic_facade=True,
        ).slug
        == "gpt-5.5"
    )
    with pytest.raises(ChatGPTOAuthInvalidRequestError, match="require"):
        resolve_model(
            snapshot,
            "claude-sonnet-4-6",
            None,
            anthropic_facade=True,
        )


def test_cache_reuses_only_fresh_snapshot(model_catalog_document) -> None:
    cache = ModelCatalogCache()
    calls = 0

    def load() -> CatalogLoadResult:
        nonlocal calls
        calls += 1
        return CatalogLoadResult(model_catalog_document, f'"etag-{calls}"')

    key = ("account-a", "https://example.test/codex", "0.153.3")
    first = cache.get(key, load, ttl_seconds=60)
    second = cache.get(key, load, ttl_seconds=60)

    assert first is second
    assert calls == 1


def test_cache_freshness_uses_monotonic_time_across_wall_clock_rollback(
    monkeypatch,
    model_catalog_document,
) -> None:
    import codex_as_api.model_capabilities as capabilities_module

    wall = [1_000.0]
    monotonic = [100.0]
    monkeypatch.setattr(capabilities_module.time, "time", lambda: wall[0])
    monkeypatch.setattr(capabilities_module.time, "monotonic", lambda: monotonic[0])
    cache = ModelCatalogCache()
    calls = 0

    def load() -> CatalogLoadResult:
        nonlocal calls
        calls += 1
        return CatalogLoadResult(model_catalog_document, '"etag"')

    key = ("account-a", "https://example.test/codex", "0.153.3")
    cache.get(key, load, ttl_seconds=60)
    wall[0] = -10_000.0
    monotonic[0] = 161.0
    cache.get(key, load, ttl_seconds=60)

    assert calls == 2


def test_cache_normalizes_response_etag_before_comparison(model_catalog_document) -> None:
    cache = ModelCatalogCache()
    calls = 0

    def load() -> CatalogLoadResult:
        nonlocal calls
        calls += 1
        return CatalogLoadResult(model_catalog_document, ' "etag" ')

    key = ("account-a", "https://example.test/codex", "0.153.3")
    first = cache.get(key, load, ttl_seconds=60)
    cache.invalidate_on_etag_mismatch(key, '  "etag"  ')
    second = cache.get(key, load, ttl_seconds=60)

    assert first is second
    assert calls == 1


@pytest.mark.parametrize("ttl", [0, -1, float("inf"), float("nan"), True, "300"])
def test_cache_rejects_invalid_ttl(model_catalog_document, ttl) -> None:
    cache = ModelCatalogCache()

    with pytest.raises(ValueError, match="positive finite"):
        cache.get(
            ("account-a", "https://example.test/codex", "0.153.3"),
            lambda: CatalogLoadResult(model_catalog_document, '"etag-a"'),
            ttl_seconds=ttl,
        )


def test_expired_refresh_failure_does_not_return_stale(model_catalog_document) -> None:
    contract = json.loads(
        (Path(__file__).resolve().parents[1] / "config" / "codex-upstream-contract.json").read_text(encoding="utf-8")
    )["models_request"]
    assert contract["allow_stale_on_refresh_error"] is False

    cache = ModelCatalogCache()
    key = ("account-a", "https://example.test/codex", "0.153.3")
    cache.get(
        key,
        lambda: CatalogLoadResult(model_catalog_document, '"etag-a"'),
        ttl_seconds=0.001,
    )
    time.sleep(0.01)

    def fail() -> CatalogLoadResult:
        raise ChatGPTOAuthCatalogUnavailableError("refresh failed")

    with pytest.raises(ChatGPTOAuthCatalogUnavailableError, match="refresh failed"):
        cache.get(key, fail, ttl_seconds=60)


def test_cache_coalesces_concurrent_refresh(model_catalog_document) -> None:
    cache = ModelCatalogCache()
    key = ("account-a", "https://example.test/codex", "0.153.3")
    calls = 0
    started = threading.Event()
    release = threading.Event()

    def load() -> CatalogLoadResult:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=2)
        return CatalogLoadResult(model_catalog_document, '"etag-a"')

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cache.get, key, load, ttl_seconds=60) for _ in range(4)]
        assert started.wait(timeout=2)
        release.set()
        snapshots = [future.result(timeout=2) for future in futures]

    assert calls == 1
    assert all(snapshot is snapshots[0] for snapshot in snapshots)


def test_etag_invalidation_prevents_inflight_snapshot_publication(
    model_catalog_document,
) -> None:
    cache = ModelCatalogCache()
    key = ("account-a", "https://example.test/codex", "0.153.3")
    cache.get(
        key,
        lambda: CatalogLoadResult(model_catalog_document, '"stale"'),
        ttl_seconds=0.001,
    )
    time.sleep(0.01)
    started = threading.Event()
    release = threading.Event()

    def load_stale() -> CatalogLoadResult:
        started.set()
        assert release.wait(timeout=2)
        return CatalogLoadResult(model_catalog_document, '"stale"')

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(cache.get, key, load_stale, ttl_seconds=60)
        assert started.wait(timeout=2)
        cache.invalidate_on_etag_mismatch(key, '"new"')
        release.set()
        with pytest.raises(ChatGPTOAuthCatalogUnavailableError, match="invalidated"):
            future.result(timeout=2)

    refreshed = cache.get(
        key,
        lambda: CatalogLoadResult(model_catalog_document, '"new"'),
        ttl_seconds=60,
    )
    assert refreshed.etag == '"new"'


def test_etag_observation_does_not_invalidate_initial_or_same_etag_refresh(
    model_catalog_document,
) -> None:
    cache = ModelCatalogCache()
    key = ("account-a", "https://example.test/codex", "0.153.3")

    for seed in (False, True):
        cache.clear()
        if seed:
            cache.get(
                key,
                lambda: CatalogLoadResult(model_catalog_document, '"same"'),
                ttl_seconds=0.001,
            )
            time.sleep(0.01)
        started = threading.Event()
        release = threading.Event()

        def load(started: threading.Event = started, release: threading.Event = release) -> CatalogLoadResult:
            started.set()
            assert release.wait(timeout=2)
            return CatalogLoadResult(model_catalog_document, '"same"')

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(cache.get, key, load, ttl_seconds=60)
            assert started.wait(timeout=2)
            cache.invalidate_on_etag_mismatch(key, '"same"')
            release.set()
            assert future.result(timeout=2).etag == '"same"'


def test_cache_key_namespaces_accounts(model_catalog_document) -> None:
    cache = ModelCatalogCache()
    calls = 0

    def load() -> CatalogLoadResult:
        nonlocal calls
        calls += 1
        return CatalogLoadResult(model_catalog_document, f'"etag-{calls}"')

    first = cache.get(
        ("account-a", "https://example.test/codex", "0.153.3"),
        load,
        ttl_seconds=60,
    )
    second = cache.get(
        ("account-b", "https://example.test/codex", "0.153.3"),
        load,
        ttl_seconds=60,
    )

    assert first.account_id == "account-a"
    assert second.account_id == "account-b"
    assert calls == 2


def test_python_package_configuration_excludes_static_model_catalog() -> None:
    project = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    targets = pyproject["tool"]["hatch"]["build"]["targets"]
    wheel_files = targets["wheel"]["force-include"]
    sdist_files = targets["sdist"]["include"]

    assert all(
        Path(source).name != "model-capabilities.json" and Path(destination).name != "model-capabilities.json"
        for source, destination in wheel_files.items()
    )
    assert all(Path(source).name != "model-capabilities.json" for source in sdist_files)
    assert not (project / "config" / "model-capabilities.json").exists()
    assert not (project / "src" / "codex_as_api" / "model-capabilities.json").exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (RESPONSES_LITE_ENV, ""),
        (RESPONSES_LITE_ENV, "sometimes"),
        (CODEX_METADATA_ENV, "   "),
        (CODEX_METADATA_ENV, "sometimes"),
    ],
)
def test_invalid_model_environment_fails_loudly(monkeypatch, name, value) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        validate_model_capability_environment()
