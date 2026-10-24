"""Regression tests for the release static-catalog guard."""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_no_static_model_catalog.py"
SPEC = importlib.util.spec_from_file_location("check_no_static_model_catalog", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_no_static_model_catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_no_static_model_catalog)


def catalog_document() -> dict[str, object]:
    return {
        "models": {
            "live-model": {
                "context_window": 100_000,
                "supported_reasoning_levels": ["low", "medium"],
            }
        }
    }


def add_tar_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def test_allows_contract_metadata_that_only_names_the_models_response_path(tmp_path: Path) -> None:
    contract = tmp_path / "codex-upstream-contract.json"
    contract.write_text(
        json.dumps(
            {
                "models_request": {
                    "response_models_path": ["models"],
                    "cache_ttl_seconds": 300,
                }
            }
        ),
        encoding="utf-8",
    )

    assert check_no_static_model_catalog.forbidden_paths(contract) == []


@pytest.mark.parametrize(
    "name",
    [
        "models.json",
        "catalog.json",
        "model-capabilities.json",
        "model_capabilities_v2.json",
        "archived-model-catalog.json",
    ],
)
def test_rejects_catalog_like_json_names_even_without_rows(tmp_path: Path, name: str) -> None:
    catalog = tmp_path / name
    catalog.write_text("{}", encoding="utf-8")

    assert check_no_static_model_catalog.forbidden_paths(tmp_path)


def test_rejects_live_array_catalog_nested_in_the_allowed_contract_name(tmp_path: Path) -> None:
    contract = tmp_path / "codex-upstream-contract.json"
    contract.write_text(
        json.dumps(
            {
                "runtime": {
                    "models": [
                        {
                            "slug": "live-model",
                            "context_window": 100_000,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    assert check_no_static_model_catalog.forbidden_paths(contract)


def test_rejects_even_an_empty_runtime_models_payload_in_the_allowed_contract(tmp_path: Path) -> None:
    contract = tmp_path / "codex-upstream-contract.json"
    contract.write_text(json.dumps({"runtime": {"models": []}}), encoding="utf-8")

    assert check_no_static_model_catalog.forbidden_paths(contract)


@pytest.mark.parametrize("archive_kind", ["zip", "tar"])
def test_rejects_structural_catalogs_inside_release_archives(tmp_path: Path, archive_kind: str) -> None:
    content = json.dumps(catalog_document()).encode()
    if archive_kind == "zip":
        archive_path = tmp_path / "package.whl"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("package/data.json", content)
    else:
        archive_path = tmp_path / "package.tgz"
        with tarfile.open(archive_path, "w:gz") as archive:
            add_tar_bytes(archive, "package/data.json", content)

    assert check_no_static_model_catalog.forbidden_paths(archive_path)


def test_rejects_renamed_mapping_catalog_in_a_source_directory(tmp_path: Path) -> None:
    forbidden = tmp_path / "innocent-name.json"
    forbidden.write_text(json.dumps(catalog_document()), encoding="utf-8")

    assert check_no_static_model_catalog.forbidden_paths(tmp_path)


def test_rejects_catalog_rows_identified_only_by_new_live_compatibility_fields(tmp_path: Path) -> None:
    forbidden = tmp_path / "innocent-name.json"
    forbidden.write_text(
        json.dumps({"models": [{"slug": "live-model", "comp_hash": "family-a"}]}),
        encoding="utf-8",
    )

    assert check_no_static_model_catalog.forbidden_paths(tmp_path)


def test_invalid_json_fails_closed(tmp_path: Path) -> None:
    invalid = tmp_path / "data.json"
    invalid.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError):
        check_no_static_model_catalog.forbidden_paths(invalid)


def test_allows_canonical_verified_hidden_models_config_in_repo_and_packages(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    canonical_config = repo_root / "config" / "verified-hidden-models.json"
    content = canonical_config.read_bytes()

    assert check_no_static_model_catalog.forbidden_paths(canonical_config) == []

    for relpath in (
        Path("config") / "verified-hidden-models.json",
        Path("codex_as_api") / "verified-hidden-models.json",
        Path("src") / "codex_as_api" / "verified-hidden-models.json",
    ):
        target = tmp_path / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    assert check_no_static_model_catalog.forbidden_paths(tmp_path) == []

    wheel_path = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("codex_as_api/verified-hidden-models.json", content)
    assert check_no_static_model_catalog.forbidden_paths(wheel_path) == []

    tar_path = tmp_path / "package.tgz"
    with tarfile.open(tar_path, "w:gz") as archive:
        add_tar_bytes(archive, "package/config/verified-hidden-models.json", content)
        add_tar_bytes(archive, "codex_as_api-0.1.0/src/codex_as_api/verified-hidden-models.json", content)
    assert check_no_static_model_catalog.forbidden_paths(tar_path) == []


def test_rejects_verified_hidden_models_payload_outside_canonical_paths_or_with_invalid_schema(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    canonical_document = json.loads(
        (repo_root / "config" / "verified-hidden-models.json").read_text(encoding="utf-8")
    )

    non_canonical = tmp_path / "verified-hidden-models.json"
    non_canonical.write_text(json.dumps(canonical_document), encoding="utf-8")
    assert check_no_static_model_catalog.forbidden_paths(tmp_path)
    non_canonical.unlink()

    bad_slug_doc = json.loads(json.dumps(canonical_document))
    bad_slug_doc["models"][0]["slug"] = "gpt-5.4"
    canonical_target = tmp_path / "config" / "verified-hidden-models.json"
    canonical_target.parent.mkdir(parents=True, exist_ok=True)
    canonical_target.write_text(json.dumps(bad_slug_doc), encoding="utf-8")
    assert check_no_static_model_catalog.forbidden_paths(tmp_path)

    for field in ("default_reasoning_summary", "default_verbosity"):
        for invalid_value in ({}, []):
            malformed = json.loads(json.dumps(canonical_document))
            malformed["models"][0][field] = invalid_value
            canonical_target.write_text(json.dumps(malformed), encoding="utf-8")
            assert check_no_static_model_catalog.forbidden_paths(tmp_path)

    extra_field_doc = json.loads(json.dumps(canonical_document))
    extra_field_doc["models"][0]["base_instructions"] = "forbid prompt payloads"
    canonical_target.write_text(json.dumps(extra_field_doc), encoding="utf-8")
    assert check_no_static_model_catalog.forbidden_paths(tmp_path)
