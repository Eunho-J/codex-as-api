"""Regression tests for the release-version consistency checker."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_package_versions.py"
SPEC = importlib.util.spec_from_file_location("check_package_versions", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_package_versions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_package_versions)


def write_release_metadata(root: Path, version: str = "1.2.3") -> None:
    (root / "src" / "codex_as_api").mkdir(parents=True)
    (root / "ts").mkdir()
    (root / "rust").mkdir()
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "codex-as-api"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "src" / "codex_as_api" / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    (root / "src" / "codex_as_api" / "server.py").write_text(
        f'from fastapi import FastAPI\napp = FastAPI(title="test", version="{version}")\n',
        encoding="utf-8",
    )
    (root / "ts" / "package.json").write_text(
        json.dumps({"name": "codex-as-api", "version": version}),
        encoding="utf-8",
    )
    (root / "ts" / "package-lock.json").write_text(
        json.dumps(
            {
                "name": "codex-as-api",
                "version": version,
                "packages": {"": {"name": "codex-as-api", "version": version}},
            }
        ),
        encoding="utf-8",
    )
    (root / "rust" / "Cargo.toml").write_text(
        f'[package]\nname = "codex-as-api"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (root / "rust" / "Cargo.lock").write_text(
        f'version = 4\n\n[[package]]\nname = "codex-as-api"\nversion = "{version}"\n',
        encoding="utf-8",
    )


def test_collect_versions_reads_every_release_version_surface(tmp_path: Path) -> None:
    write_release_metadata(tmp_path)

    versions = check_package_versions.collect_versions(tmp_path)

    assert versions == {
        "Python package (pyproject.toml)": "1.2.3",
        "Python runtime (__version__)": "1.2.3",
        "Python API (FastAPI app)": "1.2.3",
        "TypeScript package (ts/package.json)": "1.2.3",
        "TypeScript lockfile (ts/package-lock.json)": "1.2.3",
        "TypeScript lockfile root package": "1.2.3",
        "Rust package (rust/Cargo.toml)": "1.2.3",
        "Rust lockfile (rust/Cargo.lock)": "1.2.3",
    }
    assert check_package_versions.validate_versions(versions, "v1.2.3") == "1.2.3"


def test_validate_package_names_reads_every_published_identity(tmp_path: Path) -> None:
    write_release_metadata(tmp_path)

    names = check_package_versions.collect_package_names(tmp_path)

    assert names == {
        "Python package (pyproject.toml)": "codex-as-api",
        "TypeScript package (ts/package.json)": "codex-as-api",
        "TypeScript lockfile (ts/package-lock.json)": "codex-as-api",
        "TypeScript lockfile root package": "codex-as-api",
        "Rust package (rust/Cargo.toml)": "codex-as-api",
    }
    check_package_versions.validate_package_names(names)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (Path("pyproject.toml"), '[project]\nname = "wrong-package"\nversion = "1.2.3"\n'),
        (Path("ts/package.json"), json.dumps({"name": "wrong-package", "version": "1.2.3"})),
        (
            Path("ts/package-lock.json"),
            json.dumps(
                {
                    "name": "codex-as-api",
                    "version": "1.2.3",
                    "packages": {"": {"name": "wrong-package", "version": "1.2.3"}},
                }
            ),
        ),
        (Path("rust/Cargo.toml"), '[package]\nname = "wrong-package"\nversion = "1.2.3"\n'),
    ],
)
def test_validate_package_names_rejects_wrong_published_identity(tmp_path: Path, path: Path, replacement: str) -> None:
    write_release_metadata(tmp_path)
    (tmp_path / path).write_text(replacement, encoding="utf-8")

    with pytest.raises(check_package_versions.VersionCheckError, match="package names are inconsistent"):
        check_package_versions.validate_package_names(check_package_versions.collect_package_names(tmp_path))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (Path("src/codex_as_api/__init__.py"), '__version__ = "1.2.4"\n'),
        (Path("src/codex_as_api/server.py"), 'app = FastAPI(version="1.2.4")\n'),
        (
            Path("ts/package-lock.json"),
            json.dumps(
                {
                    "name": "codex-as-api",
                    "version": "1.2.4",
                    "packages": {"": {"name": "codex-as-api", "version": "1.2.4"}},
                }
            ),
        ),
        (Path("rust/Cargo.lock"), 'version = 4\n\n[[package]]\nname = "codex-as-api"\nversion = "1.2.4"\n'),
    ],
)
def test_validate_versions_rejects_mismatched_tracked_metadata(tmp_path: Path, path: Path, replacement: str) -> None:
    write_release_metadata(tmp_path)
    (tmp_path / path).write_text(replacement, encoding="utf-8")

    with pytest.raises(check_package_versions.VersionCheckError, match="package versions are inconsistent"):
        check_package_versions.validate_versions(check_package_versions.collect_versions(tmp_path))


def test_validate_versions_rejects_mismatched_tag(tmp_path: Path) -> None:
    write_release_metadata(tmp_path)

    with pytest.raises(check_package_versions.VersionCheckError, match="release tag"):
        check_package_versions.validate_versions(check_package_versions.collect_versions(tmp_path), "v1.2.4")


def test_collect_versions_rejects_missing_lockfile_root_version(tmp_path: Path) -> None:
    write_release_metadata(tmp_path)
    (tmp_path / "ts" / "package-lock.json").write_text(
        json.dumps({"name": "codex-as-api", "version": "1.2.3", "packages": {"": {"name": "codex-as-api"}}}),
        encoding="utf-8",
    )

    with pytest.raises(check_package_versions.VersionCheckError, match="packages.*version"):
        check_package_versions.collect_versions(tmp_path)
