#!/usr/bin/env python3
"""Check that the Python, npm, and Rust packages describe one release."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Mapping
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - supports local Python 3.10 runs
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON_PACKAGE_NAME = "codex-as-api"
EXPECTED_TYPESCRIPT_PACKAGE_NAME = "codex-as-api"
EXPECTED_RUST_PACKAGE_NAME = "codex-as-api"


class VersionCheckError(ValueError):
    """Raised when a tracked release version cannot be read or does not agree."""


def require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise VersionCheckError(f"{description} must be a non-empty string")
    return value


def read_toml(path: Path) -> Mapping[str, object]:
    with path.open("rb") as source:
        document: Mapping[str, object] = tomllib.load(source)
    return document


def read_toml_version(path: Path, section: str) -> str:
    document = read_toml(path)
    section_data = document.get(section)
    if not isinstance(section_data, Mapping):
        raise VersionCheckError(f"{path}: missing [{section}] table")
    return require_string(section_data.get("version"), f"{path}: [{section}].version")


def read_toml_name(path: Path, section: str) -> str:
    document = read_toml(path)
    section_data = document.get(section)
    if not isinstance(section_data, Mapping):
        raise VersionCheckError(f"{path}: missing [{section}] table")
    return require_string(section_data.get("name"), f"{path}: [{section}].name")


def read_json(path: Path) -> Mapping[str, object]:
    with path.open(encoding="utf-8") as source:
        document = json.load(source)
    if not isinstance(document, Mapping):
        raise VersionCheckError(f"{path}: expected a JSON object")
    return document


def read_json_version(path: Path, *keys: str) -> str:
    value: object = read_json(path)
    location = str(path)
    for key in keys:
        if not isinstance(value, Mapping):
            raise VersionCheckError(f"{location}: {key} is not an object")
        value = value.get(key)
        location = f"{location}.{key}"
    return require_string(value, location)


def read_python_string_assignment(path: Path, variable: str) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: list[str] = []
    for statement in module.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if not any(isinstance(target, ast.Name) and target.id == variable for target in targets):
            continue
        value = statement.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str) or not value.value:
            raise VersionCheckError(f"{path}: {variable} must be assigned a non-empty string literal")
        values.append(value.value)
    if len(values) != 1:
        raise VersionCheckError(f"{path}: expected exactly one {variable} assignment")
    return values[0]


def read_fastapi_app_version(path: Path) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    versions: list[str] = []
    for node in ast.walk(module):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "app" for target in targets):
            continue
        if (
            not isinstance(node.value, ast.Call)
            or not isinstance(node.value.func, ast.Name)
            or node.value.func.id != "FastAPI"
        ):
            continue
        version_keyword = next((keyword for keyword in node.value.keywords if keyword.arg == "version"), None)
        if version_keyword is None:
            raise VersionCheckError(f"{path}: FastAPI app is missing a version keyword")
        value = version_keyword.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str) or not value.value:
            raise VersionCheckError(f"{path}: FastAPI app version must be a non-empty string literal")
        versions.append(value.value)
    if len(versions) != 1:
        raise VersionCheckError(f"{path}: expected exactly one FastAPI app version")
    return versions[0]


def read_cargo_lock_version(path: Path) -> str:
    packages = read_toml(path).get("package")
    if not isinstance(packages, list):
        raise VersionCheckError(f"{path}: missing [[package]] entries")
    matching_packages = [
        package
        for package in packages
        if isinstance(package, Mapping) and package.get("name") == EXPECTED_RUST_PACKAGE_NAME
    ]
    if len(matching_packages) != 1:
        raise VersionCheckError(f"{path}: expected exactly one codex-as-api package entry")
    return require_string(matching_packages[0].get("version"), f"{path}: codex-as-api package version")


def read_json_object(path: Path, *keys: str) -> Mapping[str, object]:
    value: object = read_json(path)
    location = str(path)
    for key in keys:
        if not isinstance(value, Mapping):
            raise VersionCheckError(f"{location}: {key} is not an object")
        value = value.get(key)
        location = f"{location}.{key}"
    if not isinstance(value, Mapping):
        raise VersionCheckError(f"{location} is not an object")
    return value


def collect_package_names(root: Path) -> dict[str, str]:
    package_json = root / "ts" / "package.json"
    package_lock = root / "ts" / "package-lock.json"
    return {
        "Python package (pyproject.toml)": read_toml_name(root / "pyproject.toml", "project"),
        "TypeScript package (ts/package.json)": require_string(
            read_json(package_json).get("name"), f"{package_json}.name"
        ),
        "TypeScript lockfile (ts/package-lock.json)": require_string(
            read_json(package_lock).get("name"), f"{package_lock}.name"
        ),
        "TypeScript lockfile root package": require_string(
            read_json_object(package_lock, "packages", "").get("name"),
            f"{package_lock}.packages..name",
        ),
        "Rust package (rust/Cargo.toml)": read_toml_name(root / "rust" / "Cargo.toml", "package"),
    }


def validate_package_names(names: Mapping[str, str]) -> None:
    expected = {
        "Python package (pyproject.toml)": EXPECTED_PYTHON_PACKAGE_NAME,
        "TypeScript package (ts/package.json)": EXPECTED_TYPESCRIPT_PACKAGE_NAME,
        "TypeScript lockfile (ts/package-lock.json)": EXPECTED_TYPESCRIPT_PACKAGE_NAME,
        "TypeScript lockfile root package": EXPECTED_TYPESCRIPT_PACKAGE_NAME,
        "Rust package (rust/Cargo.toml)": EXPECTED_RUST_PACKAGE_NAME,
    }
    mismatches = [
        f"  {surface}: expected {expected[surface]!r}, got {actual!r}"
        for surface, actual in names.items()
        if actual != expected[surface]
    ]
    if mismatches:
        raise VersionCheckError("package names are inconsistent:\n" + "\n".join(mismatches))


def collect_versions(root: Path) -> dict[str, str]:
    return {
        "Python package (pyproject.toml)": read_toml_version(root / "pyproject.toml", "project"),
        "Python runtime (__version__)": read_python_string_assignment(
            root / "src" / "codex_as_api" / "__init__.py", "__version__"
        ),
        "Python API (FastAPI app)": read_fastapi_app_version(root / "src" / "codex_as_api" / "server.py"),
        "TypeScript package (ts/package.json)": read_json_version(root / "ts" / "package.json", "version"),
        "TypeScript lockfile (ts/package-lock.json)": read_json_version(root / "ts" / "package-lock.json", "version"),
        "TypeScript lockfile root package": read_json_version(
            root / "ts" / "package-lock.json", "packages", "", "version"
        ),
        "Rust package (rust/Cargo.toml)": read_toml_version(root / "rust" / "Cargo.toml", "package"),
        "Rust lockfile (rust/Cargo.lock)": read_cargo_lock_version(root / "rust" / "Cargo.lock"),
    }


def validate_versions(versions: Mapping[str, str], tag: str | None = None) -> str:
    distinct_versions = set(versions.values())
    if len(distinct_versions) != 1:
        details = "\n".join(f"  {package}: {version}" for package, version in versions.items())
        raise VersionCheckError(f"package versions are inconsistent:\n{details}")

    version = distinct_versions.pop()
    if tag is not None and tag != f"v{version}":
        raise VersionCheckError(
            f"release tag {tag!r} does not match package version {version!r}; expected {f'v{version}'!r}"
        )
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        help="Release tag to validate. It must be v followed by the shared package version.",
    )
    args = parser.parse_args()

    try:
        validate_package_names(collect_package_names(ROOT))
        version = validate_versions(collect_versions(ROOT), args.tag)
    except VersionCheckError as error:
        print(error, file=sys.stderr)
        return 1

    print(f"package versions are consistent: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
