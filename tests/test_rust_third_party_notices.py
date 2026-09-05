"""Tests for deterministic Rust dependency notice generation."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_rust_third_party_notices.py"
SPEC = importlib.util.spec_from_file_location("generate_rust_third_party_notices", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
notices = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = notices
SPEC.loader.exec_module(notices)


def package(tmp_path: Path, package_id: str, name: str, license_name: str = "LICENSE") -> dict[str, Any]:
    package_root = tmp_path / package_id
    package_root.mkdir()
    (package_root / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    (package_root / license_name).write_text(f"license for {name}\n", encoding="utf-8")
    return {
        "id": package_id,
        "name": name,
        "version": "1.0.0",
        "license": "MIT",
        "license_file": None,
        "manifest_path": str(package_root / "Cargo.toml"),
    }


def metadata(packages: list[dict[str, Any]], dependencies: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "packages": packages,
        "resolve": {
            "root": "root",
            "nodes": [
                {"id": "root", "deps": dependencies},
                *({"id": item["id"], "deps": []} for item in packages if item["id"] != "root"),
            ],
        },
    }


def dependency(package_id: str, kind: str | None = None) -> dict[str, Any]:
    return {"pkg": package_id, "dep_kinds": [{"kind": kind, "target": None}]}


def test_collects_target_union_of_only_normal_locked_dependencies(tmp_path: Path) -> None:
    root = package(tmp_path, "root", "codex-as-api")
    common = package(tmp_path, "common", "common")
    linux = package(tmp_path, "linux", "linux-only")
    windows = package(tmp_path, "windows", "windows-only", "COPYING.txt")
    dev = package(tmp_path, "dev", "dev-only")
    build = package(tmp_path, "build", "build-only")
    linux_metadata = metadata(
        [root, common, linux, dev, build],
        [dependency("common"), dependency("linux"), dependency("dev", "dev"), dependency("build", "build")],
    )
    windows_metadata = metadata(
        [root, common, windows, dev, build],
        [dependency("common"), dependency("windows"), dependency("dev", "dev"), dependency("build", "build")],
    )

    dependencies = notices.collect_runtime_dependencies(
        {"x86_64-pc-windows-msvc": windows_metadata, "x86_64-unknown-linux-gnu": linux_metadata}
    )
    bundled_notices = b"bundled tokenizer notice\n"
    generated = notices.render_notices(dependencies, bundled_notices)

    assert [item.name for item in dependencies] == ["common", "linux-only", "windows-only"]
    assert dependencies[0].targets == ("x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu")
    assert b"license for common" in generated
    assert b"license for linux-only" in generated
    assert b"license for windows-only" in generated
    assert bundled_notices in generated
    assert b"dev-only" not in generated
    assert b"build-only" not in generated
    assert notices.render_notices(dependencies, bundled_notices) == generated


def test_rejects_a_runtime_dependency_without_a_shipped_license_file(tmp_path: Path) -> None:
    root = package(tmp_path, "root", "codex-as-api")
    missing = package(tmp_path, "missing", "missing")
    (tmp_path / "missing" / "LICENSE").unlink()

    with pytest.raises(notices.NoticeGenerationError, match="no shipped license file"):
        notices.collect_runtime_dependencies(
            {"x86_64-unknown-linux-gnu": metadata([root, missing], [dependency("missing")])}
        )


def test_rejects_an_empty_or_non_utf8_license_file(tmp_path: Path) -> None:
    empty_file = tmp_path / "EMPTY"
    empty_file.write_bytes(b"")
    invalid_file = tmp_path / "INVALID"
    invalid_file.write_bytes(b"\xff")
    base = {
        "package_id": "dep",
        "name": "dep",
        "version": "1.0.0",
        "license_expression": "MIT",
        "targets": ("x86_64-unknown-linux-gnu",),
    }

    with pytest.raises(notices.NoticeGenerationError, match="must not be empty"):
        notices.render_notices(
            [notices.RuntimeDependency(**base, license_files=(empty_file,))],
            b"bundled notice\n",
        )
    with pytest.raises(notices.NoticeGenerationError, match="must be UTF-8"):
        notices.render_notices(
            [notices.RuntimeDependency(**base, license_files=(invalid_file,))],
            b"bundled notice\n",
        )


def test_rejects_missing_bundled_asset_notices(tmp_path: Path) -> None:
    license_file = tmp_path / "LICENSE"
    license_file.write_text("license\n", encoding="utf-8")
    dependency_record = notices.RuntimeDependency(
        package_id="dep",
        name="dep",
        version="1.0.0",
        license_expression="MIT",
        license_files=(license_file,),
        targets=("x86_64-unknown-linux-gnu",),
    )

    with pytest.raises(notices.NoticeGenerationError, match="must not be empty"):
        notices.render_notices([dependency_record], b"")


def test_normalizes_all_input_line_endings_before_hashing_and_rendering(tmp_path: Path) -> None:
    license_file = tmp_path / "LICENSE"
    license_file.write_bytes(b"first\r\nsecond\rthird\n")
    dependency_record = notices.RuntimeDependency(
        package_id="dep",
        name="dep",
        version="1.0.0",
        license_expression="MIT",
        license_files=(license_file,),
        targets=("x86_64-unknown-linux-gnu",),
    )

    generated = notices.render_notices([dependency_record], b"bundled\r\nnotice\r")
    normalized_license = b"first\nsecond\nthird\n"

    assert b"\r" not in generated
    assert b"bundled\nnotice\n" in generated
    assert normalized_license in generated
    assert hashlib.sha256(normalized_license).hexdigest().encode() in generated
