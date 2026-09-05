"""Regression tests for release archive identity validation."""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_release_artifacts.py"
SPEC = importlib.util.spec_from_file_location("check_release_artifacts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_release_artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_release_artifacts)
RUST_PACKAGE_SCRIPT_PATH = SCRIPT_PATH.with_name("package_rust_release.py")
RUST_PACKAGE_SPEC = importlib.util.spec_from_file_location("package_rust_release", RUST_PACKAGE_SCRIPT_PATH)
assert RUST_PACKAGE_SPEC is not None and RUST_PACKAGE_SPEC.loader is not None
package_rust_release = importlib.util.module_from_spec(RUST_PACKAGE_SPEC)
RUST_PACKAGE_SPEC.loader.exec_module(package_rust_release)


def core_metadata(name: str, version: str) -> bytes:
    return f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n".encode()


def add_tar_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def make_wheel(path: Path, name: str = "codex-as-api", version: str = "0.7.0") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "codex_as_api-0.7.0.dist-info/METADATA",
            core_metadata(name, version),
        )
        archive.writestr(
            "codex_as_api-0.7.0.dist-info/entry_points.txt",
            "[console_scripts]\ncodex-as-api = codex_as_api.server:main\n",
        )


def make_sdist(path: Path, name: str = "codex-as-api", version: str = "0.7.0") -> None:
    with tarfile.open(path, "w:gz") as archive:
        add_tar_bytes(
            archive,
            "codex_as_api-0.7.0/PKG-INFO",
            core_metadata(name, version),
        )
        for relative_path in check_release_artifacts.SDIST_REQUIRED_FILES:
            add_tar_bytes(
                archive,
                f"codex_as_api-0.7.0/{relative_path}",
                b"#!/usr/bin/env python3\n",
            )


def make_npm_tarball(path: Path, name: str, version: str = "0.7.0") -> None:
    with tarfile.open(path, "w:gz") as archive:
        package_data = {
            "name": name,
            "version": version,
            **check_release_artifacts.NPM_ENTRYPOINT_METADATA,
            "exports": check_release_artifacts.NPM_EXPORTS_METADATA,
            "bin": check_release_artifacts.NPM_BIN_METADATA,
        }
        add_tar_bytes(
            archive,
            "package/package.json",
            json.dumps(package_data).encode(),
        )
        entrypoint_targets = {
            *check_release_artifacts.NPM_ENTRYPOINT_METADATA.values(),
            *check_release_artifacts.NPM_BIN_METADATA.values(),
            *check_release_artifacts.NPM_EXPORTS_METADATA["."].values(),
        }
        for target in entrypoint_targets:
            add_tar_bytes(archive, f"package/{target.removeprefix('./')}", b"release entry point")
        for filename in check_release_artifacts.NPM_REQUIRED_TEXT_FILES:
            add_tar_bytes(
                archive,
                f"package/{filename}",
                (SCRIPT_PATH.parents[1] / filename).read_bytes(),
            )


def make_rust_archive(tmp_path: Path, platform: str, version: str = "0.7.0") -> Path:
    binary_name = package_rust_release.PLATFORM_BINARIES[platform]
    binary = tmp_path / platform / binary_name
    binary.parent.mkdir(parents=True)
    binary.write_bytes(fake_rust_binary(platform))
    return Path(package_rust_release.package_rust_release(binary, platform, version, tmp_path / "dist"))


def fake_rust_binary(platform: str) -> bytes:
    if platform == "linux":
        return b"\x7fELF" + bytes(60)
    if platform == "macos":
        return b"\xcf\xfa\xed\xfe" + bytes(28)
    binary = bytearray(128)
    binary[:2] = b"MZ"
    binary[0x3C:0x40] = (0x40).to_bytes(4, "little")
    binary[0x40:0x44] = b"PE\0\0"
    return bytes(binary)


def test_accepts_exact_python_and_npm_release_identities(tmp_path: Path) -> None:
    wheel = tmp_path / "codex_as_api-0.7.0-py3-none-any.whl"
    sdist = tmp_path / "codex_as_api-0.7.0.tar.gz"
    npm = tmp_path / "codex-as-api-0.7.0.tgz"
    github_npm = tmp_path / "eunho-j-codex-as-api-0.7.0.tgz"
    make_wheel(wheel)
    make_sdist(sdist)
    make_npm_tarball(npm, "codex-as-api")
    make_npm_tarball(github_npm, "@eunho-j/codex-as-api")

    check_release_artifacts.validate_python_distributions([wheel, sdist], "0.7.0")
    check_release_artifacts.validate_npm_tarball(npm, "codex-as-api", "0.7.0")
    check_release_artifacts.validate_npm_tarball(
        github_npm,
        "@eunho-j/codex-as-api",
        "0.7.0",
    )


@pytest.mark.parametrize(
    ("package_name", "version"),
    [
        ("wrong-package", "0.7.0"),
        ("codex-as-api", "0.7.1"),
    ],
)
def test_rejects_wrong_python_metadata_identity(tmp_path: Path, package_name: str, version: str) -> None:
    wheel = tmp_path / "codex_as_api-0.7.0-py3-none-any.whl"
    make_wheel(wheel, package_name, version)

    with pytest.raises(check_release_artifacts.ArtifactIdentityError):
        check_release_artifacts.validate_wheel(wheel, "0.7.0")


@pytest.mark.parametrize(
    ("package_name", "version"),
    [
        ("wrong-package", "0.7.0"),
        ("codex-as-api", "0.7.1"),
    ],
)
def test_rejects_wrong_npm_metadata_identity(tmp_path: Path, package_name: str, version: str) -> None:
    npm = tmp_path / "codex-as-api-0.7.0.tgz"
    make_npm_tarball(npm, package_name, version)

    with pytest.raises(check_release_artifacts.ArtifactIdentityError):
        check_release_artifacts.validate_npm_tarball(npm, "codex-as-api", "0.7.0")


def test_rejects_missing_or_extra_python_distribution_artifacts(tmp_path: Path) -> None:
    wheel = tmp_path / "codex_as_api-0.7.0-py3-none-any.whl"
    duplicate_wheel = tmp_path / "codex_as_api-0.7.0-py3-none-linux.whl"
    sdist = tmp_path / "codex_as_api-0.7.0.tar.gz"
    make_wheel(wheel)
    make_wheel(duplicate_wheel)
    make_sdist(sdist)

    with pytest.raises(check_release_artifacts.ArtifactIdentityError):
        check_release_artifacts.validate_python_distributions([wheel], "0.7.0")
    with pytest.raises(check_release_artifacts.ArtifactIdentityError):
        check_release_artifacts.validate_python_distributions(
            [wheel, duplicate_wheel, sdist],
            "0.7.0",
        )


def test_rejects_filename_identity_mismatch_before_reading_archive(tmp_path: Path) -> None:
    wrong_filename = tmp_path / "renamed.tgz"
    make_npm_tarball(wrong_filename, "codex-as-api")

    with pytest.raises(check_release_artifacts.ArtifactIdentityError):
        check_release_artifacts.validate_npm_tarball(
            wrong_filename,
            "codex-as-api",
            "0.7.0",
        )


def test_rejects_wheel_without_the_exact_console_script(tmp_path: Path) -> None:
    wheel = tmp_path / "codex_as_api-0.7.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "codex_as_api-0.7.0.dist-info/METADATA",
            core_metadata("codex-as-api", "0.7.0"),
        )
        archive.writestr(
            "codex_as_api-0.7.0.dist-info/entry_points.txt",
            "[console_scripts]\ncodex-as-api = codex_as_api.server:wrong\n",
        )

    with pytest.raises(check_release_artifacts.ArtifactIdentityError, match="console scripts must match"):
        check_release_artifacts.validate_wheel(wheel, "0.7.0")


def test_rejects_npm_tarball_without_the_repository_license(tmp_path: Path) -> None:
    npm = tmp_path / "codex-as-api-0.7.0.tgz"
    with tarfile.open(npm, "w:gz") as archive:
        package_data = {
            "name": "codex-as-api",
            "version": "0.7.0",
            **check_release_artifacts.NPM_ENTRYPOINT_METADATA,
            "exports": check_release_artifacts.NPM_EXPORTS_METADATA,
            "bin": check_release_artifacts.NPM_BIN_METADATA,
        }
        add_tar_bytes(
            archive,
            "package/package.json",
            json.dumps(package_data).encode(),
        )
        entrypoint_targets = {
            *check_release_artifacts.NPM_ENTRYPOINT_METADATA.values(),
            *check_release_artifacts.NPM_BIN_METADATA.values(),
            *check_release_artifacts.NPM_EXPORTS_METADATA["."].values(),
        }
        for target in entrypoint_targets:
            add_tar_bytes(archive, f"package/{target.removeprefix('./')}", b"release entry point")
        add_tar_bytes(
            archive,
            "package/THIRD_PARTY_NOTICES.md",
            (SCRIPT_PATH.parents[1] / "THIRD_PARTY_NOTICES.md").read_bytes(),
        )

    with pytest.raises(check_release_artifacts.ArtifactIdentityError, match="package/LICENSE"):
        check_release_artifacts.validate_npm_tarball(npm, "codex-as-api", "0.7.0")


def test_rejects_sdist_without_release_validator_scripts(tmp_path: Path) -> None:
    sdist = tmp_path / "codex_as_api-0.7.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        add_tar_bytes(
            archive,
            "codex_as_api-0.7.0/PKG-INFO",
            core_metadata("codex-as-api", "0.7.0"),
        )

    with pytest.raises(
        check_release_artifacts.ArtifactIdentityError,
        match="scripts/check_no_static_model_catalog.py",
    ):
        check_release_artifacts.validate_sdist(sdist, "0.7.0")


def test_rejects_sdist_without_generated_rust_notices(tmp_path: Path) -> None:
    sdist = tmp_path / "codex_as_api-0.7.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        add_tar_bytes(
            archive,
            "codex_as_api-0.7.0/PKG-INFO",
            core_metadata("codex-as-api", "0.7.0"),
        )
        for relative_path in check_release_artifacts.SDIST_REQUIRED_SCRIPT_FILES:
            add_tar_bytes(
                archive,
                f"codex_as_api-0.7.0/{relative_path}",
                b"#!/usr/bin/env python3\n",
            )

    with pytest.raises(check_release_artifacts.ArtifactIdentityError, match="rust/THIRD_PARTY_NOTICES.md"):
        check_release_artifacts.validate_sdist(sdist, "0.7.0")


@pytest.mark.parametrize("platform", ["linux", "macos", "windows"])
def test_packages_and_validates_rust_archives_with_repository_notices(tmp_path: Path, platform: str) -> None:
    archive = make_rust_archive(tmp_path, platform)

    check_release_artifacts.validate_rust_archive(archive, platform, "0.7.0")


def test_rust_release_archive_is_deterministic(tmp_path: Path) -> None:
    first = make_rust_archive(tmp_path / "first", "linux")
    second = make_rust_archive(tmp_path / "second", "linux")

    assert first.read_bytes() == second.read_bytes()


def test_rejects_rust_archive_without_license_files(tmp_path: Path) -> None:
    archive = tmp_path / "codex-as-api-0.7.0-linux.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("codex-as-api", b"release-binary")

    with pytest.raises(check_release_artifacts.ArtifactIdentityError, match="exact Rust archive members"):
        check_release_artifacts.validate_rust_archive(archive, "linux", "0.7.0")


def test_rejects_npm_tarball_with_a_missing_export_target(tmp_path: Path) -> None:
    npm = tmp_path / "codex-as-api-0.7.0.tgz"
    make_npm_tarball(npm, "codex-as-api")
    rewritten = tmp_path / "rewritten.tgz"
    with tarfile.open(npm) as source, tarfile.open(rewritten, "w:gz") as output:
        for member in source.getmembers():
            if member.name == "package/dist/index.js":
                continue
            content = source.extractfile(member) if member.isfile() else None
            output.addfile(member, content)
    rewritten.replace(npm)

    with pytest.raises(check_release_artifacts.ArtifactIdentityError, match="package/dist/index.js"):
        check_release_artifacts.validate_npm_tarball(npm, "codex-as-api", "0.7.0")


def test_rejects_rust_archive_with_non_executable_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "codex-as-api-0.7.0-linux.zip"
    with zipfile.ZipFile(archive, "w") as output:
        binary = zipfile.ZipInfo("codex-as-api")
        binary.external_attr = 0o100755 << 16
        output.writestr(binary, b"x")
        for filename, source_path in check_release_artifacts.RUST_REQUIRED_TEXT_SOURCES.items():
            output.writestr(filename, source_path.read_bytes())

    with pytest.raises(check_release_artifacts.ArtifactIdentityError, match="ELF executable"):
        check_release_artifacts.validate_rust_archive(archive, "linux", "0.7.0")
