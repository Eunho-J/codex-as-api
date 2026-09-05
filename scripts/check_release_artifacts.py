#!/usr/bin/env python3
"""Validate the identities of Python and npm release archives before publishing."""

from __future__ import annotations

import argparse
import configparser
import json
import re
import tarfile
import zipfile
from collections.abc import Mapping, Sequence
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath
from typing import TypeVar

PYTHON_PACKAGE_NAME = "codex-as-api"
NPM_PACKAGE_NAME = "codex-as-api"
GITHUB_NPM_PACKAGE_NAME = "@eunho-j/codex-as-api"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NPM_REQUIRED_TEXT_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md")
SDIST_REQUIRED_SCRIPT_FILES = (
    "scripts/check_no_static_model_catalog.py",
    "scripts/check_package_versions.py",
    "scripts/check_registry_release.py",
    "scripts/check_release_artifacts.py",
    "scripts/generate_rust_third_party_notices.py",
    "scripts/package_rust_release.py",
)
SDIST_REQUIRED_FILES = (*SDIST_REQUIRED_SCRIPT_FILES, "rust/THIRD_PARTY_NOTICES.md")
RUST_PLATFORM_BINARIES = {
    "linux": "codex-as-api",
    "macos": "codex-as-api",
    "windows": "codex-as-api.exe",
}
NPM_ENTRYPOINT_METADATA = {
    "main": "./dist/index.cjs",
    "module": "./dist/index.js",
    "types": "./dist/index.d.ts",
}
NPM_EXPORTS_METADATA = {
    ".": {
        "types": "./dist/index.d.ts",
        "import": "./dist/index.js",
        "require": "./dist/index.cjs",
    }
}
NPM_BIN_METADATA = {"codex-as-api": "dist/cli.js"}
PYTHON_CONSOLE_SCRIPTS = {"codex-as-api": "codex_as_api.server:main"}
RUST_REQUIRED_TEXT_SOURCES = {
    "LICENSE": REPOSITORY_ROOT / "LICENSE",
    "THIRD_PARTY_NOTICES.md": REPOSITORY_ROOT / "rust" / "THIRD_PARTY_NOTICES.md",
}
T = TypeVar("T")


class ArtifactIdentityError(ValueError):
    """Raised when an archive does not contain the package it is meant to publish."""


def normalize_python_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def python_archive_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", value)


def require_exactly_one(values: Sequence[T], description: str) -> T:
    if len(values) != 1:
        raise ArtifactIdentityError(f"expected exactly one {description}, found {len(values)}")
    return values[0]


def parse_core_metadata(content: bytes, label: str) -> tuple[str, str]:
    metadata = BytesParser(policy=default).parsebytes(content)
    names = metadata.get_all("Name", [])
    versions = metadata.get_all("Version", [])
    name = require_exactly_one(names, f"Name field in {label}")
    version = require_exactly_one(versions, f"Version field in {label}")
    if not name or not version:
        raise ArtifactIdentityError(f"{label}: Name and Version must be non-empty")
    return name, version


def validate_python_identity(name: str, version: str, expected_version: str, label: str) -> None:
    if normalize_python_package_name(name) != PYTHON_PACKAGE_NAME:
        raise ArtifactIdentityError(f"{label}: expected Python package {PYTHON_PACKAGE_NAME!r}, got {name!r}")
    if version != expected_version:
        raise ArtifactIdentityError(f"{label}: expected Python version {expected_version!r}, got {version!r}")


def validate_wheel(path: Path, expected_version: str) -> None:
    expected_prefix = f"{python_archive_component(PYTHON_PACKAGE_NAME)}-{python_archive_component(expected_version)}-"
    if not path.name.startswith(expected_prefix) or path.suffix.lower() != ".whl":
        raise ArtifactIdentityError(f"{path}: wheel filename must begin with {expected_prefix!r} and end in .whl")
    expected_metadata = (
        f"{python_archive_component(PYTHON_PACKAGE_NAME)}-"
        f"{python_archive_component(expected_version)}.dist-info/METADATA"
    )
    expected_entry_points = expected_metadata.removesuffix("METADATA") + "entry_points.txt"
    if not zipfile.is_zipfile(path):
        raise ArtifactIdentityError(f"{path}: wheel is not a valid ZIP archive")
    with zipfile.ZipFile(path) as archive:
        metadata_members = [
            member.filename
            for member in archive.infolist()
            if not member.is_dir() and member.filename.endswith(".dist-info/METADATA")
        ]
        metadata_name = require_exactly_one(metadata_members, f".dist-info/METADATA in {path}")
        if metadata_name != expected_metadata:
            raise ArtifactIdentityError(
                f"{path}: expected metadata member {expected_metadata!r}, got {metadata_name!r}"
            )
        name, version = parse_core_metadata(archive.read(metadata_name), f"{path}:{metadata_name}")
        entry_point_members = [
            member.filename
            for member in archive.infolist()
            if not member.is_dir() and member.filename.endswith(".dist-info/entry_points.txt")
        ]
        entry_points_name = require_exactly_one(entry_point_members, f".dist-info/entry_points.txt in {path}")
        if entry_points_name != expected_entry_points:
            raise ArtifactIdentityError(
                f"{path}: expected entry-points member {expected_entry_points!r}, got {entry_points_name!r}"
            )
        try:
            entry_points_text = archive.read(entry_points_name).decode("utf-8")
            entry_points = configparser.ConfigParser(interpolation=None, strict=True)
            entry_points.read_string(entry_points_text)
        except (UnicodeDecodeError, configparser.Error) as error:
            raise ArtifactIdentityError(f"{path}:{entry_points_name}: invalid entry points: {error}") from error
        if entry_points.sections() != ["console_scripts"]:
            raise ArtifactIdentityError(f"{path}:{entry_points_name}: expected only [console_scripts]")
        if dict(entry_points.items("console_scripts")) != PYTHON_CONSOLE_SCRIPTS:
            raise ArtifactIdentityError(
                f"{path}:{entry_points_name}: console scripts must match {PYTHON_CONSOLE_SCRIPTS!r}"
            )
    validate_python_identity(name, version, expected_version, str(path))


def validate_sdist(path: Path, expected_version: str) -> None:
    archive_stem = f"{python_archive_component(PYTHON_PACKAGE_NAME)}-{python_archive_component(expected_version)}"
    expected_filename = f"{archive_stem}.tar.gz"
    if path.name != expected_filename:
        raise ArtifactIdentityError(f"{path}: expected source distribution filename {expected_filename!r}")
    if not tarfile.is_tarfile(path):
        raise ArtifactIdentityError(f"{path}: source distribution is not a valid tar archive")
    expected_metadata = f"{archive_stem}/PKG-INFO"
    with tarfile.open(path) as archive:
        file_members = {member.name for member in archive.getmembers() if member.isfile()}
        for relative_path in SDIST_REQUIRED_FILES:
            expected_member = f"{archive_stem}/{relative_path}"
            if expected_member not in file_members:
                raise ArtifactIdentityError(f"{path}: missing required source-distribution file {expected_member!r}")
        metadata_members = [
            member
            for member in archive.getmembers()
            if member.isfile() and PurePosixPath(member.name).name == "PKG-INFO"
        ]
        metadata_member = require_exactly_one(metadata_members, f"PKG-INFO in {path}")
        if metadata_member.name != expected_metadata:
            raise ArtifactIdentityError(
                f"{path}: expected metadata member {expected_metadata!r}, got {metadata_member.name!r}"
            )
        source = archive.extractfile(metadata_member)
        if source is None:
            raise ArtifactIdentityError(f"{path}:{metadata_member.name}: could not read metadata")
        name, version = parse_core_metadata(source.read(), f"{path}:{metadata_member.name}")
    validate_python_identity(name, version, expected_version, str(path))


def validate_python_distributions(paths: Sequence[Path], expected_version: str) -> None:
    wheels = [path for path in paths if path.suffix.lower() == ".whl"]
    sdists = [path for path in paths if path.name.lower().endswith(".tar.gz")]
    unknown = [path for path in paths if path not in wheels and path not in sdists]
    if unknown:
        raise ArtifactIdentityError("unexpected Python distribution artifact(s): " + ", ".join(map(str, unknown)))
    wheel = Path(require_exactly_one([str(path) for path in wheels], "Python wheel"))
    sdist = Path(require_exactly_one([str(path) for path in sdists], "Python source distribution"))
    validate_wheel(wheel, expected_version)
    validate_sdist(sdist, expected_version)


def npm_archive_filename(package_name: str, version: str) -> str:
    return f"{package_name.removeprefix('@').replace('/', '-')}-{version}.tgz"


def validate_npm_tarball(path: Path, expected_name: str, expected_version: str) -> None:
    expected_filename = npm_archive_filename(expected_name, expected_version)
    if path.name != expected_filename:
        raise ArtifactIdentityError(f"{path}: expected npm tarball filename {expected_filename!r}")
    if not tarfile.is_tarfile(path):
        raise ArtifactIdentityError(f"{path}: npm package is not a valid tar archive")
    with tarfile.open(path) as archive:
        package_json_members = [
            member for member in archive.getmembers() if member.isfile() and member.name == "package/package.json"
        ]
        package_json_member = require_exactly_one(package_json_members, f"package/package.json in {path}")
        source = archive.extractfile(package_json_member)
        if source is None:
            raise ArtifactIdentityError(f"{path}: could not read package/package.json")
        try:
            package_data = json.load(source)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactIdentityError(f"{path}: invalid package/package.json: {error}") from error
        if not isinstance(package_data, Mapping):
            raise ArtifactIdentityError(f"{path}: package/package.json must be an object")
        for field, expected_target in NPM_ENTRYPOINT_METADATA.items():
            if package_data.get(field) != expected_target:
                raise ArtifactIdentityError(
                    f"{path}: package.json {field} must be the exact target {expected_target!r}"
                )
        if package_data.get("exports") != NPM_EXPORTS_METADATA:
            raise ArtifactIdentityError(f"{path}: package.json exports must match the supported entry points")
        if package_data.get("bin") != NPM_BIN_METADATA:
            raise ArtifactIdentityError(f"{path}: package.json bin must match the supported CLI entry point")
        entrypoint_targets = {
            *NPM_ENTRYPOINT_METADATA.values(),
            *NPM_BIN_METADATA.values(),
            *NPM_EXPORTS_METADATA["."].values(),
        }
        for target in entrypoint_targets:
            expected_member = f"package/{target.removeprefix('./')}"
            members = [member for member in archive.getmembers() if member.isfile() and member.name == expected_member]
            member = require_exactly_one(members, f"{expected_member} in {path}")
            entrypoint_source = archive.extractfile(member)
            if entrypoint_source is None or not entrypoint_source.read():
                raise ArtifactIdentityError(f"{path}: {expected_member} must not be empty")
        for filename in NPM_REQUIRED_TEXT_FILES:
            expected_member = f"package/{filename}"
            members = [member for member in archive.getmembers() if member.isfile() and member.name == expected_member]
            member = require_exactly_one(members, f"{expected_member} in {path}")
            text_source = archive.extractfile(member)
            if text_source is None:
                raise ArtifactIdentityError(f"{path}: could not read {expected_member}")
            if text_source.read() != (REPOSITORY_ROOT / filename).read_bytes():
                raise ArtifactIdentityError(f"{path}: {expected_member} does not match the repository copy")
    if package_data.get("name") != expected_name:
        raise ArtifactIdentityError(f"{path}: expected npm package {expected_name!r}, got {package_data.get('name')!r}")
    if package_data.get("version") != expected_version:
        raise ArtifactIdentityError(
            f"{path}: expected npm version {expected_version!r}, got {package_data.get('version')!r}"
        )


def validate_rust_archive(path: Path, platform: str, expected_version: str) -> None:
    binary_name = RUST_PLATFORM_BINARIES.get(platform)
    if binary_name is None:
        raise ArtifactIdentityError(f"unsupported Rust release platform: {platform}")
    expected_filename = f"codex-as-api-{expected_version}-{platform}.zip"
    if path.name != expected_filename:
        raise ArtifactIdentityError(f"{path}: expected Rust archive filename {expected_filename!r}")
    if not zipfile.is_zipfile(path):
        raise ArtifactIdentityError(f"{path}: Rust release archive is not a valid ZIP archive")

    expected_members = {binary_name, *RUST_REQUIRED_TEXT_SOURCES}
    with zipfile.ZipFile(path) as archive:
        file_members = [member for member in archive.infolist() if not member.is_dir()]
        member_names = [member.filename for member in file_members]
        if len(member_names) != len(set(member_names)):
            raise ArtifactIdentityError(f"{path}: Rust release archive contains duplicate members")
        if set(member_names) != expected_members:
            raise ArtifactIdentityError(
                f"{path}: expected exact Rust archive members {sorted(expected_members)!r}, "
                f"got {sorted(member_names)!r}"
            )
        binary = archive.read(binary_name)
        if not binary:
            raise ArtifactIdentityError(f"{path}: Rust release binary must not be empty")
        validate_rust_binary_magic(binary, platform, str(path))
        binary_member = next(member for member in file_members if member.filename == binary_name)
        if platform != "windows" and not ((binary_member.external_attr >> 16) & 0o111):
            raise ArtifactIdentityError(f"{path}: Rust release binary must be executable")
        for filename, source_path in RUST_REQUIRED_TEXT_SOURCES.items():
            if archive.read(filename) != source_path.read_bytes():
                raise ArtifactIdentityError(f"{path}: {filename} does not match {source_path}")


def validate_rust_binary_magic(binary: bytes, platform: str, label: str) -> None:
    if platform == "linux" and (len(binary) < 64 or not binary.startswith(b"\x7fELF")):
        raise ArtifactIdentityError(f"{label}: Linux Rust release binary must be an ELF executable")
    if platform == "macos":
        mach_o_magics = {
            b"\xca\xfe\xba\xbe",
            b"\xbe\xba\xfe\xca",
            b"\xca\xfe\xba\xbf",
            b"\xbf\xba\xfe\xca",
            b"\xce\xfa\xed\xfe",
            b"\xfe\xed\xfa\xce",
            b"\xcf\xfa\xed\xfe",
            b"\xfe\xed\xfa\xcf",
        }
        if len(binary) < 32 or binary[:4] not in mach_o_magics:
            raise ArtifactIdentityError(f"{label}: macOS Rust release binary must be a Mach-O executable")
    if platform == "windows":
        pe_offset = int.from_bytes(binary[0x3C:0x40], "little") if len(binary) >= 0x40 else -1
        if (
            not binary.startswith(b"MZ")
            or pe_offset < 0x40
            or pe_offset + 4 > len(binary)
            or binary[pe_offset : pe_offset + 4] != b"PE\0\0"
        ):
            raise ArtifactIdentityError(f"{label}: Windows Rust release binary must be a PE executable")


def existing_paths(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ArtifactIdentityError("artifact files do not exist: " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Exact version expected inside every archive.")
    parser.add_argument("--python-dist", nargs="+", type=Path)
    parser.add_argument("--npm-tarball", type=Path)
    parser.add_argument("--github-npm-tarball", type=Path)
    parser.add_argument(
        "--rust-archive",
        action="append",
        nargs=2,
        metavar=("PLATFORM", "PATH"),
        help="Rust platform and its deterministic ZIP archive; repeat for multiple platforms.",
    )
    args = parser.parse_args()

    rust_archives = [(platform, Path(path)) for platform, path in args.rust_archive or []]

    selected = [
        *(args.python_dist or []),
        *([args.npm_tarball] if args.npm_tarball else []),
        *([args.github_npm_tarball] if args.github_npm_tarball else []),
        *(path for _, path in rust_archives),
    ]
    if not selected:
        parser.error("at least one artifact option is required")

    try:
        existing_paths(selected)
        if args.python_dist:
            validate_python_distributions(args.python_dist, args.version)
        if args.npm_tarball:
            validate_npm_tarball(args.npm_tarball, NPM_PACKAGE_NAME, args.version)
        if args.github_npm_tarball:
            validate_npm_tarball(
                args.github_npm_tarball,
                GITHUB_NPM_PACKAGE_NAME,
                args.version,
            )
        for platform, path in rust_archives:
            validate_rust_archive(path, platform, args.version)
    except (ArtifactIdentityError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.error(str(error))

    print(f"release artifact identities match version {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
