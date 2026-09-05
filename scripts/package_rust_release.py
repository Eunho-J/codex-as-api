#!/usr/bin/env python3
"""Build a deterministic Rust release archive with repository notices."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLATFORM_BINARIES = {
    "linux": "codex-as-api",
    "macos": "codex-as-api",
    "windows": "codex-as-api.exe",
}
REQUIRED_TEXT_SOURCES = {
    "LICENSE": REPOSITORY_ROOT / "LICENSE",
    "THIRD_PARTY_NOTICES.md": REPOSITORY_ROOT / "rust" / "THIRD_PARTY_NOTICES.md",
}
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _write_entry(archive: zipfile.ZipFile, name: str, content: bytes, mode: int) -> None:
    member = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    member.create_system = 3
    member.compress_type = zipfile.ZIP_DEFLATED
    member.external_attr = mode << 16
    archive.writestr(member, content)


def package_rust_release(
    binary: Path,
    platform: str,
    version: str,
    output_dir: Path,
) -> Path:
    expected_binary = PLATFORM_BINARIES.get(platform)
    if expected_binary is None:
        raise ValueError(f"unsupported Rust release platform: {platform}")
    if not version or version != version.strip() or "/" in version or "\\" in version:
        raise ValueError("release version must be a non-empty path-safe string")
    if binary.name != expected_binary:
        raise ValueError(f"{platform} release binary must be named {expected_binary}")
    binary_content = binary.read_bytes()
    if not binary_content:
        raise ValueError("Rust release binary must not be empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"codex-as-api-{version}-{platform}.zip"
    with zipfile.ZipFile(output, "w") as archive:
        _write_entry(archive, expected_binary, binary_content, 0o100755)
        for filename, source_path in REQUIRED_TEXT_SOURCES.items():
            _write_entry(archive, filename, source_path.read_bytes(), 0o100644)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--platform", required=True, choices=sorted(PLATFORM_BINARIES))
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    try:
        output = package_rust_release(args.binary, args.platform, args.version, args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
