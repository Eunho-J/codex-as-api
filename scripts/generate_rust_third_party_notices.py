#!/usr/bin/env python3
"""Generate deterministic notices for locked Rust runtime dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPOSITORY_ROOT / "rust" / "Cargo.toml"
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "rust" / "THIRD_PARTY_NOTICES.md"
DEFAULT_BUNDLED_NOTICES_PATH = REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md"
RELEASE_TARGETS = (
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-msvc",
    "x86_64-unknown-linux-gnu",
)
LICENSE_BASENAMES = ("copying", "license", "licence", "notice", "unlicense")


class NoticeGenerationError(RuntimeError):
    """Raised when the locked dependency graph cannot produce complete notices."""


@dataclass(frozen=True)
class RuntimeDependency:
    package_id: str
    name: str
    version: str
    license_expression: str | None
    license_files: tuple[Path, ...]
    targets: tuple[str, ...]


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NoticeGenerationError(f"{label} must be an object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise NoticeGenerationError(f"{label} must be a non-empty string")
    return value


def load_metadata(manifest_path: Path, target: str) -> Mapping[str, Any]:
    command = [
        "cargo",
        "metadata",
        "--locked",
        "--format-version=1",
        "--filter-platform",
        target,
        "--manifest-path",
        str(manifest_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr.strip()
            if isinstance(error, subprocess.CalledProcessError) and error.stderr
            else str(error)
        )
        raise NoticeGenerationError(f"cargo metadata failed for {target}: {detail}") from error
    try:
        return _object(json.loads(completed.stdout), f"cargo metadata for {target}")
    except json.JSONDecodeError as error:
        raise NoticeGenerationError(f"cargo metadata returned invalid JSON for {target}") from error


def normal_dependency_ids(metadata: Mapping[str, Any]) -> set[str]:
    resolve = _object(metadata.get("resolve"), "cargo metadata resolve")
    root = _string(resolve.get("root"), "cargo metadata resolve.root")
    raw_nodes = resolve.get("nodes")
    if not isinstance(raw_nodes, list):
        raise NoticeGenerationError("cargo metadata resolve.nodes must be an array")
    nodes: dict[str, Mapping[str, Any]] = {}
    for index, raw_node in enumerate(raw_nodes):
        node = _object(raw_node, f"cargo metadata resolve.nodes[{index}]")
        package_id = _string(node.get("id"), f"cargo metadata resolve.nodes[{index}].id")
        if package_id in nodes:
            raise NoticeGenerationError(f"cargo metadata contains duplicate node {package_id}")
        nodes[package_id] = node
    if root not in nodes:
        raise NoticeGenerationError("cargo metadata resolve root has no matching node")

    visited = {root}
    stack = [root]
    dependencies: set[str] = set()
    while stack:
        node_id = stack.pop()
        raw_dependencies = nodes[node_id].get("deps")
        if not isinstance(raw_dependencies, list):
            raise NoticeGenerationError(f"cargo metadata node {node_id} deps must be an array")
        for index, raw_dependency in enumerate(raw_dependencies):
            dependency = _object(raw_dependency, f"cargo metadata node {node_id} deps[{index}]")
            package_id = _string(dependency.get("pkg"), f"cargo metadata node {node_id} dependency package")
            raw_kinds = dependency.get("dep_kinds")
            if not isinstance(raw_kinds, list):
                raise NoticeGenerationError(f"cargo metadata dependency {package_id} dep_kinds must be an array")
            is_normal = False
            for raw_kind in raw_kinds:
                kind = _object(raw_kind, f"cargo metadata dependency {package_id} kind").get("kind")
                if kind is None:
                    is_normal = True
                elif kind not in {"build", "dev"}:
                    raise NoticeGenerationError(f"cargo metadata dependency {package_id} has unknown kind {kind!r}")
            if not is_normal:
                continue
            if package_id not in nodes:
                raise NoticeGenerationError(f"cargo metadata dependency {package_id} has no matching node")
            dependencies.add(package_id)
            if package_id not in visited:
                visited.add(package_id)
                stack.append(package_id)
    return dependencies


def _is_license_filename(path: Path) -> bool:
    lowered = path.name.casefold()
    return any(
        lowered == basename or lowered.startswith(f"{basename}.") or lowered.startswith(f"{basename}-")
        for basename in LICENSE_BASENAMES
    )


def find_license_files(package: Mapping[str, Any]) -> tuple[Path, ...]:
    manifest_path = Path(_string(package.get("manifest_path"), "cargo package manifest_path")).resolve()
    package_root = manifest_path.parent
    raw_license_file = package.get("license_file")
    if raw_license_file is not None:
        license_file = package_root / _string(raw_license_file, "cargo package license_file")
        candidates = [license_file.resolve()]
    else:
        try:
            candidates = sorted(
                (path.resolve() for path in package_root.iterdir() if path.is_file() and _is_license_filename(path)),
                key=lambda path: (path.name.casefold(), path.name),
            )
        except OSError as error:
            raise NoticeGenerationError(f"cannot inspect license files for {package_root}: {error}") from error
    if not candidates:
        raise NoticeGenerationError(f"{package_root}: dependency has no shipped license file")
    for candidate in candidates:
        if not candidate.is_relative_to(package_root) or not candidate.is_file():
            raise NoticeGenerationError(f"{candidate}: dependency license file must exist inside {package_root}")
    return tuple(candidates)


def collect_runtime_dependencies(
    metadata_by_target: Mapping[str, Mapping[str, Any]],
) -> list[RuntimeDependency]:
    package_targets: dict[str, set[str]] = defaultdict(set)
    packages_by_id: dict[str, Mapping[str, Any]] = {}
    for target in sorted(metadata_by_target):
        metadata = metadata_by_target[target]
        raw_packages = metadata.get("packages")
        if not isinstance(raw_packages, list):
            raise NoticeGenerationError(f"cargo metadata packages for {target} must be an array")
        current_packages: dict[str, Mapping[str, Any]] = {}
        for index, raw_package in enumerate(raw_packages):
            package = _object(raw_package, f"cargo metadata packages[{index}] for {target}")
            package_id = _string(package.get("id"), f"cargo metadata package id for {target}")
            current_packages[package_id] = package
        for package_id in normal_dependency_ids(metadata):
            if package_id not in current_packages:
                raise NoticeGenerationError(f"cargo metadata package {package_id} is missing for {target}")
            package_targets[package_id].add(target)
            existing = packages_by_id.get(package_id)
            if existing is not None and existing != current_packages[package_id]:
                raise NoticeGenerationError(f"cargo metadata package {package_id} changed across targets")
            packages_by_id[package_id] = current_packages[package_id]

    dependencies: list[RuntimeDependency] = []
    for package_id, package in packages_by_id.items():
        name = _string(package.get("name"), f"cargo package {package_id} name")
        version = _string(package.get("version"), f"cargo package {package_id} version")
        license_expression = package.get("license")
        if license_expression is not None and not isinstance(license_expression, str):
            raise NoticeGenerationError(f"cargo package {package_id} license must be a string or null")
        license_files = find_license_files(package)
        dependencies.append(
            RuntimeDependency(
                package_id=package_id,
                name=name,
                version=version,
                license_expression=license_expression,
                license_files=license_files,
                targets=tuple(sorted(package_targets[package_id])),
            )
        )
    dependencies.sort(
        key=lambda dependency: (
            dependency.name.casefold(),
            dependency.name,
            dependency.version,
            dependency.package_id,
        )
    )
    return dependencies


def _markdown_fence(content: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def _normalize_line_endings(content: bytes) -> bytes:
    return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def render_notices(dependencies: Sequence[RuntimeDependency], bundled_notices: bytes) -> bytes:
    if not dependencies:
        raise NoticeGenerationError("the locked Rust runtime dependency graph is empty")
    if not bundled_notices:
        raise NoticeGenerationError("the bundled-asset notices must not be empty")
    normalized_bundled_notices = _normalize_line_endings(bundled_notices)
    try:
        bundled_notice_text = normalized_bundled_notices.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NoticeGenerationError("the bundled-asset notices must be UTF-8") from error
    output = [
        "# Rust Third-Party Notices\n\n",
        "Generated by `python scripts/generate_rust_third_party_notices.py`.\n\n",
        "## Bundled non-Cargo assets\n\n",
        bundled_notice_text,
        "\n" if not bundled_notice_text.endswith("\n") else "",
        "\n## Locked Rust runtime dependencies\n\n",
        "This file contains the shipped license files for every locked normal dependency ",
        "reachable on the supported Linux, macOS, and Windows release targets.\n\n",
    ]
    for dependency in dependencies:
        output.append(f"## {dependency.name} {dependency.version}\n\n")
        expression = dependency.license_expression or "not declared in Cargo metadata"
        escaped_expression = expression.replace("`", "\\`")
        output.append(f"- Cargo license expression: `{escaped_expression}`\n")
        output.append(f"- Release targets: {', '.join(f'`{target}`' for target in dependency.targets)}\n\n")
        for license_file in dependency.license_files:
            raw_content = license_file.read_bytes()
            if not raw_content:
                raise NoticeGenerationError(f"{license_file}: dependency license file must not be empty")
            normalized_content = _normalize_line_endings(raw_content)
            try:
                content = normalized_content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise NoticeGenerationError(f"{license_file}: dependency license file must be UTF-8") from error
            fence = _markdown_fence(content)
            digest = hashlib.sha256(normalized_content).hexdigest()
            output.append(f"### `{license_file.name}`\n\n")
            output.append(f"SHA-256: `{digest}`\n\n")
            output.append(f"{fence}text\n")
            output.append(content)
            if not content.endswith("\n"):
                output.append("\n")
            output.append(f"{fence}\n\n")
    return "".join(output).encode("utf-8")


def generate_notices(
    manifest_path: Path,
    bundled_notices_path: Path,
    targets: Sequence[str] = RELEASE_TARGETS,
) -> bytes:
    metadata_by_target = {target: load_metadata(manifest_path, target) for target in targets}
    return render_notices(
        collect_runtime_dependencies(metadata_by_target),
        bundled_notices_path.read_bytes(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--bundled-notices", type=Path, default=DEFAULT_BUNDLED_NOTICES_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail unless the committed notices equal generated output.",
    )
    args = parser.parse_args()

    try:
        generated = generate_notices(args.manifest_path.resolve(), args.bundled_notices.resolve())
        output_path = args.output.resolve()
        if args.check:
            if not output_path.is_file() or output_path.read_bytes() != generated:
                raise NoticeGenerationError(
                    f"{output_path} is stale; run python scripts/generate_rust_third_party_notices.py"
                )
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(generated)
    except (NoticeGenerationError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
