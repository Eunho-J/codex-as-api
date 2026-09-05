#!/usr/bin/env python3
"""Reject forbidden or structurally recognizable JSON static model catalogs."""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

UPSTREAM_CONTRACT_NAME = "codex-upstream-contract.json"
MODEL_CAPABILITY_KEYS = {
    "auto_compact_token_limit",
    "context_window",
    "default_reasoning_effort",
    "default_reasoning_level",
    "default_reasoning_summary",
    "default_service_tier",
    "default_verbosity",
    "effective_context_window_percent",
    "input_modalities",
    "max_context_window",
    "comp_hash",
    "service_tiers",
    "support_verbosity",
    "supported_in_api",
    "supported_reasoning_levels",
    "supports_image_detail_original",
    "supports_reasoning_summary_parameter",
    "use_responses_lite",
    "visibility",
}


def normalized_basename(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).name.lower()


def is_catalog_like_json_name(name: str) -> bool:
    basename = normalized_basename(name)
    if not basename.endswith(".json"):
        return False
    stem = "".join(character for character in basename[:-5] if character.isalnum())
    return stem in {"models", "catalog"} or ("model" in stem and ("catalog" in stem or "capabilit" in stem))


def is_model_record(value: object, *, require_identifier: bool) -> bool:
    if not isinstance(value, Mapping):
        return False
    if require_identifier and not any(isinstance(value.get(key), str) and bool(value[key]) for key in ("slug", "id")):
        return False
    return bool(MODEL_CAPABILITY_KEYS.intersection(value))


def is_model_collection(value: object) -> bool:
    if isinstance(value, list):
        return any(is_model_record(item, require_identifier=True) for item in value)
    if isinstance(value, Mapping):
        return any(
            isinstance(model_id, str) and bool(model_id) and is_model_record(model, require_identifier=False)
            for model_id, model in value.items()
        )
    return False


def catalog_locations(value: object, location: str = "$") -> list[str]:
    if is_model_collection(value):
        return [location]
    found: list[str] = []
    if isinstance(value, Mapping):
        models = value.get("models")
        if is_model_collection(models):
            found.append(f"{location}.models")
        for key, child in value.items():
            if key == "models" and is_model_collection(child):
                continue
            found.extend(catalog_locations(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(catalog_locations(child, f"{location}[{index}]"))
    return found


def runtime_models_locations(value: object, location: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key == "models" and isinstance(child, (list, Mapping)):
                found.append(child_location)
            found.extend(runtime_models_locations(child, child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(runtime_models_locations(child, f"{location}[{index}]"))
    return found


def json_catalog_findings(label: str, content: bytes) -> list[str]:
    try:
        document: Any = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}: invalid JSON: {error}") from error
    findings = [f"{label}: static model catalog payload at {location}" for location in catalog_locations(document)]
    if normalized_basename(label) == UPSTREAM_CONTRACT_NAME:
        findings.extend(
            f"{label}: runtime models payload is forbidden in the upstream contract at {location}"
            for location in runtime_models_locations(document)
        )
    return findings


def file_findings(path: Path, label: str | None = None) -> list[str]:
    display = label or str(path)
    found: list[str] = []
    if is_catalog_like_json_name(display):
        found.append(f"{display}: forbidden catalog-like JSON filename")
    if path.suffix.lower() == ".json":
        found.extend(json_catalog_findings(display, path.read_bytes()))
    return found


def directory_findings(path: Path) -> list[str]:
    found: list[str] = []
    for candidate in path.rglob("*"):
        if candidate.is_file():
            found.extend(file_findings(candidate))
    return found


def zip_findings(path: Path) -> list[str]:
    found: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            label = f"{path}:{member.filename}"
            if is_catalog_like_json_name(member.filename):
                found.append(f"{label}: forbidden catalog-like JSON filename")
            if not member.is_dir() and PurePosixPath(member.filename).suffix.lower() == ".json":
                found.extend(json_catalog_findings(label, archive.read(member)))
    return found


def tar_findings(path: Path) -> list[str]:
    found: list[str] = []
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            label = f"{path}:{member.name}"
            if is_catalog_like_json_name(member.name):
                found.append(f"{label}: forbidden catalog-like JSON filename")
            if not member.isfile() or PurePosixPath(member.name).suffix.lower() != ".json":
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"{label}: could not read JSON archive member")
            found.extend(json_catalog_findings(label, source.read()))
    return found


def archive_findings(path: Path) -> list[str]:
    if zipfile.is_zipfile(path):
        return zip_findings(path)
    if tarfile.is_tarfile(path):
        return tar_findings(path)
    raise ValueError(f"unsupported release archive: {path}")


def forbidden_paths(path: Path) -> list[str]:
    if path.is_dir():
        return directory_findings(path)
    if path.suffix.lower() == ".json":
        return file_findings(path)
    return archive_findings(path)


def find_forbidden_catalogs(paths: Iterable[Path]) -> list[str]:
    return [entry for path in paths for entry in forbidden_paths(path)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    missing = [str(path) for path in args.paths if not path.exists()]
    if missing:
        parser.error(f"paths do not exist: {', '.join(missing)}")

    try:
        found = find_forbidden_catalogs(args.paths)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    if found:
        parser.error("static model catalogs found: " + ", ".join(found))

    print(f"no static model catalog found in {len(args.paths)} path(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
