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
VERIFIED_HIDDEN_MODELS_NAME = "verified-hidden-models.json"
VERIFIED_HIDDEN_MODELS_CANONICAL_RELPATHS = {
    "config/verified-hidden-models.json",
    "codex_as_api/verified-hidden-models.json",
    "src/codex_as_api/verified-hidden-models.json",
}
VERIFIED_HIDDEN_MODEL_SLUGS = (
    "gpt-6-chat-thinking",
    "gpt-reserve",
    "codex-auto-review",
)
VERIFIED_HIDDEN_MODEL_WARNING = (
    "Experimental hidden model; upstream support may stop at any time."
)
VERIFIED_HIDDEN_MODEL_ALLOWED_KEYS = {
    "slug",
    "display_name",
    "description",
    "priority",
    "visibility",
    "supported_in_api",
    "default_reasoning_level",
    "supported_reasoning_levels",
    "multi_agent_reasoning_effort",
    "supports_reasoning_summary_parameter",
    "default_reasoning_summary",
    "comp_hash",
    "context_window",
    "max_context_window",
    "auto_compact_token_limit",
    "effective_context_window_percent",
    "input_modalities",
    "service_tiers",
    "default_service_tier",
    "use_responses_lite",
    "supports_image_detail_original",
    "support_verbosity",
    "default_verbosity",
}
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


def _is_js_safe_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and -(2**53 - 1) <= value <= (2**53 - 1)


def _is_int32(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and -(2**31) <= value <= (2**31 - 1)


def is_canonical_verified_hidden_models_archive_member(member_name: str) -> bool:
    normalized = member_name.replace("\\", "/").lstrip("./")
    parts = PurePosixPath(normalized).parts
    if not parts:
        return False
    joined = "/".join(parts)
    if joined in VERIFIED_HIDDEN_MODELS_CANONICAL_RELPATHS:
        return True
    if len(parts) >= 3 and parts[0] not in {"", ".", ".."}:
        return "/".join(parts[1:]) in VERIFIED_HIDDEN_MODELS_CANONICAL_RELPATHS
    return False


def is_canonical_verified_hidden_models_path(path: Path, *, root: Path | None = None) -> bool:
    if root is not None:
        rel = path.relative_to(root).as_posix()
        if rel in VERIFIED_HIDDEN_MODELS_CANONICAL_RELPATHS:
            return True
        return root.name in {"config", "codex_as_api"} and rel == VERIFIED_HIDDEN_MODELS_NAME
    parts = PurePosixPath(path.as_posix().replace("\\", "/")).parts
    return len(parts) >= 2 and (parts[-2], parts[-1]) in {
        ("config", VERIFIED_HIDDEN_MODELS_NAME),
        ("codex_as_api", VERIFIED_HIDDEN_MODELS_NAME),
    }


def _is_canonical_verified_hidden_models_label(label: str) -> bool:
    if ":" in label:
        _, member_name = label.rsplit(":", 1)
        return is_canonical_verified_hidden_models_archive_member(member_name)
    parts = PurePosixPath(label.replace("\\", "/")).parts
    return len(parts) >= 2 and (parts[-2], parts[-1]) in {
        ("config", VERIFIED_HIDDEN_MODELS_NAME),
        ("codex_as_api", VERIFIED_HIDDEN_MODELS_NAME),
    }


def _is_valid_verified_hidden_model_record(value: object, expected_slug: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    if set(value.keys()) != VERIFIED_HIDDEN_MODEL_ALLOWED_KEYS:
        return False
    if value.get("slug") != expected_slug:
        return False
    display_name = value.get("display_name")
    if not isinstance(display_name, str) or not display_name.strip():
        return False
    if value.get("description") != VERIFIED_HIDDEN_MODEL_WARNING:
        return False
    if not _is_int32(value.get("priority")):
        return False
    if value.get("visibility") != "list":
        return False
    if value.get("supported_in_api") is not True:
        return False
    for key in ("default_reasoning_level", "multi_agent_reasoning_effort"):
        effort = value.get(key)
        if effort is not None and (not isinstance(effort, str) or not effort):
            return False
    levels = value.get("supported_reasoning_levels")
    if not isinstance(levels, list):
        return False
    for level in levels:
        if not isinstance(level, Mapping) or set(level.keys()) != {"effort", "description"}:
            return False
        if not isinstance(level.get("effort"), str) or not level["effort"]:
            return False
        if not isinstance(level.get("description"), str):
            return False
    if not isinstance(value.get("supports_reasoning_summary_parameter"), bool):
        return False
    summary = value.get("default_reasoning_summary")
    if not isinstance(summary, str) or summary not in {"auto", "concise", "detailed", "none"}:
        return False
    comp_hash = value.get("comp_hash")
    if comp_hash is not None and not isinstance(comp_hash, str):
        return False
    for key in ("context_window", "max_context_window", "auto_compact_token_limit"):
        limit = value.get(key)
        if limit is not None and not _is_js_safe_int(limit):
            return False
    if not _is_js_safe_int(value.get("effective_context_window_percent")):
        return False
    modalities = value.get("input_modalities")
    if not isinstance(modalities, list) or any(
        not isinstance(modality, str) or modality not in {"text", "image", "audio"}
        for modality in modalities
    ):
        return False
    tiers = value.get("service_tiers")
    if not isinstance(tiers, list):
        return False
    for tier in tiers:
        if not isinstance(tier, Mapping) or set(tier.keys()) != {"id", "name", "description"}:
            return False
        if not all(isinstance(tier.get(field), str) for field in ("id", "name", "description")):
            return False
    default_tier = value.get("default_service_tier")
    if default_tier is not None and not isinstance(default_tier, str):
        return False
    for key in ("use_responses_lite", "supports_image_detail_original", "support_verbosity"):
        if not isinstance(value.get(key), bool):
            return False
    default_verbosity = value.get("default_verbosity")
    return default_verbosity is None or (
        isinstance(default_verbosity, str) and default_verbosity in {"low", "medium", "high"}
    )


def is_valid_verified_hidden_models_document(document: object) -> bool:
    if not isinstance(document, Mapping):
        return False
    if set(document.keys()) != {"verified_at", "warning", "models"}:
        return False
    verified_at = document.get("verified_at")
    if not isinstance(verified_at, str) or not verified_at.strip():
        return False
    if document.get("warning") != VERIFIED_HIDDEN_MODEL_WARNING:
        return False
    models = document.get("models")
    if not isinstance(models, list) or len(models) != len(VERIFIED_HIDDEN_MODEL_SLUGS):
        return False
    return all(
        _is_valid_verified_hidden_model_record(model, expected_slug)
        for model, expected_slug in zip(models, VERIFIED_HIDDEN_MODEL_SLUGS, strict=True)
    )


def json_catalog_findings(
    label: str,
    content: bytes,
    *,
    allow_verified_hidden_models: bool | None = None,
) -> list[str]:
    try:
        document: Any = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}: invalid JSON: {error}") from error
    is_canonical_exception = (
        _is_canonical_verified_hidden_models_label(label)
        if allow_verified_hidden_models is None
        else allow_verified_hidden_models
    )
    if is_canonical_exception and is_valid_verified_hidden_models_document(document):
        return []
    findings = [f"{label}: static model catalog payload at {location}" for location in catalog_locations(document)]
    if is_canonical_exception:
        findings.append(f"{label}: invalid verified hidden models configuration")
    if normalized_basename(label) == UPSTREAM_CONTRACT_NAME:
        findings.extend(
            f"{label}: runtime models payload is forbidden in the upstream contract at {location}"
            for location in runtime_models_locations(document)
        )
    return findings


def file_findings(path: Path, label: str | None = None, *, root: Path | None = None) -> list[str]:
    display = label or str(path)
    found: list[str] = []
    if is_catalog_like_json_name(display):
        found.append(f"{display}: forbidden catalog-like JSON filename")
    if path.suffix.lower() == ".json":
        found.extend(
            json_catalog_findings(
                display,
                path.read_bytes(),
                allow_verified_hidden_models=is_canonical_verified_hidden_models_path(path, root=root),
            )
        )
    return found


def directory_findings(path: Path) -> list[str]:
    found: list[str] = []
    for candidate in path.rglob("*"):
        if candidate.is_file():
            found.extend(file_findings(candidate, root=path))
    return found


def zip_findings(path: Path) -> list[str]:
    found: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            label = f"{path}:{member.filename}"
            if is_catalog_like_json_name(member.filename):
                found.append(f"{label}: forbidden catalog-like JSON filename")
            if not member.is_dir() and PurePosixPath(member.filename).suffix.lower() == ".json":
                found.extend(
                    json_catalog_findings(
                        label,
                        archive.read(member),
                        allow_verified_hidden_models=is_canonical_verified_hidden_models_archive_member(
                            member.filename
                        ),
                    )
                )
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
            found.extend(
                json_catalog_findings(
                    label,
                    source.read(),
                    allow_verified_hidden_models=is_canonical_verified_hidden_models_archive_member(
                        member.name
                    ),
                )
            )
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
