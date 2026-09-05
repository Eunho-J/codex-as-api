#!/usr/bin/env python3
"""Check whether exact release artifacts are already present in a registry."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

REQUEST_TIMEOUT_SECONDS = 20


class RegistryPreflightError(ValueError):
    """Raised when registry state cannot be proven safe for this release."""


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


_HTTP_OPENER = build_opener(_RejectRedirects())


def _http_request(
    url: str,
    headers: Mapping[str, str],
    *,
    method: str,
    opener: Callable[..., Any] = _HTTP_OPENER.open,
) -> tuple[int, bytes]:
    request = Request(url, headers=dict(headers), method=method)
    try:
        response_context = opener(request, timeout=REQUEST_TIMEOUT_SECONDS)
        with response_context as response:
            status = response.getcode()
            content = response.read()
    except HTTPError as error:
        return error.code, b""
    except (URLError, TimeoutError, OSError) as error:
        raise RegistryPreflightError(f"registry request failed ({type(error).__name__})") from None

    if not isinstance(status, int):
        raise RegistryPreflightError("registry response did not include an HTTP status")
    if not isinstance(content, bytes):
        raise RegistryPreflightError("registry response body was not bytes")
    return status, content


def _http_get(
    url: str,
    headers: Mapping[str, str],
    *,
    opener: Callable[..., Any] = _HTTP_OPENER.open,
) -> tuple[int, bytes]:
    return _http_request(url, headers, method="GET", opener=opener)


def _http_delete(
    url: str,
    headers: Mapping[str, str],
    *,
    opener: Callable[..., Any] = _HTTP_OPENER.open,
) -> None:
    status, _ = _http_request(url, headers, method="DELETE", opener=opener)
    if status != 204:
        raise RegistryPreflightError(f"GitHub asset deletion returned HTTP {status}")


HttpGetter = Callable[[str, Mapping[str, str]], tuple[int, bytes]]
HttpDeleter = Callable[[str, Mapping[str, str]], None]


def _request_json(
    url: str,
    headers: Mapping[str, str],
    *,
    http_get: HttpGetter = _http_get,
) -> object | None:
    status, content = http_get(url, headers)
    if status == 404:
        return None
    if not 200 <= status < 300:
        raise RegistryPreflightError(f"registry request returned HTTP {status}")
    try:
        document: object = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RegistryPreflightError("registry returned malformed UTF-8 JSON") from None
    return document


JsonFetcher = Callable[[str, Mapping[str, str]], object | None]


def _file_digest(path: Path, algorithm: str) -> bytes:
    digest = hashlib.new(algorithm)
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RegistryPreflightError(f"could not read release artifact {path} ({type(error).__name__})") from None
    return digest.digest()


def _sha256_hex(path: Path) -> str:
    return _file_digest(path, "sha256").hex()


def _sha512_integrity(path: Path) -> str:
    encoded = base64.b64encode(_file_digest(path, "sha512")).decode("ascii")
    return f"sha512-{encoded}"


def _require_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RegistryPreflightError(f"{description} must be an object")
    return value


def _require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise RegistryPreflightError(f"{description} must be a non-empty string")
    return value


def check_pypi_release(
    package_name: str,
    version: str,
    artifacts: Sequence[Path],
    *,
    fetch_json: JsonFetcher = _request_json,
) -> bool:
    expected: dict[str, str] = {}
    for artifact in artifacts:
        if artifact.name in expected:
            raise RegistryPreflightError(f"duplicate local artifact filename: {artifact.name}")
        expected[artifact.name] = _sha256_hex(artifact)
    if not expected:
        raise RegistryPreflightError("at least one Python release artifact is required")

    url = f"https://pypi.org/pypi/{quote(package_name, safe='')}/{quote(version, safe='')}/json"
    document = fetch_json(url, {"Accept": "application/json"})
    if document is None:
        return False

    root = _require_mapping(document, "PyPI response")
    info = _require_mapping(root.get("info"), "PyPI response.info")
    if info.get("name") != package_name or info.get("version") != version:
        raise RegistryPreflightError("PyPI package name or version does not match the release")
    urls = root.get("urls")
    if not isinstance(urls, list):
        raise RegistryPreflightError("PyPI response.urls must be an array")

    published: dict[str, str] = {}
    for index, entry_value in enumerate(urls):
        entry = _require_mapping(entry_value, f"PyPI response.urls[{index}]")
        filename = _require_string(entry.get("filename"), f"PyPI response.urls[{index}].filename")
        if filename in published:
            raise RegistryPreflightError(f"PyPI returned duplicate artifact filename: {filename}")
        digests = _require_mapping(entry.get("digests"), f"PyPI response.urls[{index}].digests")
        sha256 = _require_string(
            digests.get("sha256"),
            f"PyPI response.urls[{index}].digests.sha256",
        )
        published[filename] = sha256

    if published != expected:
        raise RegistryPreflightError("PyPI artifact filenames or SHA-256 digests do not match")
    return True


def _npm_version_metadata(
    document: object,
    package_name: str,
    version: str,
    *,
    packument: bool,
) -> Mapping[str, object] | None:
    root = _require_mapping(document, "npm registry response")
    if packument:
        if root.get("name") != package_name:
            raise RegistryPreflightError("npm registry package name does not match the release")
        versions = _require_mapping(root.get("versions"), "npm registry response.versions")
        if version not in versions:
            return None
        metadata = versions[version]
        return _require_mapping(metadata, f"npm registry response.versions[{version!r}]")
    return root


def check_npm_release(
    registry_url: str,
    package_name: str,
    version: str,
    artifact: Path,
    *,
    token: str | None = None,
    packument: bool = False,
    fetch_json: JsonFetcher = _request_json,
) -> bool:
    package_path = quote(package_name, safe="@")
    endpoint = f"{registry_url.rstrip('/')}/{package_path}"
    if not packument:
        endpoint = f"{endpoint}/{quote(version, safe='')}"
    headers = {"Accept": "application/json"}
    if token is not None:
        if not token:
            raise RegistryPreflightError("registry bearer token must not be empty")
        if not registry_url.startswith("https://"):
            raise RegistryPreflightError("registry bearer tokens require an HTTPS registry URL")
        headers["Authorization"] = f"Bearer {token}"

    document = fetch_json(endpoint, headers)
    if document is None:
        return False
    metadata = _npm_version_metadata(document, package_name, version, packument=packument)
    if metadata is None:
        return False
    if metadata.get("name") != package_name or metadata.get("version") != version:
        raise RegistryPreflightError("npm package name or version does not match the release")
    dist = _require_mapping(metadata.get("dist"), "npm registry response.dist")
    integrity = _require_string(dist.get("integrity"), "npm registry response.dist.integrity")
    if integrity != _sha512_integrity(artifact):
        raise RegistryPreflightError("npm artifact SHA-512 integrity does not match")
    return True


def check_github_release(
    repository: str,
    tag: str,
    token: str,
    *,
    fetch_json: JsonFetcher = _request_json,
) -> bool:
    document = fetch_json(_github_release_url(repository, tag), _github_headers(token))
    if document is None:
        return False
    release = _require_mapping(document, "GitHub release response")
    if release.get("tag_name") != tag:
        raise RegistryPreflightError("GitHub release tag does not match the requested tag")
    return True


def _github_release_url(repository: str, tag: str) -> str:
    parts = repository.split("/")
    if len(parts) != 2 or not all(parts):
        raise RegistryPreflightError("GitHub repository must have owner/name form")
    owner, name = parts
    return (
        f"https://api.github.com/repos/{quote(owner, safe='')}/{quote(name, safe='')}"
        f"/releases/tags/{quote(tag, safe='')}"
    )


def _github_headers(token: str) -> Mapping[str, str]:
    if not token:
        raise RegistryPreflightError("GitHub bearer token must not be empty")
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _load_github_release(
    repository: str,
    tag: str,
    token: str,
    *,
    fetch_json: JsonFetcher,
) -> Mapping[str, object]:
    document = fetch_json(_github_release_url(repository, tag), _github_headers(token))
    if document is None:
        raise RegistryPreflightError("GitHub release does not exist")
    release = _require_mapping(document, "GitHub release response")
    if release.get("tag_name") != tag:
        raise RegistryPreflightError("GitHub release tag does not match the requested tag")
    return release


def _local_release_assets(artifacts: Sequence[Path]) -> Mapping[str, str]:
    expected: dict[str, str] = {}
    for artifact in artifacts:
        if artifact.name in expected:
            raise RegistryPreflightError(f"duplicate local release asset filename: {artifact.name}")
        expected[artifact.name] = _sha256_hex(artifact)
    if not expected:
        raise RegistryPreflightError("at least one GitHub release asset is required")
    return expected


def _remote_release_assets(
    release: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    assets_value = release.get("assets")
    if not isinstance(assets_value, list):
        raise RegistryPreflightError("GitHub release response.assets must be an array")
    assets: dict[str, Mapping[str, object]] = {}
    for index, asset_value in enumerate(assets_value):
        asset = _require_mapping(asset_value, f"GitHub release response.assets[{index}]")
        name = _require_string(asset.get("name"), f"GitHub release response.assets[{index}].name")
        if name in assets:
            raise RegistryPreflightError(f"GitHub release returned duplicate asset filename: {name}")
        assets[name] = asset
    return assets


def prune_github_release_assets(
    repository: str,
    tag: str,
    token: str,
    artifacts: Sequence[Path],
    *,
    fetch_json: JsonFetcher = _request_json,
    delete_asset: HttpDeleter = _http_delete,
) -> None:
    expected = _local_release_assets(artifacts)
    release = _load_github_release(
        repository,
        tag,
        token,
        fetch_json=fetch_json,
    )
    remote = _remote_release_assets(release)
    headers = _github_headers(token)
    repository_url = _github_release_url(repository, tag).split("/releases/tags/", 1)[0]
    deletions: list[int] = []
    for name, asset in remote.items():
        if name in expected:
            continue
        asset_id = asset.get("id")
        if isinstance(asset_id, bool) or not isinstance(asset_id, int) or asset_id <= 0:
            raise RegistryPreflightError(f"GitHub release asset {name!r} must have a positive integer id")
        deletions.append(asset_id)
    for asset_id in deletions:
        delete_asset(f"{repository_url}/releases/assets/{asset_id}", headers)


def verify_github_release_assets(
    repository: str,
    tag: str,
    token: str,
    artifacts: Sequence[Path],
    *,
    fetch_json: JsonFetcher = _request_json,
) -> None:
    expected = _local_release_assets(artifacts)
    release = _load_github_release(
        repository,
        tag,
        token,
        fetch_json=fetch_json,
    )
    remote = _remote_release_assets(release)
    if remote.keys() != expected.keys():
        raise RegistryPreflightError("GitHub release asset filenames do not match the local release asset set")
    for name, sha256 in expected.items():
        asset = remote[name]
        if asset.get("state") != "uploaded":
            raise RegistryPreflightError(f"GitHub release asset {name!r} is not uploaded")
        digest = _require_string(asset.get("digest"), f"GitHub release asset {name!r}.digest")
        if digest != f"sha256:{sha256}":
            raise RegistryPreflightError(f"GitHub release asset {name!r} SHA-256 digest does not match")


def write_github_output(published: bool, output_path: str | None = None) -> None:
    destination = output_path if output_path is not None else os.environ.get("GITHUB_OUTPUT")
    if destination is None:
        return
    try:
        with Path(destination).open("a", encoding="utf-8") as output:
            output.write(f"published={'true' if published else 'false'}\n")
    except OSError as error:
        raise RegistryPreflightError(f"could not write GITHUB_OUTPUT ({type(error).__name__})") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="target", required=True)

    pypi = subparsers.add_parser("pypi")
    pypi.add_argument("--package", required=True)
    pypi.add_argument("--version", required=True)
    pypi.add_argument("--artifact", nargs="+", type=Path, required=True)

    npm = subparsers.add_parser("npm")
    npm.add_argument("--registry", required=True)
    npm.add_argument("--package", required=True)
    npm.add_argument("--version", required=True)
    npm.add_argument("--artifact", type=Path, required=True)
    npm.add_argument("--token-env")
    npm.add_argument("--packument", action="store_true")

    github_release = subparsers.add_parser("github-release")
    github_release.add_argument("--repository", required=True)
    github_release.add_argument("--tag", required=True)
    github_release.add_argument("--token-env", default="GH_TOKEN")

    github_release_assets = subparsers.add_parser("github-release-assets")
    github_release_assets.add_argument("--repository", required=True)
    github_release_assets.add_argument("--tag", required=True)
    github_release_assets.add_argument("--token-env", default="GH_TOKEN")
    github_release_assets.add_argument("--mode", choices=("prune", "verify"), required=True)
    github_release_assets.add_argument("--asset", nargs="+", type=Path, required=True)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.target == "pypi":
            published = check_pypi_release(args.package, args.version, args.artifact)
        elif args.target == "npm":
            token = None
            if args.token_env is not None:
                token = os.environ.get(args.token_env)
                if token is None:
                    raise RegistryPreflightError(f"required token environment variable is not set: {args.token_env}")
            published = check_npm_release(
                args.registry,
                args.package,
                args.version,
                args.artifact,
                token=token,
                packument=args.packument,
            )
        elif args.target == "github-release":
            token = os.environ.get(args.token_env)
            if token is None:
                raise RegistryPreflightError(f"required token environment variable is not set: {args.token_env}")
            published = check_github_release(args.repository, args.tag, token)
        else:
            token = os.environ.get(args.token_env)
            if token is None:
                raise RegistryPreflightError(f"required token environment variable is not set: {args.token_env}")
            if args.mode == "prune":
                prune_github_release_assets(
                    args.repository,
                    args.tag,
                    token,
                    args.asset,
                )
                print("unexpected GitHub release assets were removed")
            else:
                verify_github_release_assets(
                    args.repository,
                    args.tag,
                    token,
                    args.asset,
                )
                print("GitHub release asset names and SHA-256 digests match")
            return 0
        write_github_output(published)
    except RegistryPreflightError as error:
        print(error, file=sys.stderr)
        return 1

    state = "already published with identical artifacts" if published else "not published"
    print(f"release target is {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
