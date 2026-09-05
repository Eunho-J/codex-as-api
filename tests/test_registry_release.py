"""Deterministic tests for release-registry idempotency checks."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
from collections.abc import Callable, Mapping
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_registry_release.py"
SPEC = importlib.util.spec_from_file_location("check_registry_release", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_registry_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_registry_release)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha512_integrity(path: Path) -> str:
    digest = hashlib.sha512(path.read_bytes()).digest()
    return "sha512-" + base64.b64encode(digest).decode("ascii")


def static_fetcher(
    document: object | None,
) -> Callable[[str, Mapping[str, str]], object | None]:
    def fetch(url: str, headers: Mapping[str, str]) -> object | None:
        assert url.startswith("https://")
        assert headers["Accept"]
        return document

    return fetch


def pypi_document(package: str, version: str, artifacts: list[Path]) -> dict[str, object]:
    return {
        "info": {"name": package, "version": version},
        "urls": [
            {
                "filename": artifact.name,
                "digests": {"sha256": sha256(artifact)},
            }
            for artifact in artifacts
        ],
    }


def npm_version_document(package: str, version: str, artifact: Path) -> dict[str, object]:
    return {
        "name": package,
        "version": version,
        "dist": {"integrity": sha512_integrity(artifact)},
    }


def test_pypi_404_is_unpublished(tmp_path: Path) -> None:
    wheel = tmp_path / "codex_as_api-0.7.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")

    published = check_registry_release.check_pypi_release(
        "codex-as-api",
        "0.7.0",
        [wheel],
        fetch_json=static_fetcher(None),
    )

    assert published is False


def test_pypi_requires_exact_filenames_and_sha256(tmp_path: Path) -> None:
    wheel = tmp_path / "codex_as_api-0.7.0-py3-none-any.whl"
    sdist = tmp_path / "codex_as_api-0.7.0.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    document = pypi_document("codex-as-api", "0.7.0", [wheel, sdist])

    published = check_registry_release.check_pypi_release(
        "codex-as-api",
        "0.7.0",
        [wheel, sdist],
        fetch_json=static_fetcher(document),
    )

    assert published is True


@pytest.mark.parametrize("mutation", ["filename", "digest", "extra", "identity"])
def test_pypi_rejects_mismatched_registry_state(tmp_path: Path, mutation: str) -> None:
    wheel = tmp_path / "codex_as_api-0.7.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    document = pypi_document("codex-as-api", "0.7.0", [wheel])
    urls = document["urls"]
    assert isinstance(urls, list)
    if mutation == "filename":
        entry = urls[0]
        assert isinstance(entry, dict)
        entry["filename"] = "renamed.whl"
    elif mutation == "digest":
        entry = urls[0]
        assert isinstance(entry, dict)
        digests = entry["digests"]
        assert isinstance(digests, dict)
        digests["sha256"] = "0" * 64
    elif mutation == "extra":
        urls.append({"filename": "extra.tar.gz", "digests": {"sha256": "1" * 64}})
    else:
        document["info"] = {"name": "other-package", "version": "0.7.0"}

    with pytest.raises(check_registry_release.RegistryPreflightError):
        check_registry_release.check_pypi_release(
            "codex-as-api",
            "0.7.0",
            [wheel],
            fetch_json=static_fetcher(document),
        )


def test_npm_version_404_is_unpublished(tmp_path: Path) -> None:
    artifact = tmp_path / "codex-as-api-0.7.0.tgz"
    artifact.write_bytes(b"npm")

    published = check_registry_release.check_npm_release(
        "https://registry.npmjs.org",
        "codex-as-api",
        "0.7.0",
        artifact,
        fetch_json=static_fetcher(None),
    )

    assert published is False


def test_npm_requires_exact_name_version_and_sha512(tmp_path: Path) -> None:
    artifact = tmp_path / "codex-as-api-0.7.0.tgz"
    artifact.write_bytes(b"npm")
    document = npm_version_document("codex-as-api", "0.7.0", artifact)

    published = check_registry_release.check_npm_release(
        "https://registry.npmjs.org",
        "codex-as-api",
        "0.7.0",
        artifact,
        fetch_json=static_fetcher(document),
    )

    assert published is True


@pytest.mark.parametrize("mutation", ["name", "version", "integrity", "malformed"])
def test_npm_rejects_mismatched_or_malformed_registry_state(tmp_path: Path, mutation: str) -> None:
    artifact = tmp_path / "codex-as-api-0.7.0.tgz"
    artifact.write_bytes(b"npm")
    document: object = npm_version_document("codex-as-api", "0.7.0", artifact)
    if mutation == "name":
        document["name"] = "other-package"  # type: ignore[index]
    elif mutation == "version":
        document["version"] = "0.7.1"  # type: ignore[index]
    elif mutation == "integrity":
        document["dist"]["integrity"] = "sha512-wrong"  # type: ignore[index]
    else:
        document = []

    with pytest.raises(check_registry_release.RegistryPreflightError):
        check_registry_release.check_npm_release(
            "https://registry.npmjs.org",
            "codex-as-api",
            "0.7.0",
            artifact,
            fetch_json=static_fetcher(document),
        )


def test_github_packages_packument_supports_new_and_identical_versions(tmp_path: Path) -> None:
    artifact = tmp_path / "eunho-j-codex-as-api-0.7.0.tgz"
    artifact.write_bytes(b"github package")
    package = "@eunho-j/codex-as-api"
    unpublished = {"name": package, "versions": {"0.6.5": {}}}
    published = {
        "name": package,
        "versions": {"0.7.0": npm_version_document(package, "0.7.0", artifact)},
    }

    assert (
        check_registry_release.check_npm_release(
            "https://npm.pkg.github.com",
            package,
            "0.7.0",
            artifact,
            token="secret",
            packument=True,
            fetch_json=static_fetcher(unpublished),
        )
        is False
    )
    assert (
        check_registry_release.check_npm_release(
            "https://npm.pkg.github.com",
            package,
            "0.7.0",
            artifact,
            token="secret",
            packument=True,
            fetch_json=static_fetcher(published),
        )
        is True
    )


def test_github_packages_rejects_a_malformed_target_version(tmp_path: Path) -> None:
    artifact = tmp_path / "eunho-j-codex-as-api-0.7.0.tgz"
    artifact.write_bytes(b"github package")
    package = "@eunho-j/codex-as-api"

    with pytest.raises(check_registry_release.RegistryPreflightError):
        check_registry_release.check_npm_release(
            "https://npm.pkg.github.com",
            package,
            "0.7.0",
            artifact,
            token="secret",
            packument=True,
            fetch_json=static_fetcher({"name": package, "versions": {"0.7.0": None}}),
        )


def test_github_packages_uses_bearer_without_exposing_it(tmp_path: Path) -> None:
    artifact = tmp_path / "eunho-j-codex-as-api-0.7.0.tgz"
    artifact.write_bytes(b"github package")
    token = "never-print-this-token"

    def fetch(url: str, headers: Mapping[str, str]) -> object | None:
        assert url == "https://npm.pkg.github.com/@eunho-j%2Fcodex-as-api"
        assert headers["Authorization"] == f"Bearer {token}"
        raise check_registry_release.RegistryPreflightError("registry request returned HTTP 401")

    with pytest.raises(check_registry_release.RegistryPreflightError) as captured:
        check_registry_release.check_npm_release(
            "https://npm.pkg.github.com",
            "@eunho-j/codex-as-api",
            "0.7.0",
            artifact,
            token=token,
            packument=True,
            fetch_json=fetch,
        )

    assert str(captured.value) == "registry request returned HTTP 401"


class FakeResponse:
    def __init__(self, status: int, content: bytes) -> None:
        self.status = status
        self.content = content

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self.content


def test_http_layer_treats_only_404_as_absent() -> None:
    def not_found(request: Request, timeout: int) -> FakeResponse:
        raise HTTPError(request.full_url, 404, "missing", Message(), None)

    def unauthorized(request: Request, timeout: int) -> FakeResponse:
        raise HTTPError(request.full_url, 401, "unauthorized", Message(), None)

    assert (
        check_registry_release._request_json(  # noqa: SLF001
            "https://registry.example/package/0.7.0",
            {},
            http_get=lambda url, headers: check_registry_release._http_get(  # noqa: SLF001
                url,
                headers,
                opener=not_found,
            ),
        )
        is None
    )
    with pytest.raises(check_registry_release.RegistryPreflightError) as captured:
        check_registry_release._request_json(  # noqa: SLF001
            "https://registry.example/package/0.7.0",
            {},
            http_get=lambda url, headers: check_registry_release._http_get(  # noqa: SLF001
                url,
                headers,
                opener=unauthorized,
            ),
        )
    assert str(captured.value) == "registry request returned HTTP 401"


def test_http_layer_rejects_malformed_json_and_sanitizes_network_errors() -> None:
    def malformed(request: Request, timeout: int) -> FakeResponse:
        return FakeResponse(200, b"\xff")

    def network_failure(request: Request, timeout: int) -> FakeResponse:
        raise URLError("never-print-this-token")

    with pytest.raises(check_registry_release.RegistryPreflightError) as malformed_error:
        check_registry_release._request_json(  # noqa: SLF001
            "https://registry.example/package/0.7.0",
            {},
            http_get=lambda url, headers: check_registry_release._http_get(  # noqa: SLF001
                url,
                headers,
                opener=malformed,
            ),
        )
    assert str(malformed_error.value) == "registry returned malformed UTF-8 JSON"

    with pytest.raises(check_registry_release.RegistryPreflightError) as network_error:
        check_registry_release._http_get(  # noqa: SLF001
            "https://registry.example/package/0.7.0",
            {"Authorization": "Bearer never-print-this-token"},
            opener=network_failure,
        )
    assert str(network_error.value) == "registry request failed (URLError)"


def test_github_release_only_accepts_404_or_the_exact_tag() -> None:
    assert (
        check_registry_release.check_github_release(
            "Eunho-J/codex-as-api",
            "v0.7.0",
            "secret",
            fetch_json=static_fetcher(None),
        )
        is False
    )
    assert (
        check_registry_release.check_github_release(
            "Eunho-J/codex-as-api",
            "v0.7.0",
            "secret",
            fetch_json=static_fetcher({"tag_name": "v0.7.0"}),
        )
        is True
    )
    with pytest.raises(check_registry_release.RegistryPreflightError):
        check_registry_release.check_github_release(
            "Eunho-J/codex-as-api",
            "v0.7.0",
            "secret",
            fetch_json=static_fetcher({"tag_name": "v0.7.1"}),
        )


def test_github_release_asset_prune_only_deletes_unexpected_assets(tmp_path: Path) -> None:
    expected = tmp_path / "codex-as-api-linux"
    expected.write_bytes(b"linux")
    release = {
        "tag_name": "v0.7.0",
        "assets": [
            {"id": 10, "name": expected.name},
            {"id": 11, "name": "$(unsafe asset name)"},
        ],
    }
    deleted: list[str] = []

    def delete_asset(url: str, headers: Mapping[str, str]) -> None:
        assert headers["Authorization"] == "Bearer secret"
        deleted.append(url)

    check_registry_release.prune_github_release_assets(
        "Eunho-J/codex-as-api",
        "v0.7.0",
        "secret",
        [expected],
        fetch_json=static_fetcher(release),
        delete_asset=delete_asset,
    )

    assert deleted == ["https://api.github.com/repos/Eunho-J/codex-as-api/releases/assets/11"]


def test_github_release_asset_prune_validates_all_ids_before_deleting(tmp_path: Path) -> None:
    expected = tmp_path / "codex-as-api-linux"
    expected.write_bytes(b"linux")
    deleted: list[str] = []

    with pytest.raises(check_registry_release.RegistryPreflightError):
        check_registry_release.prune_github_release_assets(
            "Eunho-J/codex-as-api",
            "v0.7.0",
            "secret",
            [expected],
            fetch_json=static_fetcher(
                {
                    "tag_name": "v0.7.0",
                    "assets": [
                        {"id": 11, "name": "first-extra"},
                        {"id": None, "name": "invalid-extra"},
                    ],
                }
            ),
            delete_asset=lambda url, headers: deleted.append(url),
        )

    assert deleted == []


def test_github_release_asset_verification_requires_exact_names_and_sha256(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "codex-as-api-linux"
    expected.write_bytes(b"linux")
    release = {
        "tag_name": "v0.7.0",
        "assets": [
            {
                "id": 10,
                "name": expected.name,
                "state": "uploaded",
                "digest": f"sha256:{sha256(expected)}",
            }
        ],
    }

    check_registry_release.verify_github_release_assets(
        "Eunho-J/codex-as-api",
        "v0.7.0",
        "secret",
        [expected],
        fetch_json=static_fetcher(release),
    )


@pytest.mark.parametrize("mutation", ["extra", "missing", "digest", "state", "duplicate"])
def test_github_release_asset_verification_rejects_non_exact_remote_state(
    tmp_path: Path,
    mutation: str,
) -> None:
    expected = tmp_path / "codex-as-api-linux"
    expected.write_bytes(b"linux")
    asset = {
        "id": 10,
        "name": expected.name,
        "state": "uploaded",
        "digest": f"sha256:{sha256(expected)}",
    }
    assets: list[dict[str, object]] = [asset]
    if mutation == "extra":
        assets.append({"id": 11, "name": "extra", "state": "uploaded", "digest": "sha256:0"})
    elif mutation == "missing":
        assets.clear()
    elif mutation == "digest":
        asset["digest"] = "sha256:" + "0" * 64
    elif mutation == "state":
        asset["state"] = "open"
    else:
        assets.append(dict(asset))

    with pytest.raises(check_registry_release.RegistryPreflightError):
        check_registry_release.verify_github_release_assets(
            "Eunho-J/codex-as-api",
            "v0.7.0",
            "secret",
            [expected],
            fetch_json=static_fetcher({"tag_name": "v0.7.0", "assets": assets}),
        )


def test_http_delete_requires_github_no_content_without_exposing_token() -> None:
    def forbidden(request: Request, timeout: int) -> FakeResponse:
        assert request.get_method() == "DELETE"
        assert request.headers["Authorization"] == "Bearer never-print-this-token"
        return FakeResponse(403, b'{"message":"never-print-this-token"}')

    with pytest.raises(check_registry_release.RegistryPreflightError) as captured:
        check_registry_release._http_delete(  # noqa: SLF001
            "https://api.github.com/repos/Eunho-J/codex-as-api/releases/assets/11",
            {"Authorization": "Bearer never-print-this-token"},
            opener=forbidden,
        )

    assert str(captured.value) == "GitHub asset deletion returned HTTP 403"


def test_github_output_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "github-output"

    check_registry_release.write_github_output(False, str(output))
    check_registry_release.write_github_output(True, str(output))

    assert output.read_text(encoding="utf-8") == "published=false\npublished=true\n"
