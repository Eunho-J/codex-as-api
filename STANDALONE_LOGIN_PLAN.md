# Standalone subscription login — implementation contract

Status: implementation authorized by Eunho in voice on 2026-10-29. Dedicated-store foundation is in progress. Device-code-first login has been proposed to the user; confirmation is still pending. Do not publish before integration review/tests and a final deployment check.

## Confirmed scope

- Remove the installed Codex CLI requirement and all automatic reading/import/fallback of Codex credential files.
- Own the project's new credentials and token refresh lifecycle. Never change the existing Codex credential store.
- Expose login initiation, progress and cancellation through the API; never return access/refresh/ID tokens.
- Preserve the repository's existing Python, TypeScript and Rust parity and existing inference/model behavior.
- Existing Codex model config discovery is separate from credentials; do not broaden this work into model/config changes.

## Shared store foundation

Resolution: explicit constructor/CLI store path > CODEX_AS_API_AUTH_PATH > CODEX_AS_API_HOME/auth.json > ~/.codex-as-api/auth.json.

An explicit path is a project-store configuration, not an import command. No CODEX_HOME credential discovery, legacy credential fallback or automatic migration. Existing internal token parsing and refresh protections may be reused; old credential files may not be reused automatically. New directories are private; writes use exclusively created private temporary files and atomic replacement. No secrets in errors or logs. A failed/cancelled new login must not destroy an existing valid project login.

## Proposed minimal API (awaiting device-first confirmation)

All four routes use one independent CODEX_AS_API_ADMIN_KEY bearer credential. An unset key makes management routes unavailable, not publicly writable. Preserve existing inference authentication semantics. Responses use Cache-Control: no-store. Do not accept OAuth tokens, credential paths, issuer, client ID or redirect URI from a management request.

- POST /auth/login: start an explicit device-code login. Return 202 with id, pending status, verification_url, user_code, expires_at and interval. Only the initiating response contains the user code. One active login per store; a conflicting start returns 409 rather than cancelling the first.
- GET /auth/login/{id}: return pending/succeeded/failed/cancelled/expired and safe timestamps/error codes; no token, authorization code, verifier, device-auth ID or file paths. A success means the new store was durably committed.
- DELETE /auth/login/{id}: cancel a pending attempt, with a generation fence against late exchange/commit. Repeat cancellation is idempotent. An already committed login is reported as completed, not falsely removed.
- GET /auth/status: return minimal project-login state without secrets or token export. Document that local credential presence is not proof of provider-side validity/revocation status.

Use the pinned official rust-v0.164.0 device authorization protocol, not a generic RFC device grant. Fixed OpenAI endpoints/client identity; HTTP-client/clock injection for isolated tests. Respect polling intervals and a real total expiry deadline; cancellation must interrupt pending polling/requests and prevent late credential writes. Initial usercode 404 is a clear unsupported-device-flow result, not an automatic browser fallback. License and modified-source attribution must accompany each published artifact.

## Verification and release

- Fake issuer/transport tests for pending, success, malformed responses, denial, initial 404 versus poll 403/404, expiry, cancellation and completion races.
- Management authorization, no cross-origin credential-changing request, no secret/code leakage, and no-store responses.
- No-CLI and forbidden-legacy-path guards with a disposable HOME; provider/health/realtime integration using project-owned fake credentials.
- Refresh rotation/concurrency and account/generation protection; safe initial and replacement storage failure paths.
- Full existing suites/build/type/lint checks across the three implementations, then artifact/version/license inspection and the existing release pipeline.
- Real account approval is performed by the human on the provider page. Do not display device codes or tokens in public channels. Clearly distinguish mocked success from a real live login.

## Work ownership

- TypeScript: df361dd4-fcb9-45d8-8ede-e2cf80bfdcaf, ts/ only.
- Python: edd80bf8-9f7a-4274-b46f-41cad43b8cc6, src/codex_as_api/ and tests/ only.
- Rust: 3097c352-aef2-411c-a87a-8b0ebe30e712, rust/ only.
- Parent: shared contract, root documentation/versions/licenses/release workflow, integration review and deployment. Only parent commits or changes branches.
