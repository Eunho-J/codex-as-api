# Standalone subscription login — implementation contract

Status: Eunho resumed implementation at 18:27, explicitly requested deployment plus Melchior handoff at 18:47, and at 18:50 APPROVED CHANGING THE DEFAULT TO ORDINARY BROWSER OAUTH LOGIN. Implement browser PKCE/loopback login as default and device-code login as an explicit remote/headless alternative, in all three languages. Review/test/package/deploy, then hand off the app-login integration to Melchior. No Codex install instructions. Foundation checkpoint 41ef6ca. Do not publish before integration review/tests and final deployment verification.

## Confirmed scope

- Remove the installed Codex CLI requirement and all automatic reading/import/fallback of Codex credential files.
- Own the project's new credentials and token refresh lifecycle. Never change the existing Codex credential store.
- Expose login initiation, progress and cancellation through the API; never return access/refresh/ID tokens.
- Preserve the repository's existing Python, TypeScript and Rust parity and existing inference/model behavior.
- Existing Codex model config discovery is separate from credentials; do not broaden this work into model/config changes.

## Shared store foundation

Resolution: explicit constructor/CLI store path > CODEX_AS_API_AUTH_PATH > CODEX_AS_API_HOME/auth.json > ~/.codex-as-api/auth.json.

An explicit path is a project-store configuration, not an import command. No CODEX_HOME credential discovery, legacy credential fallback or automatic migration. Existing internal token parsing and refresh protections may be reused; old credential files may not be reused automatically. New directories are private; writes use exclusively created private temporary files and atomic replacement. No secrets in errors or logs. A failed/cancelled new login must not destroy an existing valid project login.

## Agreed implementation API (18:50 browser-default approval supersedes device-first)

All four routes use one independent CODEX_AS_API_ADMIN_KEY bearer credential. An unset key makes management routes unavailable, not publicly writable. Preserve existing inference authentication semantics. Responses use Cache-Control: no-store. Do not accept OAuth tokens, credential paths, issuer, client ID or redirect URI from a management request.

- POST /auth/login: empty body, {} or {"method":"browser"} starts ordinary browser OAuth; {"method":"device"} explicitly selects device-code login. Return 202 with id, method, pending status and expires_at. Browser adds auth_url; device adds verification_url, user_code and interval. Only this initiating response contains the authorization URL/user code. One active login per store across BOTH methods; a conflicting start returns 409 rather than cancelling the first.
- GET /auth/login/{id}: return pending/succeeded/failed/cancelled/expired and safe timestamps/error codes; no token, authorization code, verifier, device-auth ID or file paths. A success means the new store was durably committed.
- DELETE /auth/login/{id}: cancel a pending attempt, with a generation fence against late exchange/commit. Repeat cancellation is idempotent. An already committed login is reported as completed, not falsely removed.
- GET /auth/status: return minimal project-login state without secrets or token export. Document that local credential presence is not proof of provider-side validity/revocation status.

Use the pinned official rust-v0.164.0 browser and device authorization protocols, not an invented OAuth grant. Fixed OpenAI endpoints/client identity; HTTP-client/clock/listener injection for isolated tests. Respect polling intervals and real total expiry. Cancellation interrupts pending callbacks/polling/requests and prevents late writes. Neither method silently falls back to the other. License and modified-source attribution accompany each artifact.

## Browser OAuth default (approved 18:50)

- Source /Users/cayde/Documents/Scheherazade/evidence/codex-01640-source/codex-rs/login/src/server.rs and oauth/{authorization,pkce,client}.rs. Fixed issuer/client as below. Generate cryptographically random state and PKCE verifier; S256 base64url challenge. GET https://auth.openai.com/oauth/authorize with response_type=code, client_id, redirect_uri, state, code_challenge, code_challenge_method=S256, scopes `openid profile email offline_access api.connectors.read api.connectors.invoke`, and official id_token_add_organizations=true, codex_cli_simplified_flow=true, originator=codex_cli_rs. No optional workspace switching/API-key mint/TUI port.
- Bind a separate IPv4 loopback listener 127.0.0.1:1455, falling back only to registered 1457 if occupied; redirect_uri is exactly http://127.0.0.1:<bound-port>/auth/callback. If both unavailable return 409 callback_port_unavailable, do not cancel a port owner or choose arbitrary production ports. Internal tests may inject listeners/ephemeral port lists; no request/env overrides of registered callback/issuer/client. No automatic browser launch inside the API server: return auth_url for the trusted app to open. Melchior will wire Scheherazade's login button to opening this URL.
- GET /auth/callback only, expected exact Host, bounded query/request line. Require exactly one correct state BEFORE processing code or error; reject duplicate state/code/error, missing/empty state, wrong method/Host/state and conflicting code+error without completing a valid session. A valid-state denial safely fails that session. Unknown paths cannot claim success. No secret values or reflected arbitrary text in HTML/errors/logs, Cache-Control:no-store and Referrer-Policy:no-referrer. State and authorization URLs are secret temporary material, returned only to the authenticated initiator.
- Exchange code once at fixed /oauth/token with form grant_type=authorization_code, client_id, code, EXACT loopback redirect_uri and original code_verifier. No retry after ambiguous failure. Validate and commit through same private writer/generation protections as device flow. Browser callback success page and succeeded state only AFTER durable write; failed/cancelled/expired close the listener and reject late callbacks. One request timeout bounded by remaining total deadline; all body reads and commit-before-rename guard count against 900s lifetime. Don't mark a completed durable commit cancelled.
- Minimal tests use fake issuer/transport and actual loopback callback: authorize URL/PKCE/redirect fields, happy callback -> durable real temp store -> succeeded, state mismatch/duplicate/error/code+error, listener preferred/fallback/busy, cancel/expiry closes listener, callback/exchange race no late writes, no success via unrelated URL. Never launch real browser or contact real auth/OpenAI; the human performs actual account approval later.

## Cross-language behavior

- Request bodies for POST /auth/login are empty, {} or exactly {"method":"browser"|"device"}; browser is DEFAULT. Reject unknown methods/extra fields with 400 invalid_request; no client-supplied credential, path or upstream endpoint. HTTP 503 management_disabled when CODEX_AS_API_ADMIN_KEY is absent, 401 invalid_admin_key for absent/wrong bearer, 403 origin_not_allowed for a supplied Origin other than the request's own scheme+Host. Preserve inference auth behavior.
- Normal session GET/DELETE responses: 200 with id, method, status, expires_at, and terminal error_code if any. Pre-rename storage failure uses login_storage_error; replacement followed by failed durability sync uses login_persistence_uncertain (failed, not expired/cancelled; visible new credentials may exist). Unknown id: 404 login_not_found. Concurrent active start: 409 login_in_progress. Errors use {"error":{"code":"...","message":"..."}} with generic safe messages. Return Cache-Control: no-store for management successes and failures.
- GET /auth/status returns {"status":"authenticated"|"unauthenticated"|"reauthentication_required"}. This is a local credential check only: it does not contact the issuer and does not prove remote validity. A valid refreshable project credential may be considered authenticated; unusable/malformed existing credentials need reauthentication. No account/token/path payload.
- Fixed issuer https://auth.openai.com and client ID app_EMoamEEZ73f0CkXaXp7hrann; JSON usercode/poll requests, form-urlencoded authorization_code exchange with redirect_uri https://auth.openai.com/deviceauth/callback. Source: /Users/cayde/Documents/Scheherazade/evidence/codex-01640-source/codex-rs/login/src/device_code_auth.rs (rust-v0.164.0). Accept user_code/usercode. Poll interval accepts integer seconds or digit strings; absent defaults to 5; reject nonpositive/malformed values rather than spin. A total 900-second monotonic deadline includes requests and exchange, not just sleeps. Per-request timeout is bounded by remaining deadline. Do not follow redirects; sanitize upstream failures without returning their bodies.
- Initial upstream 404 => device_login_unsupported (502); other initial network/upstream failures => login_upstream_error (502); malformed upstream => invalid_login_response (502). Background poll 403/404 => pending; other failure => failed with safe error_code. Provider denial/expiry must not destroy prior valid login. No automatic retry of ambiguous authorization-code exchange.
- Credentials become visible only after validation and a private atomic durable write. Reuse existing auth parser/writer/refresh protections, adding new-login ownership generation fencing as needed so old refresh cannot replace a fresh login (including same-account relogin). Cancellation and expiration prevent late token commits; a committed result cannot be relabelled cancelled. Close/shutdown cancels outstanding work. Retain a small bounded set of terminal records for client polling; do not build a persistent job system. Supported deployment is one server process per credential store; cross-process store sharing is not claimed. Reserve the active slot before the initial request. A failure after rename is an ambiguous durability failure, not proof that old credentials were preserved.
- Tests inject transport/clock/timings internally; production does not accept issuer/client/path overrides in management requests. Never touch the real Codex store, real account keys, installed Scheherazade app, or perform live device-code requests. Use disposable HOME and fake credentials; /opt/homebrew/bin/node is the independent Node, not the app's Electron substitute.

## Verification and release

- Fake issuer/transport tests for pending, success, malformed responses, denial, initial 404 versus poll 403/404, expiry, cancellation and completion races.
- Management authorization, no cross-origin credential-changing request, no secret/code leakage, and no-store responses.
- No-CLI and forbidden-legacy-path guards with a disposable HOME; provider/health/realtime integration using project-owned fake credentials.
- Refresh rotation/concurrency and account/generation protection; safe initial and replacement storage failure paths.
- Full existing suites/build/type/lint checks across the three implementations, then artifact/version/license inspection and the existing release pipeline.
- Real account approval is performed by the human on the provider page. Do not display device codes or tokens in public channels. Clearly distinguish mocked success from a real live login.

## Work ownership

- TypeScript: c4a6c1e8-2e3f-49eb-aeb7-de27e97254dc, ts/ only.
- Python: ca6d5b14-4a59-4cc8-b2f3-e79dde7a716a, src/codex_as_api/ and tests/ only.
- Rust: 234167b8-69cf-4011-962d-1b56cf724ed0, rust/ only.
- 18:58 human constraint: no excessive tests and no regex tests. Keep focused behavior checks of actual inputs/outputs/state, and consolidate redundant new checks rather than growing a broad defensive test matrix.
- Parent: shared contract, root documentation/versions/licenses/release workflow, integration review and deployment. Only parent commits or changes branches.
